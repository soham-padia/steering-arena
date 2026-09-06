"""Rescore one season's submissions into another (e.g. seed Season 2 with Season 1's
sequences, scored under the new model/layer/direction).

Uses the EXACT live scoring path (app.scoring + ResidualReader.from_settings), so the
migrated scores match what the server would compute. The target season's config is
taken from the current settings/.env (must already point at the target season's
model/layer/d/probes). Resumable: skips sequences already present in the target season
(unique (season_id, norm_key)). Historical rows get a sentinel ip_hash so the migration
doesn't perturb real users' rate-limit counts.

    # settings/.env must be the Season 2 config (OLMo-3-32B, layer 24, L24 d, season2 probes)
    python scripts/rescore_season.py --from-season-id 3 --to-season-id 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings  # noqa: E402

RESCORE_IP = "season1-rescore"  # sentinel so migrated rows don't pollute per-IP limits


def build_scorer():
    """Replicates app.main.get_scorer for the current (target-season) settings.

    Mirrors BOTH branches: the single-layer path, and Season 3's banded path when
    settings.banded() is true. It must stay in step with app/main.py — a rescore that
    scores differently from the live scorer would fill a board with numbers no future
    submission can be compared against, which is the season invariant broken from the
    inside.
    """
    from app import scoring
    from app.ndif_client import ResidualReader

    probes = scoring.load_probes(settings.probe_set)
    reader = ResidualReader.from_settings(settings)
    hidden = reader.hidden_size

    def count_tokens(text):
        return len(reader.tokenizer(text, add_special_tokens=False)["input_ids"])

    if settings.banded():
        d1, per1, band1, _ = scoring.load_banded_direction(settings.score1_d_file)
        d2, per2, band2, _ = scoring.load_banded_direction(settings.score2_d_file)
        if hidden is not None and hidden != d1.shape[0]:
            raise SystemExit(f"d dim {d1.shape[0]} != model hidden {hidden} — wrong season config?")
        if band1 != settings.band1() or band2 != settings.band2():
            raise SystemExit(f"band mismatch: config {settings.band1()}/{settings.band2()} "
                             f"vs d files {band1}/{band2}")
        read = settings.read_layers()
        pos = {L: i for i, L in enumerate(read)}
        last = {"texts": None, "mat": None}

        def batch_layers_fn(texts, layers):
            # One remote call per text batch, sliced per band — same trick as main.py, and
            # the reason score2 costs no extra NDIF quota during a 618-row rescore.
            key = tuple(texts)
            if last["texts"] != key:
                last["mat"] = reader.batch_last_resids_layers(list(texts), read)
                last["texts"] = key
            return last["mat"][[pos[L] for L in layers]]

        base1 = scoring.banded_baseline(probes, batch_layers_fn, band1)
        base2 = scoring.banded_baseline(probes, batch_layers_fn, band2)

        def score_fn(seq):
            s1, z = scoring.banded_shift_and_specificity(
                seq, probes, batch_layers_fn, band1, base1, d1, eps=settings.specificity_eps)
            s2 = scoring.banded_shift(seq, probes, batch_layers_fn, band2, base2, d2,
                                      per_layer=per2, aggregate=scoring.PER_LAYER_MIN)
            return scoring.ScoreResult(s1, s1, z if settings.specificity_enabled else None, s2)

        return count_tokens, score_fn

    d, meta = scoring.load_direction(settings.d_file)
    if hidden is not None and hidden != d.shape[0]:
        raise SystemExit(f"d dim {d.shape[0]} != model hidden {hidden} — wrong season config?")

    def batch_fn(texts):
        return reader.batch_last_resids(texts, settings.layer)

    base_units = scoring.baseline_unit_rows(probes, batch_fn)

    def score_fn(seq):
        shift, z = scoring.shift_and_specificity(seq, probes, batch_fn, base_units, d,
                                                 eps=settings.specificity_eps)
        ranked = z if settings.scoring_mode == scoring.SPECIFICITY_Z else shift
        return scoring.ScoreResult(ranked, shift, z)

    return count_tokens, score_fn


def assert_settings_match_season(dst):
    """SKILL.md step 2: the scorer config must EQUAL the destination season row.

    This script previously only PRINTED both and trusted the operator to compare them.
    Nothing stopped a rescore running with Season 2's layer and d against a Season 3 row,
    which would silently fill the new board with old-metric numbers — indistinguishable
    from a correct run by inspection, and only detectable much later by someone noticing
    the scores looked familiar.
    """
    problems = []
    if dst["model_id"] != settings.model_id:
        problems.append(f"model_id: season {dst['model_id']!r} vs settings {settings.model_id!r}")
    if settings.banded():
        declared = [int(x) for x in str(dst.get("layers") or "").split(",") if x.strip()]
        if declared != settings.band1():
            problems.append(f"layers: season {declared} vs settings score1_layers {settings.band1()}")
        if dst.get("scoring_mode") != scoring_mode_expected():
            problems.append(f"scoring_mode: season {dst.get('scoring_mode')!r} vs "
                            f"expected {scoring_mode_expected()!r} for a banded season")
    else:
        if int(dst["layer"]) != int(settings.layer):
            problems.append(f"layer: season {dst['layer']} vs settings {settings.layer}")
    if problems:
        raise SystemExit("REFUSING TO RESCORE — settings do not match the target season:\n  "
                         + "\n  ".join(problems)
                         + "\n\nRun scripts/check_season_matches_d.py, and set the bands in .env.")


def scoring_mode_expected():
    from app import scoring
    return scoring.BANDED_MULTILAYER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-season-id", type=int, required=True)
    ap.add_argument("--to-season-id", type=int, required=True)
    ap.add_argument("--limit", type=int, default=0, help="cap N sequences (smoke test)")
    ap.add_argument("--dry-run", action="store_true", help="score + print, don't insert")
    args = ap.parse_args()

    from supabase import create_client
    c = create_client(settings.supabase_url, settings.supabase_service_key)

    src = c.table("seasons").select("*").eq("id", args.from_season_id).single().execute().data
    dst = c.table("seasons").select("*").eq("id", args.to_season_id).single().execute().data
    print(f"FROM season {src['id']} {src['name']!r} ({src['model_id']} L{src['layer']})")
    print(f"  TO season {dst['id']} {dst['name']!r} ({dst['model_id']} L{dst['layer']} {dst['d_version']})")
    # Print the d files the scorer WILL use. On a banded season that is score1/score2_d_file;
    # printing settings.d_file there names the unused single-layer direction and makes the
    # log look like the run used the wrong vector.
    if settings.banded():
        print(f"scoring with settings: model={settings.model_id} BANDED "
              f"score1={settings.band1()} d={settings.score1_d_file.split('/')[-1]} | "
              f"score2={settings.band2()} d={settings.score2_d_file.split('/')[-1]} | "
              f"probes={settings.probe_set.split('/')[-1]}\n")
    else:
        print(f"scoring with settings: model={settings.model_id} layer={settings.layer} "
              f"d={settings.d_file.split('/')[-1]} probes={settings.probe_set.split('/')[-1]}\n")

    assert_settings_match_season(dst)

    # research_consent / consent_version carried over: a rescore is the SAME person's
    # submission re-measured, not a new act of publication, so silently defaulting them to
    # false would drop every carried-over row out of export_research_data.py. created_at
    # carried over too — the leaderboard tie-breaks on it, so stamping now() would reorder
    # equal scores by migration order rather than by when they were actually submitted.
    rows = (c.table("submissions")
            .select("user_handle, sequence_text, norm_key, token_count, "
                    "research_consent, consent_version, created_at")
            .eq("season_id", args.from_season_id).order("created_at").execute().data)
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} source submissions\n", flush=True)

    count_tokens, score_fn = build_scorer()
    budget = dst.get("token_budget") or settings.token_budget
    done = skipped = over = failed = 0
    for i, r in enumerate(rows, 1):
        seq, key = r["sequence_text"], r["norm_key"]
        if c.table("submissions").select("id", count="exact", head=True).eq("season_id", args.to_season_id).eq("norm_key", key).execute().count:
            skipped += 1
            print(f"[{i}/{len(rows)}] skip (already in target): {seq[:50]!r}", flush=True)
            continue
        try:
            res = score_fn(seq)
            # ScoreResult (NamedTuple) or legacy plain float.
            score_alt = None
            if isinstance(res, tuple):
                score, shift_raw = float(res[0]), float(res[1])
                spec = float(res[2]) if res[2] is not None else None
                if len(res) > 3 and res[3] is not None:
                    score_alt = float(res[3])
            else:
                score, shift_raw, spec = float(res), float(res), None
            tok = count_tokens(seq)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[{i}/{len(rows)}] FAIL {type(e).__name__}: {str(e)[:80]} — {seq[:40]!r}", flush=True)
            continue
        flag = "  ⚠over-budget" if tok > budget else ""
        if tok > budget:
            over += 1
        spec_s = f" z={spec:+.1f}" if spec is not None else ""
        print(f"[{i}/{len(rows)}] {score:+.5f}{spec_s}  ({tok} tok){flag}  {r['user_handle']!r}: {seq[:45]!r}", flush=True)
        if not args.dry_run:
            c.table("submissions").insert({
                "season_id": args.to_season_id, "user_handle": r["user_handle"],
                "sequence_text": seq, "norm_key": key, "token_count": tok,
                "score": score, "shift_raw": shift_raw, "specificity": spec,
                "score_alt": score_alt,
                "ip_hash": RESCORE_IP,
                "research_consent": bool(r.get("research_consent")),
                "consent_version": r.get("consent_version"),
                "created_at": r.get("created_at"),
            }).execute()
            done += 1

    print(f"\nrescored: inserted={done} skipped={skipped} over_budget={over} failed={failed}")


if __name__ == "__main__":
    main()
