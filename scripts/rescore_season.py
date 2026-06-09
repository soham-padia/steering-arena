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
    """Replicates app.main.get_scorer for the current (target-season) settings."""
    from app import scoring
    from app.ndif_client import ResidualReader

    d, meta = scoring.load_direction(settings.d_file)
    probes = scoring.load_probes(settings.probe_set)
    reader = ResidualReader.from_settings(settings)
    hidden = reader.hidden_size
    if hidden is not None and hidden != d.shape[0]:
        raise SystemExit(f"d dim {d.shape[0]} != model hidden {hidden} — wrong season config?")

    def count_tokens(text):
        return len(reader.tokenizer(text, add_special_tokens=False)["input_ids"])

    def batch_fn(texts):
        return reader.batch_last_resids(texts, settings.layer)

    base_cos = scoring.baseline_cosines(probes, batch_fn, d)

    def score_fn(seq):
        return scoring.steering_shift_batched(seq, probes, batch_fn, base_cos, d)

    return count_tokens, score_fn


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
    print(f"scoring with settings: model={settings.model_id} layer={settings.layer} "
          f"d={settings.d_file.split('/')[-1]} probes={settings.probe_set.split('/')[-1]}\n")

    rows = (c.table("submissions").select("user_handle, sequence_text, norm_key, token_count")
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
            score = float(score_fn(seq))
            tok = count_tokens(seq)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[{i}/{len(rows)}] FAIL {type(e).__name__}: {str(e)[:80]} — {seq[:40]!r}", flush=True)
            continue
        flag = "  ⚠over-budget" if tok > budget else ""
        if tok > budget:
            over += 1
        print(f"[{i}/{len(rows)}] {score:+.5f}  ({tok} tok){flag}  {r['user_handle']!r}: {seq[:45]!r}", flush=True)
        if not args.dry_run:
            c.table("submissions").insert({
                "season_id": args.to_season_id, "user_handle": r["user_handle"],
                "sequence_text": seq, "norm_key": key, "token_count": tok,
                "score": score, "ip_hash": RESCORE_IP,
            }).execute()
            done += 1

    print(f"\nrescored: inserted={done} skipped={skipped} over_budget={over} failed={failed}")


if __name__ == "__main__":
    main()
