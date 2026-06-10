"""Backfill `shift_raw` + `specificity` for existing Season rows (Track 1, Option A).

Recomputes both values through the exact live scoring path for every submission in the
target season where `specificity is null`, and UPDATEs ONLY those two columns — never
`score`, so the season's ranking is untouched. Resumable by construction (the null
filter). Logs |shift_recomputed − stored score| as an NDIF-drift canary: this season's
`score` IS the raw shift, so any gap beyond float/NDIF tolerance (1e-3, spec §5.4/§14)
means the served model drifted since the row was scored.

    # settings/.env must match the season being backfilled (model/layer/d/probes)
    python scripts/backfill_specificity.py --season-id 4 --limit 3 --dry-run   # smoke
    python scripts/backfill_specificity.py --season-id 4                       # full
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings  # noqa: E402
from scripts.rescore_season import build_scorer  # noqa: E402 — same live-path scorer

DRIFT_WARN = 1e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season-id", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="cap rows (smoke test)")
    ap.add_argument("--dry-run", action="store_true", help="compute + print, don't update")
    args = ap.parse_args()

    from supabase import create_client
    c = create_client(settings.supabase_url, settings.supabase_service_key)

    season = c.table("seasons").select("*").eq("id", args.season_id).single().execute().data
    print(f"backfilling season {season['id']} {season['name']!r} ({season['model_id']} "
          f"L{season['layer']} {season['d_version']})")
    print(f"scorer settings: model={settings.model_id} layer={settings.layer} "
          f"d={settings.d_file.split('/')[-1]} probes={settings.probe_set.split('/')[-1]}\n")

    rows = (c.table("submissions").select("id, sequence_text, score")
            .eq("season_id", args.season_id).is_("specificity", "null")
            .order("created_at").execute().data)
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} rows to backfill (specificity is null)\n", flush=True)
    if not rows:
        return

    _, score_fn = build_scorer()
    done = failed = drifted = 0
    for i, r in enumerate(rows, 1):
        seq = r["sequence_text"]
        try:
            res = score_fn(seq)  # ScoreResult
            shift, spec = float(res[1]), float(res[2])
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[{i}/{len(rows)}] FAIL {type(e).__name__}: {str(e)[:80]} — {seq[:40]!r}", flush=True)
            continue
        drift = abs(shift - float(r["score"]))
        flag = ""
        if drift > DRIFT_WARN:
            drifted += 1
            flag = f"  ⚠drift {drift:.2e}"
        print(f"[{i}/{len(rows)}] shift={shift:+.5f} z={spec:+.1f}{flag}  {seq[:50]!r}", flush=True)
        if not args.dry_run:
            c.table("submissions").update(
                {"shift_raw": shift, "specificity": spec}
            ).eq("id", r["id"]).execute()
            done += 1

    print(f"\nbackfilled: updated={done} failed={failed} drift>{DRIFT_WARN:g}={drifted}")
    if drifted:
        print("⚠ drift warnings mean the served model may have changed since those rows "
              "were scored (season-break check, spec §14).")


if __name__ == "__main__":
    main()
