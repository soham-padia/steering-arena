"""Export the research dataset — CONSENTED submissions only (data governance).

Pulls rows where research_consent = true and writes a minimized, pseudonymous JSONL:
keeps handle (user-chosen pseudonym), sequence, scores, season, timestamp, and the
consent_version that was agreed to. NEVER exports ip_hash. Rows collected before the
consent column existed default to research_consent = false and are therefore excluded.

    python scripts/export_research_data.py                 # all seasons, consented only
    python scripts/export_research_data.py --season-id 4   # one season
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings  # noqa: E402

# Fields allowed into the research export. ip_hash is intentionally absent.
EXPORT_FIELDS = ["id", "season_id", "user_handle", "sequence_text", "score",
                 "shift_raw", "specificity", "token_count", "consent_version", "created_at"]
OUT_DIR = Path("data/research_export")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season-id", type=int, default=0, help="0 = all seasons")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from supabase import create_client
    c = create_client(settings.supabase_url, settings.supabase_service_key)

    q = c.table("submissions").select(", ".join(EXPORT_FIELDS)).eq("research_consent", True)
    if args.season_id:
        q = q.eq("season_id", args.season_id)
    rows = q.order("created_at").execute().data or []

    # Hard guarantee: never emit an ip_hash even if the schema/select changes.
    rows = [{k: r.get(k) for k in EXPORT_FIELDS} for r in rows]

    total = c.table("submissions").select("id", count="exact", head=True).execute().count
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d")
    out = Path(args.out) if args.out else OUT_DIR / f"steering_arena_consented_{stamp}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"exported {len(rows)} consented rows (of {total} total submissions) → {out}")
    print("fields:", ", ".join(EXPORT_FIELDS), "(ip_hash excluded by design)")
    if not rows:
        print("note: 0 consented rows — rows predating the consent notice default to not-consented.")


if __name__ == "__main__":
    main()
