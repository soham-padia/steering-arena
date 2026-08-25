"""Export the /generate demo log — CONSENTED rows only (same governance as submissions).

Rows are stored either way so the demo stays auditable; research_consent controls what
may be PUBLISHED. ip_hash is never exported, and the field list is a hard allow-list so
a schema change cannot silently leak it.

    python scripts/export_generation_log.py
    python scripts/export_generation_log.py --since 2026-08-01 --out /tmp/demo.jsonl
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings  # noqa: E402

EXPORT_FIELDS = ["id", "created_at", "arm", "prompt", "continuation", "cached",
                 "consent_version"]
OUT_DIR = Path("data/research_export")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="", help="ISO date, e.g. 2026-08-01")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from supabase import create_client
    c = create_client(settings.supabase_url, settings.supabase_service_key)

    q = (c.table("generation_events").select(", ".join(EXPORT_FIELDS))
         .eq("research_consent", True).not_.is_("prompt", "null"))
    if args.since:
        q = q.gte("created_at", args.since)
    rows = q.order("created_at").execute().data or []

    # Hard guarantee: never emit ip_hash or prompt_hash even if the select changes.
    rows = [{k: r.get(k) for k in EXPORT_FIELDS} for r in rows]

    total = c.table("generation_events").select("id", count="exact", head=True).execute().count
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d")
    out = Path(args.out) if args.out else OUT_DIR / f"steering_arena_demo_log_{stamp}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"exported {len(rows)} consented generations (of {total} logged rows) → {out}")
    print("fields:", ", ".join(EXPORT_FIELDS), "(ip_hash / prompt_hash excluded by design)")
    if not rows:
        print("note: 0 rows — pre-0006 rows have no prompt text and are skipped.")


if __name__ == "__main__":
    main()
