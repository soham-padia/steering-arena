"""Read the /generate demo log (maintainer CLI).

There is no web view of this on purpose: it holds text strangers typed, so it stays
behind the service key rather than behind a URL someone could guess.

    python scripts/show_generation_log.py                  # last 30
    python scripts/show_generation_log.py --limit 200 --arm anti_coherent
    python scripts/show_generation_log.py --consented      # only publishable rows
    python scripts/show_generation_log.py --stats          # counts by arm
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--arm", default="")
    ap.add_argument("--consented", action="store_true", help="only research_consent = true")
    ap.add_argument("--stats", action="store_true", help="summary instead of rows")
    ap.add_argument("--full", action="store_true", help="do not truncate the answer")
    args = ap.parse_args()

    from supabase import create_client
    c = create_client(settings.supabase_url, settings.supabase_service_key)

    q = c.table("generation_events").select(
        "created_at,arm,prompt,continuation,cached,research_consent,consent_version")
    if args.arm:
        q = q.eq("arm", args.arm)
    if args.consented:
        q = q.eq("research_consent", True)
    rows = q.order("created_at", desc=True).limit(args.limit).execute().data or []

    if args.stats:
        by_arm = collections.Counter(r["arm"] for r in rows)
        with_text = sum(1 for r in rows if r.get("prompt"))
        consented = sum(1 for r in rows if r.get("research_consent"))
        cached = sum(1 for r in rows if r.get("cached"))
        print(f"{len(rows)} row(s): {with_text} with text, {consented} consented, "
              f"{cached} served from cache")
        for arm, n in by_arm.most_common():
            print(f"  {arm:>14} {n}")
        return

    print(f"{len(rows)} row(s), newest first\n")
    for r in rows:
        when = r["created_at"][:19].replace("T", " ")
        flags = ("consented" if r.get("research_consent") else "not consented")
        flags += ", cached" if r.get("cached") else ""
        if not r.get("prompt"):
            print(f"[{when}] {r['arm']:>14}  (no text — logged before migration 0006)")
            continue
        answer = r["continuation"] or ""
        if not args.full and len(answer) > 160:
            answer = answer[:160] + "…"
        print(f"[{when}] {r['arm']:>14}  ({flags})")
        print(f"    typed:  {r['prompt']}")
        print(f"    model:  {answer}\n")


if __name__ == "__main__":
    main()
