#!/usr/bin/env python3
"""Take a generation out of the public feed, or put it back.

This replaces the `/admin/hide` HTTP endpoint and the web admin view, both removed
on 2026-09-07. The capability had to survive the removal: `web/behavior.html` shows
text that strangers typed into the public demo, `app/main.py:_feed` filters rows
where `hidden` is true, and nothing else in the codebase can set that column.

Run from the repo root with SUPABASE_URL and SUPABASE_SERVICE_KEY in `.env`:

    # see what is currently in the public feed
    python scripts/moderate_generation.py list --limit 20

    # hide one row, addressed by the created_at the listing prints
    python scripts/moderate_generation.py hide 2026-09-01T12:34:56.789Z

    # and put it back
    python scripts/moderate_generation.py unhide 2026-09-01T12:34:56.789Z

    # everything currently hidden
    python scripts/moderate_generation.py list --hidden-only

Why a script rather than a web page: the page needed a route, a sign-in flow, an
email allowlist, and an `ADMIN_API` flag, and it lived one misconfigured
environment variable away from being public again. This needs the service key,
which never leaves the server, so there is no surface to expose. See
`tools/admin/README.md` in git history if you want the page that used to be here.

**The service key bypasses row-level security.** That is why this works from a
shell while the publishable key's insert attempt fails with 42501. Do not put the
service key anywhere a browser can reach it.

Failure modes:
  - no credentials -> exits 2 and names the missing variable; it does not fall back
    to the publishable key, which cannot write.
  - a `created_at` that is valid but matches nothing -> exits 1 saying 0 rows
    matched, rather than reporting success. Supabase's update returns an empty list
    rather than an error, so the row count is checked explicitly.
  - a `created_at` that is not a timestamp at all -> exits 1 with PostgREST's
    `22007 invalid input syntax for type timestamp with time zone`. Verified: this
    is a different failure from "no such row", and both are reported rather than
    raised, because a traceback here reads like the script is broken when the
    argument is what is wrong.
  - a truncated timestamp -> parses, then matches nothing. `created_at` is the
    address and must be the exact string `list` printed, to the millisecond.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TABLE = "generation_events"


def client():
    from app.config import settings

    missing = [n for n, v in (("SUPABASE_URL", settings.supabase_url),
                              ("SUPABASE_SERVICE_KEY", settings.supabase_service_key))
               if not v]
    if missing:
        print(f"error: {' and '.join(missing)} not set (looked in .env)", file=sys.stderr)
        raise SystemExit(2)

    from app.db import SupabaseDatabase
    return SupabaseDatabase(settings.supabase_url, settings.supabase_service_key).client


def cmd_list(args) -> int:
    q = (client().table(TABLE)
         .select("created_at, arm, handle, hidden, prompt, continuation")
         .order("created_at", desc=True).limit(args.limit))
    if args.hidden_only:
        q = q.eq("hidden", True)
    rows = q.execute().data or []
    if not rows:
        print("no rows matched")
        return 0
    for r in rows:
        flag = "HIDDEN " if r.get("hidden") else "       "
        prompt = (r.get("prompt") or "")[:60].replace("\n", " ")
        print(f"{flag}{r['created_at']}  {r.get('arm') or '-':<14} "
              f"{(r.get('handle') or '-'):<12} {prompt}")
    print(f"\n{len(rows)} row(s). Address a row by its exact created_at.")
    return 0


def set_hidden(created_at: str, hidden: bool) -> int:
    try:
        res = (client().table(TABLE).update({"hidden": hidden})
               .eq("created_at", created_at).execute())
    except Exception as exc:  # noqa: BLE001 — postgrest.APIError, without the import
        print(f"error: {getattr(exc, 'message', None) or exc}", file=sys.stderr)
        return 1
    n = len(res.data or [])
    if not n:
        print(f"error: 0 rows matched created_at={created_at!r}. Run `list` and copy "
              f"the timestamp exactly.", file=sys.stderr)
        return 1
    print(f"{'hid' if hidden else 'unhid'} {n} row(s): {created_at}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list", help="show recent generations and their hidden flag")
    pl.add_argument("--limit", type=int, default=30)
    pl.add_argument("--hidden-only", action="store_true")
    pl.set_defaults(fn=cmd_list)

    for name, val in (("hide", True), ("unhide", False)):
        s = sub.add_parser(name, help=f"set hidden={val} on one row")
        s.add_argument("created_at", help="exact created_at from `list`")
        s.set_defaults(fn=lambda a, _v=val: set_hidden(a.created_at, _v))

    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
