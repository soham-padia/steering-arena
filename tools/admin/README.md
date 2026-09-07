# The admin view, unpublished

This directory is **not served by the public site**. `app/main.py` mounts `web/` as a
static catch-all at `/`, so anything inside `web/` is world-reachable by construction —
no route needed, no error if you forget. These two files used to live there. They now
sit outside it, and the `/admin/*` routes are registered only when `ADMIN_API` is true,
which the deployed Hugging Face Space does not set.

The result on the public deployment: `/admin.html`, `/admin.css`, `/admin/generations`
and `/admin/hide` all return **404**. Not 401 — 404. A 401 confirms an endpoint exists
and is refusing you; 404 says nothing.

## Running it locally

```bash
ADMIN_API=true ADMIN_EMAILS=you@example.com uvicorn app.main:app --reload
```

Then open `http://127.0.0.1:8000/admin.html`. With `ADMIN_API` unset the same URL is a
404, which is what production looks like — so you can reproduce the deployed behaviour
by starting the server without the flag.

## The flag removes routes; it does not weaken them

Authorization is unchanged and does not depend on `ADMIN_API`. Both endpoints still call
`userauth.require_admin`, which:

- verifies the session against Supabase rather than validating a JWT locally, so a
  revoked session stops working at once;
- fails closed on a missing token, missing config, or an unreachable Supabase — a
  network error can never authorize;
- then requires the email to be on `ADMIN_EMAILS`. A valid Supabase account was never
  sufficient, because anyone can make one.

`ADMIN_EMAILS` decides *who* may read the log once the routes exist. `ADMIN_API` decides
*whether they exist*. Two separate questions, two separate settings.

## Why remove the surface if the auth held

It did hold. Probed against the live deployment on 2026-09-07: both admin routes
returned `401` unauthenticated, and the published publishable key could not read a row
(`[]` from all three tables) or write one (`42501 new row violates row-level security
policy`). This is attack-surface removal and defence in depth, **not** a fix for a
breach.

There is also a second layer, in `app/main.py`: when `ADMIN_API` is off, `/admin.html`
and `/admin.css` are bound to a route that returns 404. The files being absent from
`web/` already achieves that, but the mount at `/` is a catch-all — a stray copy, a bad
merge, or a revert would republish them silently. A route registered before the mount
wins, so the 404 is a property of the application rather than of the filesystem.
`tests/test_admin_surface.py` pins both layers, and checks for admin files inside `web/`
specifically because that is the failure the mount makes invisible.

## What the public site still exposes, on purpose

`/auth/config` returns `{enabled, supabase_url, anon_key}` without authentication,
because a browser cannot start a Supabase session without the publishable key.
`sb_publishable_*` is Supabase's own name for a key meant to ship to clients; it is in
the page source of every Supabase web app. The **secret** key is never served —
`app/db.py` builds the server client from `settings.supabase_service_key`, and nothing
hands it to a client.

This lived at `/admin/config` until 2026-09-07. The payload was fine; the path was not.
A deliberately-public endpoint sitting under `/admin/*` between two authenticated
siblings reads as a leak to anyone who opens it — and it did, which is what prompted
this change. The old path is **gone rather than aliased**, since an alias would preserve
the thing that was wrong with it. It also no longer reports whether admin is configured;
nothing public needs to know.

What keeps the publishable key harmless is row-level security, not secrecy. `seasons`,
`submissions` and `generation_events` all have RLS enabled with **no policies**, so an
anonymous caller reads nothing. If you ever add a policy, re-run the probes in
`docs/how-to/verify-the-public-surface.md` — and note the trap documented there: a
policy-denied read returns `200` with `[]`, so a status-code-only check cannot tell
"locked down" from "wide open".

## Moderation without the web view

`/admin/hide` sets `hidden` on a `generation_events` row, taking it out of the public
feed without destroying the record. If you would rather not run the web view at all, the
same update through the service key does it:

```bash
python3 - <<'PY'
from app.config import settings
from app.db import SupabaseDatabase
db = SupabaseDatabase(settings.supabase_url, settings.supabase_service_key)
db.client.table("generation_events").update({"hidden": True}) \
    .eq("created_at", "2026-09-01T12:34:56.789Z").execute()
PY
```

`created_at` is the row key the admin view uses too. The service key bypasses RLS, which
is why this works from a shell and the publishable key's insert attempt does not.
