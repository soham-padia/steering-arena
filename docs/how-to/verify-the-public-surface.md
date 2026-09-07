# How to verify the public surface

Check what an anonymous stranger can reach on the live site: which endpoints answer,
what the published Supabase key can read or write, and whether the admin surface
exists. Run this after any deploy that touches `app/main.py`, `web/`, or a Supabase
policy.

**Prerequisites:** `curl` and `python3`. No repo checkout, no credentials, no GPU.
Every command below is a read-only probe from outside, except one deliberate insert
attempt that is expected to fail.

```bash
BASE=https://sohampadianeu-steering-arena.hf.space
curl -s "$BASE/auth/config"
```

```
{"enabled":true,"supabase_url":"https://pjsgeslryfqcydshafkq.supabase.co","anon_key":"sb_publishable_..."}
```

That endpoint is unauthenticated on purpose and this output is not a leak. A browser
cannot start a Supabase session without the publishable key, and `sb_publishable_*` is
Supabase's own name for a key meant to ship to clients. What makes it safe is
row-level security, which is what the rest of this page checks.

## 1. Take the project URL and key from the site itself

Do not paste a key from a config file. Read it from the deployment you are testing, so
the probe uses whatever is actually live:

```bash
eval "$(curl -s "$BASE/auth/config" | python3 -c '
import json, sys
c = json.load(sys.stdin)
print(f"U={c[\"supabase_url\"]}")
print(f"K={c[\"anon_key\"]}")
')"
echo "project: $U"
echo "key length: ${#K}"
```

```
project: https://pjsgeslryfqcydshafkq.supabase.co
key length: 51
```

An empty `$K` means sign-in is unconfigured on that deployment, and the read probes
below will return an auth error rather than telling you anything about RLS.

## 2. Confirm the key cannot read a row

```bash
for t in submissions seasons generation_events; do
  printf '%-20s %s\n' "$t" "$(curl -s "$U/rest/v1/$t?select=*&limit=2" -H "apikey: $K")"
done
```

```
submissions          []
seasons              []
generation_events    []
```

**`[]` is the passing result, and the status code is 200.** This is the trap on this
page: PostgREST returns `200` with an empty array when a policy denies every row, and
reserves `403` for table-grant denial. So a status-code-only probe cannot tell "locked
down" from "wide open" — both are 200. Print the body.

Anything other than `[]` means an anonymous caller is reading your data. That is a real
finding. Go to §5.

## 3. Confirm the key cannot write a row

Reads and writes are governed by separate policies, so a passing §2 does not imply
this:

```bash
curl -s -w '\n[%{http_code}]\n' -X POST "$U/rest/v1/generation_events" \
  -H "apikey: $K" -H 'content-type: application/json' \
  -d '{"prompt":"rls-probe"}'
```

```
{"code":"42501","details":null,"hint":null,"message":"new row violates row-level security policy for table \"generation_events\""}
[401]
```

`42501` is the PostgreSQL insufficient-privilege code. Seeing it is the pass. A `201`
means anyone can write rows into the public feed.

## 4. Confirm the admin surface is absent

```bash
for p in /admin.html /admin.css /admin/config /admin/generations; do
  printf '%-22s %s\n' "$p" "$(curl -s -o /dev/null -w '%{http_code}' "$BASE$p")"
done
printf '%-22s %s\n' "/admin/hide (POST)" \
  "$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/admin/hide" \
     -H 'content-type: application/json' -d '{"created_at":"x","hidden":true}')"
```

```
/admin.html            404
/admin.css             404
/admin/config          404
/admin/generations     404
/admin/hide (POST)     405
```

**The POST answers 405, and that is correct.** The static mount at `/` catches every
unmatched POST, so `POST /anything-that-does-not-exist` also answers 405. Check the
equivalence rather than the number:

```bash
curl -s -o /dev/null -w 'control POST -> %{http_code}\n' \
  -X POST "$BASE/definitely-not-a-route" \
  -H 'content-type: application/json' -d '{}'
```

```
control POST -> 405
```

Equal codes mean the response says nothing about whether an endpoint is behind that
path, which is the property that matters.

**Expect 404, not 401, on the GETs.** These routes do not exist: the admin view and the
`/admin/*` endpoints were deleted on 2026-09-07, along with the `ADMIN_API` flag that
used to make them conditional. The flag was the weak part — one variable in the Space's
settings would have republished the whole surface.

A `401` is therefore *worse* than it looks: it means a route has been added back and is
merely refusing you. A `200` on `/admin.html` means a copy of the page has reappeared
inside `web/`, which `app/main.py` mounts as a world-readable catch-all — no route
required. Either way, something was reintroduced.

`/admin/config` returning 404 is also load-bearing: it moved to `/auth/config` and was
deliberately not aliased, so a 200 there means an old build is deployed.

## 5. If a probe fails

**An anonymous read returns rows.** A policy grants `select` to `anon`. Find it, and
decide whether it is intended:

```sql
select schemaname, tablename, policyname, roles, cmd, qual
from pg_policies where schemaname = 'public' order by tablename;
```

Zero rows is this project's baseline: all three tables have `enable row level security`
with no policies at all, which denies everything. The board still works because the
server reads with the service key, which bypasses RLS.

**An anonymous insert returns 201.** Same query, looking at `cmd = 'INSERT'`. Until it
is fixed, the public feed can be written to by anyone.

**An admin path returns 200 or 401.** Something was added back. Run
`python -m pytest tests/test_admin_surface.py` in a checkout: it asserts absence
unconditionally, checks the app's own route table for any `/admin` path, looks for
admin front-end files anywhere in the tree, and confirms `require_admin`,
`admin_emails`, `ADMIN_API` and `ADMIN_EMAILS` are all still gone.

## What this buys you

Four commands, no credentials, and no repo, that together distinguish "the published
key is public by design" from "the published key reads the database". Those two look
identical from the outside — same endpoint, same key, same 200 status — and only the
response body separates them. Running these after a deploy is what lets you answer the
question honestly instead of reasoning from the code.

## Cross-links

`app/main.py` (the `/auth/config` docstring records the same probes and their results) ·
`scripts/moderate_generation.py` (hiding a row now that `/admin/hide` is gone) ·
`tests/test_admin_surface.py` (the same boundary, pinned in CI) ·
`db/migrations/0001_init.sql` and `db/migrations/0005_generations.sql` (where RLS is
enabled) · `docs/how-to/open-a-new-season.md` (the other procedure whose ordering is
load-bearing).
