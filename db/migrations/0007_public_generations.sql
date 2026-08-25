-- 0007: generations become publicly readable, and writing them requires an account.
--
-- Read: anyone can see the feed (served by the API with the service key, so the table
--   itself stays RLS-on / no-policies and nothing is exposed except the allow-listed
--   columns in app/main.py).
-- Write: /generate now requires a verified Supabase session. user_hash is a salted hash
--   of the Supabase user id — durable rate limiting that survives IP changes, without
--   storing anyone's identity next to what they typed.
-- hidden: takedown without deletion. Public reads filter it out; the row stays for the
--   audit trail and for the maintainer.

alter table generation_events add column if not exists user_hash text;
alter table generation_events add column if not exists hidden boolean not null default false;

create index if not exists generation_events_user_time
  on generation_events (user_hash, created_at desc);
create index if not exists generation_events_public
  on generation_events (hidden, created_at desc);
