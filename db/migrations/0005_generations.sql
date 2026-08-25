-- 0005: live generation endpoint (/generate).
--
-- The public "try a prefix" demo spends the maintainer's NDIF quota on every uncached
-- call, so its rate limits must survive a Space restart exactly like the submission
-- limits do (see app/ratelimit.py). This table is the durable counter; it is NOT a
-- research dataset and stores no raw prompt text — only a salted hash, so a repeat can
-- be recognised without retaining what anyone typed.

create table if not exists generation_events (
  id          bigserial primary key,
  created_at  timestamptz not null default now(),
  ip_hash     text not null,
  arm         text not null,
  prompt_hash text not null,      -- sha256(salt:prompt); never the prompt itself
  cached      boolean not null default false
);

create index if not exists generation_events_ip_time on generation_events (ip_hash, created_at desc);
create index if not exists generation_events_time on generation_events (created_at desc);

-- Service-role only, same posture as submissions: the anon key must never read or write.
alter table generation_events enable row level security;
