-- Steering Arena — initial schema (PROJECT_SPEC.md §6).
-- Apply in the Supabase SQL editor (PostgREST can't run DDL, so this is manual).

create table if not exists seasons (
  id             bigint generated always as identity primary key,
  name           text not null,
  model_id       text not null,            -- NDIF-hosted model id/string
  model_build    text,                     -- NDIF build/revision id if exposed
  layer          int  not null,
  d_version      text not null,
  scoring_mode   text not null default 'cosine_steering_shift',
  token_budget   int  not null default 10,
  probe_set_id   text,
  active         boolean not null default true,
  created_at     timestamptz not null default now(),
  unique (model_id, layer, d_version)
);

create table if not exists submissions (
  id             bigint generated always as identity primary key,
  season_id      bigint not null references seasons(id),
  user_handle    text not null,
  sequence_text  text not null,
  norm_key       text not null,            -- normalized form for dedup + score cache
  token_count    int  not null,
  score          double precision not null,
  ip_hash        text,                     -- salted SHA-256; never raw IPs
  created_at     timestamptz not null default now()
);

create index if not exists submissions_board_idx on submissions (season_id, score desc);
create unique index if not exists submissions_unique_seq on submissions (season_id, norm_key);
create index if not exists submissions_ip_time_idx on submissions (ip_hash, created_at);
create index if not exists submissions_time_idx on submissions (created_at);

-- Stub season so the app works before the real Season 1 is opened (Phase 5).
insert into seasons (name, model_id, model_build, layer, d_version, scoring_mode, token_budget, active)
values ('Season 0 — scaffold', 'OLMo-3-32B', null, 16, 'v0-stub', 'cosine_steering_shift', 10, true)
on conflict (model_id, layer, d_version) do nothing;
