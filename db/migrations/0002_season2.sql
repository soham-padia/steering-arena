-- Season 2 — OLMo-3-32B, layer 24, logistic direction (confound-audited; best causal
-- steerer from the layer sweep). Deactivates the current season and opens Season 2.
-- Season 1's submissions stay attached to their season_id; the leaderboard filters by
-- season, so scores are never mixed across seasons (PROJECT_SPEC.md §6).
--
-- Run against the project's Supabase Postgres (SQL editor or psql). Idempotent on the
-- insert via the unique(model_id, layer, d_version) constraint.

begin;

-- Close whatever season is currently live.
update seasons set active = false where active = true;

-- Open Season 2.
insert into seasons (name, model_id, model_build, layer, d_version,
                     scoring_mode, token_budget, probe_set_id, active)
values ('Season 2', 'allenai/Olmo-3-1125-32B', null, 24, 'olmo3_L24_logistic',
        'cosine_steering_shift', 100, 'season2', true)
on conflict (model_id, layer, d_version) do update
  set active = true, name = excluded.name, token_budget = excluded.token_budget,
      probe_set_id = excluded.probe_set_id;

commit;

-- Sanity check (run after): expect exactly one active row, the Season 2 tuple.
--   select id, name, model_id, layer, d_version, token_budget, active from seasons order by id;
