-- Season 3 — OLMo-3-32B, MULTI-LAYER banded scoring, approach-de-confounded direction.
--
-- Season 3 changes four things at once (new d, new layers, new scoring mode, a second
-- score), so by the season invariant every one of them on its own would force a new
-- season. Season 2's rows stay attached to season_id = 4 and are never touched; the
-- leaderboard filters by season_id, so nothing is mixed.
--
-- RUN THIS IN TWO PARTS. Part A opens Season 3 INACTIVE. Season 2 stays live while
-- scripts/rescore_season.py backfills the 618 carried-over entries. Part B flips the
-- switch once the board is populated. Running it as one transaction would leave the
-- live board empty with /submit open for the hours the rescore takes, so early
-- submitters would legitimately hold rank 1 and then be buried by the backfill.
--
-- NOTE ON IDS: the identity sequence has already handed out 1, 3, 4 (2 was consumed by
-- an aborted insert). Season 3 will therefore get id 5, NOT 3. Nothing may assume the
-- season's display number matches its primary key — .env's SEASON_ID has never matched.
--
-- NOTE ON `layer`: seasons.layer is `int not null` and participates in
-- unique (model_id, layer, d_version), so a band cannot live there. It holds the band's
-- median as a representative value; the authoritative band is `layers`, added below.
--
-- DEPARTURE FROM 0002: that migration used `on conflict ... do update set active = true`,
-- which silently re-opens an existing season — the exact thing
-- .claude/skills/new-season/SKILL.md:18 calls a mistake. This one has no conflict clause,
-- so a duplicate tuple fails loudly instead.

-- ══════════════════════════════════════════════════════════════════════════
-- PART A — schema + open Season 3 INACTIVE. Safe to run any time; changes
--          nothing a visitor can see.
-- ══════════════════════════════════════════════════════════════════════════
begin;

-- The authoritative band for a multi-layer season. Null for every existing season,
-- which is correct: seasons 1/3/4 really were single-layer.
alter table seasons add column if not exists layers text;

-- The second, informational score (per-layer min over the spread band). Nullable:
-- every Season 1 and Season 2 row predates it and must stay null rather than 0,
-- which would read as "scored 0" instead of "not scored".
alter table submissions add column if not exists score_alt double precision;

-- There has never been a constraint enforcing one live season, and
-- db.get_active_season() does `.limit(1)` with NO order — so two active rows would make
-- which season the site serves undefined by PostgREST. Since this migration flips active
-- from one row to another, convert that silent split-brain into a failed transaction.
-- Partial index: only rows with active = true participate, so any number may be false.
create unique index if not exists seasons_one_active_idx
  on seasons (active) where active;

insert into seasons (name, model_id, model_build, layer, layers, d_version,
                     scoring_mode, token_budget, probe_set_id, active)
values ('Season 3', 'allenai/Olmo-3-1125-32B', null,
        23,                       -- representative only; see note above
        '19,23,27,31',            -- Score 1 band (ranks). Score 2 = 15,23,31,39.
        'olmo3_s3_banded',
        'banded_mean_multilayer',
        1000, 'season3', false);  -- INACTIVE until Part B

commit;

-- Sanity after Part A (expect Season 2 id=4 still active=true, Season 3 active=false):
--   select id, name, layer, layers, d_version, scoring_mode, active
--     from seasons order by id;


-- ══════════════════════════════════════════════════════════════════════════
-- PART B — go live. Run ONLY after:
--   1. data/directions/d_olmo3_s3_score{1,2}.npz are committed and their metadata
--      matches this row (model_id, layer, d_version) — SKILL.md step 2;
--   2. the causal steering check has PASSED for score1's direction;
--   3. scripts/rescore_season.py has backfilled Season 2's 618 entries into it;
--   4. data/probes/season3.json is committed and frozen.
-- ══════════════════════════════════════════════════════════════════════════
-- begin;
--
-- -- Order matters: the partial unique index above permits exactly one active row, so
-- -- Season 2 must be closed BEFORE Season 3 opens or the update collides mid-statement.
-- update seasons set active = false where active = true;
-- update seasons set active = true
--   where model_id = 'allenai/Olmo-3-1125-32B' and d_version = 'olmo3_s3_banded';
--
-- commit;
--
-- Sanity after Part B (expect exactly one row, Season 3):
--   select id, name, layers, active from seasons where active;
-- And confirm Season 2's board is intact and unchanged:
--   select count(*) from submissions where season_id = 4;   -- expect 618
