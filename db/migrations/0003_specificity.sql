-- Direction-specificity metric (Track 1, Option A — additive, non-ranking).
-- Adds two nullable columns to submissions; Season 2 ranking (score index) untouched.
--   shift_raw   — the raw cosine steering-shift (same quantity as `score` this season)
--   specificity — closed-form z: shift_d / max(||Delta||_F / sqrt(P*H), 1e-4),
--                 i.e. how specifically the sequence moves activations ALONG d,
--                 vs. just moving them anywhere (token-soup artifact detector).
--                 Bounded |z| <= sqrt(hidden) ~= 71.55 for OLMo-3-32B.
-- NULL = scored before this metric existed (backfilled by scripts/backfill_specificity.py).
--
-- MUST be applied BEFORE deploying the app code that writes these columns
-- (otherwise every /submit insert fails on the unknown keys).

alter table submissions add column if not exists shift_raw   double precision;
alter table submissions add column if not exists specificity double precision;

-- Sanity check (run after): both columns present, all rows NULL until backfill.
--   select count(*) filter (where specificity is null) as unbackfilled, count(*) from submissions;
