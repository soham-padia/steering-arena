-- 0006: record what the /generate demo actually produced.
--
-- 0005 deliberately stored only a salted prompt hash, because the table existed purely
-- as a rate-limit counter. Keeping the text turns it into a research record, so it
-- follows the same governance as submissions (db/migrations/0004_consent.sql):
--   • the row is stored either way — that is what makes the demo auditable
--   • research_consent gates whether it may enter a PUBLISHED dataset, nothing else
--   • consent_version records which notice the person actually saw
--   • ip_hash stays salted and is never exported
-- The demo UI states plainly that prompts and outputs are recorded.

alter table generation_events add column if not exists prompt text;
alter table generation_events add column if not exists continuation text;
alter table generation_events add column if not exists research_consent boolean not null default false;
alter table generation_events add column if not exists consent_version text;

-- Read path for the export script: consented rows, newest first.
create index if not exists generation_events_consent_time
  on generation_events (research_consent, created_at desc);
