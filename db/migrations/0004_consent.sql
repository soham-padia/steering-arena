-- Research-consent tracking (data governance).
-- Records, per submission, whether the submitter consented to research use and which
-- version of the consent notice was in effect. Research exports include ONLY rows with
-- research_consent = true (scripts/export_research_data.py).
--
-- Default is FALSE: every row that predates this column (the ~170 Season-2 entries
-- collected before any consent notice existed) is therefore EXCLUDED from research data
-- — you cannot retroactively consent people. New submissions send an explicit value.
--
-- Apply in Supabase BEFORE deploying the app code that writes these columns.

alter table submissions add column if not exists research_consent boolean not null default false;
alter table submissions add column if not exists consent_version  text;

-- After deploy, sanity check the split:
--   select research_consent, count(*) from submissions group by research_consent;
