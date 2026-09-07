# How to open a new season

A season is a frozen tuple — `(model_id, model_build, layer, d_version, scoring_config)`.
Scores are comparable only within one, so **any** change to any element forces a new season:
a new `d`, a new layer or band, a new scoring mode, a new probe set, or an NDIF re-serve that
shifts scores. Old rows stay attached to their old `season_id` and are never mixed.

This page is the order in which to do it. **The order is the whole point** — every step below
is safe in this position and destructive in a different one.

**Prerequisites:** the `steering-arena` conda env, `SUPABASE_SERVICE_KEY` in `.env`, push
access to the HF Space, and a direction file already validated by the extraction gates. The
backfill step needs NDIF quota; nothing else here does.

```bash
python scripts/check_season_matches_d.py --d data/directions/d_olmo3_s3_score1.npz
```

That is the gate. It compares the season row against the `.npz` metadata field by field, and
it is the check that would have caught both of the first Season 3 attempt's mistakes — a row
saying `layer=23` against a d file saying `layer=27`, and a `d_version` that differed by a
`_score1` suffix. Both were found by hand.

## 1. Freeze the probe set as a new file

```bash
cp data/probes/season2.json data/probes/season3.json
# edit the "season" and "note" fields; leave "prompts" untouched
git add data/probes/season3.json && git commit -m "freeze the season 3 probe set"
```

Do this **even when the prompts are byte-identical**. `data/probes/season3.json` carries the
same 16 prompts as `season2.json`, verbatim and deliberately: Season 3 already changed the
direction, the layers and the scoring mode, and holding the probes fixed keeps the probe set
from becoming a fourth simultaneous variable. It is a separate file rather than a reference so
that `season2.json` can never be edited out from under an archived board.

The probe set is committed before anything else because `scripts/check_season_matches_d.py`
fails a season whose `probe_set_id` file is untracked.

## 2. Run the causal steering check on the ranking direction

```bash
python scripts/behavioral_eval.py generate \
  --d data/directions/d_olmo3_s3_score1.npz --limit 10 --mults 1.0
```

If adding `α·d` does not move generations, `d` is bad and no scoring config rescues it. Only
the **ranking** direction needs this — Score 2 is informational and does not order the board.
Record what you ran and how far it was descoped; Season 3's is
`data/analysis/season3_causal_spotcheck.md`, a 10-prompt spot check justified by
`cos(d_s3_score1, d_s2_L24) = +0.8688`, not the full battery.

## 3. Deploy the code first

```bash
git checkout --orphan space-deploy
git add -A
git commit -m "Steering Arena — deploy (squashed from $(git rev-parse --short main))"
GIT_ASKPASS=/tmp/hf_askpass.sh GIT_TERMINAL_PROMPT=0 git push --force space space-deploy:main
git checkout main && git branch -D space-deploy
curl -s https://sohampadianeu-steering-arena.hf.space/health
```

The deployed build must be able to score the new objective **before** the database knows the
season exists. Full procedure and the askpass shim in `docs/DEPLOY.md`. Opening a season needs
no environment variables — `/health`, `/season` and the scorer all read the active season row
from Supabase, and the Space's `LAYER` / `D_FILE` / `SEASON_ID` variables are legacy fallbacks
the database overrides.

## 4. Run the migration part that opens the season inactive

Run **Part A only** of the migration in the Supabase SQL editor:

```bash
sed -n '/PART A/,/^commit;/p' db/migrations/0009_season3.sql
```

Part A adds the columns the new season needs, creates the one-active-season index, and inserts
the row with `active = false`. Season 2 stays live and visible the whole time. Then confirm:

```sql
select id, name, layer, layers, d_version, scoring_mode, active from seasons order by id;
```

Expect the old season still `active = true` and the new one `active = false`.

## 5. Assert the row matches the direction files

```bash
python scripts/check_season_matches_d.py --season-id 5 \
  --d data/directions/d_olmo3_s3_score1.npz \
  --d data/directions/d_olmo3_s3_score2.npz
```

It exits 0 only if `model_id`, `layer`, `d_version` and `model_build` agree between the row
and every `.npz`, `seasons.layers` equals the ranking band, exactly one season is active, the
probe file exists and is committed, and every `d` loads finite and unit-norm at the model's
hidden size. Read-only — it touches no table.

`--season-id` is the **primary key**, which is not the display number. Get it from the
`select` in step 4, never from `.env`.

## 6. Backfill the carried-over board

Point `.env` at the new season's config, then rescore:

```bash
# .env: SCORING_MODE=banded_mean_multilayer, SCORE1_LAYERS=19,23,27,31,
#       SCORE2_LAYERS=15,23,31,39, PROBE_SET=data/probes/season3.json
python scripts/rescore_season.py --from-season-id 4 --to-season-id 5 --limit 5 --dry-run
python scripts/rescore_season.py --from-season-id 4 --to-season-id 5
```

`rescore_season.py` builds its scorer through `app.scoring` and `ResidualReader.from_settings`
— the exact live path — and refuses to start if the settings disagree with the target row:

```
REFUSING TO RESCORE — settings do not match the target season:
  layer: season 23 vs settings 24

Run scripts/check_season_matches_d.py, and set the bands in .env.
```

That is what you see when `.env` is still the old single-layer config: with `SCORE1_LAYERS`
empty, `settings.banded()` is false and the check falls to the single-layer comparison. Fix `.env`, do not pass a flag around it:
a rescore that scores differently from the live scorer fills the new board with numbers no
future submission can be compared against, which breaks the season invariant from the inside.

The run is resumable on `(season_id, norm_key)` and re-running it skips what landed. Migrated
rows keep their original `created_at` (the board tie-breaks on it), their consent fields, and
get the sentinel `ip_hash = "season1-rescore"` so they do not perturb real rate-limit counts.

## 7. Only now flip it active

Run **Part B**, which is commented out in the migration on purpose:

```sql
begin;
update seasons set active = false where active = true;
update seasons set active = true
  where model_id = 'allenai/Olmo-3-1125-32B' and d_version = 'olmo3_s3_banded';
commit;
```

```sql
select id, name, layers, active from seasons where active;   -- exactly one row, the new season
select count(*) from submissions where season_id = 4;        -- expect 618, unchanged
```

The close-then-open order inside the transaction is required; see the failure modes below.

## What goes wrong

**The ordering trap.** You flip the DB active before the deployed code can score the new
objective. `/submit` resolves the active season from the database but scores with whatever the
deployed build computes, so rows land under the **new** `season_id` carrying the **old**
metric. There is no error, nothing in a log, and afterwards those rows are indistinguishable
from correct ones — the sequence, the handle and the timestamp are all real and only the number
is wrong. This happened once during the Season 3 build and was caught before any submission
arrived. Deploy first, flip second.

**`on conflict ... do update set active = true`.** Migration `0002` used it, and it silently
**re-opens an existing season** instead of failing on a duplicate tuple — you believe you
opened a new board and you have reactivated an archived one, with its old rows live again.
`0009` has no conflict clause for exactly this reason: a duplicate tuple must fail loudly.

**Two active seasons.** `db.get_active_season()` does `.limit(1)` with **no order**, so which
season the site serves is undefined by PostgREST and can differ between requests. The partial
unique index `seasons_one_active_idx on seasons (active) where active` converts that silent
split-brain into a failed transaction — which is why the old season must be closed **before**
the new one opens. Reverse the two `update` statements and the second collides mid-statement
and the transaction aborts. `check_season_matches_d.py` reports the same condition as:

```
  FAIL  2 active seasons: [4, 5] — db.get_active_season() does .limit(1) with NO order
```

**Taking the display number for the primary key.** They have never matched. `.env` says
`SEASON_ID=2` while the DB ids are `1, 3, 4, 5` — `2` was consumed by an aborted insert, so
"Season 3" is row `id=5`. Anything that writes, rescores or queries by season must read the id
from the `seasons` table. A `--to-season-id 3` meant as "Season 3" writes into Season 1.

**Putting a band in `seasons.layer`.** It is `int not null` and participates in
`unique (model_id, layer, d_version)`, so it cannot hold `19,23,27,31`. It holds the band's
median as a **representative** value (`23` for Season 3) and `seasons.layers` is authoritative.
Reading `seasons.layer` on a multi-layer season gives you one layer of four and no error.

## Verify it yourself

One command decides whether the season is coherent, and it costs nothing:

```bash
python scripts/check_season_matches_d.py
```

With no arguments it checks whichever row is `active`. `ALL CHECKS PASS` means the live season
row, its direction files and its probe set agree with each other. Run it after step 4 and again
after step 7.

## Cross-links

`db/migrations/0009_season3.sql` (the two-part migration, and the notes on ids and `layer`) ·
`docs/DEPLOY.md` (the squashed-orphan deploy, and the ordering trap for the record) ·
`docs/reference/seasons.md` (the season table, ids, and what each season froze) ·
`docs/reference/scoring.md` (what the scoring config actually is) ·
`data/analysis/season3_causal_spotcheck.md` (the steering check as it was actually run).
