# Season reference

Every season the board has ever run, one entry each, in a fixed format. This page is for
lookup. For the scoring math a season freezes, read `docs/reference/scoring.md`. For what a
season change requires, read `.claude/skills/new-season/SKILL.md`.

Run this first. It lists every season row from the live API, needs no key and no GPU, and
every entry below refers to what it shows:

```bash
curl -s https://sohampadianeu-steering-arena.hf.space/seasons | python3 -m json.tool
```

```
{
    "seasons": [
        {
            "id": 5,
            "name": "Season 3",
            "model_id": "allenai/Olmo-3-1125-32B",
            "layer": 27,
            "layers": "19,23,27,31",
            "d_version": "olmo3_s3_banded",
            "scoring_mode": "banded_mean_multilayer",
            "token_budget": 1000,
            "active": true
        },
        {
            "id": 4,
            "name": "Season 2",
            "model_id": "allenai/Olmo-3-1125-32B",
            "layer": 24,
            "layers": null,
            "d_version": "olmo3_L24_logistic",
            "scoring_mode": "cosine_steering_shift",
            "token_budget": 1000,
            "active": false
        },
        {
            "id": 3,
            "name": "Season 1",
            "model_id": "meta-llama/Llama-3.1-8B",
            "layer": 16,
            "layers": null,
            "d_version": "llama8b-v1",
            "scoring_mode": "cosine_steering_shift",
            "token_budget": 100,
            "active": false
        },
        {
            "id": 1,
            "name": "Season 0 — scaffold",
            "model_id": "OLMo-3-32B",
            "layer": 16,
            "layers": null,
            "d_version": "v0-stub",
            "scoring_mode": "cosine_steering_shift",
            "token_budget": 10,
            "active": false
        }
    ]
}
```

`/seasons` is exposed and is the only place a client can enumerate archived seasons.
`/season` returns the active row alone. `/health` returns the short form:

```bash
curl -s https://sohampadianeu-steering-arena.hf.space/health
```

```
{"status":"ok","season":5,"season_name":"Season 3","model":"allenai/Olmo-3-1125-32B"}
```

Submission counts come from `/leaderboard?season=<id>`. The query parameter is **`season`**,
not `season_id`; an unrecognised parameter is dropped and you get the active board instead.
Measured 2026-09-07: id 1 → 0 entries, id 3 → 103, id 4 → 618, id 5 → 626.

---

## Entry format

Each entry has the same six fields, in this order: `Definition`, `Value`, `Where it is
set`, `Source of truth`, `Failure mode`, `Positive implication`.

---

## Season 0 — scaffold (`season_id = 1`)

**Definition:** the stub row that lets the app boot against an empty database, inserted
before any real season existed.

**Value:** id 1, `model_id` `OLMo-3-32B`, layer 16, `d_version` `v0-stub`,
`token_budget` 10, `scoring_mode` `cosine_steering_shift`, `layers` null, `active` false.
0 submissions.

**Where it is set:** `db/migrations/0001_init.sql`, the final `insert into seasons` guarded
by `on conflict (model_id, layer, d_version) do nothing`.

**Source of truth:** that migration for the row's origin; `/seasons` for its current state.

**Failure mode:** `model_id` is the bare string `OLMo-3-32B`, not a resolvable Hugging Face
id. Anything that hands a historical row's `model_id` to `from_pretrained` or to NDIF fails
on this row, and the closest committed direction, `data/directions/d_v1.npz`, records
`"extraction_method": "placeholder-random"` and `"placeholder": true`.

**Positive implication:** the schema and the app are testable with no season open, which is
why `pytest tests/` needs neither NDIF nor a populated database.

---

## Season 1 (`season_id = 3`)

**Definition:** the first real board. A different model from every season since.

**Value:** id 3, `model_id` `meta-llama/Llama-3.1-8B`, layer 16, `d_version` `llama8b-v1`,
`token_budget` 100, `scoring_mode` `cosine_steering_shift`, `layers` null, `active` false.
**103 submissions.** Probe set `data/probes/season1.json`, 16 prompts.

**Where it is set:** **nowhere in the repository.** `db/migrations/` contains exactly three
`insert into seasons` statements — `0001` (Season 0), `0002` (Season 2), `0009` (Season 3).
None of them opens Season 1.

**Source of truth:** the `seasons` table, reachable only through `/seasons`. There is no
committed migration to check it against.

**Failure mode:** a Supabase project rebuilt by replaying `db/migrations/` in order has no
Season 1 row, so the 103 submissions would have no parent and the foreign key would reject
them. The `d_version` string `llama8b-v1` also matches no committed artifact: the nearest
file, `data/directions/d_llama_v1.npz`, is Llama-3.1-8B at layer 16 but its `meta.d_version`
reads `v1`. Treat the season's direction as unreconstructable from the repo alone.

**Positive implication:** the id is stable and the leaderboard filters by `season_id`, so
the 103 rows stay attached to the tuple they were scored under even though nothing in the
repo describes it.

---

## Season 2 (`season_id = 4`)

**Definition:** the single-layer OLMo-3 board. The season whose top 36 ranks were GCG token
soup, and the corpus every later rescore starts from.

**Value:** id 4, `model_id` `allenai/Olmo-3-1125-32B`, layer 24, `d_version`
`olmo3_L24_logistic`, `scoring_mode` `cosine_steering_shift`, `layers` null, `active` false,
`probe_set_id` `season2`. Live `token_budget` **1000**. **618 submissions.** Direction
`data/directions/d_olmo3_L24_logistic.npz`, whose `meta` reads layer 24, estimator
`logistic`, `confounds_removed` `["length", "sentiment"]`.

**Where it is set:** `db/migrations/0002_season2.sql`.

**Source of truth:** `/seasons` for the row, the npz `meta` for the direction. The
migration is **not** a faithful record of the row: it wrote `token_budget` 100, and the live
row reads 1000.

**Failure mode:** `0002` ends its insert with `on conflict (model_id, layer, d_version) do
update set active = true`, which silently re-opens an already-archived season instead of
failing. `.claude/skills/new-season/SKILL.md` names that as a mistake, and `0009`
deliberately has no conflict clause so a duplicate tuple errors out.

**Positive implication:** 618 rows scored under one frozen tuple are a ready-made corpus,
which is what made the Season 3 rescore a comparison of scoring functions rather than a new
data collection.

---

## Season 3 (`season_id = 5`)

**Definition:** the live board. Multi-layer banded scoring against an
approach-de-confounded direction, carrying two objectives per row.

**Value:** id 5, `model_id` `allenai/Olmo-3-1125-32B`, `layers` `19,23,27,31`, `layer` 27,
`d_version` `olmo3_s3_banded`, `scoring_mode` `banded_mean_multilayer`, `token_budget` 1000,
`probe_set_id` `season3`, `active` true. **626 submissions** as of 2026-09-07 — the 618
carried over from Season 2 plus 8 native ones. Two directions:

| Role | File | Band | Aggregate |
|---|---|---|---|
| Score 1 (ranks) | `data/directions/d_olmo3_s3_score1.npz` | 19, 23, 27, 31 | `banded_mean` |
| Score 2 (informational) | `data/directions/d_olmo3_s3_score2.npz` | 15, 23, 31, 39 | `per_layer_min` |

Both npz files carry `d` (5120,), `per_layer` (4, 5120), `band`, and a `meta` recording
`confounds_removed` `["length", "sentiment", "approach"]` and `held_out_separation` 1.0.

**Where it is set:** `db/migrations/0009_season3.sql`, in two parts. Part A adds
`seasons.layers` and `submissions.score_alt` and opens the season **inactive**. Part B flips
`active` after `scripts/rescore_season.py` has backfilled the 618 carried rows. Running both
as one transaction would leave the live board empty with `/submit` open.

**Source of truth:** `/seasons` for the row; `data/analysis/season3_directions.json` for the
direction provenance; `data/analysis/season3_gcg_baseline.json` for the two baselines. Do
not copy a baseline constant into prose — cite the JSON.

**Failure mode:** `0009` inserts `layer = 23` as a representative value, and the live row
reads **27**. Both direction files' `meta` say `"layer": 27`, and
`scripts/check_season_matches_d.py` asserts that the row and the files agree, so the row was
corrected after the migration was written. Reading 23 out of the migration and expecting it
to match the database is wrong in both directions.

**Positive implication:** the two bands union to six layers, which one
`batch_last_resids_layers` call reads, so Score 2 costs no extra forward pass.

---

## `seasons.layer` and `seasons.layers`

**Definition:** the two columns describing which depth a season scores at.

**Value:** `layer` is `int not null` and participates in `unique (model_id, layer,
d_version)`. `layers` is `text`, added by `0009`, and is null on seasons 1, 3 and 4 — which
really were single-layer.

**Where it is set:** `db/migrations/0001_init.sql` for `layer`,
`db/migrations/0009_season3.sql` for `layers`. Read back by `app/main.py:season` and
`app/main.py:seasons`.

**Source of truth:** `layers` on a multi-layer season. `layer` is back-compat only.

**Failure mode:** `layer` is `int not null`, so **it cannot hold a band**. On Season 3 it
holds 27, one layer of a four-layer band. A score computed from `layer` alone is a different
metric and raises no error — it is a plausible number for the wrong objective. The banded
scorer reads `layers`.

**Positive implication:** the old column keeps every pre-Season-3 client working, and the
new one is unambiguous about which seasons are banded, because it is null exactly when the
season is not.

---

## `score` and `score_alt`

**Definition:** the two submission columns holding the two objectives.

**Value:** `score` is Score 1 and is `double precision not null`. `score_alt` is Score 2 and
is nullable.

**Where it is set:** `db/migrations/0001_init.sql` for `score`,
`db/migrations/0009_season3.sql` for `score_alt`. Served by `app/main.py:leaderboard`.

**Source of truth:** the `submissions` table.

**Failure mode:** `score_alt` is **null on Seasons 1 and 2**, which predate the banded
scorer. Null means "not scored under that objective", **never zero**. Averaging or plotting
it as zero drags every historical row down and produces a fictional trend. `app/main.py`
passes the null through unchanged and the comment there says the UI must render it as "not
scored".

**Positive implication:** one row can carry two objectives, so a new objective is a backfill
rather than a re-collection.

---

## `active`, and the one-live-season index

**Definition:** which row the site actually serves.

**Value:** at most one row with `active = true`, enforced by the partial unique index
`seasons_one_active_idx on seasons (active) where active`.

**Where it is set:** `db/migrations/0009_season3.sql`, Part A. Read by
`db.get_active_season()`.

**Source of truth:** the index. Before it, nothing enforced this.

**Failure mode:** `db.get_active_season()` does `.limit(1)` with **no `order by`**, so with
two active rows the season the site serves is whatever PostgREST returns first. That is
split-brain with no error. The index converts it into a failed transaction, which is also
why Part B must close Season 2 *before* opening Season 3 — the other order collides
mid-statement.

**Positive implication:** the partial index constrains only `true` rows, so any number of
archived seasons may sit at `false`.

---

## `SEASON_ID` in `.env`

**Definition:** a display fallback for when the database is unreachable.

**Value:** `2`. It matches **no** database id. The ids are 1, 3, 4, 5 — the identity
sequence handed out 2 to an aborted insert and no row ever held it.

**Where it is set:** `app/config.py:season_id`, default `2`, overridable by `.env`, by an
environment variable, and by a Hugging Face Space variable of the same name.

**Source of truth:** the active `seasons` row, via `db.get_active_season()`. `/health` and
`/season` both read the database and fall back to `settings` only on an exception.

**Failure mode:** filtering a board or writing a submission against `settings.season_id`
targets id 2, which does not exist, so you get an empty result or a foreign-key error rather
than Season 2. `/health` used to report it, and was therefore advertising a season that does
not exist. `docs/DEPLOY.md` lists `SEASON_ID`, `SEASON_NAME`, `LAYER`, `D_FILE`, `D_VERSION`
and `PROBE_SET` as legacy Space variables the database overrides.

**Positive implication:** because the database is authoritative, opening a season is a SQL
statement and not a redeploy — the deploy and the season flip are separate, orderable steps.

---

## The probe files

**Definition:** the frozen prompt sets a season averages its scores over.

**Value:** `data/probes/season1.json`, `season2.json`, `season3.json`. 16 prompts each.
Season 3's `prompts` array is byte-identical to Season 2's; Season 1's is not.

**Where it is set:** `seasons.probe_set_id` (`season2`, `season3`) selects the file;
`app/config.py:probe_set` is the fallback path.

**Source of truth:** the files.

**Failure mode:** the *files* are not identical even where the prompts are — `season3.json`
carries a different `season` and `note` header, so the md5s differ (`2b1284a4…` for
`season2.json`, `95836327…` for `season3.json`) and the byte sizes differ (1171 vs 1426).
Checking the freeze with a file checksum reports a change that is not in any prompt. Compare
the `prompts` array.

**Positive implication:** Season 3 got its own copy precisely *because* the prompts are the
same, so `season2.json` can never be edited out from under an archived board while Season 3
still reads the identical text.

---

## What this buys you

One row carries two objectives, which is how all 618 Season 2 submissions were rescored
under Season 3's banded metric without asking anyone to resubmit — and why a third objective
would be a backfill, not a new board. Every season's frozen tuple is recoverable from
`/seasons` plus a committed npz, with one honest exception: Season 1 has no migration, so
the database is its only record.
