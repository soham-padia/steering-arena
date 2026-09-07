# Result artifact reference

Every committed result artifact, one row each, grouped by the question it answers. This page
is a **map, not a copy**: it tells you where a number lives, and it deliberately carries no
numbers of its own. The repo's rule is one number, one home — quote it from the artifact, not
from an index. For the numbers themselves, start at `data/analysis/REVISIONS_2026-09-05.md`.
For the scoring quantities, read `docs/reference/scoring.md`.

Run this first. It counts what is on disk, needs no GPU and no API key, and every row below
refers to one of the files it counts:

```bash
ls data/analysis/*.json | wc -l; ls data/analysis/*.md | wc -l
```

```
46
38
```

46 JSON artifacts, 38 write-ups. Three CSVs (`data/analysis/*_blind*.csv`) and three audit
JSONs under `_falsifier/` sit outside those counts and are listed below too.

---

## How a row reads

Each row has the same five columns.

- **artifact** — the path, relative to the repo root. `data/analysis/` is elided in the
  tables; `_falsifier/` paths are written out.
- **written by** — the script and subcommand that produces it. The naming convention is
  `data/analysis/X.json` ← `scripts/X.py`, with the GCG artifacts from `scripts/gcg/`. Where
  a file breaks the convention, the column says what actually writes it.
- **interpreted in** — the `.md` beside it, or **none**. That column is the gap list, and it
  is useful precisely because it is honest: 21 artifacts have no write-up.
- **answers** — the question, not the answer.
- **cost** — one of `$0`, `GPU`, `NDIF quota`, `paid API`: the most expensive resource the
  *writing step* touches directly, not the sum of the pipeline behind it.

Two consequences of that cost rule, both load-bearing. A `stats` or `report` step that reads
verdicts off disk is `$0` even when the verdicts cost money — the paid step's cost is
recorded on the raw verdict file, where it belongs. And the generation caches under
`data/cache/` are **gitignored**, so on a fresh clone anything downstream of a generation
needs the generation re-run first, at `GPU` or `NDIF quota`.

---

## Direction and band

| artifact | written by | interpreted in | answers | cost |
|---|---|---|---|---|
| `banded_direction.json` | `scripts/banded_direction.py` | `banded_direction.md` | does a band of layers separate the seed pairs better than one layer does? | `$0` |
| `season3_band_select.json` | `scripts/season3_band_select.py` | `season3_band_select.md` | which band and which aggregate go into Season 3? | `$0` |
| `season3_directions.json` | `scripts/build_season3_directions.py` | `season3_directions.md` | what are the two shipped Season 3 directions, and which gates did they pass? | `$0` |
| `season3_gcg_baseline.json` | `scripts/gcg/baseline_const.py` | `season3_gcg_baseline.md` | what constant converts a search number into a board number? | `GPU` |
| `direction_purity.json` | `scripts/direction_purity.py` | `direction_purity.md` | is the residual cosine between `d` and `approach` load-bearing? | `$0` |
| `direction_null.json` | `scripts/direction_null.py` | `direction_null.md` | how much cross-layer cosine structure survives label shuffling? | `$0` |
| `layer_concept_profile.json` | `scripts/layer_concept_profile.py` | `layer_concept_profile.md` | at which layers is the concept decodable above its own null? | `$0` |
| `layer_profile_all64.json` | `scripts/layer_profile_all64.py` | `layer_profile_all64.md` | the same question over all 64 layers, full- and sliding-attention split apart | `$0` |
| `layer_sweep_prefix.json` | `scripts/layer_sweep_prefix.py` | `layer_sweep_prefix.md` | does the board winner's advantage exist anywhere but the scored layer? | `NDIF quota` |
| `banded_score_arms.json` | `scripts/banded_score_arms.py` | `banded_direction.md` (merged) | who would have won Season 2 if the board had scored over a band? | `$0` |

Every `.md` in these tables sits beside the `.json` a script produced. If the two disagree,
**the JSON wins and the `.md` is a bug** — fix the prose, do not edit the artifact. Eight
`.md` files have no `.json` beside them at all and are prose in their own right, not
interpretations: `REVISIONS_2026-09-05.md`, `citations.md`, `cosine_scale.md`,
`raw_examples.md`, `season3_causal_spotcheck.md`, `season3_gcg_setup.md`,
`specificity_calibration.md`, `steering_random_control_preregistration.md`.

## Search (Season 3 Part A)

| artifact | written by | interpreted in | answers | cost |
|---|---|---|---|---|
| `season3_gcg_aggregate_asymmetry.json` | no script recorded — read out of the committed GCG run logs, as its own `.md` states | `season3_gcg_aggregate_asymmetry.md` | why is anti easier to search than pro, and does that need a mechanism? | `$0` |
| `season3_k3_control.json` | `scripts/gcg/k3_control.py` | `season3_k3_control.md` | does multi-position mutation fix the conjunction, and does the anti arm rule out generic search improvement? | `$0` |
| `season3_prefix_scores.json` | `scripts/score_banded_local.py` | `season3_prefix_scores.md` | what does each arm score under both Season 3 objectives, in board units? | `GPU` |

## String fidelity

| artifact | written by | interpreted in | answers | cost |
|---|---|---|---|---|
| `season3_gcg_ablation.json` | `scripts/gcg/ablate_prompt.py --run <run_dir>` | `season3_gcg_ablation.md` | which tokens of a GCG winner actually carry the score? | `GPU` |
| `prefix_degeneration_s3.json` | `scripts/prefix_behavior_eval.py --tag s3 degeneration` | `prefix_degeneration_s3.md` | how often does an arm loop or collapse instead of answering? | `$0` |
| `prefix_content_s3.json` | `scripts/prefix_behavior_eval.py --tag s3 content` | `prefix_content_s3.md` | does the prefix steer, or does the model copy its vocabulary downstream? | `$0` |
| `coherence_confound.json` | `scripts/coherence_confound.py` | `coherence_confound.md` | is the behavioural gain a capability-damage artifact rather than steering? | `$0` |

Four artifact families are **inputs**, not results, and deliberately have no write-up: the
blind keys (`*_blind_key*.json`), the raw judge verdict files (`prefix_judge_*.json`), the
arms files (`prefix_eval_arms*.json`) and the seen-prompts file
(`prefix_eval_seen_prompts.json`). Nothing is concluded from them directly, so a `.md` beside
one would have nothing to say; they are listed here so you can find the raw records a result
was computed from.

## Behaviour (Season 3 Part A′)

| artifact | written by | interpreted in | answers | cost |
|---|---|---|---|---|
| `prefix_eval_s3.json` | `scripts/prefix_behavior_eval.py --tag s3 stats` | `prefix_eval_s3.md` | do the Season 3 winners change judged behaviour, and which score orders the arms? | `$0` |
| `prefix_eval_arms_s3.json` | `scripts/prefix_behavior_eval.py --tag s3 select-gcg` | **none** | which strings are frozen into the six Season 3 arms? | `$0` |
| `prefix_blind_s3.csv` | `scripts/prefix_behavior_eval.py --tag s3 blind` | **none** | the blind rating sheet a rater fills in, via `scripts/rate_blind.py` | `$0` |
| `prefix_blind_key_s3.json` | `scripts/prefix_behavior_eval.py --tag s3 blind` | **none** | which blind row was which arm | `$0` |
| `prefix_judge_claude_s3.json` | `scripts/prefix_behavior_eval.py --tag s3 claude-batches` then `claude-merge` | **none** | the raw per-pair verdicts of the `$0` judge path | `$0` |

## Behaviour (Season 2)

| artifact | written by | interpreted in | answers | cost |
|---|---|---|---|---|
| `behavioral_eval.json` | `scripts/behavioral_eval.py` (`generate`, then `judge` — OLMo-as-judge on NDIF) | `behavioral_eval.md` | does adding or removing `d` at the season layer change generation? | `NDIF quota` |
| `behavioral_blind.csv` | `scripts/behavioral_eval.py blind` | **none** | the blind sheet for the human pass — **gitignored** | `$0` |
| `behavioral_blind_key.json` | `scripts/behavioral_eval.py blind` | **none** | which blind row was which arm — **gitignored**, stays private until the human pass closes | `$0` |
| `behavioral_cache_repair.json` | `scripts/repair_behavioral_cache.py` | `behavioral_cache_repair.md` | which cached generations were mis-keyed, and what did repairing them change? | `$0` |
| `prefix_eval.json` | `scripts/prefix_behavior_eval.py stats` | `prefix_eval.md` | do the Season 2 board winners change judged behaviour? | `$0` |
| `prefix_eval_arms.json` | `scripts/prefix_behavior_eval.py select` | **none** | which leaderboard strings are frozen into the Season 2 arms | `$0` |
| `prefix_blind.csv` | `scripts/prefix_behavior_eval.py blind` | **none** | the blind rating sheet for Season 2 | `$0` |
| `prefix_blind_key.json` | `scripts/prefix_behavior_eval.py blind` | **none** | which blind row was which arm | `$0` |
| `prefix_eval_seen_prompts.json` | `scripts/prefix_behavior_eval.py` | **none** | which eval prompts had already been seen when the arms were chosen | `$0` |
| `prefix_judge_verdicts.json` | `scripts/prefix_behavior_eval.py judge` | **none** | the raw DeepSeek verdicts, per pair, with billed call and token counts | `paid API` |
| `prefix_judge_verdicts_v1.json` | `scripts/prefix_behavior_eval.py judge`, rubric v1 | **none** | the superseded rubric's raw verdicts, kept rather than overwritten | `paid API` |
| `prefix_judge_claude.json` | `scripts/prefix_behavior_eval.py claude-batches` then `claude-merge` | **none** | the raw second-rater verdicts | `$0` |
| `prefix_loop_control.json` | `scripts/prefix_behavior_eval.py stats --report data/analysis/prefix_loop_control.json` | `prefix_loop_control.md` | what happens to the Season 2 result when looped continuations are scoped out? | `$0` |
| `prefix_gallery_judge.json` | `scripts/prefix_gallery.py judge` | **none** | how do the gallery arms score on judged kindness against the frozen rubric? | `paid API` |
| `site_prefixes.json` | `scripts/prefix_gallery.py select` | **none** | which prefixes the live site may prepend, and which of them are public | `$0` |
| `prefix_transfer.json` | `scripts/prefix_transfer_eval.py` | `prefix_transfer.md` | does the anti arm's collapse reproduce on the Llama models? | `paid API` |

`SESSION_REPORT.md` carries an older artifacts table covering Season 2 only. This page extends
past it rather than replacing it: where the two overlap, they agree; where they differ, this
page is the one that also knows about Season 3.

## Mechanism

| artifact | written by | interpreted in | answers | cost |
|---|---|---|---|---|
| `compile_check.json` | `scripts/compile_check.py` | `compile_check.md` | is the winning string a compilation of `d` into tokens? | `NDIF quota` |
| `normalization_check.json` | `scripts/normalization_check.py` | `normalization_check.md` | does RMSNorm mean the model never reads the on-`d` component? | `$0` |
| `causal_layer_curve.json` | `scripts/causal_layer_curve.py` | `causal_layer_curve.md` | does ablation matter at any depth, or only at the layer `d` was fit on? | `NDIF quota` |
| `steering_ablation.json` | `scripts/steering_ablation.py report` | `steering_ablation.md` | does removing `d` change behaviour more than removing a norm-matched random direction? | `$0` |
| `steering_ablation_measure.json` | `scripts/steering_ablation.py measure` | **none** | how big is the thing being removed? | `NDIF quota` |
| `steering_ablation_check.json` | `scripts/steering_ablation.py check` | **none** | did the intervention land where it was aimed? | `NDIF quota` |
| `steering_ablation_blind_key.json` | `scripts/steering_ablation.py blind` | **none** | which blind item was which arm | `$0` |
| `steering_dose.json` | `scripts/steering_dose.py` | `steering_dose.md` | do tokens beat vectors, or was the injection dose wrong? | `paid API` |
| `steering_random_control.json` | `scripts/steering_random_control.py` (`generate`, `judge`, `claude-merge`, `report`) | `steering_random_control.md` | does a random vector of the same norm derail generation as much? | `paid API` |
| `meandiff_ablation.json` | `scripts/meandiff_ablation.py` | `meandiff_ablation.md` | does the meandiff direction ablate like the shipped logistic one? | `NDIF quota` |
| `transfer_report.json` | `scripts/transfer_report.py` | `transfer_report.md` | does a high-scoring sequence still score on another model using that model's own `d`? | `NDIF quota` |

## Audit

| artifact | written by | interpreted in | answers | cost |
|---|---|---|---|---|
| `_falsifier/verify_result.json` | `_falsifier/verify.py` | **none** | which committed claims still match the artifacts they were drawn from? | `$0` |
| `_falsifier/recompute_result.json` | `_falsifier/recompute.py` | `_falsifier/recompute_result.md` | do the published effects survive a fixed per-prompt baseline and the withdrawn arm's removal? | `$0` |
| `_falsifier/honesty_result.json` | `_falsifier/honesty_score.py`, over the blind file from `_falsifier/honesty_blind.py` | `_falsifier/honesty_result.md` | does the kindness effect also appear on an honesty-only rubric? | `$0` |

All three audit artifacts are `$0` and stay that way by construction: they re-read committed
per-item records and make no new generations and no new judge calls. An audit that needed
compute would not get run often enough to matter.

## The gap list, accounted for

21 rows above read **none** in the `interpreted in` column, and as of 2026-09-07 not one of
them is a missing analysis write-up. The accounting:

- **13 are inputs or intermediates**, not results, and deliberately have no prose: the four
  blind keys, the four raw judge-verdict files, the two arms files, the seen-prompts file,
  `site_prefixes.json` and `prefix_gallery_judge.json`. A key or a per-pair verdict dump is
  evidence for a write-up, not a claim needing one.
- **3 are the blind CSVs**, which are rating worksheets. Two of them are gitignored.
- **`banded_score_arms.json` is merged** into `banded_direction.md`, following the precedent
  of `steering_ablation.md` covering three artifacts under its primary stem.
- **`steering_ablation_check.json` and `steering_ablation_measure.json` are transcribed in
  full** inside `steering_ablation.md`, at its "What was removed" and "Manipulation check"
  sections. They are the `measure` and `check` subcommands of one script, not separate
  analyses.
- **`_falsifier/verify_result.json` is the audit output**, regenerated by running
  `verify.py`, and its prose home is `_falsifier/README.md`.

So every orphaned *analysis* in `data/analysis/` now has a companion. That was not true this
morning: 17 did not, including `season3_gcg_aggregate_asymmetry.json`, which held the whole
Part A mechanism result and was cited by four other files with no prose anywhere.

## What this buys you

Corrections cost nothing here, because per-item records were kept. 14 published effects were
recomputed with zero new generations and zero new judge calls, and a cache corruption was
located and repaired with 30 metadata rewrites and no compute at all. The artifact discipline
is why: the raw records survive the analysis, so a wrong baseline or a withdrawn arm is a
re-read rather than a re-run. The 21 rows above whose `interpreted in` column says **none**
are the standing work list, and they are listed rather than hidden for the same reason.
