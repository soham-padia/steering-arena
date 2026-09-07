# How to run a blind behavioural eval on a set of prefixes

The operator's procedure for asking whether a set of prefixes changes what the model *says*,
not only where its activations point: freeze the arms, gate them, generate, blind, get them
rated, merge, and run the statistics. This page covers the operator's side only. **It does
not contain the rater's instructions** — those are `docs/PREFIX_BLIND_RUBRIC.md`, which you
hand to the rater and never paraphrase here.

**Prerequisites:** the `steering-arena` conda env, run from the repo root. Exactly one stage
needs a GPU — generation, which is the sbatch in step 3 (one B200, ~40 s of model load,
then 50 prompts × 7 conditions). Everything else — `select-gcg`, `blind`, `claude-batches`,
`claude-merge`, `degeneration`, `content`, `stats` — is CPU-only and belongs on
`sh ~/cpu.sh` or the login node. No NDIF calls and no quota are spent; a continuation is
not a score. Generations land in `data/cache/prefix_behavioral_s3/`, which is **gitignored**.

Log the GPU job before you launch it:

```bash
python ~/trauma-experiment-gpt/bin/runlog.py add --why "Part A' prefix eval, s3 arms"
```

## 1. Smoke it on three prompts first

```bash
sbatch slurm/prefix_behavior_s3.sbatch --limit 3
```

Everything after the script name is forwarded to `generate`, so `--limit 3` caps the prompt
set. This exercises both phases — the fidelity gate and the generation loop — for a few
minutes of GPU instead of the full run, and the cache it writes is reused by the real run
rather than thrown away. Read the log at
`/work/neu/p2026_0037_neu/steering-arena/logs/pfx_s3_<jobid>.out` and confirm Phase 1
printed `ok` for every arm before you spend the full allocation.

## 2. Freeze the arms

```bash
python scripts/prefix_behavior_eval.py --tag s3 select-gcg --season-id 5
```

This reads the current `best.json` of each GCG run directory and writes
`data/analysis/prefix_eval_arms_s3.json` with eight arms (`score1_top`, `score2_top`,
`score2_top_final`, `pro_coherent`, `random32`, `score1_anti`, `score2_anti`,
`score2_anti_final`). To pull a newer string for one arm after its run has advanced:

```bash
python scripts/prefix_behavior_eval.py --tag s3 select-gcg --refresh score2_top_final
```

`--refresh` takes a comma-separated list, and **every arm not named in it is preserved
byte-exact.** That is deliberate: `best.json` keeps improving while a job runs, so a
re-run must not be able to silently advance a string you have already generated 50
continuations from. Overwriting the whole file requires `--force`, and doing so orphans
every generation already in the cache.

## 3. Gate the arms and generate (the GPU stage)

```bash
sbatch slurm/prefix_behavior_s3.sbatch
```

Two phases on one model load. **Phase 1** re-scores every frozen string under both banded
objectives and checks it reproduces the number its run recorded
(`scripts/score_banded_local.py --arms … --update-arms`); it also fills in the recorded
score for the `random32` control, which has none. **Phase 2** generates the continuations —
50 prompts × 8 prefixed arms plus the unprefixed base, greedy, 40 new tokens.

Generation is resumable and keyed per `(prompt, arm, max_new, backend)`, so re-running fills
gaps rather than redoing work. The equivalent direct form on an interactive node:

```bash
python scripts/prefix_behavior_eval.py --tag s3 generate --backend local --max-new 40
```

`--backend local` runs the full-depth HF model on this node. NDIF stays canonical for
*scores*; it is not involved in generating text.

## 4. Build the blinded CSV

```bash
python scripts/prefix_behavior_eval.py --tag s3 blind
```

Writes `data/analysis/prefix_blind_s3.csv` — the rater's file, containing only the
continuations with the model input stripped, plus the shared eval prompt for context — and
`data/analysis/prefix_blind_key_s3.json`, the unblinding key, which the rater must never
see. Pairs where the strip failed are dropped; pairs where the model echoed the prefix
verbatim are kept and flagged `leak` in the key. `--force` rebuilds over existing ratings.

## 5. Emit the batches

```bash
python scripts/prefix_behavior_eval.py --tag s3 claude-batches --only-unjudged
```

Every pair is emitted in **both presentation orders**, split so that no single batch — and
therefore no single rater context — ever holds both orientations of the same pair. That is
what makes the swap an independent second opinion instead of a memory test. Batches land in
`data/cache/prefix_behavioral_s3/claude/in/`, named by pair range
(`batch_fwd_p1-25.json`). `--only-unjudged` skips pairs that already have verdicts in
*both* orientations; a pair with one orientation is re-emitted, because `claude-merge` drops
half-judged pairs.

## 6. Hand each batch to a separate rater context

Give each batch file, and `docs/PREFIX_BLIND_RUBRIC.md`, to a **separate** agent context —
one batch per context, no exceptions. Each rater writes its verdicts to
`data/cache/prefix_behavioral_s3/claude/out/` under **the same filename** as its input.
Batches must not be pooled into one context: co-locating two batches recreates the
cross-orientation leak that step 5 exists to prevent.

## 7. Merge the verdicts

```bash
python scripts/prefix_behavior_eval.py --tag s3 claude-merge --label claude-opus-5
```

Merges `out/` against the unblinding key, un-flips the reversed orientation, averages the
per-text kindness ratings crosswise, intersects and unions the markers, and writes
`data/analysis/prefix_judge_claude_s3.json`. It prints how many pairs were decided, how
many **abstained on the A/B swap**, and how many are missing an orientation.

## 8. Run the two mechanical measures

Neither needs a judge, a GPU, or the network — both read the generation cache:

```bash
python scripts/prefix_behavior_eval.py --tag s3 degeneration
python scripts/prefix_behavior_eval.py --tag s3 content
```

`degeneration` reports word counts, distinct-4gram and distinct-word ratios, a looping flag
and maximum repetition per arm. `content` asks whether the prefix's own vocabulary reappears
downstream — a word counts only if it appears in the prefix and in **none** of the 50
unprefixed base continuations. Run both before reading the judge results: a kindness verdict
cannot distinguish steering an abstract direction from injecting topic words, and these two
are what separate those mechanisms.

## 9. Run the statistics

```bash
python scripts/prefix_behavior_eval.py --tag s3 stats
python scripts/prefix_behavior_eval.py --tag s3 stats --report /tmp/reanalysis.json
```

`stats` needs the arms file, the blind key, the judge files and the generation cache; it
computes per-arm win/loss/tie counts per rater, sign tests, inter-rater agreement, and
scopes results with and without the leak and loop flags. With no `--report` it writes
`data/analysis/prefix_eval_s3.json`, the experiment's own report path. Pass `--report` when
re-analysing a published run so the published JSON is not overwritten.

## What goes wrong

**Omitting `--tag s3`.** `--tag ""` is the default and it is the **frozen Season 2
namespace** — `prefix_eval_arms.json`, `prefix_blind.csv`, `prefix_eval.json`, all cited
line-by-line in `data/analysis/prefix_eval.md`. Running `generate` or `blind` without the
tag writes over a published record and makes those numbers unreproducible. Every command in
this procedure carries the tag; check for it before pressing return, and `git status`
afterwards if you are unsure.

**Running `stats` on a fresh clone.** It fails looking for
`data/cache/prefix_behavioral_s3/`, which is gitignored. The generation cache is not in the
repo and cannot be, so `stats`, `degeneration` and `content` are not reproducible from
source alone — they require either the cache from `/work` or a re-run of step 3 on a GPU.
This is a property of the pipeline, not a misconfiguration.

**Naming a second round's batches by sequence number.** Pair ids are stable across a
`blind` rebuild, which is why verdicts already in `out/` stay valid and `--only-unjudged`
works at all. But a second round emitting its own `batch_01` and a rater writing
`out/batch_fwd_01.json` would **overwrite the first round's verdicts for a completely
different set of pairs**, silently and with no size change to notice. Batch files are named
by pair range for this reason; keep that convention if you generate them by hand.

**A rater that opens the unblinding key.** The run is invalidated — not weakened, not
caveated. The rubric names the exact files a rater must not open, and a rater who has seen
the key cannot un-see which arm a text came from. If it happens, discard that rater's
verdicts entirely and re-rate those batches in a fresh context; do not attempt to keep the
pairs you believe were unaffected. Blinding is imperfect even when the procedure holds — an
instruction-style prefix leaves its register on the continuation — which is why the rubric
asks for kindness only and never provenance.

**Treating abstentions as missing data.** An abstention is a pair where the two presentation
orders **disagreed** about which text was kinder. Discarding it is the intended debias: an
order-dependent verdict measures position bias, not kindness. `claude-merge` reports
abstentions separately from missing orientations for exactly this reason, and a high
abstention rate is a finding about the arm, not a defect in the run.

## Verify it yourself

The one check worth running before the GPU stage, because it is what the whole eval rests
on — that the strings in the arms file are the strings the runs found:

```bash
python scripts/score_banded_local.py --arms data/analysis/prefix_eval_arms_s3.json
```

Every arm prints its re-scored value against its recorded one with an `ok` or `MISMATCH`
flag. A `MISMATCH` is a corrupted string; stop and fix the arm before generating 50
continuations from it.

## Cross-links

`docs/PREFIX_BLIND_RUBRIC.md` (the rating protocol — the rater's document, not this one) ·
`data/analysis/prefix_eval_s3.md` (the Season 3 results) ·
`data/analysis/prefix_degeneration_s3.md` (the looping measures) ·
`data/analysis/prefix_content_s3.md` (the content-injection measures) ·
`data/analysis/prefix_loop_control.md` (the loop control) ·
`slurm/prefix_behavior_s3.sbatch` (the two-phase GPU job) ·
`docs/how-to/move-a-searched-string-without-corrupting-it.md` (the fidelity gate) ·
`docs/HANDOFF_BEHAVIORAL_S3.md` (why the experiment exists).
