# How to search a banded objective with GCG

Launching, resuming and reading a GCG + simulated-annealing run against a Season 3 banded
objective — Score 1 or Score 2, pro or anti. The run produces a `best.json` in a run
directory and nothing more: no submission, no board score, no analysis. Turning that file
into either is a separate procedure.

**Prerequisites:** the `steering-arena` conda env — **not** `sa-ndif`, whose torch is
CPU-only. Steps 1, 3, 4 and 5 launch the optimiser and need **one B200** with the
`allenai/Olmo-3-1125-32B` weights present under `$HF_HOME` (`/scratch/$USER/hf`, purged
after 30 days). Step 6 reads files and needs no GPU. No NDIF calls and no quota are spent
at any point. Log the job before you launch it:

```bash
python ~/trauma-experiment-gpt/bin/runlog.py add --why "score2 GCG, k=3 mutations"
```

## 1. Pick the role

`--role score2` (the default) is the per-layer **min** over layers `{15,23,31,39}`, each
against its own direction — the conjunctive objective, the one the season exists to test.
`--role score1` is the banded **mean** over `{19,23,27,31}` against a single averaged
direction; it is the positive control, against a target already known to be beatable.

```bash
sbatch slurm/gcg.sbatch score2 32 --n-mutations 3
```

The sbatch takes the role and the token count **positionally**, in that order, and forwards
everything after them to the script. The equivalent direct form, for an interactive node:

```bash
python scripts/gcg/optimize_banded.py --role score2 --n-controlled-tokens 32 --n-mutations 3
```

`--max-iters` defaults to `0`, meaning run until killed, so the sbatch's `--time=04:00:00`
is what ends the run. `--cand-chunk` (default 4) is the memory/speed knob; raise it only if
you have headroom on the card.

## 2. Note where the run went

```
run dir: /work/neu/p2026_0037_neu/steering-arena/gcg/score2-mut3-2026-09-07T00-17-08Z
```

The name carries the role, `-anti` when the sign is flipped, `-mutN` for `N > 1`, and a
UTC timestamp. Run directories live on `/work` and never on `/home`, which is nearly full.
You need this path for step 5 and for every downstream step.

## 3. Search the opposite sign with `--anti`

```bash
sbatch slurm/gcg.sbatch score2 32 --anti
```

`--anti` negates the direction **and, for Score 2, swaps the aggregate from `min` to
`max`**, because

```
min_L cos(R, −d) = −max_L cos(R, d)
```

Negating the direction alone would leave the aggregate as a `min` over the negated
cosines, which optimises the model's **best** layer instead of its worst — a different
objective that still prints plausible-looking, monotonically improving numbers. There is no
error message for getting this wrong. The `-anti` in the run-directory name and the `anti`
field in every record are there so the two arms can never be confused on disk, since an
anti run's own logs read positive-is-better like everything else.

## 4. Use `--softmin-search` as a search surrogate only

```bash
sbatch slurm/gcg.sbatch score2 32 --softmin-search
```

Score 2 only. The hard `min` routes the gradient to one layer per step; the soft-min
spreads it across all four, weighted toward the weakest. **The reported and recorded scores
always use the true `min`** — the flag changes only what the search climbs, and the record
carries both `search_score` and `score` so the two never merge. A soft-min number is never
a reported score. `watch.py` prints an extra `search` column while a surrogate is in use.

A soft-min run and a hard-min run at the same role and token count *are* comparable to each
other: same objective, different search. That comparison is the reason the flag exists.

## 5. Resume an interrupted run

```bash
python scripts/gcg/optimize_banded.py --role score2 --n-controlled-tokens 32 --n-mutations 3 \
  --resume-from /work/neu/p2026_0037_neu/steering-arena/gcg/score2-mut3-2026-09-07T00-17-08Z
```

Resuming reads `latest.json` from that directory, restores the control token ids, continues
the iteration counter, and restores the best-so-far threshold from `best.json`'s
`board_score`. Two assertions fire before any GPU work: the checkpoint's token count must
match `--n-controlled-tokens`, and its `d_tag` must match the direction now loaded — one
run directory must not contain two objectives. Pass the same role and flags you launched
with; `--resume-from` restores the *state*, not the configuration.

## 6. Read the run

```bash
python scripts/gcg/watch.py                       # one snapshot of every arm
python scripts/gcg/watch.py --every 120 --for 3600
```

Read the **`board`** column. `score` is the optimiser's number on the raw token ids and
almost always runs ahead of it — measured across the eight Season 3 runs, `score > board`
on **4219 of 4236** logged iterations, with 17 exceptions — so the difference is
retokenisation drift rather than progress. It is not guaranteed one-signed, so read the
gap, do not assume it.

Three files per run directory:

- **`best.json`** — the record with the highest `board_score` seen so far. This is the
  artifact of the run, and the only one you take a string from.
- **`latest.json`** — the current iteration, overwritten atomically each step. This is what
  `--resume-from` reads.
- **`history.jsonl`** — one record per iteration, append-only, for trajectories.

Every record carries its own provenance: `role`, `band`, `aggregate`, `search_aggregate`,
`anti`, `n_mutations`, `d_version`, `d_tag`, `board_score`, `board_score_true_sign`,
`roundtrip_ok`, `ctrl_token_ids` and `ctrl_token_ids_retokenised`. Read `board_score` for
the magnitude and `board_score_true_sign` when you need the sign an anti entry would show
on the board.

## What goes wrong

**Quoting the optimiser's number as a board score.** Neither `score` nor `board_score` is a
leaderboard score. Both drop the per-probe baseline `mean_p agg_L cos(probe)`, which is
constant with respect to the prefix and so cannot change which candidate wins. Convert with
`live = sign * best − baseline` — **flip the sign first, subtract once.** The other order is
a 2×baseline error. Constants are in `data/analysis/season3_gcg_baseline.md`.

**Reading `rt_ok=False` as a broken guard.** You will see `rt_ok=False` on nearly every
iteration and `n_not_roundtrip` climbing, and that is the guard working. Rejecting
non-round-tripping candidates was the first design and it rejects essentially everything
from iteration 0 — GCG's all-`!` init already merges under BPE — so the guard silently
disabled itself and `best.json` was never written at all. The guard now **scores the
re-tokenised prefix every iteration** instead, over the same probes with the same
aggregate, and picks `best` on that number. Nothing to fix; read `board`.

**Leaving `--n-mutations` at 1 on Score 2.** Single-position mutation saturates on the
conjunctive objective at around **+0.045** and then stops moving, because no single-token
edit lifts every band layer at once. Multi-position mutation is what helps; it costs
acceptance rate. Numbers and the control in `data/analysis/season3_k3_control.md`.

**Reaching for `--t-sa-scale` to fix saturation.** The cooling mechanism is **refuted, not
merely unsupported**: the step-size gap it was premised on appears only after iteration 384,
the two arms' steps are near-identical through iteration 300, and the cooled arm came back
worse than baseline — the small late steps are a consequence of saturation, not its cause.
The flag's own help text asserted the withdrawn mechanism as fact until 2026-09-07. It is
kept because "does cooling help a saturated conjunction" remains a valid question, and it is
not a recommended setting.

**A cooled arm you cannot identify later.** `--t-sa-scale` is **not** in the checkpoint
schema — the record stores `n_mutations`, `band`, `aggregate`, `search_aggregate` and
`anti`, but not the temperature scale, and it is not in the run-directory name either. A
cooled run is therefore indistinguishable from an uncooled one by its own artifacts. If you
set it, record the value in the runlog entry and in the run directory yourself, at launch
time; there is no recovering it afterwards.

**`set -u` in the sbatch script.** It kills the batch shell on this cluster: the job exits
almost immediately with an empty or truncated `.out`. `slurm/gcg.sbatch` uses `set -eo
pipefail` deliberately. Do not add `u` when you edit it.

## Verify it yourself

Before you trust any number from a run, print what it actually optimised:

```bash
python3 - <<'PY'
import json
run = "/work/neu/p2026_0037_neu/steering-arena/gcg/score2-mut3-2026-09-07T00-17-08Z"
b = json.load(open(f"{run}/best.json"))
for k in ("role", "band", "aggregate", "search_aggregate", "anti", "n_mutations",
          "d_version", "d_tag", "iter", "board_score", "board_score_true_sign",
          "roundtrip_ok"):
    print(f"{k:24} {b[k]}")
PY
```

`aggregate` must be `max` on a Score 2 anti run and `min` on a Score 2 pro run. If an anti
run shows `min`, the sign handling is wrong and every number in the directory is against the
wrong objective.

## Cross-links

`scripts/gcg/README.md` (the port, the objectives, the retokenisation guard) ·
`data/analysis/season3_gcg_setup.md` (how the arms were configured) ·
`data/analysis/season3_gcg_aggregate_asymmetry.md` (what the two aggregates do differently) ·
`data/analysis/season3_k3_control.md` (the `--n-mutations` control) ·
`data/analysis/season3_gcg_baseline.md` (the baseline constants for the conversion) ·
`docs/how-to/move-a-searched-string-without-corrupting-it.md` (getting the string out).
