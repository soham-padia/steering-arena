# The constant that turns a GCG number into a board number

`python scripts/gcg/baseline_const.py` → `data/analysis/season3_gcg_baseline.json`. One
local forward pass over the 16 frozen `season3` probes, one B200, bf16, **zero NDIF calls,
$0**.

**I ran this.** `REVISIONS_2026-09-05.md` is authoritative on what results are now taken to
mean; this page is authoritative on what the artifact contains.

## The two constants

```
live_score = board_score − BASELINE
```

| role | band | aggregate | baseline |
|---|---|---|---|
| Score 1 | 19, 23, 27, 31 | `banded_mean` | **−0.00101278659349191** |
| Score 2 | 15, 23, 31, 39 | `per_layer_min` | **−0.009410522776306607** |

`BASELINE` is the mean over probes of the aggregated cosine of each probe's own last-token
residual with the direction — what the probes score before any prefix is attached. Each
value is exactly the mean of its own 16 per-probe entries, which I verified: −0.0010128 and
−0.0094105 to the precision the file stores. So the file is internally consistent and you
can check it without a GPU.

## Why the optimiser omits it

`scripts/gcg/optimize_banded.py` does not subtract it. That is deliberate and it is not a
bug: the baseline does not depend on the prefix, so it cannot change which candidate wins,
and skipping it saves a forward pass over up to 1024 candidates per iteration. The
consequence is that a run's own `board_score` is offset from a leaderboard score by exactly
this constant, per role — which is the whole reason this file exists. Without it, every
number in every Part A run log is merely relative.

**The sign rule that goes with it**, because getting the order wrong is a 2×baseline error:
`live = sign * best − baseline`. Flip first, subtract once. See
`PROJECT_SPEC.md` §5 and the withdrawal in `season3_prefix_scores.md`.

## The per-probe spread is the interesting part

| role | per-probe min | per-probe max | probes above zero | sd (*derived*) |
|---|---|---|---|---|
| Score 1 | −0.037256 | +0.047810 | 2 of 16 | **0.017436** |
| Score 2 | −0.049120 | +0.009901 | 1 of 16 | 0.012303 |

Two things follow, and the second is a warning.

Score 2's baseline is **9.3× more negative** than Score 1's. That is structural, not
noise: `min ≤ mean` for any set, so an aggregate that takes the worst band layer sits below
one that averages them. Nothing about the probes changed between the two roles — the probe
set is byte-identical — only the aggregate.

**The spread is comparable to the entire field.** Score 1's per-probe sd is 0.017436
(*derived* here, not stored in the file), against the 618-entry field sd of **0.01826**
quoted in `docs/HANDOFF_BEHAVIORAL_S3.md` §2. The 16 probes disagree with each other about
as much as the whole leaderboard disagrees with itself. That is why
`scripts/score_banded_local.py` and `app/scoring.py` subtract the baseline **per probe**
rather than applying this scalar: the scalar converts an aggregate score, and it does not
license per-probe reasoning. If you find yourself subtracting −0.00101 from a single probe's
cosine, you are using this file wrong.

Probe 3 (index 2) is the extreme on both roles, at −0.037256 and −0.049120. Probe 16 is the
most positive on Score 1 at +0.047810 and **negative** on Score 2 at −0.004864, so
per-probe alignment does not carry across objectives either.

## What this buys

Every number in every Part A run log becomes convertible to a leaderboard number instead of
merely relative — which is what `season3_gcg_aggregate_asymmetry.md`,
`season3_k3_control.md`, `season3_prefix_scores.md` and the handoff's reference block all
depend on. It cost one forward pass over 16 short prompts and no NDIF quota, and it made a
whole run family comparable to the live board.

## Limits

1. **16 probes, one model build, one dtype.** The constants are properties of a frozen
   tuple. If NDIF re-serves the model, or the local build changes, or the probe set is
   edited, both constants change and the season is broken by definition
   (`PROJECT_SPEC.md` §5.4).
2. The per-probe sd is derived here and is not stored in the artifact. Recompute it rather
   than quoting this page if it matters.
3. Measured locally in bf16, not on NDIF. The local/NDIF agreement bound of |gap| ≤ 3.71e-4
   applies to prefix scores, not specifically to these constants; nobody has re-measured
   them remotely.
4. The file records no `d_tag`, so it does not itself prove which direction files produced
   it. The band and aggregate fields identify the role, and the directions are
   `d_olmo3_s3_score{1,2}.npz` — but that is inference from the role name, not a recorded
   fact.

## Cross-links

`season3_gcg_aggregate_asymmetry.md` and `season3_k3_control.md` (both convert with these
constants) · `season3_prefix_scores.md` (the sign-order withdrawal) ·
`season3_gcg_setup.md` §3 · `docs/HANDOFF_BEHAVIORAL_S3.md` §2 · `season3_directions.md` ·
`PROJECT_SPEC.md` §5.
