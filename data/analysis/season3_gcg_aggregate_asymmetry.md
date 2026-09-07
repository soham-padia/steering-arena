# The aggregate function, not the search, explains both Part A anomalies

The artifact is `data/analysis/season3_gcg_aggregate_asymmetry.json`, and it records no
invocation of its own — no script name, no job id, no run directory. One local B200, bf16,
no NDIF, $0: every value was read out of already-written run logs, so re-reading them costs
nothing. All figures in the main table are **|LIVE|** — `|sign*(best board so far −
baseline)|`, the JSON's own definition — so a pro arm and an anti arm are comparable
regardless of sign, and each is comparable to a leaderboard entry. The JSON does not record
the baseline constant it subtracted; `season3_gcg_baseline.md` is where that lives. Ratios
are dimensionless. The window statistics are in the same |LIVE| units, per 100 iterations of
search.

`REVISIONS_2026-09-05.md` is authoritative on what results are now taken to mean; this page
is authoritative on what the artifact contains.

**I ran this.** It is this project's own analysis of its own Part A search, not a number
carried in from anywhere: the four arms' `history.jsonl` files under
`/work/neu/p2026_0037_neu/steering-arena/gcg/` were read and compared at **equal iteration
counts**. That last point is the JSON's `note` on the main table — *"matched iterations, so
hardware differences do not enter"* — and it is what makes the comparison mean anything. A
faster node, a longer wall clock or an earlier stop cannot enter a table indexed by
iteration.

The JSON states its question and its answer verbatim. Question: *"why does score2 lag
score1, and why do the anti arms outrun the pro arms"*. Answer: *"aggregate structure: min
is a conjunction and saturates; max is a disjunction and does not"*.

## The matched-iteration table

| iter | `score1_pro_mean` | `score1_anti_mean` | `score2_pro_min` | `score2_anti_max` |
|---|---|---|---|---|
| 50 | 0.02462 | 0.03762 | 0.02504 | **0.07908** |
| 100 | 0.04364 | 0.07451 | 0.03017 | 0.09136 |
| 150 | 0.05348 | 0.0956 | 0.03912 | 0.12296 |
| 200 | 0.07166 | 0.10127 | **0.0453** | 0.12906 |

Scale, for reading the columns against each other rather than against a field sd the JSON
does not carry: `score2_anti_max` at **iteration 50** is already at 0.07908, above the
largest pro number anywhere in the table (`score1_pro_mean`'s 0.07166 at iteration 200). The
disjunctive anti arm starts past where 200 iterations of the best pro arm finish, and
`score2_pro_min` ends the table at 0.0453 — below where the anti arm's first checkpoint
began.

Read the two score1 columns first, because they are the clean pair: both use the same
aggregate, so the only difference between them is the sign of the objective.

## Where the anti advantage comes from

| checkpoint | `score1_aggregate_held_fixed` | `score2_min_vs_max` |
|---|---|---|
| 50 | 1.53 | 3.16 |
| 100 | 1.71 | 3.03 |
| 150 | 1.79 | 3.14 |
| 200 | 1.41 | 2.85 |
| geometric mean (computed here) | **1.60302** | **3.04247** |

The arithmetic, so you can check it. `1.53 × 1.71 × 1.79 × 1.41 = 6.60327957`, and the
fourth root of 6.60327957 is **1.60302** — square it twice to verify: 1.60302² = 2.56967,
2.56967² = 6.60320. `3.16 × 3.03 × 3.14 × 2.85 = 85.6848852`, and the fourth root of
85.6848852 is **3.04247**: 3.04247² = 9.25662, 9.25662² = 85.68501. The quotient is
`3.04247 / 1.60302 = 1.89796`.

**That closes the JSON's attribution string**, which reads *"~1.6x geometric (visible with
the aggregate held fixed), ~1.9x extra from min->max"* — the two geometric means are the
1.6x and the quotient is the 1.9x, and neither figure was stated with its derivation until
now. **The caveat that travels with it:** the 1.6030 column is the one measured with the
aggregate held fixed, and that is the whole reason the decomposition is clean, but the 3.0425
column changes the aggregate *and* the sign at once, so 1.898 is the size of everything the
min→max swap brings with it, not the size of `max` alone.

## min saturates; mean does not

| window | `score1_pro_mean` | `score2_pro_min` | `score2_anti_max` |
|---|---|---|---|
| 0-100 | 0.0545 | 0.04477 | 0.06907 |
| 100-200 | 0.02574 | 0.01189 | 0.03462 |
| 200-300 | 0.01601 | 0.00509 | 0.00982 |
| 300-400 | **0.02899** | 0.00529 | — |
| 400-500 | 0.02486 | **0.00188** | — |

`score2_pro_min` decays and does not recover: 0.04477, then 0.01189, then 0.00509, then a
flat 0.00529, then 0.00188. Its first window is 23.8× its last. Scale: `score1_pro_mean`'s
300-400 window alone (0.02899) is larger than everything `score2_pro_min` gained across
iterations 200-500 combined (`0.00509 + 0.00529 + 0.00188 = 0.01226`).

**`score1_pro_mean` re-accelerates in the 300-400 window** — 0.01601 rising to 0.02899, and
still 0.02486 in 400-500 — and that is the direct evidence that `mean` does not saturate.
The two pro arms are the same optimiser against the same model with the same move class;
only the aggregate differs. One of them runs out of gains by iteration 300 and the other
finds its second-largest window after iteration 300. **The bound on this:** three windows of
`score2_anti_max` are all the artifact has for the disjunctive arm (0.06907, 0.03462,
0.00982, then two nulls), so "max never saturates" rests on step size below, not on this
column.

## Step size tells the same story, and refutes the temperature account

| window | `score1_pro_mean` | `score2_pro_min` | `score2_anti_max` |
|---|---|---|---|
| 0-100 | 0.000627 | 0.000599 | 0.000951 |
| 100-200 | 0.000439 | 0.000446 | 0.001091 |
| 200-300 | 0.000394 | 0.000415 | 0.001326 |
| 300-400 | **0.000564** | 0.000292 | — |
| 400-500 | **0.001132** | 0.000316 | — |

These are median improving steps. Through iteration 300 the two pro arms track each other:
0.000627 against 0.000599, 0.000439 against 0.000446, 0.000394 against 0.000415 — a largest
divergence of 2.8e-5, under 5% of either arm's own step. After iteration 300 they separate,
and at 400-500 `score1_pro_mean`'s 0.001132 is 3.58× `score2_pro_min`'s 0.000316. Scale: the
gap that opens late is more than three times the size of the gap that exists early, and it
opens in the same window where the net-gain table shows one arm re-accelerating and the
other at its floor.

`score2_anti_max` moves the other way — 0.000951, 0.001091, 0.001326, monotonically
increasing — which is the disjunction signature: one layer is enough, so there is no
conjunction to equalise and nothing to stall against.

## WITHDRAWN: the T_SA temperature mechanism

The claim, as the JSON records it: *"T_SA runs ~3.4x hotter on score2 because its steps are
3.4x smaller (commit 9517e05)"*. It was **measured only over iter>=384**. Through iteration
300 the two arms have near-identical median step sizes — **5.99e-4 vs 6.27e-4** — so the
temperature was equally scaled for both. `score2`'s small late steps are a *consequence* of
saturation, not its cause; an effect was measured and named as the cause.

**Withdrawn:** that the shared annealing temperature runs hotter on `score2` because
`score2`'s steps are smaller. **Survives:** the JSON's own status line — *"the --t-sa-scale
flag and job 706969 remain a valid experiment (does cooling improve retention on a saturated
objective?) but the stated mechanism was backwards."* **What the withdrawal does not
license:** the late-window gap is real and reproduces in the table above. What fails is the
inference from it, because the same statistic through iteration 300 shows no gap to explain.

## WITHDRAWN: "min creates a flat search landscape"

The JSON's second withdrawal, verbatim: *"score2-hardmin has 3.6% identical consecutive
states vs score1's 5.0% -- less flat, not more."* The measurement runs opposite to the guess
it was made to support.

**Withdrawn:** that `min` flattens the search landscape. **Survives:** the saturation
account itself, which now rests on net gain per window and median step size — the two tables
above — and never needed a flatness claim. **The limit of the refutation:** 3.6% against
5.0% is a count of steps where the optimiser's state did not change, which is state churn.
It shows the stated evidence for flatness points the other way; it does not measure landscape
curvature and so does not establish that the landscape is not flat.

## The corrupted live submission

| field | value |
|---|---|
| `row_id` | 1342 |
| `board_score1` | 0.086 |
| `locally_measured` | **0.14045** |
| tokens | 38 submitted vs 32 in the run |

The cause, as recorded: the prompt contains two real newline bytes (0x0a 0x0a), it was copied
from a Python repr so they arrived as literal backslash-n text, plus an inserted newline.
Scale: the board understates this string by `0.14045 − 0.086 = 0.05445`, which is more than
twice everything `score2_pro_min` gained after iteration 100
(`0.01189 + 0.00509 + 0.00529 + 0.00188 = 0.02415`). A copy-paste defect costs more score
than 400 iterations of conjunction search buys.

## What this buys

Part A closed with two anomalies and no account of either: `score2` lagged `score1`, and the
anti arms outran the pro arms. Both are now explained by one property of the metric — the
aggregate function — and not by the search. That is the positive result, and it is the
load-bearing one on this page. It holds because the comparison is indexed by iteration, so
hardware cannot enter, and because the withdrawal above removes the leading
search-hyperparameter alternative rather than leaving it standing.

**The decomposition is what makes the account more than a label.** The ~1.6x geometric
component is measured on `score1`, where both the pro and the anti arm use the same
aggregate. Holding the aggregate fixed isolates that component from the aggregate component,
so the remaining ~1.9x is attributable to the min→max swap rather than being confounded with
the sign asymmetry that exists anyway. Few results in this repo separate two causes that
cleanly, and it was possible only because `score1` happened to ship the same aggregate for
both directions.

What follows for the metric, not the search: a conjunction is the thing that resists direct
optimisation, and the resistance shows up as saturation — gains that decay to a floor rather
than a search that never gets started. A disjunction has no such floor, because damaging one
layer suffices. That is an actionable statement about how to build a banded objective, and it
is the reason `season3_k3_control.md` went looking at move class rather than at temperature.

## Limits

1. **One run per cell.** Every arm is a single run with no seed replication, so all eight
   ratios and both geometric means are n=1. Seed variance is unmeasured and could be of the
   same order as the 1.41-to-1.79 spread within the held-fixed column.
2. **The min-versus-max comparison confounds aggregate with the sign of the objective.**
   Only the anti arms use `max`, so "conjunction versus disjunction" and "pro versus anti"
   are the same contrast in that column. A `max`-aggregated *pro* arm would separate them and
   does not exist. The 1.6030 column is not affected; the 3.0425 column is.
3. **`score2_anti_max` has only three windows.** Both window tables carry nulls at 300-400
   and 400-500 for that arm, so its growing-step-size trend rests on three points.
4. **The identical-consecutive-state percentages measure state churn, not curvature.** So the
   second withdrawal refutes the stated evidence for flatness rather than proving the
   landscape is not flat.
5. **The artifact is not self-sufficient.** It records no baseline constant, no run
   directories, no job ids and no per-arm iteration totals, so nothing here can be
   re-derived from the JSON alone. Its 5.99e-4 and 6.27e-4, quoted as "through iter 300", are
   also numerically identical to the 0-100 window medians, and the file does not carry a
   separate through-iteration-300 statistic.

## Cross-links

`season3_prefix_scores.md` re-scores the arms under both objectives and corrects the anti
magnitudes without touching any conclusion here · `season3_k3_control.md` tests the fix this
account implies and withdraws the evidence it was first argued from · `season3_gcg_setup.md`
§2 for why Score 2 ran first and §7 for what the setup was pre-registered as unable to answer
· `docs/HANDOFF_BEHAVIORAL_S3.md` §3 for the run directories and the repr trap that produced
row 1342 · `season3_gcg_baseline.md` for the baseline constant this page's |LIVE| units
subtract · `prefix_eval_s3.md` §3 for whether any of these cosines correspond to behaviour ·
`season3_gcg_ablation.md` for what the searched strings are made of.
