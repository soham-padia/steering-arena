# Multi-position mutation does fix the conjunction — but not for the reason it was argued from

`python scripts/gcg/k3_control.py` → `data/analysis/season3_k3_control.json`. No GPU, no
NDIF, $0: it reads the committed `history.jsonl` of four Season 3 `score2` run directories.
All values are **|LIVE|** — `|board score in board sign − baseline|`, baseline
`−0.009410522776306607` from `season3_gcg_baseline.json` — so each is comparable to a
leaderboard entry regardless of sign.

`REVISIONS_2026-09-05.md` is authoritative on what results are now taken to mean; this
page is authoritative on what this artifact contains.

The pre-registered criterion, from commit `6304f2c`, written before either k=3 arm finished:

> Arms 707082 (pro) and 707083 (anti) run k=3. The anti arm is the control: `max` is a
> disjunction and never saturated, so if k=3 helps it as much as pro, the effect is generic
> search improvement and not a conjunction fix.

Both arms have now finished — `707083` ran as `score2-anti-mut3-2026-09-07T04-17-25Z`,
after every other Season 3 `.md` was last written — so the criterion can finally be
applied. It returns the wrong answer, and the reason is instructive.

## The matched-iteration table

Best-so-far |LIVE|, capped at **501 iterations**, the shortest run in either pair. Read
from `history.jsonl` and not from `best.json`, because the four runs stop at different
iterations and an endpoint comparison would confound k with run length.

| iter | pro k=1 | pro k=3 | ratio | anti k=1 | anti k=3 | ratio |
|---|---|---|---|---|---|---|
| 50 | +0.02504 | +0.02457 | **0.981×** | +0.06026 | +0.08956 | 1.486× |
| 100 | +0.03017 | +0.03722 | 1.233× | +0.07254 | +0.08956 | 1.235× |
| 150 | +0.03912 | +0.04508 | 1.152× | +0.10414 | +0.08956 | **0.860×** |
| 200 | +0.04530 | +0.05105 | 1.127× | +0.11024 | +0.10215 | **0.927×** |
| 250 | +0.04530 | +0.05105 | 1.127× | +0.11902 | +0.10215 | **0.858×** |
| 300 | +0.04530 | +0.05715 | 1.262× | +0.11902 | +0.10215 | **0.858×** |
| 350 | +0.04530 | +0.06475 | 1.429× | +0.11902 | +0.10215 | **0.858×** |
| 400 | +0.04539 | +0.06515 | 1.435× | +0.11902 | +0.14330 | 1.204× |
| 450 | +0.05372 | +0.06856 | 1.276× | +0.12628 | +0.14748 | 1.168× |
| 460 | +0.05372 | +0.06856 | 1.276× | +0.12628 | +0.15910 | 1.260× |

At the last matched checkpoint the two ratios coincide — **1.276× on pro against 1.260× on
anti**. Taken alone that is the pre-registered signature of a *generic* search improvement,
and it would retire the conjunction explanation.

## The trajectory shapes are opposite, and that is the finding

The endpoint equality is a coincidence of two different paths. Counting distinct
improvements to the best-so-far, over the same 501 iterations:

| arm | improvements | mean improvement | largest | longest plateau (iters) |
|---|---|---|---|---|
| pro k=1 | 35 | 0.001468 | 0.005759 | **197** |
| **pro k=3** | **54** | 0.001251 | 0.009684 | **105** |
| anti k=1 | 44 | 0.002821 | 0.010669 | 170 |
| **anti k=3** | **31** | **0.005200** | 0.017494 | **188** |

On the **conjunction**, k=3 makes **more and smaller** improvements — 54 against 35, mean
0.001251 against 0.001468 — and **halves the longest plateau, 197 iterations down to 105**.
That is escaping a stall, which is exactly what a conjunction fix should look like: once
the four band layers equalise, no single-token edit lifts them together, and changing three
positions at once finds the moves that do.

On the **disjunction** it does the reverse: **fewer and larger** improvements, 31 against
44 at mean 0.005200 against 0.002821, and a **longer** plateau, 188 against 170. That is
added variance, not escape. A disjunction never had a stall to escape, so k=3 has nothing
to fix there and instead just takes bigger, rarer swings.

That shape difference also explains the ratio column that looked damning: anti k=3 jumps
early (1.486× at iteration 50), then sits on a plateau at +0.08956 while anti k=1 climbs
steadily past it across iterations **150–350** — bottoming at **0.858×** — and then jumps
again in the last window. Its net gain by 100-iteration window makes the lumpiness plain:
`0.08741 → 0.01259 → 0.0 → 0.04115 → 0.02006`, a zero window followed by the largest
late window in the study. Pro k=3's is comparatively even:
`0.03506 → 0.01383 → 0.00610 → 0.00800 → 0.00455`.

**So the handoff's conclusion was right in substance and wrong in its evidence.**
Multi-position mutation does do something specific to the conjunction. The endpoint ratio
it was argued from cannot show that, and read on its own it argues the opposite.

> **CORRECTED 2026-09-07.** `docs/HANDOFF_BEHAVIORAL_S3.md` §8 states "**Multi-position
> mutation is the fix, and it works.** `--n-mutations 3` reaches `+0.05715` by iter 300
> where k=1 stalled at `+0.04530`." Both numbers are correct and reproduce here at
> iteration 300. The claim built on them is not supported by them: a k=1-versus-k=3 gap on
> the pro arm alone is what the anti arm was pre-registered to control, and the anti arm's
> endpoint ratio (1.260×) is indistinguishable from pro's (1.276×). What supports the claim
> is the trajectory shape above, which the handoff did not measure.

**Withdrawn:** "the k=1/k=3 gap on the pro arm shows the conjunction was the problem" — the
control it was supposed to survive returns the same endpoint ratio.
**Survives:** multi-position mutation acts specifically on the conjunction, on the evidence
of improvement count, improvement size and plateau length, all of which move in opposite
directions between the two families.

## What this buys

The conjunction account of Score 2 comes out of this stronger than it went in, on better
evidence than it previously had, and the fix is now understood rather than merely observed:
k=3 works on Score 2 because it shortens plateaus, so the knob to reach for on a saturating
conjunctive objective is *move class*, not temperature. That is a directly actionable
result, and it is the second time the temperature hypothesis has lost to the aggregate —
`season3_gcg_aggregate_asymmetry.json` withdrew the `T_SA` mechanism for the same reason.

It also produced the two highest-scoring strings in the project: pro k=3 finished at
**+0.06969** (iteration 498) and anti k=3 at **−0.16336** (iteration 461), against the
+0.05372 and −0.12628 their k=1 counterparts reached. Both are now behavioural arms
(`score2_top_final`, `score2_anti_final`) in `prefix_eval_arms_s3.json`.

## Limits

1. **One run per cell.** Four runs, no seed replication, so every ratio and every shape
   statistic is n=1. The shape argument rests on four numbers per arm and would be far
   stronger with three seeds each; nobody has run that.
2. **The min-versus-max comparison confounds aggregate with sign.** Only the anti arms use
   `per_layer_max`, so "conjunction versus disjunction" and "pro versus anti" are the same
   contrast here. A `max`-aggregated *pro* arm would separate them and does not exist.
3. **Plateau length is sensitive to run length.** All four are capped at 501 iterations, but
   pro k=1 logged 752, so its 197-iteration plateau is measured inside a window it had
   already escaped by iteration 450. The cap makes the comparison fair; it does not make
   the plateau a property of the objective alone.
4. **`--t-sa-scale` is not recorded in the checkpoint schema**, so none of these four arms
   can be confirmed to have run at the default temperature from its own artifacts. They are
   assumed to, on the basis that only `score2-2026-09-07T00-17-01Z` was launched with the
   flag; that identification is by timestamp, not by record.
5. Neither k=3 string round-trips (`roundtrip_ok: false`), so for both the board scores a
   different token sequence than the optimiser searched. `board_score` already reports the
   re-tokenised form, and `season3_prefix_scores.md` re-gated both, so the numbers here are
   ones the board can reproduce — but the searched ids and the submitted string differ.
6. Nothing here is behavioural. Whether +0.06969 says more to a reader than +0.06515 is
   `prefix_eval_s3.md`'s question, and the two new arms exist to answer it.

## Cross-links

`season3_gcg_aggregate_asymmetry.json` (the conjunction/disjunction account this tests, and
the earlier withdrawal of the temperature mechanism) · `docs/HANDOFF_BEHAVIORAL_S3.md` §7
and §8 · `season3_gcg_setup.md` §2 · `season3_prefix_scores.md` (the fidelity gate on both
new strings) · `prefix_eval_s3.md` (their behavioural test) · `scripts/gcg/optimize_banded.py`
(`make_candidates`, and the `--n-mutations` sampling guarantees in commit `6304f2c`).
