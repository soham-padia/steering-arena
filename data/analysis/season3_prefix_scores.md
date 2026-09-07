# Season 3 arms, re-scored under both banded objectives

`scripts/score_banded_local.py --arms data/analysis/prefix_eval_arms_s3.json`
→ `data/analysis/season3_prefix_scores.json`. Job `709380`, one B200, bf16, 16 frozen
season3 probes, 6 strings × 2 objectives. All numbers are LIVE (per-probe baseline
subtracted), so each is directly comparable to a leaderboard entry.

This ran as the fidelity gate in front of the Part A′ behavioral eval — verify the strings
before spending a GPU generating from them. It settled one thing it was pointed at and two
it was not.

## The table

| arm | Score 1 | Score 2 | tok | recorded → re-scored |
|---|---|---|---|---|
| `score1_top` | **+0.16395** | +0.05001 | 32 | +0.16395 → +0.16395 (gap 0.0e+00) |
| `score2_top` | +0.07896 | **+0.06518** | 33 | +0.06515 → +0.06518 (gap +3.1e-05) |
| `pro_coherent` | +0.03976 | +0.02145 | 17 | +0.04046 → +0.03976 (gap −7.0e-04) |
| `random32` | −0.00335 | −0.00885 | 32 | — (no recorded score; this run is its first) |
| `score1_anti` | −0.10135 | −0.12618 | 32 | −0.10144 → −0.10135 (gap +8.8e-05) |
| `score2_anti` | −0.07379 | −0.12595 | 33 | −0.12628 → −0.12595 (gap +3.3e-04) |

Scale, for reading the columns: Season 2's winner rescored gives Score 1 **+0.06747** and
Score 2 **+0.02308**; field sd over the 618 rescored entries is **0.01826** and **0.01421**.
The live board's best is `id=1346` at Score 1 **+0.13217** / Score 2 **+0.04113**.

Every arm with a recorded score reproduced it, max |gap| **7.0e-4**. `pro_coherent`'s
−7.0e-4 is the only one that is a genuine cross-backend comparison — its recorded value is
the NDIF score in the database, not a local number — and at ~2× the calibration's max |gap|
of 3.71e-4 it is the same order, not a discrepancy. The rest are local-vs-local and land at
1e-4 or below.

## WITHDRAWN: the anti LIVE scores in the Part A′ handoff

`docs/HANDOFF_BEHAVIORAL_S3.md` §3 reports `score1_anti` at **−0.10346** and `score2_anti`
at **−0.14510**. Both are wrong, and this run is the empirical proof: scored under the
**pro** objective, which is what the board computes, the strings come out at **−0.10135**
and **−0.12595**. The score2 error is 1.9e-2 — about 1.3 field sd, and 57× the largest gap
anywhere else in the table.

The cause was `scripts/gcg/watch.py`, which computed

```python
live = sign * (best - baseline)        # wrong: distributes the flip over the baseline
```

The optimiser always maximises, so an anti run's own `best` is positive-is-better and the
board would show its negation. The flip has to come **first**, then the baseline is
subtracted **once**:

```python
live = sign * best - baseline         # and sign * best is board_score_true_sign
```

which makes the anti and pro cases one rule — *live = (board score, in board sign) −
baseline* — and `best.json` had already been recording `board_score_true_sign` all along.
Fixed in `watch.py` the same day, with the arithmetic in the comment. Corrected values:
`score1_anti` **−0.10144**, `score2_anti` **−0.12628** (algebraic), confirmed here at
−0.10135 and −0.12595 (measured).

This does not change any pro number, any ranking, or the aggregate-asymmetry conclusions in
`season3_gcg_aggregate_asymmetry.json` — anti was always easier than pro and still is. It
changes the magnitude quoted for the two anti arms.

## The random-token control, which Season 2 never ran

A length-matched random 32-token prefix, drawn from the same (whole) vocabulary GCG
searches, scores **−0.00335** on Score 1 and **−0.00885** on Score 2 — i.e. roughly nothing,
and slightly *negative*. In sd units that is −0.18 sd and −0.62 sd.

This closes the obvious hole in the Season 2 result. "Any 32-token prefix perturbs the
residual stream, so of course the score moves" is now measured and false: random tokens do
not move this metric in the pro direction at all. Whatever the GCG winners are doing, it is
not a generic prefix-length effect.

The asymmetry between the two columns is worth keeping. Random tokens hurt Score 2 (−0.62
sd) about three times as much as Score 1 (−0.18 sd), which is what a **conjunction**
predicts and independent of the evidence already collected for it: Score 2 is a `min` over
four layers, so a perturbation only has to damage the weakest one to pull the aggregate
down, while Score 1's `mean` lets the other three absorb it.

Caveat: n = 1 random draw, seeded (`20260906`), chosen to re-tokenise to exactly 32 tokens.
One draw fixes no distribution. A proper null band needs ~10 draws, the way
`steering_random_control.py` does it for the steering arms — not run here.

## Cross-objective transfer is real, asymmetric, and awkward for the gaming story

Reading the off-diagonal of the table:

| | scores on Score 1 | scores on Score 2 |
|---|---|---|
| optimised for Score 1 | +0.16395 (100%) | +0.05001 (**77%** of the Score-2 optimum) |
| optimised for Score 2 | +0.07896 (48% of the Score-1 optimum) | +0.06518 (100%) |

**A string found by searching Score 1 reaches 77% of the dedicated Score-2 optimum, and it
already beats the live board's best Score 2 entry (+0.05001 vs +0.04113).** The reverse
transfer is much weaker: the Score-2 winner gets less than half of `score1_top` on Score 1.

That cuts against the cleanest reading of "Score 2 genuinely resists gaming". It does resist
*direct* search — the saturation curve in `season3_gcg_aggregate_asymmetry.json`
(`+0.045 → +0.012 → +0.005 → +0.005 → +0.002` per 100 iters) is not withdrawn, and getting
from +0.050 to +0.065 took 389 iterations of k=3 multi-position search. But most of the
distance to a competitive Score 2 comes free from the *easier* objective. A player who
cannot beat Score 2 directly can search Score 1 and submit that.

For the anti arms the transfer is total: `score1_anti` scores **−0.12618** on Score 2,
statistically indistinguishable from the dedicated `score2_anti`'s **−0.12595** (gap 2.3e-4,
inside this run's noise). So a Score-2 anti board would be topped just as well by a Score-1
anti search — consistent with the recorded finding that a disjunction needs only one layer
to fail.

**The confound that must travel with all of this.** Score 1 and Score 2 differ in *two*
ways at once — band (19/23/27/31 vs 15/23/31/39) and direction (one shared `d` vs one `d_L`
per layer). This table cannot say which of the two the transfer rides on. The clean
experiment is a 2×2: each band crossed with each direction type, which nobody has run. Until
then "transfers at 77%" is a fact about these two objectives as shipped, not a claim about
`mean` vs `min`.

## Side finding: five of the eight live Season 3 board rows are repr-corrupted

The handoff records that only `id=1342` was ever submitted. There are **8** live rows in
`season_id=5` (ids 1340, 1342–1348, all handle `Soham`, 2026-09-06 18:58–23:10). Checked
against the frozen arm strings with the same tokenizer the board uses:

| id | tok | Score 1 | Score 2 | literal `\n` | real newlines |
|---|---|---|---|---|---|
| 1340 | 40 | +0.02232 | +0.01385 | 4 | 0 |
| 1342 | 38 | +0.08600 | +0.03588 | 2 | 2 |
| 1343 | 37 | +0.09419 | +0.02709 | 2 | 0 |
| 1344 | 34 | +0.11397 | +0.03166 | 0 | 2 |
| 1345 | 52 | +0.00729 | −0.00522 | 2 | 2 |
| 1346 | 34 | **+0.13217** | +0.04113 | 0 | 2 |
| 1347 | 37 | −0.06711 | −0.08365 | 2 | 0 |
| 1348 | 42 | −0.05863 | −0.06901 | 0 | 2 |

Every final run string is **32 or 33** tokens. Every live row is **34 to 52**. Five rows
(1340, 1342, 1343, 1345, 1347) contain literal two-character `\n` sequences; the three with
zero real newlines alongside them (1340, 1343, 1347) are the clearest signature, because the
runs those strings came from record real newline bytes. `id=1347` is recognisably the
`score1_anti` family and sits at −0.06711 where the true string measures **−0.10135**.

Consequence for the board as it stands: the current Score-1 leader, `id=1346` at +0.13217,
is a 34-token string, while the run's own final 32-token string measures **+0.16395**. The
board understates the search by ~0.032, and no row on it is the string any run actually
found.

Not acted on — the leaderboard is the maintainer's, and `CLAUDE.md` puts submission
decisions there, not here. Flagged because the fidelity check that produced the table above
is exactly the check that would have caught it at submission time, and it is now one
command: `python scripts/score_banded_local.py "<string>"`.

## What this does not touch

Nothing here is behavioral. Every number is a cosine against a direction, which is exactly
the quantity Part A′ exists to question. `score2_top`'s +0.06518 says where the residual
stream points, not what the model says.
