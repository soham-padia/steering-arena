# Why the metric is shaped this way

`docs/reference/scoring.md` says what every quantity in the scorer is, for lookup. This
page says why each has that shape and not the obvious alternative, and what it cost. Two
rejected alternatives survive nowhere but a commit message; recording them is why this
page exists. Run the example first — no GPU, no key, no model, and it is the whole
argument for the first choice:

```bash
python3 - <<'PY'
import numpy as np
from app.scoring import cosine
d = np.array([1.0, 0.0]); r = np.array([0.6, 0.8])
print(cosine(r, d), cosine(1000 * r, d))     # norm inflation buys nothing
print(float(r @ d), float((1000 * r) @ d))   # raw projection would pay for it
PY
```

```
0.6 0.6
0.6 600.0
```

## Cosine, not raw projection

Two lines of output, two designs. Under raw projection, scaling the residual 1000× scores
1000× higher, so the winning strategy becomes inflating activations, which is unrelated to
the direction being measured. Cosine is invariant to it, so the board needs no norm
penalty and no scale rule — a defensive term you never write is one nobody has to audit.
The cost is that cosine discards magnitude; RMSNorm makes the model's read angular anyway.

## The induced shift, not the sequence's own alignment

`app/scoring.py` has both. `self_score` is the cosine of the sequence's own last-token
residual with `d`; the board metric is how much prepending it moves a *probe's* residual.
Ranking on `self_score` would make the board a contest in writing text that reads as
pro-human, which is easier to game and is not the question.

## A frozen probe set, not a sample

Sixteen prompts, fixed for the season, in `data/probes/season3.json`. Sampling per
submission would make two scores on one board incomparable, and a season is *defined* as a
frozen tuple of model, build, layer, direction and scoring config precisely so a score
means one thing. The cost is that the objective is overfittable to sixteen prompts by
construction, which is why behaviour is a separate claim.

## A baseline that depends only on the probe

The score is a shift, so the probe's own alignment with `d` is subtracted. That term
depends on the probe alone, so it is computed once per season. A submission-dependent
baseline — a matched random prefix, say — multiplies the forward passes per submission,
and on a donated quota at $0 that is not efficiency but whether the board exists.

## One layer became a band, and the band is worse on paper

Season 2 scored at a single layer, and `layer_sweep_prefix.md` measured what its winning
prefix did: it satisfied layer 24 and essentially nowhere else — CV 1.73 across depth,
layer 24 at 4.69× the next-best layer, negative by L48. A metric one string can satisfy at
one depth pays for depth-specific overfitting, not for the concept. So Season 3 scores
over a four-layer band, chosen **against** the best numbers in its own selection artifact
(`season3_band_select.md`): the single-layer Season 2 baseline holds the highest margin of
five candidates at 0.27191, and the band that shipped has 0.23650, 13% lower, with
confound cosines an order of magnitude larger. That is a deliberate trade against a
measured failure mode: a harder target, not an unbeatable one.

## `mean` and `min`, and why both ship

`banded_mean` gives partial credit — stall on one layer and you still score.
`per_layer_min` is a conjunction: your score is your worst layer. The conjunction is the
stronger claim; `season3_gcg_aggregate_asymmetry.md` measures its cost. Net gain per 100
search iterations decays +0.045, +0.012, +0.005, +0.005, +0.002 under `min`, because once
the four layers equalise the argmin oscillates and no single-token edit lifts all four at
once. The disjunctive `max` of the anti arm never saturates, and about 1.9× of the
anti/pro gap is attributable to the aggregate swap. Both ship: `mean` is searchable, `min`
is the claim you would rather make.

## Why the board does not rank on specificity

`specificity_z` is implemented and computed on every submission, and nothing ranks on it.
It was meant to demote token soup on the theory that soup is isotropic junk with a lucky
projection; `specificity_calibration.md` falsified that: soup is *coherent*, so artifacts
hold or gain rank under z — `.) {}` moves #17→#10 at z=+1.63 — because these strings
perturb all sixteen probes coherently, partially along `d`. The second reason is
structural: the closed form needs linearity in the direction, so it does not extend to
`per_layer_min`.

## Two alternatives rejected, whose measurements lived only in a commit message

Both come from commit `4605eb7`. Neither has an analysis file; this page is their record.

**The geometric mean, rejected.** A product over the band looks like a natural
conjunction, and it fails here for a reason about the corpus rather than the formula.
Verified against the cached activations, on the seed pairs **99.6%** of texts have the
same sign at all four band layers: 49.6% all positive, 50.0% all negative, 0.4% mixed. A
four-way product is therefore positive 99.6% of the time, half of that from four
*negatives* multiplying. A text that is anti-human at every layer, per-layer cosines
`[-0.196, -0.175, -0.154, -0.141]`, has a geometric mean of **+0.165** against an
arithmetic mean of **−0.167**. It ranks the most anti-human text in the corpus as strongly
pro-human, and because the layers correlate that is half the corpus, not an edge case.

That 99.6% carries its own caveat, and the caveat is a live open question about the band
premise. It is measured on the **seed pairs**, which the direction was **fit** on, at
layers it was fit at, where held-out separation is 1.000 — so near-perfect sign agreement
there is partly circular. The version that would really establish "the band supplies no
independent constraint" is lockstep agreement on **submissions**, which are out of
distribution and are not cached.

**The soft-min surrogate, adopted for search only.** A hard `min` has a gradient that
reaches the argmin layer and nothing else, so a GCG step learns about one depth out of
four. Measured on a real seed-pair text, per-layer gradient share is `[0.6509, 0.2278,
0.0797, 0.0416]` under softmin against `[1.0, 0.0, 0.0, 0.0]` under hard min — all four
band layers get signal. It interpolates correctly: at T=0.001 it returns −0.19461 against
a true min of −0.19600, and at T=10 it returns −0.16652 against a mean of −0.16650.
`SOFTMIN_T = 0.02`, in `scripts/gcg/gcg_utils.py`. It is a search surrogate and never a
reported score, because the board computes `min`: gradients propose candidates, and only
the recorded number has to be true.

It also introduced a reporting bug of the kind that survives review. With a surrogate
active the run's `score` column was the softmin while `board` was the true min, so the
printed drift conflated the retokenisation gap with the surrogate-versus-true gap and
**overstated the former nearly fivefold — +0.0245 against a real +0.0053**.

## How layer 24 got chosen, which is not how it was justified

Layer 24 was not chosen. All five candidate bands separate held-out pairs at 1.000, and
`scripts/extract_direction.py` selects on separation with a strict `>`, so under a tie it
keeps whichever candidate it saw first (`season3_band_select.md`): the layer the whole
Season 2 board was scored at arrived by iteration order. `layer_concept_profile.md` later
found L24 is the genuine margin peak, 0.2726 with the highest Cohen's d at 3.3777, so the
choice was vindicated on a better criterion than the one that made it. That is luck: a
saturated selection metric hands you an arbitrary answer without reporting that it did.

## The confound sentence that has to be said in full

Both halves, always, or neither. The fitted `d` does **not** ride the `approach` confound:
`cos(d, approach) = 0.1501`, and projecting `approach` out leaves held-out separation at
1.000, all-135 separation at 1.000, and the kind-over-cruel control gap at 0.2067 against
0.2075 (`data/analysis/direction_purity.json`). But the seed **corpus** is confounded:
`approach` alone separates the pairs at 0.824. That is a corpus problem, not a direction
problem, and no band or aggregate fixes it. State the first half alone and you have
overclaimed; state the second alone and you have withdrawn something that survived.

## What this buys you

An objective a stranger can recompute on a laptop, that nobody wins by scaling a vector,
and whose harder variant turned out to predict measured behaviour better than its easier
one. Every shape above has a named cost: cosine discards magnitude, the probe set is
overfittable by construction, the band gave up 13% of margin, and the conjunction
saturates under search. The costs sit beside the reasons, which is the only arrangement in
which a later reader can tell whether a choice still holds.
