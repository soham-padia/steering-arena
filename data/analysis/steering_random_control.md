# Steering arm: norm-matched random direction control

Pre-registered in `steering_random_control_preregistration.md` before any result was
inspected. Data: `steering_random_control.json`. Script: `scripts/steering_random_control.py`.

## The gap this closes

`behavioral_eval.md` reported that adding `α·d` to the layer-24 residual changes
generation at α = 0.5 and 1.0 × ‖R‖, with no control for perturbation SIZE. A random
vector of equal norm might derail generation comparably, which would make that result
about magnitude rather than about `d`.

## Manipulation check (measured, not read off the code)

The intervention was measured by saving the layer-24 output with and without the edit and
differencing them, **before** running the control.

| property | measured |
|---|---|
| positions edited | **all 9 prompt positions**, not last-token only |
| per-position edit norm | **30.07**, exactly α |
| direction | **cos(diff, d) = 1.000** at every position |
| generated tokens edited | **none** (trace covers 9 of a 17-token sequence) |
| per-position baseline residual norms | 18.5, 21.5, 33.2, 35.1, 33.5, 32.6, 35.9, **5.2**, 31.1 |

**This is prefill-only steering.** The vector perturbs the prompt's forward pass and
changes the state generation proceeds from. It is NOT applied at each decoding step. Any
description of this eval as "steering every token" is incorrect, including how the
existing result has previously been discussed.

**α = 1.0·‖R‖ is large and uneven.** ‖R‖ is the mean *last-token* residual norm applied
uniformly to every position. Against per-position norms spanning 5.2 to 35.9, the same α
is ≈1× the local residual at some positions and **≈6×** at others.

## Setup

~~3 random unit vectors, seed 20260826, each scaled to the same α = 30.07 = 1.0·‖R‖.
Measured |cos(random_i, d)| = 0.0144, 0.0057, 0.0012.~~

**Corrected 2026-08-27: this said 3; it is 8.** The Setup section listed 3 draws and 3
cosines while the rest of this file and `steering_random_control.json`
(`setup.n_dirs = 8`) say 8. The three listed were the first three. Internal inconsistency,
never flagged.

**8 random unit vectors**, seed 20260826, each scaled to the same α = 30.07 = 1.0·‖R‖.
Measured `|cos(random_i, d)|`, all 8, from `steering_random_control.json`:

| draw | rand1 | rand2 | rand3 | rand4 | rand5 | rand6 | rand7 | rand8 |
|---|---|---|---|---|---|---|---|---|
| \|cos(random_i, d)\| | 0.0144 | 0.0057 | 0.0012 | 0.0032 | 0.0053 | 0.0005 | 0.0014 | 0.0021 |

50 prompts per arm. Both signs of the real arm re-judged in the same run under the v2 rubric,
since the previous ±1 numbers came from the OLMo judge that produced no usable signal.

**Provenance caveat, moved here 2026-08-27 from `steering_dose.md` Limit 4, where it was
misfiled.** This experiment **does** span the June/August boundary: `base` and `±1` were
generated **2026-06-10**, all eight random arms **2026-08-26**. If NDIF re-served OLMo-3-32B
between those dates, the real arms and the null band were produced by different model builds,
the exact condition `PROJECT_SPEC.md` §5.4 calls a season break. This file did not list that
caveat; `steering_dose.md` listed it against the ±0.5 vs ±1 comparison, where all five arms are
in fact from one 2026-06-10 run and it does not apply.

**Mitigating evidence, in the data and cited by no document until now.** α was recomputed
fresh in August and came back **bit-identical** at **30.07038116455078**. α is derived from the
mean layer-24 last-token residual norm over the eval prompts, so a re-serve that changed the
model would be very unlikely to reproduce it to the last bit. That is evidence the build did
not change. It is not proof, and it does not cover the judge, which is a different service.

## Results

| arm | distinct-4 | Δ vs base | looping | judged kindness Δ | p |
|---|---|---|---|---|---|
| base | 0.912 | — | 8/50 | — | — |
| **+1 · d** | 0.872 | −0.039 (p=0.29) | 8/50 | **+0.45** | **0.0070** |
| **−1 · d** | 0.889 | −0.023 (p=0.41) | 12/50 | −0.15 | 0.39 |
| rand1 | 0.844 | −0.068 (p=0.036) | 17/50 | −0.18 | 0.32 |
| rand2 | 0.867 | −0.045 (p=0.12) | 12/50 | +0.10 | 0.46 |
| rand3 | 0.940 | +0.029 (p=0.21) | 9/50 | +0.13 | 0.45 |

*Scope note added 2026-08-27, part of the same 3-vs-8 correction as Setup:* this table covers
**the first 3 of the 8 draws**. The distinct-4 and looping columns were only computed for those
three. The judged-kindness column for all 8 is in the null-band section below.

## Cross-judge check

Judge disagreement is what collapsed the `anti_top` arm in `prefix_eval.md`, and a
steering result resting on one rater has no such check. Both claim-bearing arms were
re-judged by `claude-opus-5` over the **same cached generations**, same v2 rubric, both
orders, no shared context between agents.

| arm | deepseek Δ | p | claude Δ | p |
|---|---|---|---|---|
| **+1 · d** | +0.45 | 0.0070 | **+0.54** | **0.0015** |
| **−1 · d** | −0.15 | 0.39 | **+0.05** | **0.84** |

Both judges find `+1·d` significant and `−1·d` null. Claude's `−1` result is 15/30
preferred — an exact coin flip — with a point estimate nearer zero than DeepSeek's. The
asymmetry no longer rests on a single rater on either side.

**Human blind ratings for these arms exist, and were never analysed (added 2026-08-27).** They
are in `data/analysis/behavioral_blind.csv`. Reported here for the first time, alongside the
±0.5 arms for context:

| arm | steered wins / base wins / tie | decided pairs | p |
|---|---|---|---|
| `+0.5 · d` | 2 / 2 / 6 | 4 | 1.00 |
| **`+1.0 · d`** | **5 / 3 / 2** | **8** | **0.73** |
| `−0.5 · d` | 4 / 8 / 1 | 12 | 0.39 |
| **`−1.0 · d`** | **2 / 6 / 1** | **8** | **0.29** |

**None significant; all badly underpowered** at 4 to 12 decided pairs per arm. 42 human ratings
in total across the four arms. They cannot arbitrate anything, and they are recorded because
they were sitting unread while the arms were described as resting on LLM judges alone.
Directionally `+1·d` agrees with both LLM judges. `−1·d` does **not** agree with "inside the
random band": the human raters preferred base 6 times to 2. But at 8 decided pairs that
disagreement carries no weight either.

## The null band, characterised — 8 draws, BOTH judges

Three draws bound the null loosely, and the `−1·d` result is *defined* by falling inside
it, so the band was extended to 8 seeds (250 further generations) and every random arm was
then judged by both raters. Placing a cross-judged estimate inside a single-judge
distribution would have been an inconsistent comparison.

| | deepseek-v4-pro | claude-opus-5 |
|---|---|---|
| null band (8 draws) | mean **+0.056**, sd 0.100, range [−0.18, +0.13] | mean **+0.014**, sd 0.110, range [−0.16, +0.19] |
| **+1 · d** | +0.45, **z = +3.93**, above all 8 draws | +0.54, **z = +4.76**, above all 8 draws |
| **−1 · d** | −0.15, z = −2.06, 1/8 draws below | +0.05, **z = +0.33, 5/8 draws below** |
| random arms reaching p<0.05 | **none of 8** | **none of 8** |

**`+1·d` is unambiguous.** It exceeds every one of eight random directions of identical
norm, under two independent judges, at roughly 4-5 sd above the null mean.

**`−1·d` is not distinguishable from a random direction.** It sits inside the band under
both judges, and Claude places it essentially at the **median** of the null (5 of 8 draws
below it). An earlier version of this document hedged that `−1` sat "at the low edge" of
the band; that was DeepSeek's placement alone (1/8 below) and does not survive the second
judge. The plain statement is the accurate one.

**Correction 2026-08-27: "essentially at the median (5 of 8 below)" is the FLOATING-baseline
placement only.** Under the fixed baseline Claude's `−1·d` sits at **2 below, 1 tied, 5
above**, i.e. the lower third of the null, not its median. The 5/8 figure is correct for the
estimator it was computed under and was carried into statements about the other one. Both
placements leave `−1·d` inside the band under Claude, so the conclusion is unchanged; the
rhetorical "essentially at the median" is withdrawn.

## Fixed-baseline sensitivity (added 2026-08-27)

Every Δ above is computed against a **per-arm** baseline: each arm was judged beside its own
run's base ratings, and those ratings drift on identical text (see the correction in
`prefix_eval.md`). Recomputing every arm against a **fixed** per-prompt baseline (the mean
over all 10 steering arms' base ratings) gives:

| | deepseek-v4-pro | claude-opus-5 |
|---|---|---|
| null band (8 draws), floating | mean +0.056, sd 0.100 | mean +0.014, sd 0.110 |
| null band (8 draws), **fixed** | mean +0.066, sd 0.080 | mean +0.037, sd 0.040 |
| `+1·d` floating | +0.450, above all 8 | +0.540, above all 8 |
| **`+1·d` fixed** | **+0.355, above all 8** | **+0.376, above all 8** |
| `−1·d` floating | −0.150, 1/8 below | +0.050, 5/8 below (1 tied) |
| **`−1·d` fixed** | **−0.135, 0/8 below (OUTSIDE, below all 8)** | **+0.026, 2/8 below (1 tied, INSIDE)** |
| randoms reaching p<0.05 | 0/8 under both estimators | 0/8 under both estimators |

**The positive result SURVIVES.** `+1·d` clears all 8 randoms under both judges under both
estimators. 20-30% of the reported effect turns out to be baseline drift on identical text
(+0.450 → +0.355; +0.540 → +0.376), but the placement does not move at all.

**The negative result does NOT survive as stated. It is ESTIMATOR-DEPENDENT.** Under DeepSeek
with a fixed baseline, `−1·d` falls **below all eight** randoms, outside the band, on the
other side. Under Claude with the same fixed baseline it stays inside. So

> "`−1·d` is not distinguishable from a random direction"

must be stated as **estimator-dependent**: it holds under the floating baseline under both
judges and under the fixed baseline under Claude, and fails under the fixed baseline under
DeepSeek. The estimator was never chosen deliberately; it was whichever one the script
happened to compute.

### Two statistical caveats on the table above

**1. Do not read the z-scores. Read the placement counts.** A fixed baseline differences every
arm against the *same* per-prompt vector, which removes the between-arm base variance that the
floating baseline was injecting into the null. The null sd therefore shrinks **mechanically**
(deepseek 0.100 → 0.080; claude **0.110 → 0.040**) and z inflates for arithmetic reasons, not
because anything got stronger. Under Claude the point estimate **falls 30%** (+0.540 → +0.376)
while z **rises 77%** (+4.76 → +8.44). A statistic that grows while the effect it describes
shrinks is not measuring the effect. Placement counts are reported throughout instead.

**2. But the placement count is nearly powerless.** Over 8 draws, a one-sided placement test
has a floor of **p ≈ 1/9 ≈ 0.11**. "Above all 8" is the best outcome available and it cannot
distinguish z = +3.6 from z = +8.4; both saturate the same ceiling. The statistic is robust
and almost powerless. More draws is the only fix; 8 was chosen to bound the band, not to
discriminate within it.

**A withdrawn observation.** An earlier version reported the null mean as +0.056 and noted
that a random perturbation appears to nudge judged kindness slightly upward. Claude's null
mean is **+0.014**, essentially zero. The upward nudge was a judge artifact, not a property
of the perturbation, and the claim is withdrawn.

## Reading: the two measures dissociate

**Coherence damage is a MAGNITUDE effect.** `+1·d` (−0.039) sits inside the random spread
(−0.068 to +0.029), and the only arm reaching p<0.05 on distinct-4 is a *random* one.
Perturbing at 1.0·‖R‖ degrades text regardless of direction. Any claim that steering along
`d` specifically harms or helps coherence is not supported.

**The kindness shift is a DIRECTION effect.** `+1·d` gives +0.45 (p=0.007) while all three
random arms cluster near zero (−0.18, +0.10, +0.13, all p>0.3). The real direction is well
outside the random spread. *(Written against the first 3 draws; it holds against all 8 under
both judges and both baseline estimators, see below. Corrected count 2026-08-27.)* This is
pre-registered outcome (A) for the claim that matters:
the behavioural result is about `d`, not about perturbation size.

**The asymmetry survives and is sharper than before.** `−1·d` is −0.15 at p=0.39, and sits
**inside the random spread**. *(Qualified 2026-08-27: that placement is estimator-dependent.
Under a fixed baseline `−1·d` falls below all 8 randoms under DeepSeek. See the fixed-baseline
sensitivity section.)* Steering away from pro-human is not "the opposite behaviour"
and not even reliably "degeneration" — on both measures it is indistinguishable from a
random perturbation of the same norm. Both signs were judged under the same rubric in the
same run, so this no longer rests on the retired judge on either side.

## Robustness fact, independent of `d`

A vector of norm 30.07, added at **every** prompt position (up to ~6× the local residual
at some positions), leaves generation largely fluent: distinct-4 of 0.844 to 0.940 against
base 0.912, and looping of 9-17/50 against base 8/50. The model absorbs a perturbation the
size of its own residual stream without collapsing.

## Corrections owed to earlier write-ups

1. `behavioral_eval.md` describes an intervention it does not perform. It is prefill-only,
   not per-token.
2. Any statement that `−d` steering "produces degeneration" overstates: its coherence
   effect is within the random-direction band.
3. The `±1` numbers from the OLMo judge should not be cited; both signs are re-measured
   here.
4. **(added 2026-08-27)** This document owed corrections of its own: it said 3 draws where
   there are 8, it did not list the June/August generation split that applies to it, it did not
   cite the bit-identical α recomputation that mitigates that split, it reported a
   floating-baseline placement (5/8) inside a statement about the fixed baseline, and it did not
   report the 42 human blind ratings that exist for `±0.5` and `±1`. All are now above.

## What this control can and cannot establish (added 2026-08-27)

The eight nulls are isotropic Gaussian draws in 5120 dimensions. In that many dimensions a
Gaussian draw is near-orthogonal to `d` **and to every other concept direction in the model**.
Our own measured `|cos(random_i, d)|` values (0.0144, 0.0057, 0.0012 and the rest, all under
0.015) are a textbook instance of exactly that regime.

So this control tests `d` against **noise**. It cannot test `d` against other **meaningful**
directions. The rival hypothesis it leaves standing is not "any vector of this norm moves
kindness" (that is refuted) but "any *interpretable* direction of this norm moves kindness",
which is a different and more damaging claim.

**Credit where it is owed, in both directions.** Luo, Liang and Xuan, *SteerCheck: Attribution
Specificity and Alignment Leakage in Activation-Steering Audits* (arXiv:2608.24335), make this
argument and supply the better controls: **PCA-subspace** and **sign-randomised
same-construction** nulls under a **matched KL budget**. Sign-randomised directions, rebuilt
from the same contrast pairs with chosen/rejected flipped within pairs, "often retain
substantial target alignment", which is precisely what an isotropic draw cannot do. The older
precedent for the whole idea is Hewitt and Liang (2019) on **control tasks** for probes.

**But SteerCheck also says norm-matched random controls are "the standard evidence for
attribution".** So what this project did is **correct standard practice, not an error**. The
limitation is on what that control can *establish*, not on whether it should have been run. The
honest next control is the sign-randomised same-construction null.

## The pre-registration lesson (added 2026-08-27)

`steering_random_control_preregistration.md` named outcome **(C)**, "activation space is
organised such that many directions move the judged construct", as the most damaging threat,
identified it as the one it would be most tempted not to see, and **wrote it down in advance,
before any result was inspected.** That is genuine design-level foresight and it is the best
design artifact in the project.

The instrument then chosen could not test it. `scripts/steering_random_control.py:45-47` states
the reasoning outright, "a Gaussian draw is near-orthogonal to any fixed vector, which is
exactly the point", and that near-orthogonality is precisely what makes the family structurally
unable to produce outcome (C).

The transferable lesson, scoped honestly: **pre-registering the THREAT does not pre-register the
INSTRUMENT'S POWER against it.** A pre-registration that names a rival hypothesis should also
state what result would count as that hypothesis being *supported*, and whether the chosen
manipulation is capable of producing it. This one did the first and not the second. The control
itself was executed correctly; the gap is between the threat named and the manipulation
selected.

## Limits

1. n=50 per arm, one model, one layer, one α.
2. ~~Three random draws bound the null loosely; more would tighten it.~~ **Corrected
   2026-08-27: eight draws, not three** (see Setup). Eight still bound the null loosely: the
   one-sided placement test floors at p ≈ 1/9, so "above all 8" is the strongest result the
   design can return. More draws is the fix.
3. ~~The judge is a single rater (`deepseek-v4-pro`) under the frozen v2 rubric, unlike the
   prefix study's two.~~ **Corrected 2026-08-27:** stale. Both claim-bearing arms and all eight
   random arms are cross-judged by `deepseek-v4-pro` and `claude-opus-5`; see the cross-judge
   and null-band sections above. 42 human blind ratings also exist for `±0.5` and `±1` and are
   reported above.
4. **Base and `±1` were generated 2026-06-10; all eight random arms 2026-08-26.** See the
   provenance caveat in Setup, and the bit-identical α recomputation that mitigates it.
5. **The `−1·d` null placement is estimator-dependent.** See the fixed-baseline sensitivity
   section.
6. **The null family tests `d` against noise, not against other meaningful directions.** See
   the section above.
