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

3 random unit vectors, seed 20260826, each scaled to the same α = 30.07 = 1.0·‖R‖.
Measured |cos(random_i, d)| = 0.0144, 0.0057, 0.0012. 50 prompts per arm. Both signs of
the real arm re-judged in the same run under the v2 rubric, since the previous ±1 numbers
came from the OLMo judge that produced no usable signal.

## Results

| arm | distinct-4 | Δ vs base | looping | judged kindness Δ | p |
|---|---|---|---|---|---|
| base | 0.912 | — | 8/50 | — | — |
| **+1 · d** | 0.872 | −0.039 (p=0.29) | 8/50 | **+0.45** | **0.0070** |
| **−1 · d** | 0.889 | −0.023 (p=0.41) | 12/50 | −0.15 | 0.39 |
| rand1 | 0.844 | −0.068 (p=0.036) | 17/50 | −0.18 | 0.32 |
| rand2 | 0.867 | −0.045 (p=0.12) | 12/50 | +0.10 | 0.46 |
| rand3 | 0.940 | +0.029 (p=0.21) | 9/50 | +0.13 | 0.45 |

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
outside the random spread. This is pre-registered outcome (A) for the claim that matters:
the behavioural result is about `d`, not about perturbation size.

**The asymmetry survives and is sharper than before.** `−1·d` is −0.15 at p=0.39, and sits
**inside the random spread**. Steering away from pro-human is not "the opposite behaviour"
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

## Limits

n=50 per arm, one model, one layer, one α. Three random draws bound the null loosely;
more would tighten it. The judge is a single rater (`deepseek-v4-pro`) under the frozen v2
rubric, unlike the prefix study's two.
