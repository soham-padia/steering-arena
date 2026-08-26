# Pre-registration: norm-matched random direction control for the steering arm

Written **before** any result was inspected. Generation was already running; no judged
output and no distinct-4 numbers had been computed or seen at the time of writing.

## The gap

`behavioral_eval.md` reports that adding `α·d` to the layer-24 residual changes generation
at α = 0.5 and 1.0 × ‖R‖. It contains no control for perturbation SIZE. If a random vector
of equal norm derails generation comparably, that result is about magnitude, not about `d`.

## Manipulation check (measured, not read off the code)

Before running the control, the intervention was measured directly by saving the layer-24
output with and without the edit and differencing them.

| property | measured value |
|---|---|
| positions edited | **all 9 prompt positions** (not last-token only) |
| per-position edit norm | **30.07**, exactly α |
| direction of edit | **cos(diff, d) = 1.000** at every position |
| generated tokens edited | **none** |
| trace coverage | 9 positions, of a 17-token final sequence |
| per-position baseline residual norms | 18.5, 21.5, 33.2, 35.1, 33.5, 32.6, 35.9, **5.2**, 31.1 |

**Two consequences.**

1. **This is prefill-only steering.** The vector is added during the prompt's forward pass
   and changes the state generation proceeds from. It is NOT added at each decoding step.
   Any description of this eval as "steering every token" is wrong, and that includes how
   the existing behavioural steering result has been discussed.
2. **α = 1.0·‖R‖ is a very large perturbation, and unevenly so.** ‖R‖ is the mean
   *last-token* residual norm, applied uniformly to every position. Against per-position
   norms spanning 5.2 to 35.9, the same α is roughly 1× the local residual at some
   positions and **~6× at others**.

The random control uses the identical code path, so the comparison stays like-for-like.
What changes is what both arms must be *called*.

## Design

- k = 3 random unit vectors, seed 20260826, each scaled to the same α = 1.0·‖R‖.
- Measured |cos(random_i, d)| = 0.0144, 0.0057, 0.0012.
- 50 prompts per arm.
- **Both signs of the real arm re-judged**, +1 and −1, under the same v2 rubric in the
  same run. The existing ±1 numbers came from the OLMo judge that produced no usable
  signal, so leaving one sign on the retired instrument would leave the asymmetry claim
  resting on it.
- Two measures: distinct-4-gram ratio (no judge) and judged kindness delta vs unsteered
  base, both A/B orders, verdict counted only when consistent.

## Pre-registered outcomes

**(A) Random fluent, no kindness shift.** The steering result is about `d`. Clean, and the
strongest available support for the existing claim.

**(B) Random derails generation.** The ±1 result is about perturbation magnitude, not
about `d`. That arm means nothing until re-run at an α small enough that a random vector
of equal norm leaves text intact.

**(C) Random fluent BUT shifts judged kindness.** The most damaging and most interesting:
activation space would be organised such that many directions move the judged construct,
so "moves the construct" would not be evidence that a direction encodes it. This is the
outcome I would be most tempted not to see, which is why it is written down here first.

## What is already known regardless of outcome

If a random vector at 1.0·‖R‖, added at every prompt position, leaves generation fluent,
that is a non-obvious robustness fact about the model and worth reporting on its own,
independent of what it implies about `d`.
