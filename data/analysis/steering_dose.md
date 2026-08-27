# Dose-response inside the injection family

Does `α = 0.5·‖R‖` beat `α = 1.0·‖R‖`? Data: `steering_dose.json`. Script:
`scripts/steering_dose.py`.

## Why this was run

The compiled token string produces about twice the behavioural shift of the `α = 1.0`
injection (`compile_check.md`). That admits two readings:

- **(a) tokens beat vectors**
- **(b) the vector was applied past its useful range**

(b) is the one with reach beyond this project — "steering vectors are routinely overdriven"
would be a claim about steering practice generally. It is also testable *without* any token
comparison, which matters because a prefix and an injection are not the same mechanism: 33
extra tokens give the model attention patterns, positional structure and its own computation
over that context, none of which a residual nudge provides. Comparing `+0.5` against `+1`
stays inside the injection family, so that confound does not apply.

The `±0.5` generations already existed from the original behavioural eval. Only the judging
was missing, and the judge that saw them then (base OLMo) is the one that memo itself
reports as unusable.

## Result

50 prompts per arm, frozen v2 rubric, `deepseek-v4-pro`, both A/B orders, verdict counted
only when consistent.

| arm | α | on-`d` push | behavioural Δ | p |
|---|---|---|---|---|
| `+0.5 · d` | 15.03 | +15.03 | **+0.13** | 0.32 (n.s.) |
| `+1.0 · d` | 30.07 | +30.07 | **+0.45** | 0.007 |
| `−0.5 · d` | −15.03 | −15.03 | −0.19 | 0.28 (n.s.) |
| `−1.0 · d` | −30.07 | −30.07 | −0.15 | 0.39 (n.s.) |

## Reading

**Reading (b) is falsified.** The injection's response is still *climbing* at α = 1.0·‖R‖:
half the dose gives less than a third of the effect, and at 0.5 the effect is not
significant at all. The vector is not overdriven. If anything it is under-applied, and the
useful range extends at or beyond the largest dose tested.

**So the saturation story proposed in `compile_check.md` is dead**, and with it the claim
that would have travelled furthest. Recorded here rather than quietly dropped, because it
was the interesting version.

**What replaces it: the two families are on separate curves.**

| | on-`d` dose | behaviour | behaviour per unit dose |
|---|---|---|---|
| `pro_top` prefix | 0.93 | +0.89 | **0.96** |
| `+1·d` injection | 30.07 | +0.50 | **0.017** |

≈**58× more behaviour per unit of on-`d` displacement** from the prefix. Within prefixes the
dose predicts the response (r = +0.99); within injections it also does (0.13 → 0.45,
increasing); the curves are simply different. Matching a direction's on-`d` displacement
does not reproduce its effect.

**The negative side shows no dose-response at all.** −0.5 gives −0.19 and −1.0 gives −0.15;
neither is significant and the ordering does not even go the right way. Consistent with
`steering_random_control.md`, where `−1·d` sits inside the random-direction band under both
judges.

## Limits

1. Two doses per sign. A curve through two points is a line by construction; this
   establishes the *direction* of the trend, not its shape.
2. Single judge (`deepseek-v4-pro`) for these two arms. The `±1` arms are cross-judged;
   these are not.
3. One model, one layer, one direction.
4. The `±0.5` generations date from the June behavioural eval, so they were produced by the
   same code path but on an earlier NDIF model serving. If NDIF re-served the model between
   then and August, this comparison spans a possible model-build boundary — the exact
   condition `PROJECT_SPEC.md` §5.4 calls a season break.

## What would settle it

Doses placed deliberately along the axis (0.25, 0.75, 1.5, 2.0 × ‖R‖) rather than the two
that happen to exist, judged by both raters, generated in one run. That gives the actual
shape of the injection's dose-response and locates its peak, if it has one below the point
where a perturbation of that size starts degrading text in any direction (see
`steering_random_control.md`).
