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

**Reading (b) is falsified, on the positive side.** *Qualified 2026-08-29: everything in this
paragraph concerns `+d` only. The negative arms go the other way (see below), so "still
climbing" must never be stated of the injection in general.*

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

**Correction 2026-08-27: two inherited numbers, and one non-inheritance worth recording.**

| inherited from `compile_check.md` | as printed above | corrected |
|---|---|---|
| behaviour-per-unit ratio | ≈58× | **roughly 60-fold (50 to 75)**, median 61.3 over 18 estimator/judge/arm combinations. Safest wording: "more than an order of magnitude". |
| within-prefix dose-response | r = +0.99 (positive side, n=4) | **r = +0.986 (p = 0.0003)** over the 6 surviving arms, both signs, and it is an **ordering** result, not a dose-response |

**This document does NOT inherit the withdrawn `anti_top` behavioural number** (independently
verified: it appears nowhere here). It inherits only those two figures, both corrected above.
Recorded because "which downstream documents caught the withdrawal" is the question the
corrections were chasing, and this one is clean by accident rather than by check.

**The negative side shows no dose-response at all.** −0.5 gives −0.19 and −1.0 gives −0.15;
neither is significant and the ordering does not even go the right way. Consistent with
`steering_random_control.md`, where `−1·d` sits inside the random-direction band under both
judges. *Qualified 2026-08-27:* that placement is **estimator-dependent**. Under a fixed
baseline `−1·d` is inside the band under Claude and **outside** it (below all 8 randoms) under
DeepSeek. See `steering_random_control.md`.

## Limits

1. Two doses per sign. A curve through two points is a line by construction; this
   establishes the *direction* of the trend, not its shape.
2. Single judge (`deepseek-v4-pro`) for these two arms. The `±1` arms are cross-judged;
   these are not.

   **Correction 2026-08-27: human blind ratings for these arms DO exist and were never
   analysed.** They are in `data/analysis/behavioral_blind.csv` and had been sitting there
   unread while this limit said the arms rested on one judge. Reported now:

   | arm | steered wins / base wins / tie | decided pairs | p |
   |---|---|---|---|
   | `+0.5 · d` | 2 / 2 / 6 | 4 | 1.00 |
   | `+1.0 · d` | 5 / 3 / 2 | 8 | 0.73 |
   | `−0.5 · d` | 4 / 8 / 1 | 12 | 0.39 |
   | `−1.0 · d` | 2 / 6 / 1 | 8 | 0.29 |

   **None significant, all badly underpowered** at 4 to 12 decided pairs per arm. They neither
   confirm nor refute. Directionally `+1·d` agrees with the LLM judges; `−1·d` does **not**
   agree with "inside the random band", but at 8 decided pairs that disagreement carries no
   weight either. The limit is not "single judge", it is "single judge plus a human pass too
   small to arbitrate".
3. One model, one layer, one direction.
4. ~~The `±0.5` generations date from the June behavioural eval, so they were produced by the
   same code path but on an earlier NDIF model serving. If NDIF re-served the model between
   then and August, this comparison spans a possible model-build boundary — the exact
   condition `PROJECT_SPEC.md` §5.4 calls a season break.~~

   **CORRECTED 2026-08-27: this limit was factually wrong, and it was wrong about which
   experiment it applied to.** The ±0.5 vs ±1 comparison does **not** span a model-build
   boundary. `data/cache/behavioral/` shows `base`, `+0.5`, `+1`, `−0.5` and `−1` were **all**
   generated **2026-06-10**, 50 files each. The comparison is entirely within one run. There is
   no season-break risk here.

   The real June/August split is in the **random control**, not here: base and ±1 date from
   2026-06-10 and all eight random arms from 2026-08-26, and `steering_random_control.md` did
   not list it. The caveat was **misfiled**: written against the experiment it does not apply
   to, absent from the one it does. It has been moved to
   `steering_random_control.md` (with the mitigating α check, which is bit-identical across the
   two dates). Worth naming the failure mode: caveats in this project were not verified against
   the artifacts the way results were, which is how a `ls -l` refuted a limit that had stood
   unchallenged.

## The field-practice tension, and its resolution (added 2026-08-27)

This document applies **α = 1.0·‖R‖**. Published steering practice does not go near that, and
that gap needs stating rather than leaving implicit, because a reader who knows the literature
will assume the dose here is absurd.

| practice | coefficient | intervention |
|---|---|---|
| CAA-style published practice (Panickssery et al. 2023) | applied at "all token positions after the user's prompt" | **during generation** |
| reported ceiling before degeneracy becomes non-negligible | ≈**0.1×** the normalised residual norm | during generation |
| one open-model reproduction (GLM-5) | ≈**0.025×**, with >0.025 frequently incoherent | during generation |
| **this project** | **1.0×‖R‖** | **prefill-only**, touching no generated token |

**The tension is apparent, not real, and the resolution is mechanistic.** Our intervention is
prefill-only: `steering_random_control.md`'s manipulation check measures the edit at all 9
prompt positions and at **none** of the generated tokens. A prefill-only edit does not compound
across the decoded sequence, so it has far more magnitude headroom than an edit applied at each
decoding step. Stated as a finding: **prefill-only steering appears to have 10× to 40× more
magnitude headroom than published per-token practice, and its effect is still climbing at
1.0·‖R‖.**

The consequence for the sentence above: "the vector is under-applied" means **under-applied for
prefill-only steering**. It is *not* a criticism of published per-token practice, which is
calibrated against a different failure mode.

**Sourcing caveat.** The 0.1× and 0.025× figures are **second-hand**, quoted via a LessWrong
reproduction post rather than fetched from the primary system card. Treat both as
**unverified**. The prefill-only-versus-per-token distinction does not depend on their exact
values.

## What would settle it

Doses placed deliberately along the axis (0.25, 0.75, 1.5, 2.0 × ‖R‖) rather than the two
that happen to exist, judged by both raters, generated in one run. That gives the actual
shape of the injection's dose-response and locates its peak, if it has one below the point
where a perturbation of that size starts degrading text in any direction (see
`steering_random_control.md`).
