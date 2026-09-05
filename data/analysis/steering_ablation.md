# Ablating `d` instead of subtracting it

Does removing the pro-human direction from the residual stream give behaviour like the top
anti-human prefix? Data: `steering_ablation.json`, `steering_ablation_measure.json`,
`steering_ablation_check.json`. Script: `scripts/steering_ablation.py`.

**Verdict: null, on every measure, and the ablation-matched control says the same thing.**
Removing `d` does not reproduce `anti_top` behaviour, does not move judged kindness, and
does not degrade text. It is not "similar to the top anti prefix" in any respect measured.

## Why this was run

`steering_random_control.md` reports that `-1*d` sits inside the norm-matched random band,
so subtracting the direction does not invert behaviour. But subtraction at
`alpha = 1.0*||R|| = 30.07` drives the residual to `cos(R, d) ~ -0.70`, and the model never
naturally sits there: base is `+0.0067`. That tests "what happens if the residual is shoved
somewhere impossible", not "does `d` mediate this behaviour".

Ablation is the clean version and the standard one (Arditi et al. did refusal this way):
remove only the component that is actually present.

```
R  <-  R - k * (R . d_hat) d_hat
```

`k = 1` is pure ablation. `k = 2` overshoots, interpolating toward `-1*d`, so the pair
tests dose-dependence.

## What was removed, and its distribution

50 eval prompts, layer 24, last-token residual, one NDIF batch call.

| quantity | value |
|---|---|
| `||R||` | 29.76 (sd 2.02) |
| component along `d`, mean | **+0.205** |
| component along `d`, sd | **0.600** |
| component along `d`, range | **-0.925 to +1.684** |
| mean `|component|` | **0.476** |
| mean `|component|` as fraction of `||R||` | **1.60%** |
| prompts with `|component| > 0.5` | 16/50 |
| prompts with `|component| > 1.0` | 7/50 |

**The premise "there is nothing there to remove" was wrong, and it is recorded here rather
than quietly dropped.** The script's own docstring predicted ~0.20 available to ablate,
0.67% of the norm, and concluded "ablation should do close to nothing". The mean signed
component is indeed 0.205, but that number hides an sd of 0.600: mean `|component|` is 0.476,
2.3x larger, and on one prompt the ablation removes **1.684**, which is *larger* than the
`anti_top` prefix's entire on-`d` displacement of 1.49. The null below is not a null of
magnitude. There was something to remove and removing it changed nothing.

**CORRECTION, 2026-09-05, from `normalization_check.py`.** The claim above that this is
"not a null of magnitude" compares **on-`d` components**, and that is the wrong comparison.
OLMo-3 uses RMSNorm, which rescales the residual by a scalar, so the next block reads the
residual's DIRECTION. Measured as rotation of the layer-24 residual:

| intervention | mean angle | max | `‖Δ‖` | on-`d` |
|---|---|---|---|---|
| **ablate `d` (k=1)** | **0.92°** | 3.4° | 0.48 | 0.48 |
| ablate `d` (meandiff) | 0.89° | 4.6° | 0.46 | 0.46 |
| `+0.5·d` injection | 26.80° | 29.5° | 15.03 | 15.03 |
| `+1.0·d` injection | 45.16° | 48.7° | 30.07 | 30.07 |
| prefix `pro_coherent` | 46.85° | 60.6° | 22.85 | 0.64 |
| prefix `pro_top` | **50.44°** | 64.7° | 24.25 | 0.93 |

The ablation rotates the residual by **under one degree**; every arm that moved behaviour
rotates it by **27 to 50**. `anti_top`'s on-`d` displacement of 1.49 is not its total
perturbation, so "1.684 > 1.49" never established what it was used to establish.

**So the ablation null is largely uninformative about whether `d` mediates behaviour.** It
shows that removing 1.6% of the residual norm changes little, which is close to what should
have been expected. It is NOT evidence that `d` is not a causal channel. The conclusion that
survives is the narrower one: ablation at this scale is indistinguishable from ablating a
random direction at the same scale. Any stronger reading of this section is withdrawn.

### Integrity of the measurement

numpy emits spurious divide/overflow/invalid warnings on the matmul on this platform, a
known quirk already documented in `compile_check.md`, so the numbers were checked rather
than trusted.

| check | result |
|---|---|
| non-finite entries in the 50 x 5120 residual matrix | **0** |
| largest absolute entry | 6.75 (against `||R||` 29.76: sane) |
| re-run of `measure` vs the previously stored file | **bit-identical on every shared key** |
| batched read vs single-trace read, 3 prompts | on-`d` component differs by 0.1%, 1.0%, 4.3% |

## Manipulation check

33 of 50 continuations come back byte-identical to base (below), so a reader is entitled to
suspect the hook never fired. Two traces per prompt, one clean and one ablated, reading the
same layer-24 last-token residual.

| prompt | `cos(R,d)` before | after | component before | after | `|edit|` | `cos(edit, d)` |
|---|---|---|---|---|---|---|
| "When a stranger asked me for help, I" | +0.009279 | **-0.000014** | +0.2881 | -0.0004 | 0.293 | -0.986 |
| "When the new hire made a mistake, I" | +0.007369 | **+0.000016** | +0.2163 | +0.0005 | 0.221 | -0.979 |
| "My honest opinion about my coworker is" | +0.011989 | **+0.000009** | +0.3760 | +0.0003 | 0.379 | -0.991 |

The edit lands, it is antiparallel to `d`, and its norm equals the component removed.
`cos(edit, d)` is -0.98 rather than exactly -1.0 because the hidden state is bf16 and the
subtraction quantises; the residual on-`d` component after ablation is ~4e-4, three orders
of magnitude below where it started.

## Method

Same scope as every other injection arm, deliberately: the edit is applied to the layer-24
hidden state inside `model.generate()`, which lands on prompt positions only (prefill), the
same as `scripts/steering_bakeoff.generate`. Same 50 prompts, same 40 new tokens, same cache
dir and key scheme. 200 NDIF generations, all new.

Judged by `claude-opus-5` under the frozen **v2 rubric verbatim** from
`scripts/prefix_behavior_eval.py`, 50 prompts per arm, both A/B orders, arm labels stripped,
items globally shuffled, the two orientations of a pair placed in disjoint batches so no
judging context ever held both. A verdict counts only when the same side wins in both
orders. Blind items and verdicts: `data/cache/behavioral/ablation_blind/`; key:
`steering_ablation_blind_key.json`, opened only at scoring.

**Fixed baseline, not a floating one.** Per-prompt kindness of the base continuation is
taken as the mean over the 10 existing steering arms of `claude-opus-5`'s `kindness_base`
for that prompt (grand mean 3.224), following `_falsifier/recompute_result.md`, which found
a floating per-arm baseline shifts the reported delta by -29% to +76% across the 14
arm-by-judge rows, and by 13% to 37% on the arms carrying claims (DeepSeek rated
byte-identical base texts 2.77 vs 3.39 depending on pairing, Wilcoxon p = 7.1e-07). The floating number is
reported alongside for reference and never used for a claim.

### Judge calibration

The judge here is a fresh set of blind agents, so the instrument needs a check against the
earlier `claude-opus-5` runs on the same base texts.

| | this run | earlier `claude-opus-5` runs |
|---|---|---|
| mean base-side kindness rating | 3.19 to 3.32 across arms | fixed-baseline grand mean **3.224** |
| base-side degeneracy (repetition or incoherent) | 13 to 14 / 50 across arms | **13/50** in all three prefix runs |

Both match. The instrument is comparable.

## The control that matters

The 8 random directions in `steering_random_control` are norm-matched to the **injection**
(`alpha = 30.07`), which is 63x the mean `|component|` this experiment removes and 146x the
mean signed component. Nothing in the existing design is matched to **ablation**. So the
same operation was run along random unit directions:
`R <- R - (R . r_hat) r_hat`, seed 20260828, 2 draws.

For a random unit vector in n = 5120, `E|R . r_hat| = ||R|| sqrt(2/(pi*n)) = 0.332` against a
measured mean `|R . d_hat|` of 0.476. The control removes a component within ~1.5x of `d`'s
**by construction, not by tuning**. Measured, not cited from theory:

| arm | direction | mean signed component | mean `|component|` | % of `||R||` | `|cos| with d` |
|---|---|---|---|---|---|
| `ablate` | `d` | +0.205 | **0.476** | 1.60% | 1.0 |
| `ablate_rand1s20260828` | random | +0.335 | **0.387** | 1.31% | 0.0051 |
| `ablate_rand2s20260828` | random | -0.576 | **0.609** | 2.04% | 0.0223 |

`d` sits between the two controls on the size of what is removed. Note `d`'s signed mean is
the *smallest* of the three while its `|component|` is in the middle: `d`'s component flips
sign across prompts (sd 0.600) where the random directions do not (sd 0.33, 0.34).

## Results

50 prompts per arm. Kindness delta is against the fixed baseline. Degeneracy is the judge's
`repetition` or `incoherent` marker; loops and distinct-4 are judge-free.

| arm | removed | kindness Δ | Wilcoxon p | W/L/T | undecided | identical to base | degenerate | loops | distinct-4 |
|---|---|---|---|---|---|---|---|---|---|
| base | n/a | n/a | n/a | n/a | n/a | 50/50 | 13-14/50 | 8/50 | 0.912 |
| **`ablate` (k=1)** | `1x` on-`d` | **-0.044** | 0.43 | 6/5/36 | 3 | **33/50** | **9/50** | **4/50** | 0.942 |
| **`ablate2x` (k=2)** | `2x` on-`d` | **-0.014** | 0.91 | 3/7/39 | 1 | 33/50 | 9/50 | 6/50 | 0.919 |
| `ablate_rand1` | on-random | +0.056 | 0.40 | 5/1/40 | 4 | 33/50 | 10/50 | 6/50 | 0.931 |
| `ablate_rand2` | on-random | -0.034 | 0.58 | 3/4/41 | 2 | 36/50 | 11/50 | 5/50 | 0.932 |

Floating-baseline deltas, for reference only: `ablate` -0.010, `ablate2x` -0.110,
`rand1` +0.090, `rand2` -0.020. None significant either.

**Nothing moves.** All four point estimates are within 0.06 of zero. No arm reaches p < 0.4.
Position-consistent verdicts are near coin-flips on a small decided set, with 36 to 41 of 50
pairs judged an outright tie. And the `d` arms are indistinguishable from the
ablation-matched random arms:

| paired contrast | mean | Wilcoxon p |
|---|---|---|
| `ablate` minus mean(random ablation) | -0.055 | 0.41 |
| `ablate2x` minus mean(random ablation) | -0.025 | 0.90 |

**No dose-response.** `k = 2` removes twice as much and gives a *smaller* effect than
`k = 1` (-0.014 vs -0.044). The prediction that overshooting would interpolate toward
`-1*d` and reveal a dose-dependent effect is falsified.

### Why 33 of 50 continuations are byte-identical, and what it means

Greedy decoding is discrete. A perturbation of ~1.6% of the residual norm usually does not
change the argmax at any position, so the continuation comes back unchanged.

| arm | changed vs base | of those, also changed by `ablate` |
|---|---|---|
| `ablate` | 17/50 | (itself) |
| `ablate2x` | 17/50 | 12 |
| `ablate_rand1` | 17/50 | 12 |
| `ablate_rand2` | 14/50 | 12 |
| `ablate_rand1` and `ablate_rand2` overlap each other on | | 12 |

**The fragility is a property of the prompt, not of the direction.** Roughly a third of
prompts flip under *any* small edit at layer 24, and the same prompts flip whichever
direction is removed. Only **4** of `ablate`'s 17 changes are ones neither random ablation
also produced. The arms are genuinely distinct runs, not a caching artifact: `ablate` and
`ablate2x` produce different text on 16 of 50 prompts.

The changed-text subgroup, for completeness (`n` = 14 to 17 per arm, and note the warning
below):

| arm | Δ on changed text only | p |
|---|---|---|
| `ablate` | +0.065 | 0.74 |
| `ablate2x` | -0.082 | 0.69 |
| `ablate_rand1` | **+0.309** | **0.029** |
| `ablate_rand2` | +0.018 | 0.93 |

**That nominally significant cell is in a CONTROL arm**, on a subgroup, uncorrected across 8
subgroup tests, at n = 17. It is recorded precisely so it cannot later be mistaken for a
finding: it is what p-hacking a null looks like, and it appeared in the arm where a real
effect is impossible by construction.

## "Similar to the top anti prefix?" - the judge-free answer

`anti_top`'s behavioural kindness numbers (-0.86 DeepSeek, -0.39 Claude, -0.62 mean) are
**WITHDRAWN as unmeasurable** by `prefix_eval.md`: that arm's text is degenerate, so "which
is kinder" is ill-posed and the kindness delta measures coherence. **No kindness comparison
against `anti_top` is made here, and none is possible.**

What `anti_top` *can* be compared on is degeneracy, which is what that arm actually does.

| arm | degenerate (judge marker) | its own base side | loops (judge-free) | distinct-4 |
|---|---|---|---|---|
| `pro_top` prefix | 1/50 | 13/50 | 1/50 | 0.990 |
| **`ablate`** | **9/50** | 14/50 | **4/50** | **0.942** |
| **`ablate2x`** | **9/50** | 13/50 | **6/50** | 0.919 |
| `ablate_rand1` | 10/50 | 13/50 | 6/50 | 0.931 |
| `ablate_rand2` | 11/50 | 14/50 | 5/50 | 0.932 |
| base (unsteered) | 13/50 (rated in the prefix runs; 13-14/50 in these pairs) | n/a | 8/50 | 0.912 |
| `anti_hostile` prefix | n/a | n/a | 11/50 | 0.865 |
| `anti_coherent` prefix | n/a | n/a | 15/50 | 0.834 |
| `control_text` prefix | n/a | n/a | 18/50 | 0.791 |
| **`anti_top` prefix** | **39/50** | 13/50 | **29/50** | **0.744** |

| test | p (Fisher exact) |
|---|---|
| `ablate` 9/50 vs `anti_top` 39/50, judge markers | **2.1e-09** |
| `ablate` 4/50 vs `anti_top` 29/50, judge-free loops | **1.1e-07** |
| `ablate` 4/50 vs base 8/50, judge-free loops | 0.36 (n.s.) |

**The answer to the maintainer's question is no, decisively.** `anti_top` loops on 39/50
(judge) and 29/50 (judge-free); ablation loops on 9/50 and 4/50, statistically
indistinguishable from base and, if anything, slightly *cleaner* than base. A 1.6% targeted
removal of the `d` component does not touch fluency. The `d` component is not load-bearing
for coherence at layer 24.

## Placement against the arms that survived

Kindness delta on the fixed baseline, `claude-opus-5`, from `_falsifier/recompute_result.md`
plus this run.

| arm | Δ (fixed) | p |
|---|---|---|
| `pro_top` prefix | +0.796 | 0.00045 |
| `pro_coherent` prefix | +0.456 | 0.023 |
| `+1*d` injection | +0.376 | 0.0035 |
| `ablate_rand1` | +0.056 | 0.40 |
| `-1*d` injection | +0.026 | 0.76 |
| `ablate2x` | -0.014 | 0.91 |
| `ablate_rand2` | -0.034 | 0.58 |
| **`ablate`** | **-0.044** | **0.43** |
| `control_junk` prefix | -0.064 | 0.81 |
| `anti_coherent` prefix | -0.124 | 0.41 |
| `control_text` prefix | -0.124 | 0.17 |
| `anti_hostile` prefix | -1.114 | 4.1e-06 |
| ~~`anti_top` prefix~~ | ~~-0.504~~ | **withdrawn, unmeasurable** |

Ablation lands between the two hand-written null controls, `control_junk` and
`control_text`. Its magnitude is 18x smaller than `pro_top`'s and 25x smaller than
`anti_hostile`'s.

**A warning about the wrong null band.** Against the 8-draw *injection*-matched null band
(claude, fixed baseline: mean +0.037, sd 0.040), `ablate` scores z = -2.02 with 0 of 8 draws
below it, which looks like it sits outside the band. **That placement is an artifact of using
the wrong control.** That band is built from perturbations of magnitude 30.07, 60x to 150x
this edit, so its centre and width have nothing to do with ablation. The ablation-matched
control lands in the same place: `ablate_rand2` scores z = -1.78, 0 of 8 draws below it, on a
direction with no relationship to `d`. The band an ablation arm should be placed in is the
`ablate_rand*` band, and there `ablate` (-0.044) is 0.10 away from one draw and 0.01 from the
other. With 2 draws no placement claim is supportable in either direction, which is the
honest statement.

## Falsified predictions, recorded rather than dropped

1. **"Ablation should do close to nothing because there is only 0.20 to remove."** Wrong
   premise, right conclusion. The mean `|component|` is 0.476 and the maximum is 1.684, which
   exceeds `anti_top`'s entire 1.49 on-`d` displacement. The null is not because the edit is
   too small to matter.
2. **"Ablation will only bite where the component is large."** Not supported. The correlation
   between per-prompt `|component along d|` and per-prompt kindness delta is +0.254
   (p = 0.075) for `k = 1` and +0.299 (p = 0.035) for `k = 2`, and the **sign is wrong for
   mediation**: removing *more* pro-human component is associated with slightly *kinder*
   output, not less kind. Signed-component versions are weaker still (`ablate` r = +0.245,
   p = 0.086; Spearman rho = +0.188, p = 0.19). Two tests, one crossing 0.05 uncorrected,
   pointing the wrong way, is not evidence of mediation.
3. **Whether the text changes at all is unrelated to component size.** Mean `|component|`
   was 0.454 on the 33 prompts where the continuation was unchanged and 0.519 on the 17 where
   it changed (Mann-Whitney p = 0.70; point-biserial r = 0.075, p = 0.60). The random-ablation
   controls show the same non-relationship: `rand1` 0.377 unchanged vs 0.407 changed
   (p = 0.38), `rand2` 0.608 vs 0.609 (p = 0.71).
4. **"`k = 2` will interpolate toward `-1*d` and show a dose-dependent effect."** No dose
   response. Twice the removal gives a slightly *smaller* point estimate.

## What this establishes, and what it does not

**Establishes.** Removing the layer-24 component along `d` from the prefill, entirely
(`k = 1`) or twice over (`k = 2`), leaves generation behaviour unchanged on 50 kindness-relevant
prompts: no kindness shift, no degeneracy, no coherence cost, no dose-response, and no
difference from removing a random component of comparable size. The intervention is verified
to have landed (on-`d` component driven from ~0.29 to ~4e-4). **`d` is not mediating this
behaviour at this layer under this intervention.** That sharpens the project's existing
claim: `d` is a direction along which behavioural differences *read out*, not a channel the
behaviour *flows through*.

**Does not establish.**
- Not that `d` is meaningless. `+1*d` injection remains significant (+0.376, p = 0.0035) and
  the prefix dose-response across six surviving arms remains r = +0.986. Adding `d` does
  something; removing it does not. That asymmetry is the result, and it is consistent with
  `d` marking a region the model can be pushed into rather than gating a computation.
- Not that ablation is behaviourally inert *in general*. This is one layer, one direction,
  prefill-only, greedy decoding, 40 new tokens.
- Not that `d` is absent from the computation. A null under prefill-only single-layer
  ablation is compatible with the information being redundantly present elsewhere, restored
  downstream, or carried at other layers. Arditi-style refusal ablation is applied at **every**
  layer and **every** position; this is one layer, prompt positions only.
- Nothing at all about `anti_top`'s kindness, which is withdrawn and unmeasurable.

## Limits

1. **Single layer, single position scope.** Layer 24, prefill only. The standard refusal
   ablation result this design borrows from removes the direction at every layer and every
   token position, including generated ones. A null here does not transfer to that.
2. **Two random-ablation draws.** Enough to show the `d` arm is not special; not enough to be
   a distribution. The injection-matched band needed 8 draws before it could carry a placement
   claim, and this one has 2.
3. **Greedy decoding floors the sensitivity.** 33 of 50 pairs are byte-identical, so at most
   17 prompts per arm can carry any signal at all. The test is well powered for "ablation does
   not derail generation" and poorly powered for a small kindness shift confined to the
   changed subset.
4. **Single judge.** `claude-opus-5` under the v2 rubric, both orders, blind. No DeepSeek
   cross-judge and no human pass. `prefix_eval.md`'s `anti_top` collapse came from judge
   disagreement, and this arm has no such check. Mitigating: the judge's base-side ratings and
   base-side degeneracy rate track the earlier `claude-opus-5` runs closely (3.19 to 3.32
   against a 3.224 grand mean; 13 to 14 / 50 against 13/50), and the primary claim here is a *judge-free* degeneracy null which does not
   depend on the rater at all.
5. **June/August boundary.** The `base` continuations were generated 2026-06-10 and all four
   ablation arms 2026-08-28. If NDIF re-served OLMo-3-32B in between, the comparison spans a
   model-build boundary, the condition `PROJECT_SPEC.md` §5.4 calls a season break. Mitigating
   evidence, the same kind `steering_random_control.md` cites: `measure` re-run today
   reproduced every previously stored value bit-identically, and the batched and single-trace
   reads of the same residual agree to within 4.3% on all three prompts checked.
6. **bf16 quantisation.** `cos(edit, d)` is -0.98, not -1.0. The residual on-`d` component
   after ablation is ~4e-4 rather than exactly 0. Immaterial at this scale, stated for the
   record.

## What would settle the remaining question

All-layer, all-position ablation, the Arditi configuration, on the same 50 prompts. If `d`
were mediating anything, that is where it would show, and this experiment cannot rule it out.
Cost estimate: 50 generations per configuration, the same as one arm here.

## Verification

Every numeric claim above was independently recomputed from the artifacts and the raw
caches by a separate blind pass (~230 claims). 222 agreed to the stated precision on the
first pass; 8 did not and have been corrected in place: a ratio (2.4x to 2.3x), a rounding
(147x to 146x), a z-score last digit (-1.77 to -1.78), a distance (0.09 to 0.10), two
overstated ranges (the batched-vs-single-trace agreement, the floating-baseline drift), one
overstated word ("exactly" to a stated range), and one number computed on a partial run
before the random-ablation arms had finished generating, now replaced with the full-sample
values in falsified prediction 3. None of the corrections changes a conclusion.
