# What the 2026-09-05 checks revise

Ten checks run in one session, all but two from cache. Three of them revise claims this
project had already published, and one of those is in the report's executive summary. This
document is the accounting: what is withdrawn, what survives, and what is newly open.

Every number below is reproducible from a committed script and none of it needed a rerun of
anything that had already been paid for on NDIF.

---

## 1. THE HEADLINE REVISION: "adding works, removing does not" is not established

**The published claim.** Executive summary point 6: *"Removing `d` does nothing either...
So adding `d` works and removing it does not."* `steering_ablation.md` supported it by
arguing the null "is not a null of magnitude", because the largest ablated component (1.684)
exceeds `anti_top`'s on-`d` displacement of 1.49.

**Why it does not hold.** OLMo-3 uses RMSNorm, which rescales the residual by a scalar, so
the next block reads its **direction**. The honest unit for the size of an intervention is
therefore the angle it rotates the residual through, not the norm it adds or removes:

| intervention | mean rotation | `‖Δ‖` | on-`d` |
|---|---|---|---|
| **ablate `d`, k=1** | **0.92°** | 0.48 | 0.48 |
| ablate `d`, k=2 | 1.83° | 0.95 | 0.95 |
| ablate `d`, meandiff | 0.89° | 0.46 | 0.46 |
| `+0.5·d` injection | 26.80° | 15.03 | 15.03 |
| `+1.0·d` injection | **45.16°** | 30.07 | 30.07 |
| prefix `pro_coherent` | 46.85° | 22.85 | 0.64 |
| prefix `pro_top` | **50.44°** | 24.25 | 0.93 |

The "adding" arm perturbs the residual **49x** harder than the "removing" arm (25x even
against the k=2 control). Comparing them and concluding the model treats the two directions
asymmetrically compares a 45-degree edit with a 1-degree edit. `1.684 > 1.49` compares
**on-`d` components**, and the model does not read the on-`d` component.

**The estimator hypothesis was tested and is dead.** Standard practice says a logistic probe
classifies better and steers worse than difference-of-means, which would make our null an
artifact of shipping a logistic `d`. A meandiff direction at the same layer
(`cos` with shipped = 0.7523, held-out separation also 1.000) was **pre-registered** at 0.89deg
rotation against logistic's 0.92deg, so predicted to be equally null. It was: **30/50
byte-identical against logistic's 33/50**. Edit size, not estimator choice, is the whole
story. Source: `scripts/meandiff_ablation.py`, `meandiff_ablation.json`.

**Withdrawn:** any reading of the ablation null as evidence that `d` is not a causal channel,
and the add/remove asymmetry as a mechanistic claim.
**Survives:** ablation at this scale is indistinguishable from ablating a **random direction
at the same scale**. That comparison is size-matched and remains valid.

Source: `scripts/normalization_check.py`, `normalization_check.json`. Correction is also
inline in `steering_ablation.md`.

## 2. The causal curve: null at all five layers, and uninformative for the same reason

200 NDIF generations, k=1, each layer ablated with its **own** native direction.

| layer | `‖R‖` | `\|comp\|` | % of norm | rotation | identical | distinct4 | loops |
|---|---|---|---|---|---|---|---|
| base | 29.8 | - | - | - | - | 0.912 | 8/50 |
| L16 | 19.0 | 0.214 | 1.13% | 0.65° | 33/50 | 0.925 | 6/50 |
| L24 | 29.8 | 0.476 | 1.60% | 0.92° | 33/50 | 0.942 | 4/50 |
| L32 | 52.2 | 0.941 | 1.80% | 1.05° | 38/50 | 0.925 | 6/50 |
| L40 | 66.1 | 1.533 | 2.32% | 1.34° | 37/50 | 0.931 | 5/50 |
| L48 | 79.1 | 2.181 | 2.76% | 1.57° | 41/50 | 0.939 | 5/50 |

**The direct answer: the L24 null is not about L24.** But these are not five independent
nulls. The edit is 1.1% to 2.8% of the norm at every depth, about a degree everywhere, so a
uniform null is the expected outcome rather than a discovery.

**The genuinely informative part is the ordering.** The absolute component grows **10x** from
L16 to L48 and the rotation more than doubles, yet continuations become **more** identical
(33/50 -> 41/50). Bigger edits, less effect. Consistent with late layers holding the
information while the model has already committed.

Source: `scripts/causal_layer_curve.py`, `causal_layer_curve.json`.

## 3. Nulls decide the answer, and the standard one is too weak

Refitting the probe on **label-shuffled** pairs (same texts, same geometry, only the label
destroyed), n=40:

| | real | label-shuffled |
|---|---|---|
| held-out separation | **1.000** | **0.519** (0.265-0.794) |
| layer-to-layer cosine | 0.556 | **0.435** (max 0.453) |
| L16 row mean | 0.284 | 0.158 |
| late band {32,40,48} | 0.870 | 0.784 |

Isotropic random vectors give `|cos|` with `d_L24` of only **0.013**, which is 1/sqrt(5120);
against that null everything looks significant.

**Revised:** the cross-layer cosine structure was used to argue the layers share a pro-human
feature and to justify the band. Most of that **shape** survives label shuffling. It is
residual-stream geometry, not concept structure.
**Survives:** the real direction clears the null on every layer pair (excess +0.121,
40/40 draws), and separation is 1.000 against a coin flip. The direction encodes the label;
the geometric story was overstated. The band recommendation stands on a different reason.

Source: `scripts/direction_null.py`, `direction_null.json`.

## 4. What `d` is, and is not

**Approach is a passenger, not the driver.** `cos(d, approach) = 0.1501`, the only audited
confound never orthogonalised out. Project it out and nothing moves:

| direction | held-out | all 135 | kind>cruel | gap |
|---|---|---|---|---|
| `d` shipped | 1.000 | 1.000 | 6/6 | +0.2075 |
| `d`, approach removed | 1.000 | 1.000 | 6/6 | +0.2067 |
| approach alone | 0.824 | 0.837 | 4/6 | +0.0206 |

**But the seed pairs are confounded with it**: approach *alone* separates them at 0.824. A
probe fit on this corpus could have been an approach detector. Fact about the fit, not the
corpus. Worth fixing in the pairs for a future season.

**"Pro-human" is a family, not an axis.** The 15 seed axes (9 pairs each) give per-axis
directions with mutual cosine **mean +0.529, range +0.327 to +0.677, 0 of 105 pairs
negative**, top eigenvalue **56.4%**. A real shared axis exists; roughly half the structure
is axis-specific. The shipped `d` correlates +0.488 to +0.633 with every axis, so it is a
genuine centroid - though `empathy` and `boundaries` are its **weakest** correlates, which is
not what the name would predict. Caveat: 9 pairs in 5120 dimensions is noisy and this number
has no label-shuffled null yet.

Source: `scripts/direction_purity.py`, `direction_purity.json`.

## 5. Methodology that should carry to any future direction

**Probe accuracy cannot rank layers here.** It is 1.000 at all five. Use the **margin** and a
per-layer null instead:

| layer | accuracy | margin | null | excess | Cohen's d |
|---|---|---|---|---|---|
| L16 | 1.000 | 0.2353 | 0.0035 | +0.2318 | 2.73 |
| **L24** | 1.000 | **0.2726** | 0.0020 | **+0.2706** | **3.38** |
| L32 | 1.000 | 0.2464 | 0.0011 | +0.2453 | 3.22 |
| L40 | 1.000 | 0.2330 | 0.0009 | +0.2321 | 3.15 |
| L48 | 1.000 | 0.2044 | 0.0010 | +0.2034 | 2.97 |

This retroactively supports L24 on a better criterion than the one that chose it:
`extract_direction.py` selects on held-out separation with a strict `>`, so under a five-way
tie at 1.000 it simply keeps the first layer. Margin says L24 is genuinely the peak.

**The LayerNorm concern does not apply.** OLMo-3 is RMSNorm (no centering), and `d` is
0.41% to 2.55% aligned with the all-ones axis anyway. Clean negative, recorded so it is not
re-asked.

Source: `scripts/layer_concept_profile.py`, `scripts/plot_layer_concept.py`,
`scripts/normalization_check.py`.

## 6. What was CONFIRMED rather than revised

**The over-steering confound is ruled out.** Drop every item where the injection, the prefix
or the base loops, and the prefix-over-injection gap **widens**: +0.420 -> +0.514 (deepseek)
and +0.370 -> +0.389 (claude), n=36. Per-item coherence and kindness are uncorrelated inside
the injection arm (r = +0.0175 / +0.0414). Residual: this is lexical repetition, not
capability. Source: `coherence_confound.md`.

**A banded objective would have ranked the board differently.** Scoring over {32,40,48}
instead of L24 alone: `pro_top` goes +0.10769 -> **-0.00308** (banded mean) and **-0.00979**
(per-layer min), while `pro_coherent` goes +0.04032 -> +0.02167 / +0.02148. The readable
instruction wins under every banded variant. Note this says **this string** does not
generalise across depth, not that a banded objective is unbeatable - nobody has searched
against one. Sources: `banded_direction.json`, `banded_score_arms.json`.

**Weighting the banded mean changes nothing** on the band worth using: min-cos moves 1.6%
across uniform/margin/inv-var/minimax and arm scores move under 0.0006. Band choice dominates
weighting by an order of magnitude.

## 7. Open, in priority order

1. **A dose sweep on k at one layer.** The only way to learn whether a *larger* ablation
   bites. More layers will not help; the curve is flat because the edit is small. Note k
   well above 1 stops being ablation and becomes negative steering, the exact criticism
   `steering_ablation.md` levelled at `-1·d`.
2. **The placebo control** (Mody et al., arXiv:2607.25907): a string searched against a
   **random** direction to a matched board score. Still the single highest-value experiment
   and still not run.
3. **A fluency rubric scored separately from kindness**, 250 cached items, judge calls only.
   Closes the capability-flavoured version of the coherence confound.
4. **De-confound the seed pairs against approach** before any future season.
5. **A label-shuffled null for the per-axis cosine** in section 4.
