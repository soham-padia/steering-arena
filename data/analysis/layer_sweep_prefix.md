# Does the winning prefix act at every depth, or only at layer 24?

Everything in this project is measured at layer 24: that is where `d` was fit and where the
leaderboard scores. `pro_top` was found by GCG optimising against *that specific readout*.
`compile_check.md` names this as the obvious next experiment in its Limits section. This is it.

Data: `layer_sweep_prefix.json`. Script: `scripts/layer_sweep_prefix.py`.
Figure: `figures/layer_sweep.png`. OLMo-3-32B (64 layers), 50 eval prompts, 27 batched NDIF
forward passes, all residuals finite.

## Prediction, recorded before the first forward pass

Written into the script docstring before any NDIF call, and reproduced verbatim here so the
misses stay visible.

| # | prediction | outcome |
|---|---|---|
| P1 | (A) ramps into 24, peaks at/near 24, decays after but not to zero | **held** for `pro_top`; **missed** for `pro_coherent`, whose (A) peak is at **L16**, not 24 |
| P2 | (B): `pro_top`'s L24 point is the max, and the other four are each under half of it | **held, and understated.** Point predictions were L16 < +0.015, L32 < +0.015, L40 < +0.012, L48 < +0.010. Actual: +0.0076, +0.0071, +0.0000, **−0.0049**. The sign flip at L48 was not predicted |
| P3 | peakiness(`pro_top`) > peakiness(`pro_coherent`) | **held**, 4.69 against 0.89 |

Stated confidence at the time: 65% on P2, 55% on P3. Both came in, and P2 came in harder than
the point predictions allowed for.

## (B) The principled measurement: each layer projected onto its OWN direction

Five layers have a layer-native logistic direction. Each residual is projected onto the
direction fit in its own basis. This is the measurement that answers the question.

| layer | ‖R‖ base | base cos(R, d_L) | `pro_top` shift | t | `pro_coherent` shift | t |
|---|---|---|---|---|---|---|
| 16 | 19.00 | −0.0057 | +0.0076 | 3.9 | **+0.0214** | 11.6 |
| **24** | 29.76 | +0.0067 | **+0.0354** | 17.5 | +0.0217 | 9.1 |
| 32 | 52.18 | −0.0108 | +0.0071 | 4.8 | **+0.0223** | 11.2 |
| 40 | 66.10 | −0.0169 | **+0.0000** | 0.0 | **+0.0207** | 9.7 |
| 48 | 79.08 | −0.0225 | **−0.0049** | −2.2 | **+0.0243** | 10.6 |

Paired t over 50 prompts, df = 49.

| arm | board score | mean over the 5 native layers | sd | **CV** | L24 / max of the other four | L24 / mean of the other four |
|---|---|---|---|---|---|---|
| `pro_top` | +0.10769 | +0.0090 | 0.0156 | **1.73** | **4.69** | **14.5** |
| `pro_coherent` | +0.04032 | +0.0221 | 0.0014 | **0.06** | 0.89 | 0.98 |

**It came out SPIKE.** `pro_top`'s alignment with "pro-human" exists at the layer it was
optimised against and essentially nowhere else. Four layers away in either direction it is
1/5 the size; at L40 it is zero to four decimal places; at L48 it is negative and
significantly so.

**And the control does exactly what a control is for.** `pro_coherent`, a readable
instruction never optimised against any readout, varies by **6%** across five layers spanning
32 transformer blocks: +0.0214, +0.0217, +0.0223, +0.0207, +0.0243. That flatness is what
"the string is doing something general" looks like, and it is the reason the `pro_top` spike
cannot be dismissed as "the L32/L40/L48 probes are just worse". The probes detect a pro-human
prefix fine. They do not detect `pro_top`.

**The board ranking is a layer-24 artifact.** Head to head on native directions:

| layer | winner |
|---|---|
| 16 | `pro_coherent` |
| **24** | `pro_top` |
| 32 | `pro_coherent` |
| 40 | `pro_coherent` |
| 48 | `pro_coherent` |

`pro_top` outscores `pro_coherent` on the leaderboard by **2.7×** (+0.1077 against +0.0403)
and loses to it at **four of the five** layers where the question can be asked. It wins at
exactly the one layer the board reads.

## (A) Every sampled layer, projected onto `d_L24`

The same forward passes, projected onto the shipped `d_olmo3_L24_logistic` at all nine depths.

| layer | ‖R‖ base | base cos(R, d_L24) | `pro_top` shift | `pro_coherent` shift |
|---|---|---|---|---|
| 0 | 4.41 | −0.0021 | +0.0008 | +0.0011 |
| 8 | 13.29 | −0.0115 | +0.0076 | +0.0123 |
| 16 | 19.00 | −0.0008 | +0.0152 | **+0.0321** |
| **24** | 29.76 | +0.0067 | **+0.0354** | +0.0217 |
| 32 | 52.18 | +0.0060 | +0.0201 | +0.0178 |
| 40 | 66.10 | +0.0043 | +0.0150 | +0.0179 |
| 48 | 79.08 | +0.0028 | +0.0105 | +0.0161 |
| 56 | 92.80 | +0.0051 | +0.0064 | +0.0153 |
| 63 | 169.41 | +0.0135 | +0.0023 | +0.0081 |

**Caveat, load-bearing: `d_L24` was fit in layer 24's basis, so away from 24 this curve partly
measures basis drift, not "pro-humanness".** The native directions are only partly aligned
with `d_L24`: cos = 0.43 (L16), 0.70 (L32), 0.60 (L40), 0.53 (L48). Read (A) for its shape,
not its level, and settle any disagreement between (A) and (B) in favour of (B).

**The (A) tail after layer 24 is the residual stream carrying the same write forward, not new
alignment.** (A) says `pro_top` still has +0.0201 of `d_L24` alignment at L32; (B) says the
layer-32 readout sees +0.0071, and by L40 (B) sees nothing at all while (A) still reports
+0.0150. The residual stream is additive, so a displacement written at L24 persists mechanically
into later layers and a fixed L24 probe keeps reading it. Asking each layer's own readout is
what removes that.

## Reading

**This is a Goodhart signature and it qualifies the project's headline.** The correct statement
is now: *`pro_top` shifts the residual toward pro-human at the layer the leaderboard scores,
and does not do so anywhere else in the stack.* Every previously published `pro_top` activation
number (`compile_check.md`, `cosine_scale.md`) is a layer-24 number and must be read as one.

**It does not touch the behavioural results, and it sharpens their interpretation.**
`pro_top`'s judged kindness shift is +0.89 against `pro_coherent`'s +0.60 (`prefix_eval.md`),
so the arm with the *narrower* activation footprint produces *more* behaviour. Depth-wide on-`d`
alignment therefore does not explain the behaviour either. That is the same conclusion
`compile_check.md` reached from a different angle, now with a second independent line of
evidence: the on-`d` component is a **marker**, not the mechanism. Here it turns out to be a
marker that exists at only one depth.

**It strengthens, rather than resolves, the outstanding threat in `compile_check.md` Limit 5.**
A string that moves one readout and no other readout of the same concept is what you would
expect from optimisation against that readout, whatever the readout was. That is the rival
explanation Mody et al. (arXiv:2607.25907) raise, and the missing random-direction search
control remains missing.

## Limits

1. **Two arms.** `pro_top` and `pro_coherent`. The anti arms, `control_junk` and `control_text`
   were not run. `pro_coherent` alone carries the whole "a non-optimised prefix is flat" claim.
   The cheap extension is the same 27-call sweep over `anti_top` and `anti_coherent`, which
   would test whether the anti soup shows the mirror-image spike.
2. **Five native directions, all logistic, all fit on the same seed pairs.** They share a
   construction and could share a blind spot. They are not independent probes of "pro-human".
3. **Held-out separation is 1.000 at all five layers** (`layer_sweep_olmo-3-1125-32b_logistic.json`),
   so no native probe can be dismissed as a bad classifier. That is what licenses the L40 and
   L48 nulls. It is a linear-separability claim, not a claim that all five encode the same thing.
4. **Last-token residual only.** The prefix changes every position; this reads one. A prefix
   could carry depth-general effects through earlier positions or through attention in ways
   this does not see. Inherited unchanged from `compile_check.md` Limit 1.
5. **9 of 64 layers sampled.** The spike is resolved at ±8 layers. Nothing here says how sharp
   it is at ±1, and the L24 peak could be a plateau spanning L20 to L28.
6. **50 eval prompts, not the 16 committed probes.** The board score is a cosine shift over the
   probes; this is a cosine shift over the eval prompts, for comparability with the behavioural
   work. Related quantities, not identical ones. Same caveat as `cosine_scale.md` Caveat 1.
7. Numerical note: numpy again emitted spurious divide/overflow/invalid warnings from the matmul
   on this platform (`compile_check.md` Limit 6). All 27 residual batches were explicitly checked:
   float32, zero non-finite entries, `nonfinite: []` in the JSON, and the two sanity gates below
   reproduce published values to 4 decimals.

## Sanity checks

| check | expected | measured | verdict |
|---|---|---|---|
| base cos(R, d) at L24 (`cosine_scale.md`) | +0.0067 | **+0.006749** | pass |
| `pro_top` L24 cos shift (`cosine_scale.md`) | +0.0355 | **+0.035415** | pass |
| `pro_coherent` L24 cos shift (`cosine_scale.md`) | +0.0217 | **+0.021688** | pass |
| all residuals finite | yes | yes, 0 non-finite of 27 × 50 × 5120 | pass |
| (A) and (B) agree at L24 by construction | identical | identical | pass |

## Surprises

1. **The sign flip.** `pro_top` at L48 is −0.0049 with t = −2.2. The prefix that wins a
   pro-human leaderboard is mildly *anti*-aligned with the deepest pro-human readout measured.
   n = 1 layer and the effect is small, so this is recorded, not claimed.
2. **How flat `pro_coherent` is.** A 6% coefficient of variation across 32 transformer blocks
   was not the expected shape of the control. The prediction only required it to be flatter
   than `pro_top`.
3. **`pro_coherent`'s (A) peak is at L16, not L24**, and at L16 it is more than double
   `pro_top`'s. P1 was written for `pro_top` and does not describe the control at all.
4. **The neutral state drifts anti-aligned with depth.** base cos(R, d_L) on native directions:
   −0.0057, +0.0067, −0.0108, −0.0169, −0.0225 for L16/24/32/40/48. Layer 24 is the only
   sampled depth where an unprefixed prompt is positively aligned with its own `d`. Not
   pursued here, but it means the L24 readout is unusual in a second, independent way.
