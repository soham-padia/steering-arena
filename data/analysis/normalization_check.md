# The LayerNorm objection does not apply, and rotation is the honest unit for an intervention

`python scripts/normalization_check.py` → `data/analysis/normalization_check.json`. Reads cached activations only: **zero NDIF calls, zero GPU, $0**.

> **I did not run this.** This page is reconstructed from `normalization_check.json` plus the docstring of `scripts/normalization_check.py`. I did not execute the script, watch the job, or see any output that is not in the committed JSON. Every number below is transcribed from that file; where the JSON does not carry a quantity, this page says so rather than sourcing it elsewhere.

The run reads only cached activations and costs zero NDIF calls, so anyone who wants a verified version can produce one with the command above.

`REVISIONS_2026-09-05.md` is authoritative on what this result is now taken to mean; this page is authoritative on what the artifact contains.

## Q1: nothing is subtracted, and `d` is near-orthogonal to all-ones regardless

The JSON records `norm_type` as **RMSNorm (no centering)**. RMSNorm divides by a root-mean-square and subtracts no mean, so the objection that normalization silently removes the component of `d` along the all-ones vector has nothing to act on in this model. The overlap is recorded anyway, so the question is answered rather than re-asked.

| layer | `cos(d, 1)` | derived, as a percentage |
|---|---|---|
| `L16` | 0.0179 | 1.79% |
| `L24` | 0.0255 | 2.55% |
| `L32` | 0.0077 | 0.77% |
| `L40` | 0.0045 | 0.45% |
| `L48` | 0.0041 | 0.41% |

The spread is **0.41%–2.55%** (*derived*), so even under a centering norm almost nothing would be lost. The largest value sits at `L24`, the shipped scoring layer — the one layer where the answer bears on a live score, and 2.55% is still too small to move a ranking.

## Q2: measure the rotation, because the model reads the residual's direction

A displacement can be large in norm and irrelevant, or small in norm and decisive, depending on how far it turns the vector the next block reads. `base_resid_norm` is **29.763**, which sets the scale for the norm column.

| intervention | mean angle (°) | max angle (°) | mean ‖Δ‖ | mean on-`d` |
|---|---|---|---|---|
| `ablate d, logistic` | **0.916** | 3.427 | 0.476 | 0.476 |
| `ablate d, meandiff` | 0.893 | 4.622 | 0.465 | 0.465 |
| `+0.5*d` injection | 26.8 | 29.536 | 15.035 | 15.035 |
| `+1.0*d` injection | **45.165** | 48.692 | 30.07 | 30.07 |
| prefix `pro_coherent` | 46.848 | 60.594 | 22.853 | 0.638 |
| prefix `pro_top` | **50.437** | 64.681 | 24.248 | 0.931 |

In this unit, ablating `d` turns the residual by **under one degree** (0.916°, and 0.893° for the meandiff variant), while every intervention that moved behaviour turns it by **27–50°**. Injection to ablation is **49×** (45.165 / 0.916 = 49.3, *derived*); in norm terms the ablation displaces 1.60% of `base_resid_norm` against the injection's 101.0% (*derived*). That is what drives the headline withdrawal in `REVISIONS_2026-09-05.md` §1 and the inline correction in `steering_ablation.md`: "adding `d` works and removing it does not" is a **size** gap between two arms that were never matched in magnitude, not a mechanistic asymmetry between addition and removal. The caveat travels with it: this table shows the arms were incomparable, not that ablation is inert, and the random-direction control supporting that separate statement is not in this JSON.

## The prefixes rotate more than a full injection, and almost none of it lies along `d`

Remarked nowhere else in the record: the prefix arms turn the residual **further** than a full `+1.0*d` injection does — `pro_top` at **50.437°** and `pro_coherent` at 46.848°, against 45.165° — while placing only about **3.8%** of their displacement along `d` (0.931 of 24.248 for `pro_top`; 2.8% at 0.638 of 22.853 for `pro_coherent`; both *derived*). This is the geometric form of the content-injection finding: the score is a handle on the residual stream, and the mechanism sits in the other **96%** (*derived*), which the score does not measure. Its limit, stated here rather than deferred: the JSON records no behavioural outcome for these prefixes, so this locates where the movement is not, not what the rest of it does.

## A number REVISIONS quotes that this file does not carry

`REVISIONS_2026-09-05.md` §1 includes an `ablate d, k=2` row at roughly 1.83° / 0.95 / 0.95. **That row is not in `normalization_check.json`**, which carries two ablation variants — logistic and meandiff, both `k=1` — and no `k=2` entry. One command was run to check whether the value is committed elsewhere in the analysis artifacts:

```
grep -rl "1.83" data/analysis/*.json
```

It lists 12 files — `layer_profile_all64`, `compile_check`, `prefix_eval_arms_s3`, `layer_sweep_prefix`, `prefix_gallery_judge`, `season3_gcg_ablation`, `steering_ablation_measure`, `season3_prefix_scores`, `steering_ablation_blind_key`, `transfer_report`, `season3_k3_control`, `site_prefixes` — and `normalization_check.json` is **not** among them. The pattern is the bare substring `1.83`, which any decimal containing those digits matches, so the listing is not evidence that the row exists in those files; it is evidence only that this one command does not locate it. The row should therefore be quoted from `REVISIONS_2026-09-05.md` with attribution to REVISIONS, or not at all. It must not be cited to this file.

## What this buys

Two settled answers, both cheap to re-verify. The normalization objection is closed on the model's own architecture rather than argued around, and the intervention comparison is put in the unit the model reads, which is what turned a claimed mechanism into a measured size gap. Neither answer is behavioural.

## Limits

1. No `n`. The JSON carries means and maxima but no count of contexts or probes, so this page cannot say how many prompts each row averages over, and gives no dispersion beyond the max.
2. Five layers for `cos(d, 1)`, not all of them. Nothing here rules out a larger overlap at an unmeasured layer, though the five span 16–48 and none exceeds 2.55%.
3. No random-direction control. The surviving claim in REVISIONS — ablation is indistinguishable from ablating a random direction at matched scale — is sourced elsewhere.
4. `mean_on_d` equals `mean_delta_norm` exactly on the four `d`-parallel rows. That is a property of interventions built along `d`, not an independent measurement.
5. Two injection scales and two ablation variants only. No `k=2`, no multi-direction, no scale sweep between 0.5 and 1.0.
6. Angles are geometry. No generation, no judge, and no behavioural outcome appears in this artifact.

## Cross-links

`REVISIONS_2026-09-05.md` §1 (the withdrawn add/remove asymmetry) and §5; `steering_ablation.md` (carries the inline correction); `meandiff_ablation.md` (the second ablation variant above); `causal_layer_curve.md` (layer dependence of the causal effect); `season3_gcg_ablation.md` (the prefix side); `PROJECT_SPEC.md` §5 (what the score reads, and at which layer).
