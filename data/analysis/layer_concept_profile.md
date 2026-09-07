# Accuracy cannot rank these layers and margin can

`python scripts/layer_concept_profile.py --n-split 20 --n-shuffle 20` →
`data/analysis/layer_concept_profile.json`. Five layers (16, 24, 32, 40, 48), `n_split = 20` random
splits and `n_shuffle = 20` label-shuffled nulls each, off cached activations: **zero NDIF calls, zero GPU, $0**.

> **I did not run this.** This page is reconstructed from `layer_concept_profile.json` plus the
> docstring of `scripts/layer_concept_profile.py`. I did not execute the script, watch the job, or
> see any output that is not in the committed JSON. Every number below is transcribed from that
> file; where the JSON does not carry a quantity, this page says so rather than sourcing it
> elsewhere.

The run reads only cached activations and costs zero NDIF calls, so anyone who wants a verified
version can produce one with the command above.

`REVISIONS_2026-09-05.md` is authoritative on what this result is now taken to mean; this page is
authoritative on what the artifact contains.

## The profile: accuracy is flat, margin is not

| layer | acc real | acc null | margin real | margin null | excess, *derived* = real − null | Cohen's d real | Cohen's d null |
|---|---|---|---|---|---|---|---|
| 16 | 1.000 ± **0.0000** | 0.5529 ± 0.1602 | 0.2353 ± 0.0078 | 0.0035 ± 0.0139 | +0.2318 | 2.727 ± 0.2331 | 0.1036 ± 0.4083 |
| 24 | 1.000 ± **0.0000** | 0.5294 ± 0.1581 | **0.2726 ± 0.0056** | 0.0020 ± 0.0131 | **+0.2706** | **3.3777 ± 0.3296** | 0.0607 ± 0.3968 |
| 32 | 1.000 ± **0.0000** | 0.5059 ± 0.1443 | 0.2464 ± 0.0058 | 0.0011 ± 0.0115 | +0.2453 | 3.2191 ± 0.3748 | 0.0392 ± 0.3666 |
| 40 | 1.000 ± **0.0000** | 0.5103 ± 0.1399 | 0.2330 ± 0.0056 | 0.0009 ± 0.0108 | +0.2321 | 3.1464 ± 0.3748 | 0.0303 ± 0.3613 |
| 48 | 1.000 ± **0.0000** | 0.4971 ± 0.1470 | 0.2044 ± 0.0050 | 0.0010 ± 0.0099 | +0.2034 | 2.9745 ± 0.3389 | 0.0356 ± 0.3628 |

Real accuracy is 1.000 at every layer with a standard deviation of exactly **0.0000** over all 20 splits, which is the
proof that accuracy is *uninformative* here rather than merely uninteresting: no split at any depth misclassified a
held-out item, so the statistic has no variance left to rank with. Shuffled labels give 0.4971 to 0.5529, a coin flip
at every layer, so the task is not trivially easy either.

Margin does rank them, and every layer sits enormously above its own null: excess of +0.20 to +0.27 against
null margins of 0.0009 to 0.0035, two orders of magnitude clear, so the concept is linearly present at all
five depths rather than at one privileged depth. L24 is the genuine peak, margin 0.2726 and the highest
Cohen's d at 3.3777, and past L24 the decay is monotonic on both — 0.2726 → 0.2464 → 0.2330 → 0.2044 and
3.3777 → 3.2191 → 3.1464 → 2.9745. The null shuffles labels only, so it shows the separation is not an
artifact of fitting arbitrary labels, not what the labels mean.

## What this says about the L24 choice

| criterion | what it does across these five layers |
|---|---|
| held-out accuracy, strict `>` | ties at 1.000 five ways; keeps whichever layer came first |
| margin | ranks all five; peak L24 (0.2726) |
| Cohen's d | ranks all five; peak L24 (3.3777) |

`scripts/extract_direction.py` selects on separation with a strict `>`, so under a five-way tie at 1.000 it
did not pick L24 on the evidence — it kept the first layer it saw. Margin and Cohen's d both rank the layers
and both put L24 first, so the shipped layer is retroactively supported on a better criterion than the one
that chose it, though this is post-hoc support for a choice already made and the swept grid contains it.

## What this buys

A depth profile with a null attached, on statistics that still vary after accuracy has saturated. It establishes
that the concept is linearly decodable from L16 to L48 and best at L24. It does not establish that any layer
*uses* the concept: decodability is not causal use, and a feature can be linearly present at a depth that never
drives behaviour. The project has the causal version at L24 only, where `causal_layer_curve.md` finds it null
for a size reason, so the one layer measured both ways is the one where the causal test came back empty.

## Limits

1. Decodability, not causation. No intervention is run here.
2. 20 splits of a small validation set; the sds are resamples of the same items.
3. Five layers of 64, all `sliding_attention`, none of them a layer Season 3 scores on.
4. The seed-corpus confound is untouched; a label-shuffle null says nothing about it.
5. The JSON carries acc, margin and Cohen's d only — no per-split values, seeds, or timing.

## Cross-links

`REVISIONS_2026-09-05.md` §5 is authoritative on the interpretation. `direction_null.md` is the label-shuffle null
this depends on; `layer_profile_all64.md` lifts the method from five layers to all 64. `causal_layer_curve.md` holds
the causal counterpart at L24, and `season3_band_select.md` records the same separation saturation one level up, at
band level. Figure: `figures/layer_concept.png`.
