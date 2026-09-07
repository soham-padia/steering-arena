# Choosing the Season 3 band: separation cannot decide it, and the winner loses on margin

`python scripts/season3_band_select.py` → `data/analysis/season3_band_select.json`. Reads
the cached all-64-layer activations, so **zero NDIF calls, zero GPU, $0**. Variant A only —
the banded mean, which is the one steerable variant of the three in `banded_direction.md`.

**I ran this.** `REVISIONS_2026-09-05.md` is authoritative on what results are now taken to
mean; this page is authoritative on what the artifact contains.

135 contrastive pairs, 40 repeated train/validation splits.

## The five candidates

| candidate | band | layer types | separation | margin | min cos(d̄, member) | max confound cos |
|---|---|---|---|---|---|---|
| `single_L24_seas2` | 24 | **sliding_attention** | 1.000 | **0.27191** | **0.9998** | 0.0001 |
| **`peak_19_23_27_31`** | 19, 23, 27, 31 | full_attention | 1.000 | **0.23650** | **0.77340** | 0.0042 |
| `union_15_19_23_27_31_39` | 15, 19, 23, 27, 31, 39 | full_attention | 1.000 | 0.21636 | 0.60240 | 0.0066 |
| `spread_15_23_31_39` | 15, 23, 31, 39 | full_attention | 1.000 | 0.20971 | 0.61060 | 0.0072 |
| `wide_11_23_35_47` | 11, 23, 35, 47 | full_attention | 1.000 | 0.18013 | 0.53920 | 0.0096 |

`peak_19_23_27_31` ships as Score 1's band. `spread_15_23_31_39` ships as Score 2's, scored
under a different aggregate — see `season3_directions.md` for why its margin appears as a
different number there.

## Separation saturates, so it cannot be the criterion

All five candidates separate held-out pairs at **1.000**. Five structurally different bands,
one spanning 36 layers and one a single layer, and the metric that the extraction pipeline
selects on cannot tell them apart at all.

This is the same saturation `layer_concept_profile.json` records per layer, reproduced at
band level, and it has a direct consequence for `scripts/extract_direction.py`: that script
selects on separation with a strict `>`, so under a tie it silently keeps whichever
candidate it saw first. Margin and cross-layer coherence — `min_cos(d̄, member)`, the
smallest cosine between the averaged direction and any band member's own probe — are the
two quantities that carry information here, and they rank the four multi-layer candidates
**identically**: peak, then union, then spread, then wide on margin; peak, spread, union,
wide on min-cos. Close enough that the choice between multi-layer bands is not a judgement
call between competing metrics.

## The tension, stated where the choice is made

**The single-layer Season-2 baseline wins on every number in the table.** It has the
highest margin of all five at **0.27191**, the highest coherence at 0.9998 — trivially, it
is one layer — and the lowest confound cosines, at 0.0000 for `approach`, 0.0001 for
sentiment and 0.0001 for length. The band that shipped has a **13% lower margin** and
confound cosines an order of magnitude larger.

So the band was chosen **against** the best numbers in its own selection artifact, and it
should be read that way rather than as the band winning. The reason is gameability, and it
is not in this file: `layer_sweep_prefix.md` measured the Season 2 winning prefix and found
it satisfies layer 24 and essentially nowhere else — CV 1.73 across depth, with layer 24 at
4.69× the next-best layer. `REVISIONS_2026-09-05.md` §6 records the conclusion. A metric a
single string can satisfy at one depth is a metric that rewards depth-specific overfitting,
and the whole point of Season 3 was to stop paying for that. Trading 13% of margin for a
four-layer conjunction is a deliberate trade against a measured failure mode, and Part A
then confirmed the band is still beatable (+0.16395 LIVE, `season3_prefix_scores.md`) — so
the trade did not buy unbeatability, it bought a harder target.

The one number in this file that supports the band on its own terms is the layer type. Every
multi-layer candidate is built entirely from **`full_attention`** layers, while Season 2's
layer 24 is **`sliding_attention`**. That is also why the older per-text activation cache
held none of the candidate layers and the all-64 capture had to be run first.

## What this buys

The band choice is empirical and auditable rather than asserted: five candidates, one
artifact, and the ranking reproducible from cached activations at no cost. It also produced
a reusable methodological result — held-out separation is saturated on this corpus and
therefore useless for ranking anything, so margin and coherence are the criteria to reach
for. And among the multi-layer options the shipped band is the best on **both** informative
criteria, so nothing was traded off between them.

## Limits

1. **Variant A only.** Every row is a banded mean, so this file says nothing about the
   per-layer-min variant that Score 2 actually uses. Score 2's band was chosen here on a
   banded-mean fit and then scored under a different aggregate.
2. **Five hand-picked candidates, not a search.** No band outside this set was tried, and
   the set was chosen to span a range rather than by any optimisation.
3. **Separation is saturated, so four of the six columns are uninformative** — separation
   for all five rows, and coherence for the single-layer row where it is 0.9998 by
   construction. The table is wider than its evidence.
4. **135 pairs, one corpus, one model.** The confound cosines are small everywhere, but the
   corpus itself is confounded by `approach` at 0.824 (`direction_purity.json`), which no
   band choice fixes.
5. The gameability argument that decided the choice is **not in this artifact**. It lives in
   `layer_sweep_prefix.md` and `REVISIONS` §6, so this page cannot be read as
   self-justifying.

## Cross-links

`season3_directions.md` (what shipped, and why Score 2's margin differs) ·
`banded_direction.md` (the three variants, and why only A is steerable) ·
`layer_sweep_prefix.md` and `REVISIONS_2026-09-05.md` §6 (the gameability finding that
decided it) · `layer_profile_all64.md` (the all-64 capture and the attention-type split) ·
`layer_concept_profile.md` (the same separation saturation, per layer) ·
`season3_prefix_scores.md` (the band was beaten at +0.16395).
