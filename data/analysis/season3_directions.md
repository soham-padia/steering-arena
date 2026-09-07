# Both Season 3 directions separate held-out pairs perfectly, and `approach` is out of both by construction

`python scripts/build_season3_directions.py` → `data/analysis/season3_directions.json`, plus
the two shipped files `data/directions/d_olmo3_s3_score1.npz` and `d_olmo3_s3_score2.npz`.
Login-node CPU, **zero NDIF calls and zero GPU**, $0: it reads two committed activation
caches under `/work/neu/p2026_0037_neu/steering-arena/cache/` —
`seedpair_acts_all64_695903.npz` (seed pairs) and `confound_acts_all64_695948.npz` (confound
texts), whose filenames carry the slurm job ids of the captures that produced them. The JSON
records no job id and no timestamp of its own.

`REVISIONS_2026-09-05.md` is authoritative on what results are now taken to mean; this page
is authoritative on what the artifact contains.

**I ran this.** Every number below is transcribed from `season3_directions.json` at the
precision stored. `n_pairs: 135`, `n_split: 40`, `hidden: 5120`, `model_id:
allenai/Olmo-3-1125-32B`, `d_version: olmo3_s3_banded` — one version string over both files,
which differ by `role`, not by version.

## Method, from the script's docstring

Per band layer: fit a logistic probe (`C=0.1`) on the train split at that layer;
orthogonalise `length`, `sentiment` and `approach` out **at that layer**; unit-normalise.
Then average the per-layer directions across the band, orthogonalise once more at the band's
middle layer, and unit-normalise. That is variant A of `banded_direction.py`.
`held_out_separation` and `margin` are means over 40 repeated splits (25% held out); the
shipped vectors are refit on all 135 pairs. `confounds_removed` is `["length", "sentiment",
"approach"]` on both.

## The two shipped directions

| role | band | aggregate | `ranks_the_board` | held-out separation | margin | `min_cos_dbar_with_member` |
|---|---|---|---|---|---|---|
| Score 1 | 19, 23, 27, 31 | `banded_mean` | **true** | **1.0** | 0.2365 | 0.7734 |
| Score 2 | 15, 23, 31, 39 | `per_layer_min` | false | **1.0** | 0.26028 | 0.6106 |

Every layer in both bands is `full_attention`. **Held-out separation is 1.0 on both** — every
held-out chosen response above its rejected counterpart, on all 40 splits, under both
aggregates, against a `docs/EXTRACTION.md` §7(a) gate of ≥ 0.70. The caveat belongs in this
paragraph: 1.0 is a ceiling, `season3_band_select.md` shows all five candidate bands reach
it, and a saturated statistic ranks nothing — which is why margin is reported beside it.

`min_cos_dbar_with_member` is the cosine between the shipped mean direction and its *weakest*
band member, i.e. how much of the band one vector can stand for. Score 1's tighter band is
represented better (0.7734 against 0.6106), which is what its role needs: it is the direction
that gets steered with.

| direction | `cos(d, approach)` | `cos(d, sentiment)` | `cos(d, length)` |
|---|---|---|---|
| Score 1 | 0.0042 | 0.0041 | 0.0013 |
| Score 2 | **0.0072** | 0.0047 | 0.005 |

**The largest confound cosine anywhere in this artifact is 0.0072**, Score 2's residual
`approach`, against a §7(b) gate of < 0.20. Each value is a mean of `|cos|` over the band's
layers, so it describes the band on average and not its worst layer.

## Why `approach` coming out matters

`REVISIONS_2026-09-05.md` §4 records `approach` as "the only audited confound never
orthogonalised out". Seasons 1 and 2 removed `length` and `sentiment` and left it in, and
`layer_profile_all64.md` §4 found it is the one confound that does not decay with depth —
there is no layer at which it is near zero, so no choice of layer removes it by luck. Season
3 removes it by construction at every band layer and again on the mean. **This is the first
extraction in the project to do so.**

Both halves of the older finding travel together, and neither number is in this JSON:
REVISIONS §4 records that the *fitted* Season-1/2 `d` does not ride `approach` — project it
out and held-out separation, all-135 accuracy and the kind>cruel gap barely move — while the
*seed corpus* is confounded with it, `approach` alone separating the pairs well above chance.
Corpus problem, not direction problem, and it still is: orthogonalising `approach` out of `d`
does not de-confound the pairs. REVISIONS §7 item 4 stays open.

## Score 1's margin and Score 2's margin are different numbers, not an inconsistency

0.2365 and 0.26028 are not two measurements of one quantity. Score 1's is a **banded mean** of
`cos(unit(R_l), d_bar)` over 19/23/27/31 against one shared `d_bar`; Score 2's is a
**per-layer min** over 15/23/31/39, each layer against its own `d_l`. A mean and a min of four
cosines are different statistics, so the magnitudes do not compare.

`season3_band_select.json` supplies the cross-check and a third value: it scores Score 2's
band 15/23/31/39 under variant A, the banded mean, and gets margin **0.20971**. The same band
is 0.20971 as a mean and 0.26028 as a min, while the band Score 1 ships is 0.2365 as a mean in
both artifacts. Score 2's larger margin is the aggregate, not a better band — under the one
aggregate they share, Score 1's band is ahead.

## Per-axis coherence: 15 of 15 axes positive, on both directions

`per_axis_all_positive: true` on both. Each entry is that axis's mean chosen-minus-rejected
margin under its direction's own aggregate.

| axis | S1 | S2 | axis | S1 | S2 | axis | S1 | S2 |
|---|---|---|---|---|---|---|---|---|
| `accountability` | 0.22984 | 0.27035 | `feedback` | 0.23503 | 0.26449 | `ownership` | 0.27752 | 0.31524 |
| `boundaries` | 0.2519 | 0.26801 | `inclusion` | 0.22357 | 0.25037 | `privacy` | **0.20306** | **0.22089** |
| `conflict_resolution` | 0.28757 | 0.32699 | `integrity` | 0.23204 | 0.25637 | `respect` | 0.23592 | 0.24151 |
| `empathy` | **0.29551** | **0.33151** | `leadership` | 0.26504 | 0.2897 | `safety` | 0.26412 | 0.2829 |
| `fairness` | 0.26045 | 0.28145 | `learning` | 0.23585 | 0.25832 | `trust` | 0.23534 | 0.26884 |

Score 1 runs `privacy` 0.20306 to `empathy` 0.29551 (`per_axis_min: 0.20306`); Score 2 runs
`privacy` 0.22089 to `empathy` 0.33151 (`per_axis_min: 0.22089`). Same weakest axis, same
strongest axis, close orderings throughout: the "pro-human" family is coherent under both
aggregates, which is §7(c)'s gate (> 0).

## The `layer` field in the `.npz` is a DB-schema workaround

Both files store `layer: 27`. The seasons table's `layer` column is `int not null` and sits
inside `unique(model_id, layer, d_version)`, so a band cannot live there. 27 is Score 1's
upper-middle band layer — the layer its `d_bar` was orthogonalised at — and is
**representative only**; `band` is what the scorer reads, and
`scripts/check_season_matches_d.py` asserts both. Do not read 27 as either direction's
scoring layer; Score 2's band does not even contain it.

## What this buys

A materially stronger extraction than Season 2's on every axis measurable from cache.
`approach` is out by construction rather than by layer luck, closing the longest-standing item
on the confound list; the largest residual confound cosine anywhere is 0.0072; separation is
1.0 and all 15 axes are positive on both directions, so the label is coherent across the seed
family rather than carried by three or four axes. And the object being fit is now a band, not
a single layer — the structural answer to `layer_sweep_prefix.md`'s finding that the Season 2
winner's alignment existed at the layer it was optimised against and essentially nowhere else.

## Limits

1. **Decodability is not causal use.** The script's own closing line says it: these are
   decodability numbers, and nothing here shows `d` causes anything. The full causal gate
   `docs/EXTRACTION.md` §7(d) requires was **descoped** for Season 3 to a 10-prompt × 3-arm
   spot check injecting at layer 27 (`season3_causal_spotcheck.md`), with no random control.
2. **Per-axis margins are 9 pairs each in 5120 dimensions with no label-shuffled null** —
   REVISIONS §7 item 5 lists that null as open. Read the axis ordering as suggestive.
3. **`empathy` being strongest here must not be read as fixing REVISIONS §4**, which found
   `empathy` and `boundaries` are the *weakest* correlates of the Season-1/2 `d`. That
   measured a cosine between each axis's mean-difference direction and a single-layer `d`;
   this measures a score margin under a banded aggregate, on a different direction. Different
   quantities, not comparable, and the older finding is untouched.
4. **Score 2 is a per-layer min and so is not steerable at all** — there is no single vector
   to add. That is exactly why variant A was forced on Score 1: a causal steering check is a
   shipping gate for whatever ranks the board, so that direction has to be one vector.
5. **Score 2's confound cosines are measured against its band's mean `d_bar`**, not against
   the per-layer `d_l` the min aggregate reads. Each `d_l` is orthogonalised at its own layer
   during the fit, but 0.0072 / 0.0047 / 0.005 describe the summary vector.
6. **The seed corpus is unchanged** — the same 135 pairs, with the same confound in them.

## Cross-links

`season3_band_select.md` (how the five bands were compared, and the third margin for Score 2's
band) · `season3_causal_spotcheck.md` (the descoped causal gate) ·
`season3_gcg_baseline.json` (the per-probe baseline the board subtracts) ·
`REVISIONS_2026-09-05.md` §4 (what `d` is and is not; `approach` as the unremoved confound) and
§6 (a banded objective would have ranked the board differently) · `layer_profile_all64.md`
(`cos(d, approach)` never reaches zero at any depth) · `banded_direction.json` (variants A/B/C
and the weighting schemes) · `docs/EXTRACTION.md` §7 (the four gates) ·
`layer_sweep_prefix.md` (the single-layer spike this band answers).
