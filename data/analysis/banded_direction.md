# A direction that spans layers: three meanings, and who would have won the board

Data: `data/analysis/banded_direction.json` and `data/analysis/banded_score_arms.json`.
Scripts: `scripts/banded_direction.py` and `scripts/banded_score_arms.py`.

```
python scripts/banded_direction.py                # band 32,40,48
python scripts/banded_direction.py --band all     # all five, for contrast
python scripts/banded_score_arms.py
```

**I did not run this.** This page is reconstructed from `banded_direction.json`,
`banded_score_arms.json` and the docstrings of `scripts/banded_direction.py` and
`scripts/banded_score_arms.py`. I did not execute either script, watch the job, or see any
output that is not in the committed JSON. Every number below is transcribed from those
files; where they do not carry a quantity, this page says so rather than sourcing it
elsewhere.

`REVISIONS_2026-09-05.md` is authoritative on what this result is now taken to mean; this
page is authoritative on what the artifacts contain.

**Verdict: a multi-layer objective is constructible on both bands, the degeneracy trap in
the concatenated variant was measured and did not fire, and under every banded variant the
readable hand-written instruction beats the GCG board winner.** This is the direct ancestor
of Season 3's two scores.

## The three variants are not the same object, and only one steers

From the `banded_direction.py` docstring. `d_olmo3_L24_logistic` is a single-layer readout;
a band is harder to satisfy at one depth, so harder to game. "Multi-layer direction" has
three non-equivalent readings:

| | construction | score | steerable |
|---|---|---|---|
| **A** banded mean | average the per-layer unit directions, renormalise; one vector in R^5120 | mean over the band of `cos(R_l, d_bar)` | **yes** |
| **B** concatenated probe | stack the band's residuals into `(N, |band|*5120)`, fit one probe | weighted sum of per-layer cosines | no |
| **C** per-layer probes, min | one `d_l` per layer | `min` over the band of `cos(R_l, d_l)` | no |

Every layer block is unit-normalised before concatenation in B, so norm inflation buys
nothing in any of the three — the same magnitude-proofing PROJECT_SPEC §5 gets from cosine.
C is adversarially strictest: a string must satisfy every layer and cannot trade a spike at
one depth against a deficit at another. That steerability column is exactly why Season 3
split the way it did: **Score 1 ranks the board, so it needs a causal gate and is variant A;
Score 2 is informational and never steered, so it can afford to be variant C.** Only A is
written to disk (`data/directions/d_olmo3_banded_<tag>.npz`).

## Fit quality on both bands

135 pairs, 101 train / 34 val, `split_seed = 0`, identical on both bands.

| variant | held-out separation, band {32,40,48} | held-out separation, band {16,24,32,40,48} |
|---|---|---|
| A banded mean | **1.0** | **1.0** |
| B concat probe | 0.9706 | 0.9706 |
| C per-layer min | **1.0** | **1.0** |

B's 0.9706 is one missed pair out of 34 val pairs on both bands (derived: 0.9706 × 34 =
33.0). A records `confounds_removed = ["length", "sentiment"]` on both bands; the JSON
records no confound list for B or C, so this page does not claim one. Variant A's cosines
with the native per-layer probes:

| band | L16 | L24 | L32 | L40 | L48 |
|---|---|---|---|---|---|
| {32,40,48} | — | — | 0.9392 | 0.9793 | 0.9484 |
| all five | 0.5311 | 0.8085 | 0.9136 | 0.9035 | 0.8573 |

The late band's mean sits within 0.94–0.98 of every member. Across all five, L16 falls to
0.5311 — the docstring attributes this to L16 being the outlier in pairwise cosine among the
natives ({32,40,48} 0.870 mean / 0.803 min; all five 0.557 / 0.202), figures the JSON does
not carry.

## The degeneracy trap in B, measured rather than assumed

B's specific failure mode: if one layer separates best, the probe can put its weight there
and the "multi-layer" objective quietly collapses back to a single-layer objective, gameable
exactly as before. The script measures the per-layer weight share against the uniform value.

| band | shares | max share | uniform | max / uniform (derived) | `degenerate` |
|---|---|---|---|---|---|
| {32,40,48} | L32 0.3741, L40 0.3465, L48 0.2794 | 0.3741 | 0.3333 | 1.122× | **false** |
| all five | L16 0.1860, L24 0.2374, L32 0.2156, L40 0.1999, L48 0.1611 | 0.2374 | 0.2000 | 1.187× | **false** |

**It did not degenerate, and that is the finding.** The heaviest layer carries at most 12%
(late) or 19% (all five) more than an equal share, so B stays a genuinely distributed
objective on both bands. The trap was checked, not waved away.

## Weighting is not a knob worth turning; band choice is

Four weight schemes on the late band, quoted in full so nobody has to trust the docstring's
rounding:

| scheme | weights (L32/L40/L48) | cos with natives | min-cos |
|---|---|---|---|
| uniform | 0.3333 / 0.3333 / 0.3333 | 0.9392, 0.9793, 0.9484 | 0.9392 |
| margin | 0.3615 / 0.3401 / 0.2985 | 0.9455, 0.9790, 0.9418 | 0.9418 |
| inv-var | 0.3053 / 0.3372 / 0.3575 | 0.9333, 0.9799, 0.9533 | 0.9333 |
| minimax | 0.4996 / 0.0004 / 0.5000 | 0.9492, 0.9522, 0.9492 | **0.9492** |

The four min-cos values span 0.9333 to 0.9492: an absolute spread of **0.0159**, i.e.
**1.70%** of the smallest (derived), which the docstring rounds to 1.6%. Band choice moves
the same quantity **0.4081** — uniform min-cos 0.9392 on the late band against 0.5311 across
all five — **25.7× the weighting spread** (derived), which is the order of magnitude the
docstring claims. On the wider band minimax lifts min-cos from 0.5311 to **0.7584** and
equalises all five members at exactly 0.7584, but it pays by nearly discarding two of them
(weights 0.4161 / 0.1720 / 0.0346 / 0.0295 / 0.3479), representing everyone mediocrely
instead of the late layers well; the same move on the late band zeroes L40 to 0.0004, the
member uniform fits *best* at 0.9793. The docstring adds that arm scores move under 0.0006
across schemes and that `pro_coherent`'s signal drops under minimax on the wide band —
neither JSON carries per-scheme arm scores, so those two remain docstring-only.

## The board arms, re-scored over a band

`pro_top` took Season 2 with **+0.107693473529894** against `pro_coherent`'s
**+0.0403244759161852**, a 2.67× margin (derived) on the single-layer L24 objective.

| arm | board L24, 16 probes | A banded mean {32,40,48} | C per-layer min {32,40,48} | A banded mean, all five | C per-layer min, all five |
|---|---|---|---|---|---|
| `pro_top` (GCG winner) | +0.107693473529894 | **−0.00308** | **−0.00979** | +0.00437 | −0.00633 |
| `pro_coherent` (hand-written) | +0.0403244759161852 | **+0.02167** | **+0.02148** | +0.01728 | +0.02327 |

Two caveats travel with this table. (a) The board column uses the 16 frozen probes and every
banded column uses the 50 eval prompts, which is what the layer sweep had cached, so the
**columns are not numerically comparable** — the only valid comparison is between arms
*within* a column, where `pro_coherent` wins all four banded columns and `pro_top` collapses
to roughly nothing or below zero. (b) `pro_top` was found by GCG optimising the single-layer
L24 objective, so its collapse says **this string** does not generalise across depth; it
does **not** say a banded objective is unbeatable, because at the time nobody had searched
against one. That open experiment is now closed: it became Season 3 Part A, and it
**succeeded** — a GCG search against a banded mean reached **+0.16395** LIVE
(`season3_prefix_scores.md`). The 2026-09-05 reading of this page has been answered, and the
implication that a band resists search should not be left standing. `banded_score_arms.json`
carries no column for variant B, so B is unscored on the arms.

## Arms that are missing, not dropped

`banded_score_arms.json` lists five arms whose residuals were not cached, named in
`missing_arms` rather than silently omitted: **`anti_coherent`, `anti_hostile`, `anti_top`,
`control_junk`, `control_text`**. Completing them costs **15** NDIF calls for the late band
alone and **25** for both bands (`ndif_calls_to_complete`) — consistent with 5 arms × 3 or 5
layers (derived). Until then the arm comparison rests on two arms, both pro.

## What this buys

A multi-layer objective is constructible: variant A separates held-out pairs perfectly on
both bands while staying within 0.94–0.98 of every late-band native, and it stays a
*direction*, so it can be injected and causally tested rather than only read out. The
degeneracy trap that would have made the concatenated variant a single-layer objective in
disguise was measured on both bands and did not fire. Weighting was tested and retired as a
decision, which moves the design question to band choice where the evidence is an order of
magnitude larger. And the arm re-score gave the first evidence that the board's ranking was
a property of layer 24 rather than of the strings. All four of those are the groundwork
Season 3's Score 1 and Score 2 were built on.

## Limits

1. **Two arms, both pro.** Five of seven gallery arms are uncached, including every anti arm
   and both controls, so nothing here bounds what a banded objective does to hostile or junk
   text.
2. **Columns are not commensurable.** 16 frozen probes against 50 eval prompts; no run
   scored the arms on the same prompt set under both objectives.
3. **Cosines only.** Every number is a geometric fit or a cosine against a direction. No
   generation, no judge, no behaviour is touched anywhere in either artifact.
4. **One split.** `split_seed = 0`, 101/34, no cross-validation and no repeated splits, so
   the 1.0 separations carry no error bar and B's single missed pair is one pair.
5. **The pairwise-cosine argument for the band lives in the docstring**, not the JSON, and
   `REVISIONS_2026-09-05.md` §4 has since recorded that most of the cross-layer cosine
   *shape* survives label shuffling. The band recommendation stands on the min-cos evidence
   above, not on that geometry.
6. **No confound audit for B or C.** Only A records `confounds_removed`.

## Cross-links

`REVISIONS_2026-09-05.md` §6 (what was confirmed rather than revised),
`season3_band_select.json` and `season3_directions.json` (the bands and directions Season 3
actually shipped), `season3_prefix_scores.md` (the +0.16395 that answers caveat b),
`season3_gcg_setup.md` (the search that produced it), `layer_sweep_prefix.md` (the L24-only
result that motivated banding), `direction_null.json` (the null this kind of fit needs).
Note that `season3_band_select`, `season3_directions` and `direction_null` exist as JSON
only; there is no `.md` for any of the three.

The run reads only cached activations and costs zero NDIF calls, so anyone who wants a
verified version can produce one with the commands above.
