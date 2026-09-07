# Captions for the Part A′ figure

Three variants, one layout function (`scripts/plot_prefix_eval_s3.py`):

| file | size | use |
|---|---|---|
| `prefix_eval_s3_doc.png` | 6.5in | a Letter page at 1in margins — insert at 100%, labels render at the size they were set |
| `prefix_eval_s3.png` | 11in | a slide on a light background |
| `prefix_eval_s3_dark.png` | 10 × 5.625in | full-bleed on a 16:9 slide with a black master (`--dark`) |

Do not scale one up to fill another's role; regenerate. The dark variant is a *selected*
palette re-validated against `#000000`, not an inversion — orange moves from `#eb6834` to
`#e66633` because the original sits one thousandth over the dark lightness ceiling.

Keep the caption OUT of the image, for the same reason as `mechanism_caption.md`: at page
width, baked-in text lands around 5pt, and a real caption is selectable, searchable and
editable without regenerating the figure.

---

## Use this one (78 words)

> **Season 3's eight prefixes, scored by the two banded metrics (A1, A2) against the blind
> behavioural shift they produce.** Points are joined in score order, so a metric that
> ranks behaviour draws a rising line. Score 2 inverts two of 28 pairs and Score 1 four,
> including its own top pair. A length-matched random prefix sits at the origin on both.
> The two `FINAL` arms are the last strings their runs reached: on the pro side more score
> buys more behaviour, on the anti side it does not. B removes pairs where either
> continuation loops; C shows what the prefix puts back into the text.

## If space is tight (43 words)

> **Eight prefixes, by banded score (x) against blind behavioural shift (y), joined in
> score order.** Score 2 ranks behaviour better than Score 1 (2 versus 4 inverted pairs of
> 28). A random prefix does nothing. More score buys more behaviour on the pro side only.

---

## One extra line, only if a reviewer would otherwise be misled

Add after either caption:

> ρ is eight points, six of them optimised against these very metrics, so it measures
> ordering among these arms and not the metric's validity in general.

That caveat is the one a reviewer reaches for first and it is **not** written inside the
figure. Everything else — what the arrow in B means, what the hatch in C means, the base
loop rate — is labelled in the panels. Do not repeat those.

---

## Why panel A is split in two rather than overlaid

Both objectives are cosine shifts in the same LIVE units, so one shared x-axis would be
legitimate arithmetic. It was still wrong to draw: each arm would appear twice, and sixteen
points plus sixteen labels collide in the region that carries the result. Two panels
sharing a y-axis cost one axis of ink and make the disagreement the thing you see first.

## Why `score2_anti` and `score1_anti` are annotated rather than separated

On Score 2 they differ by 2.3e-4, far inside the run's own noise, so the marks genuinely
coincide and jitter would be a lie. The dotted callout names the coincidence and what
follows from it: a Score-2 anti board is topped just as well by a Score-1 anti search.
`score2_anti_final` at −0.163 is clear of both and needs no such treatment.

## Why hue never means two things

Hue is the arm family in every panel — blue for a GCG pro arm, aqua for the hand-written
prefix, grey for the null control, orange for an anti arm. The two *objectives* in row A
and the two *measures* in panel C are separated by line style, marker fill and hatch
instead, which keeps the categorical set at three chromatic slots. Marker shape is a second
independent identity channel, which is what lets each family carry three arms without a
fourth hue: `o`/`s`/`P` for pro, `v`/`^`/`<` for anti. Aqua sits at contrast 2.74 on the
light surface, under 3:1, so the relief rule applies and every family also carries a direct
label — identity is never colour alone.
