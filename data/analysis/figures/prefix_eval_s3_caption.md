# Caption for `prefix_eval_s3_doc.png`

Insert the PNG at 100%. It is built at exactly 6.5in, the usable width of a Letter
page with 1in margins, so every label renders at the size it was set. Then paste one
of the captions below as normal 10pt document text.

Use `prefix_eval_s3.png` (11in) for a slide instead — same panels, larger marks and a
two-line header. Do not scale the 6.5in version up to fill a slide; regenerate.

Keep the caption OUT of the image, for the same reason as `mechanism_caption.md`: at
page width, baked-in text lands around 5pt, and a real caption is selectable,
searchable, and editable without regenerating the figure.

---

## Use this one (74 words)

> **Season 3's six prefixes, scored by the two banded metrics (A1, A2) against the
> blind behavioural shift they produce.** Points are joined in score order, so a
> metric that ranks behaviour correctly draws a rising line: Score 2 has no
> inversions, Score 1 inverts its own top pair. A length-matched random prefix sits at
> the origin on both. B removes pairs where either continuation loops — `score2_anti`
> walks to 0.500. C shows the cost: the prefixes inject their own vocabulary
> downstream.

## If space is tight (41 words)

> **The six prefixes, by banded score (x) against blind behavioural shift (y), joined
> in score order.** Score 2 ranks all six correctly; Score 1 inverts its top pair. A
> random prefix does nothing. B: the loop control erases `score2_anti`.

---

## One extra line, only if a reviewer would otherwise be misled

Add after either caption:

> ρ is six points, four of them optimised against these metrics, so it measures
> ordering among these arms and not the metric's validity in general.

That caveat is the one a reviewer will reach for first and it is **not** written
inside the figure. Everything else — what the arrow in B means, what the hatch in C
means, the base loop rate — is labelled in the panels. Do not repeat those.

---

## Why panel A is split in two rather than overlaid

Both objectives are cosine shifts in the same LIVE units, so one shared x-axis would
be legitimate arithmetic. It was still wrong to draw: each arm would appear twice, and
the twelve points plus twelve labels collide in the region that carries the whole
result — the top pair, where the two metrics disagree. Two panels sharing a y-axis
cost one axis of ink and make the disagreement the thing you see first.

## Why the two anti arms are annotated rather than separated

On Score 2 they differ by 2.3e-4, far inside the run's own noise, so the marks
genuinely coincide and no jitter would be honest. The dotted callout names the
coincidence and what follows from it, which is the finding: a Score-2 anti board is
topped just as well by a Score-1 anti search.
