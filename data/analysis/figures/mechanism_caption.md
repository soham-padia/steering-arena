# Caption for `mechanism_doc.png`

Paste the block below directly under the figure in the Google Doc, as normal
document text at 10pt. It is deliberately NOT baked into the PNG: at page width
anything under about 7pt becomes unreadable, and a caption in real text is also
selectable, searchable and editable without regenerating the figure.

The figure is built at exactly 6.5in wide, the usable width of a Letter page with
1in margins, so insert it at 100% and every label renders at the size it was set.

---

**Every intervention I ran, placed by how far it moved the model along `d` (x) against
how much it changed judged behaviour (y).** The two families sit on separate curves:
injections travel up to 30 units along `d` and barely move behaviour, while token
prefixes travel under 1 unit and move it more. The dashed line marks the headline
comparison, `pro_top` against `+1·d`.

**Colour intensity = how notable that combination is, and the two kinds of quadrant
define it oppositely.** In the *aligned* quadrants (green, plain) it is efficiency,
`|shift| ÷ (|displacement| + 1)`: a large shift from a small push is notable, the same
shift from a 30-unit push is not. In the *inverted* quadrants (violet, hatched) it is
contradiction, growing with **both** `|x|` and `|y|`: a small push that inverts could be
noise, but a hard push that still inverts is more likely a broken rig than a finding.
The grey horizontal band is the measured noise floor, the span of 8 norm-matched random
directions; colour is suppressed to zero inside it, so nothing is claimed where the
randoms say nothing happened.

**Reading it.** `pro_top` and `pro_coherent` sit high in the aligned quadrant at tiny
displacement. `anti_hostile` is the largest behavioural shift in the study and it too
comes from a push under 1 unit. The injections sit at the far edges, having spent 30
units of displacement to land in or beside the noise band. The ablation arms and the 8
random directions cluster at the origin: removing `d` entirely changes nothing.

**Caveats.** Deltas use a fixed baseline and are the mean of two blind judges. Hollow
squares (`±0.5·d`) are single-judge on a floating baseline, so they are not strictly
comparable to the rest. `anti_top` is excluded because its behaviour is withdrawn as
unmeasurable: its text degenerates, so "which is kinder" is ill-posed. No arm with a
measurable effect landed in either inverted quadrant; the only two points in one are the
score-zero controls, both inside the noise band.

---

## Shorter version, if the full caption is too long for the page

**Every intervention, placed by displacement along `d` (x) against judged behavioural
shift (y).** Injections travel up to 30 units along `d` and barely move behaviour;
token prefixes travel under 1 unit and move it more. Colour intensity marks how notable
each combination is: in the aligned quadrants that means efficiency (a big shift from a
small push), in the inverted quadrants it means contradiction (a hard push that still
produced the opposite outcome, which is likelier a broken rig than a finding). The grey
band is the measured noise floor from 8 random directions. Fixed-baseline deltas, mean
of two blind judges; hollow squares are single-judge; `anti_top` excluded as
unmeasurable.
