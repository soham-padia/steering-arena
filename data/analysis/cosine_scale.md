# What scale is the board score on?

The leaderboard score is a **cosine shift**, so `pro_top`'s +0.108 has no intuitive size
until you know what the cosine was to begin with. This measures that, and puts the vector
injection on the same axis.

Data: run of `scripts/cosine_scale.py`, 50 eval prompts, OLMo-3-32B layer 24, α = 30.07.

**Provenance note (added 2026-08-27).** This document was created 2026-08-27, one day *after*
the 2026-08-26 withdrawal of the `anti_top` behavioural arm in `prefix_eval.md`, and carried
the withdrawn number in its table anyway. Corrections below are dated 2026-08-27.

## Result

| condition | cos(R, d) | shift vs base | behavioural Δ |
|---|---|---|---|
| base (no prefix) | **0.0067** | — | — |
| `pro_top` prefix | 0.0422 | **+0.0355** | **+0.89** |
| `pro_coherent` prefix | 0.0284 | +0.0217 | +0.60 |
| `anti_top` prefix | −0.0444 | −0.0512 | ~~−0.62~~ **WITHDRAWN 2026-08-27** |
| `+0.5·d` injection | 0.4567 | **+0.4500** | +0.13 |
| `+1·d` injection | 0.7135 | **+0.7068** | +0.50 |

**Correction 2026-08-27: the `anti_top` behavioural Δ is withdrawn.** `prefix_eval.md`
retracted that arm as unmeasurable (degenerate text, so "which is kinder" was ill-posed). The
−0.62 is left visible and is not used in anything below. Its **cosine** figures (−0.0444,
−0.0512) are measurements and stand.

**Correction 2026-08-27: the two injection rows are ALGEBRA, not measurement.** They are
computed as `base + α·d̂` with no model call, and the readout layer is the same layer the
vector was added to. So `cos = 0.7135` is **definitional**, not a consequence of anything the
model did. Caveat 3 below already said this; the table did not, and the headline was built on
it anyway.

## Reading

**The model's neutral state is essentially orthogonal to `d`** (cos = 0.0067). Everything
the leaderboard measures happens in a very narrow band just above zero.

**The token strings move the metric by a rounding error.** `pro_top` takes the cosine from
0.0067 to 0.0422 — a 6× *relative* increase, and +0.0355 in absolute terms.

**The injection moves it ~20× further and does less.**

| | metric shift | behaviour | behaviour per unit of metric |
|---|---|---|---|
| `pro_top` prefix | +0.0355 | **+0.89** | **25.1** |
| `+1·d` injection | +0.7068 | +0.50 | 0.71 |
| ratio | **19.9×** | 0.56× | **35×** |

Even the half-dose injection moves the metric **12.7×** more than `pro_top` while producing
+0.13 against `pro_top`'s +0.89.

**So on the metric's own axis, the ranking inverts.** The token string sits near the bottom
of the cosine scale and at the top of the behavioural one; the injection sits at the top of
the cosine scale and below it behaviourally. Per unit of cosine shift the prefix delivers
≈35× more behaviour.

**Correction 2026-08-27: the per-unit figure is scoped, and the comparison it rests on is the
invalid one.** Two things, in order of severity.

*First, the comparison runs ACROSS families.* "Behaviour per unit cosine shift" puts a prefix
and an injection on one axis, which is exactly the comparison `compile_check.md` establishes
is not valid: the two families sit on separate curves roughly 60-fold apart, and "across
families the dose does not predict the response" is that document's own finding. The
instrument varies simultaneously with the cosine shift. So the per-unit figure is a
**description of the gap between two instruments**, not a property of the metric.

*Second, "≈35×" is a single-estimator point value.* Recomputed under all three baseline
estimators the ratio is **36 to 39× per unit cosine** (35.8 floating, 36.8 within-experiment,
39.1 17-arm shared). Report the range, not the bare 35×. It is stable across estimators, which
is the useful part.

What survives without any of this: the **19.9× ratio of cosine shifts** verifies exactly and
needs no judge, and the ordering inversion is 6/6 across judges and estimators. "The top of the
metric is not the top of the behaviour" stands on those two alone.

**The practical consequence: a small score is not evidence of a small effect.** On this
evidence the metric's high end is where the behaviour *is not*. That is the same Goodhart
pattern as the leaderboard itself (`prefix_eval.md`), with the steering vector in the role
of the entrant that games the score — it maximises the measured quantity far better than any
human-written or crowd-optimised string, and behaves worse than the best of them.

## Caveats

1. **This is not the board score.** The board score averages a cosine shift over the 16
   committed probes; this measures the 50 eval prompts, for comparability with the
   behavioural work. `pro_top` scores +0.108 on probes and +0.0355 here. Related quantities,
   different measurement sets; the *ordering* is preserved
   (`pro_top` > `pro_coherent` > 0 > `anti_top`).
2. **Last-token residual at one layer.** Same limit as `compile_check.md`.
3. The injection rows are computed exactly (`base + α·d̂`), not generated — they are what the
   injection does to the residual by construction, with no model call needed.
4. Behavioural Δ values are the mean across judges, carried over from `prefix_eval.md`,
   `steering_random_control.md` and `steering_dose.md`. The `anti_top` value among them is
   withdrawn as of 2026-08-27; see the correction above.
5. **(added 2026-08-27) The "high end" probed here is far outside anything the board has seen,
   and by how much is UNCHECKABLE from this repo.** The injection's +0.7068 cosine shift is
   ~20× `pro_top`'s. Any statement of the form "20× outside anything any of the 617
   submissions reached" cannot be verified here: there is no per-submission export in the repo
   (the submissions live in Supabase) and this document measured **3** of them. The **19.9×**
   ratio itself verifies and stands, but it is a ratio against `pro_top`, not against the
   submission distribution. Scope it to the three arms measured, or mark it unverified.
   Within the injection family more cosine does give more behaviour, and within the submission
   range more score does give more behaviour; the inversion lives only in the
   between-instrument comparison flagged above.
