# What scale is the board score on?

The leaderboard score is a **cosine shift**, so `pro_top`'s +0.108 has no intuitive size
until you know what the cosine was to begin with. This measures that, and puts the vector
injection on the same axis.

Data: run of `scripts/cosine_scale.py`, 50 eval prompts, OLMo-3-32B layer 24, α = 30.07.

## Result

| condition | cos(R, d) | shift vs base | behavioural Δ |
|---|---|---|---|
| base (no prefix) | **0.0067** | — | — |
| `pro_top` prefix | 0.0422 | **+0.0355** | **+0.89** |
| `pro_coherent` prefix | 0.0284 | +0.0217 | +0.60 |
| `anti_top` prefix | −0.0444 | −0.0512 | −0.62 |
| `+0.5·d` injection | 0.4567 | **+0.4500** | +0.13 |
| `+1·d` injection | 0.7135 | **+0.7068** | +0.50 |

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
   `steering_random_control.md` and `steering_dose.md`.
