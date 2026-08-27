# Is the winning sequence a compiled steering vector?

Steering Arena publishes a direction and lets a crowd search token space for strings that
maximise projection onto it. One participant ran GCG against the objective directly. That
invites a reading of the winner as a **compilation of `d` into tokens** — and a mechanistic
question the behavioural work cannot answer: does prepending the string reproduce the
activation shift that injecting `α·d̂` produces, or does it reach the same behaviour by a
different route?

Data: `compile_check.json`. Script: `scripts/compile_check.py`.

## Method

For each of the 50 eval prompts, the layer-24 **last-token residual** with and without the
prefix. `Δ = R(prefix ⊕ p) − R(p)` is what the tokens do to activations. The injection adds
exactly `α·d̂` by construction, with `α = 1.0·‖R‖ = 30.07`, so the comparison is `Δ` against
`α·d̂`.

A true compilation would show `cos(Δ, d) ≈ 1.0`, `Δ∥ ≈ 30.07`, `‖Δ‖/α ≈ 1.0`.

## Result

| arm | board score | ‖Δ‖ | ‖Δ‖/α | cos(Δ, d) | **Δ∥ (along d)** | share on d |
|---|---|---|---|---|---|---|
| `pro_top` | +0.10769 | 24.25 | 0.81 | 0.0381 | **0.93** | 3.8% |
| `pro_coherent` | +0.04032 | 22.85 | 0.76 | 0.0253 | 0.59 | 2.8% |
| `anti_top` | −0.12912 | 27.54 | 0.92 | −0.0535 | −1.49 | 5.4% |
| `anti_hostile` | −0.02741 | 22.64 | 0.75 | −0.0382 | −0.87 | 3.9% |
| `control_junk` | +0.00714 | 17.37 | 0.58 | 0.0048 | 0.10 | 1.8% |
| *injection reference* | — | 30.07 | 1.00 | **1.000** | **30.07** | 100% |

## Reading

**The winning sequence is NOT a compiled steering vector.** `pro_top` displaces the
last-token residual by a magnitude comparable to the injection (81% of α) but delivers
**0.93 along `d`, roughly 3% of the injection's on-`d` push**. About 96% of its
displacement is orthogonal to the direction it was optimised against.

**And it produces MORE behaviour with that 3%.** Prefixing `pro_top` shifts judged kindness
by **+0.87 / +0.91** (two judges), while injecting the full `α·d̂` shifts it by
**+0.45 / +0.54**. One thirty-second of the on-`d` displacement, roughly double the effect.

**Displacement size is not the explanation.** `control_junk` moves the residual 17.37
(58% of α) and does nothing behaviourally (−0.15/−0.18, n.s.). `pro_top` moves it 24.25 and
produces the largest prefix effect measured. A 40% difference in norm, opposite behavioural
outcomes.

## Dose-response across all seven arms — and a correction

An earlier version of this document concluded from `pro_top` alone that "the behaviour comes
from somewhere in the other 96%". **That was wrong, and testing the dose against the response
across every arm is what caught it.**

| arm | on-`d` dose (Δ∥) | behavioural Δ (mean of judges) |
|---|---|---|
| `pro_top` | **+0.93** | **+0.89** |
| `pro_coherent` | +0.59 | +0.60 |
| `control_junk` | +0.10 | −0.16 |
| `control_text` | +0.05 | −0.14 |
| `anti_coherent` | −0.22 | −0.20 |
| `anti_hostile` | −0.87 | **−1.31** |
| `anti_top` | −1.49 | **−0.62** |

| predictor of behaviour | pearson r | spearman rho |
|---|---|---|
| board score | +0.745 (p=0.055) | +0.857 (p=0.014) |
| **on-`d` displacement Δ∥** | **+0.863 (p=0.012)** | **+0.929 (p=0.003)** |

**The small on-`d` component is not incidental — it is the best single predictor of
behaviour we have, better than the board score itself.** On the positive side the
dose-response is near-perfect: **r = +0.990 (p=0.010)** across four arms. On the negative
side there is none: r = +0.389 (p=0.75), and the ordering inverts (`anti_top` carries the
largest anti dose, −1.49, and produces *less* behaviour than `anti_hostile` at −0.87).

**What survives, restated.** The token strings deliver ≈3% of the injection's on-`d` push,
and within that small-dose regime the response tracks the dose almost perfectly. The
injection, with ~32× the dose, produces *less* behaviour than `pro_top` (+0.45/+0.54 vs
+0.87/+0.91). So the dose-response is not linear across regimes: it rises steeply where the
token strings live and has fallen by the time you reach α = 30.07 — consistent with the
coherence result, where a perturbation that size degrades text in any direction, random ones
included.

That is a saturation-and-damage story, not an orthogonality story. `pro_top` is closer to a
compilation than the previous version of this document claimed; what it is not is a
compilation *at the injection's magnitude*, and the magnitude turns out to be the injection's
problem rather than the string's.

**It also offers an explanation for the asymmetry.** If the behaviour is not carried by the
`d`-component, there is no reason negating `d` should invert the behaviour — and it doesn't
(`−1·d` is inside the random band; the anti soup produces stance collapse rather than
cruelty). A direction that steers when added and is inert when subtracted is what you would
expect if the added vector is doing something other than moving the model along that
direction.

## What this does to the "vector-to-token compiler" framing

The framing holds, with a sharper claim than either "it compiles" or "it doesn't":

> Optimising token sequences against a direction yields strings that move activations along
> it by ~3% of a conventional steering magnitude, and within that small-dose regime the
> behavioural response tracks the dose almost perfectly (r = +0.99). Injecting the full
> magnitude produces *less* behaviour than the compiled string, not more.

Two consequences worth stating separately.

**The compilation is real but tiny, and tiny is better.** A 0.93 push along `d` outperforms a
30.07 push along `d`. If that holds up, the practical reading is that steering vectors are
routinely applied at magnitudes past the useful range, and a token string that lands a small
precise push beats an injection that lands a large one.

**The compilation is one-sided.** The dose-response is near-perfect for positive doses and
absent for negative ones. Compilation succeeds in the sign that steers and fails in the sign
that does not — the same asymmetry as everywhere else in this project, now visible as a
dose-response curve rather than a pair of point estimates.

## Limits

1. **Last-token residual at one layer only.** The prefix changes every position; the
   injection was applied at every prompt position. This measures the readout the *score*
   uses, not the whole state. A prefix could carry its effect through earlier positions or
   through attention in ways this does not see.
2. **n = 7 arms.** The correlations rest on seven points, and the sign split on four and
   three. r = +0.99 on four points is suggestive, not established. The dose-response
   deserves arms placed deliberately along the dose axis rather than seven that happen to
   exist.
3. **One direction, one model, one layer, one α.**
3. `Δ` is measured on eval prompts; the board score is a cosine shift over the 16 committed
   probes. Related quantities, not identical ones.
4. Numerical note: numpy emitted spurious divide/overflow warnings from the matmul on this
   platform. Residuals were checked and are finite float32 with per-prompt values matching
   the aggregates (cos 0.016-0.054, Δ∥ 0.35-1.37).

## The obvious next experiment

Measure `Δ` at **every position and several layers**, not just the last-token readout at
L24, and ask whether the prefix's effect concentrates anywhere that the injection also
touches. If the two remain orthogonal everywhere, the separation is robust; if they
converge at some layer, that layer is where the behaviour actually lives.
