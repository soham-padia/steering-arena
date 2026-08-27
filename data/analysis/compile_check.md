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

**A saturation story was proposed here and then falsified.** The reasoning was: the token
strings sit at a tiny dose with a large response, the injection sits at 32× the dose with a
smaller response, therefore the response must peak somewhere between and the injection is
applied past its useful range. That predicts `+0.5·d` should beat `+1·d`. It does not
(`steering_dose.md`):

| injection dose | on-`d` push | behaviour | p |
|---|---|---|---|
| `+0.5·d` | 15.03 | **+0.13** | 0.32 (n.s.) |
| `+1.0·d` | 30.07 | **+0.45** | 0.007 |

The injection's response is still **climbing** at α = 1.0. Half the dose gives less than a
third of the effect. The vector is not overdriven; if anything it is under-applied.

**So the two families are not on one curve.**

| | on-`d` dose | behaviour | behaviour per unit dose |
|---|---|---|---|
| `pro_top` prefix | 0.93 | +0.89 | **0.96** |
| `+1·d` injection | 30.07 | +0.50 | **0.017** |

**Per unit of on-`d` displacement, the token prefix is ≈58× more behaviourally effective
than the residual injection.** Within prefixes the dose predicts the response (r = +0.99);
within injections it also predicts it (0.13 → 0.45, increasing); but the two sit on
completely separate curves.

That promotes the mechanism confound from a caveat to the leading explanation. A prefix is
not a small injection: 33 extra tokens give the model attention patterns, positional
structure and its own computation over that context, none of which a residual nudge
provides. On this evidence the on-`d` component of a prefix is a **marker** of what the
prefix is doing rather than the thing doing it — which is why matching the injection's on-`d`
push does not match its effect, in either direction.

**It also offers an explanation for the asymmetry.** If the behaviour is not carried by the
`d`-component, there is no reason negating `d` should invert the behaviour — and it doesn't
(`−1·d` is inside the random band; the anti soup produces stance collapse rather than
cruelty). A direction that steers when added and is inert when subtracted is what you would
expect if the added vector is doing something other than moving the model along that
direction.

## What this does to the "vector-to-token compiler" framing

"Compiler" turns out to be the wrong word, and the reason is the finding.

> A token string optimised against `d` moves activations along `d` by ~3% of a conventional
> steering magnitude and produces ~2× the behaviour of the full-magnitude injection. Within
> each family the dose predicts the response; **across** families it does not, by a factor
> of ≈58. Matching a direction's on-`d` displacement does not reproduce its effect.

Three consequences.

**A prefix is not a compiled vector, it is a different instrument.** Both move the residual
along `d`, and that shared coordinate is what the leaderboard measures, but the prefix also
brings attention, position and its own computation. The on-`d` number is where the two
mechanisms happen to be comparable, not where either of them works.

**The compilation is one-sided.** Near-perfect dose-response for positive doses (r = +0.99,
n=4), none for negative (r = +0.39, n=3) — the same asymmetry as everywhere else in this
project, now as a curve rather than point estimates.

**And the practical claim has to be narrowed.** "Steering vectors are applied past their
useful range" was the interesting version and it is false here: the injection is still
improving at α = 1.0·‖R‖. What is true is narrower and still worth saying — *at every dose
tested, a token prefix delivered more behaviour per unit of on-`d` displacement than an
injection did, by roughly two orders of magnitude*.

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
