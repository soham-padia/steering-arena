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

**So "maximises projection onto `d`" and "does what `d` does" are separable — and here they
are nearly orthogonal.** The leaderboard ranks the on-`d` component. The behaviour comes
from somewhere in the other 96%.

**Displacement size is not the explanation either.** `control_junk` moves the residual 17.37
(58% of α) and does nothing behaviourally (−0.15/−0.18, n.s.). `pro_top` moves it 24.25 and
produces the largest prefix effect measured. A 40% difference in norm, opposite behavioural
outcomes, and both with a negligible on-`d` component.

**It also offers an explanation for the asymmetry.** If the behaviour is not carried by the
`d`-component, there is no reason negating `d` should invert the behaviour — and it doesn't
(`−1·d` is inside the random band; the anti soup produces stance collapse rather than
cruelty). A direction that steers when added and is inert when subtracted is what you would
expect if the added vector is doing something other than moving the model along that
direction.

## What this does to the "vector-to-token compiler" framing

The framing survives but inverts. The interesting claim is not *a steering vector can be
compiled into tokens*; on this evidence it was not. The claim is stronger and more
uncomfortable:

> Optimising token sequences against a direction produces strings that **do not** reproduce
> that direction's activation shift, yet **do** produce the behaviour the direction was
> meant to encode, more strongly than injecting the direction itself.

That makes the metric a poor proxy for its own mechanism, and it makes crowd-optimised
token strings an object worth studying in their own right rather than a compilation artefact.

## Limits

1. **Last-token residual at one layer only.** The prefix changes every position; the
   injection was applied at every prompt position. This measures the readout the *score*
   uses, not the whole state. A prefix could carry its effect through earlier positions or
   through attention in ways this does not see.
2. **One direction, one model, one layer, one α.**
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
