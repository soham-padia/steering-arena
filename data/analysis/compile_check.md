# Is the winning sequence a compiled steering vector?

Steering Arena publishes a direction and lets a crowd search token space for strings that
maximise projection onto it. One participant ran GCG against the objective directly. That
invites a reading of the winner as a **compilation of `d` into tokens** — and a mechanistic
question the behavioural work cannot answer: does prepending the string reproduce the
activation shift that injecting `α·d̂` produces, or does it reach the same behaviour by a
different route?

Data: `compile_check.json`. Script: `scripts/compile_check.py`.

**Provenance note (added 2026-08-27).** This document was created 2026-08-27, one day *after*
the 2026-08-26 withdrawal of the `anti_top` behavioural arm in `prefix_eval.md`, and used the
withdrawn number anyway. Every correction dated 2026-08-27 below follows from purging it. The
withdrawal did not propagate on its own; it had to be chased into each downstream document.

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
| ~~`anti_top`~~ | −1.49 | ~~**−0.62**~~ **WITHDRAWN 2026-08-27** |

**Correction 2026-08-27: the `anti_top` behavioural point is withdrawn.** `prefix_eval.md`
retracted that arm as unmeasurable: its text was degenerate, so "which of these is kinder" was
an ill-posed question. The −0.62 (exactly −0.625, the mean of −0.86 and −0.39) is left visible
above and is **excluded from every statistic below**. Its *activation* numbers survive
untouched: dose −1.49, board score −0.12912, `cos(Δ, d) = −0.0535` are measurements, not
judgements. Only the judged behaviour is withdrawn.

| predictor of behaviour | arm set | pearson r | spearman rho |
|---|---|---|---|
| board score | 7, as published | +0.745 (p=0.055) | +0.857 (p=0.014) |
| on-`d` displacement Δ∥ | 7, as published | +0.863 (p=0.012) | +0.929 (p=0.003) |
| board score | **6 surviving** | +0.836 (p=0.038) | +0.886 (p=0.019) |
| **on-`d` displacement Δ∥** | **6 surviving** | **+0.986 (p=0.0003)** | **+0.943 (p=0.005)** |

**Correction 2026-08-27: "better than the board score itself" is softened.** The original
sentence read: "*The small on-`d` component is not incidental — it is the best single
predictor of behaviour we have, better than the board score itself.*" That is not established
at this n. The two predictors are themselves correlated at **r = +0.953**. Williams' test for
dependent overlapping correlations gives p = 0.152 at n=7 and p = 0.0228 at n=6, but the
paired bootstrap Δr is +0.117 with a 95% lower bound of **−0.00**, touching zero. The honest
statement is that the two predictors are **not distinguishable at this n**. Both r values are
reported side by side above and neither is claimed over the other.

**Correction 2026-08-27: the sign asymmetry was an artifact of the withdrawn arm.** The
original reading was: "*On the positive side the dose-response is near-perfect: r = +0.990
(p=0.010) across four arms. On the negative side there is none: r = +0.389 (p=0.75), and the
ordering inverts.*" The inversion was `anti_top` and nothing else. Drop it and the relation is
**monotone across both signs**.

| arm set | dose r | p |
|---|---|---|
| all 7, with the withdrawn arm | +0.863 | 0.0124 |
| positive side only, n=4 (as published) | +0.990 | 0.0095 |
| negative side only, n=3, with the withdrawn arm (as published) | +0.389 | 0.7457 |
| **6 surviving arms, both signs** | **+0.986** | **0.0003** |

The two surviving negative arms order correctly: `anti_coherent` dose −0.22 / behaviour −0.20,
`anti_hostile` dose −0.87 / behaviour −1.31. More dose, more behaviour, on the negative side
too. n=2 there, so ordering is reported rather than a correlation, which would be ±1 by
construction.

**Correction 2026-08-27: read r = +0.986 as an ORDERING result, not a dose-response.** The
six arms differ in *text content*, not only in dose, so nothing here licenses a causal
reading. What it establishes is that on-`d` displacement **orders** the arms behaviourally:
it is a good index of where an arm will land. It does not establish that dose *causes* the
response. That is consistent with, not in tension with, the marker-not-mechanism reading
below.

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
than the residual injection.**

**Correction 2026-08-27: "58×" does not survive as a point estimate. The figure is roughly
60-fold (50 to 75).** 58 is the value under exactly one estimator (floating baseline,
judge-mean, `pro_top`), and that is the inflated one. Recomputed across **18 combinations**
(3 baseline estimators × 2 judges plus their mean × 2 prefix arms) the ratio ranges **50.5 to
75.0, median 61.3**. Two significant figures are not supported. Safest published wording:
**"more than an order of magnitude"**.

What survives intact is the robustness, which matters more than the digits:

| check | result |
|---|---|
| robust to the baseline correction | 58 floating, 60 fixed, 63 shared |
| reproduces from a different string | `pro_coherent` 60.5 vs `pro_top` 59.7 |

Different string, different dose (0.586 vs 0.931), different behavioural delta, same ratio. It
is **not a `pro_top` artifact**.

Within prefixes the dose orders the response (r = +0.986 over the six surviving arms, per the
correction above); within injections it also predicts it (0.13 → 0.45, increasing); but the
two sit on completely separate curves.

### The separation without any ratio (added 2026-08-27)

The ratio arithmetic compares *across* families, which is the comparison this document itself
shows is unsafe. Two statements make the same point without it, and both are stronger.

**Paired head-to-head.** `pro_top`'s continuation judged directly against `+1·d`'s
continuation, same 50 prompts, same rubric:

| judge | mean diff | median | W/L (non-tied) | Wilcoxon p |
|---|---|---|---|---|
| `deepseek-v4-pro` | +0.210 | 0.00 | 24 / 13 (n=37) | 0.155 |
| `claude-opus-5` | **+0.470** | **+0.75** | 27 / 13 (n=40) | **0.010** |

Both judges favour the prefix; one significantly. No normalisation, no cross-family curve.

**The cheapest statement of the separation, no ratio at all.** `+0.5·d` carries an on-`d` push
of **15.03**, which is **16.1×** `pro_top`'s 0.93, and produces a **dead tie**: 8 wins, 8
losses out of 50, p = 0.32. `pro_top` over the same 50 prompts is p < 0.001 under **both**
judges. A sixteenfold larger on-`d` displacement, and no measurable behaviour.

That promotes the mechanism confound from a caveat to the leading explanation. A prefix is
not a small injection: 33 extra tokens give the model attention patterns, positional
structure and its own computation over that context, none of which a residual nudge
provides. On this evidence the on-`d` component of a prefix is a **marker** of what the
prefix is doing rather than the thing doing it — which is why matching the injection's on-`d`
push does not match its effect, in either direction.

**It also offers an explanation for the asymmetry.** If the behaviour is not carried by the
`d`-component, there is no reason negating `d` should invert the behaviour — and it doesn't
(`−1·d` is inside the random band; the anti soup produces stance collapse rather than
cruelty). *Qualified 2026-08-27:* "`−1·d` is inside the random band" is
**estimator-dependent**. Under a fixed baseline it is inside the band under Claude and
*outside* it under DeepSeek. See `steering_random_control.md`.

A direction that steers when added and is inert when subtracted is what you would
expect if the added vector is doing something other than moving the model along that
direction.

## What this does to the "vector-to-token compiler" framing

"Compiler" turns out to be the wrong word, and the reason is the finding.

> A token string optimised against `d` moves activations along `d` by ~3% of a conventional
> steering magnitude and produces ~2× the behaviour of the full-magnitude injection. Within
> each family the dose orders the response; **across** families it does not, by ~~a factor of
> ≈58~~ **roughly 60-fold (50 to 75)** *(corrected 2026-08-27)*. Matching a direction's on-`d`
> displacement does not reproduce its effect.

Three consequences.

**A prefix is not a compiled vector, it is a different instrument.** Both move the residual
along `d`, and that shared coordinate is what the leaderboard measures, but the prefix also
brings attention, position and its own computation. The on-`d` number is where the two
mechanisms happen to be comparable, not where either of them works.

**~~The compilation is one-sided.~~ RETRACTED 2026-08-27.** The consequence as published read:
"*The compilation is one-sided. Near-perfect dose-response for positive doses (r = +0.99,
n=4), none for negative (r = +0.39, n=3) — the same asymmetry as everywhere else in this
project, now as a curve rather than point estimates.*" The entire negative-side null was
produced by the withdrawn `anti_top` arm. Over the six surviving arms the ordering is monotone
across **both** signs at r = +0.986 (p = 0.0003), and the two surviving negative arms order
correctly. **The compilation is not one-sided.** This document's headline consequence was
built on a number that had already been retracted the day before the document was written.

**And the practical claim has to be narrowed.** "Steering vectors are applied past their
useful range" was the interesting version and it is false here: the injection is still
improving at α = 1.0·‖R‖. What is true is narrower and still worth saying — *at every dose
tested, a token prefix delivered more behaviour per unit of on-`d` displacement than an
injection did, by ~~roughly two orders of magnitude~~ **roughly one order of magnitude***.

**Correction 2026-08-27: "roughly two orders of magnitude" is pulled back to one.**
log10(60) = 1.78, and even the most favourable slope-based estimate, 82, is below 100. One
order of magnitude, not two.

## Prior work (added 2026-08-27)

Searched after the fact, which is itself the finding: the two most relevant results were both
already in print.

**The literal "compiler" goal is provably impossible, and finding 1 is a clean empirical
instance of a proved theorem.** Mishra, Khashabi and Liu, *Steered LLM Activations are
Non-Surjective* (arXiv:2604.09839), prove that "activation steering pushes the residual stream
off the manifold of states reachable from discrete prompts. Almost surely, no prompt can
reproduce the same internal behavior induced by steering." Our `cos(Δ, d) = 0.0381` instantiates
that theorem on a model and a direction they did not test, with a quantification (3.1% of α at
the scored readout) the theorem does not supply. The consequence, stated plainly:
**compiling a steering vector into tokens is ruled out at the ACTIVATION level.** The goal has
to be reframed **behaviourally**: match the behaviour, not the activations. Nothing rules
*that* out, and the paired head-to-head above is direct evidence it is reachable.

**Prompting beating vector steering behaviourally is already published, at our magnitude.**
AxBench (arXiv:2501.17148) reports mean steering scores of Prompt **0.698** versus DiffMean
**0.297** on Gemma-2-2B, and Prompt **1.075** versus DiffMean **0.322** on Gemma-2-9B, a
2.35× and 3.34× prompting advantage, with the steering factor tuned **per concept to its own
optimum**. Our ~2× behaviour ratio therefore replicates a published result on a different model
family, a different direction-construction method and a different judge. Good for credibility,
weak for novelty. It also independently kills the "you under-applied the vector" objection at
the level of the field: AxBench tuned the coefficient to each concept's own optimum and
prompting still won.

**Novelty verdict.**

| element | verdict |
|---|---|
| A prefix is not a compiled vector | already established, and *proved* (Mishra et al.) |
| Prompting beats steering behaviourally | already established (AxBench), at a similar magnitude |
| Prefix and injection are mechanistically different instruments | known, but **unquantified** |
| **Behaviour per unit of on-`d` displacement, compared across the two instruments** | **appears new.** This is the contribution. |

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
4. `Δ` is measured on eval prompts; the board score is a cosine shift over the 16 committed
   probes. Related quantities, not identical ones.
5. **THE LARGEST OUTSTANDING THREAT (added 2026-08-27): there is no random-direction control
   for the token-optimisation arm.** Mody et al., arXiv:2607.25907, ran GCG-style token
   optimisation against a latent direction and report that "a placebo random direction is
   suppressed just as hard and shifts behavior just as far". This project has a norm-matched
   random-direction control for the **injection** arm (`steering_random_control.md`) and
   **none** for the search arm. `control_junk` and `control_text` are hand-written, not
   optimised, so they do not test this. The missing experiment: **run the same search against
   a random direction to a matched board score, then judge it under the same rubric.** Until
   that exists, "the behaviour comes from optimising against `d`" is **not established**
   against the rival explanation "optimising a token string against *any* direction produces
   this".
6. Numerical note: numpy emitted spurious divide/overflow warnings from the matmul on this
   platform. Residuals were checked and are finite float32 with per-prompt values matching
   the aggregates (cos 0.016-0.054, Δ∥ 0.35-1.37).

## The obvious next experiment

Measure `Δ` at **every position and several layers**, not just the last-token readout at
L24, and ask whether the prefix's effect concentrates anywhere that the injection also
touches. If the two remain orthogonal everywhere, the separation is robust; if they
converge at some layer, that layer is where the behaviour actually lives.
