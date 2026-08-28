# Advocate brief: the MECHANISM

Token prefixes versus residual-vector injection as instruments for steering behaviour.

Scope: `data/analysis/compile_check.{md,json}`, `steering_dose.{md,json}`, `cosine_scale.md`,
`steering_random_control.{md,json}`, `_falsifier/recompute_result.{md,json}`, plus the
per-prompt judge records those documents are built from.

This brief builds on the established corrections: the `anti_top` behavioural numbers are
withdrawn; the "compilation is one-sided" claim is dead; fixed-baseline deltas are preferred
over per-arm-baseline deltas; z-scores under a fixed baseline inflate mechanically, so
placement counts are used instead; on-`d` displacement is not claimed to beat the board score
as a predictor.

## Estimator conventions used here

Behavioural deltas are reported under three estimators, because the ratio claims should not
depend on which one you pick:

| label | baseline |
|---|---|
| `floating` | each arm against its own run's base ratings (as originally published) |
| `within-experiment` | per-prompt mean of the 7 prefix arms' bases (prefix arms) / the 10 steering arms' bases (steering arms) |
| `17-arm shared` | per-prompt mean over all 17 arms' bases, legitimate because `recompute_result.json` confirms the 50 base continuations are byte-identical between the two experiments (`identical_base_continuation: 50`) |

Everything in the tables below was recomputed from the committed per-prompt records using
`_falsifier/recompute.py`'s own loaders and its own Wilcoxon convention. Nothing new was
generated; no API was called.

---

# Ranked claims

## 1. The winning token string is not a compiled steering vector, and the number is 3.1%

**STRONGEST DEFENSIBLE VERSION.**
Prepending the leaderboard's top sequence displaces the layer-24 last-token residual by
**24.25**, which is **81%** of the injection's magnitude (`α = 30.07 = 1.0·‖R‖`), but only
**0.93 of that displacement lies along `d`**. That is `cos(Δ, d) = 0.0381` and
`Δ∥/α = 0.031`: **3.1% of the injection's on-`d` push, with 96% of the displacement
orthogonal to the direction the string was optimised against**. A string produced by running
GCG directly against a projection objective does not land near the state that adding the
vector produces. Optimising tokens against a direction does not compile that direction into
tokens.

**FALLBACK VERSION.** Even granting every measurement caveat, `cos(Δ, d) = 0.0381` for
`pro_top` and `0.0253` for `pro_coherent` are not near 1.0 and are not near each other's
noise floor. The qualitative conclusion (a token-space optimum against `d` is not an
approximation of `α·d̂`) does not depend on the exact fraction.

**WHAT WOULD BREAK IT.** Measure `cos(R_L24(t), d)` at the 33 *prefix* token positions
themselves, not just at the last prompt token. If those positions carry a large on-`d`
component, the prefix does deliver a large aggregate on-`d` signal into the context that
later attention can read, and the "3.1%" is an artifact of reading only one position. This is
the single most important open experiment in this scope and it is cheap.

**Source.** `data/analysis/compile_check.json` (`pro_top`: `delta_norm` 24.248,
`cos_with_d` 0.03809, `along_d` 0.9315, `along_over_alpha` 0.03098, `alpha` 30.070,
50 prompts, `allenai/Olmo-3-1125-32B`, layer 24).

**Why this is the claim to lead with.** It is arithmetic on a measured activation, with no
judge in the loop, and it directly answers the maintainer's stated long-term question. See
Prior work: a 2026 preprint *proves* this cannot be done in general.

---

## 2. Matched total displacement, 32x less on-`d` content, same or more behaviour

**STRONGEST DEFENSIBLE VERSION.**
Two interventions of comparable total displacement at the same readout (`‖Δ‖` 24.25 for the
prefix versus `α` 30.07 for the injection, a ratio of 0.81) differ by a factor of **32.3** in
how much of that displacement lies along `d`. The one with 32x *less* on-`d` content produces
**more** judged behaviour, under both judges and all three baseline estimators:

| estimator | judge | `pro_top` Δ | `+1·d` Δ | prefix advantage |
|---|---|---|---|---|
| floating | deepseek-v4-pro | +0.870 | +0.450 | +0.420 |
| floating | claude-opus-5 | +0.910 | +0.540 | +0.370 |
| within-experiment | deepseek-v4-pro | +0.556 | +0.355 | +0.201 |
| within-experiment | claude-opus-5 | +0.796 | +0.376 | +0.420 |
| 17-arm shared | deepseek-v4-pro | +0.561 | +0.351 | +0.210 |
| 17-arm shared | claude-opus-5 | +0.825 | +0.355 | +0.470 |

**6 of 6 point estimates favour the prefix.** The direct paired head-to-head on the same 50
prompts (the prefix's continuation rated against the injection's continuation, which is
exactly the 17-arm-baseline delta difference) gives:

| judge | mean diff | median | W/L (non-tied) | Wilcoxon p |
|---|---|---|---|---|
| deepseek-v4-pro | +0.210 | 0.00 | 24 / 13 (n=37) | 0.155 |
| claude-opus-5 | +0.470 | +0.75 | 27 / 13 (n=40) | 0.010 |

**FALLBACK VERSION (this is the one to publish if pressed).** The prefix is **not worse** than
the full-magnitude injection behaviourally, under any judge or estimator tried, while carrying
32x less on-`d` displacement. Even under the null hypothesis that the two produce *identical*
behaviour, the on-`d` displacement they deliver differs by 32x. The "more behaviour" part is
significant under one judge of two; the "at 1/32 the on-`d` dose" part involves no judge at
all.

**WHAT WOULD BREAK IT.** A third judge, or a larger prompt set, in which the injection's
judged delta exceeds the prefix's. The DeepSeek paired test at p = 0.155 shows this is live:
the *ordering* is 6/6 but the *magnitude* of the advantage is judge-dependent (+0.21 versus
+0.47, a factor of 2 between raters).

**Sources.** `compile_check.json`; per-prompt records in `prefix_gallery_judge.json`,
`prefix_judge_verdicts.json`, `prefix_judge_claude.json`, `prefix_blind_key.json`,
`steering_random_control.json`, recomputed via `_falsifier/recompute.py` loaders. The
`within-experiment` and `17-arm shared` rows for `pro_top` and `+1·d` match
`_falsifier/recompute_result.md` exactly (+0.556 / +0.796 and +0.355 / +0.376; +0.561 /
+0.825 and +0.351 / +0.355).

---

## 3. Per unit of on-`d` displacement, the prefix is roughly 60x more behaviourally effective. Not 58x.

**STRONGEST DEFENSIBLE VERSION.**
Across 18 combinations (3 baseline estimators x 2 judges-plus-mean x 2 prefix arms), the ratio
of behaviour-per-unit-on-`d`-displacement, prefix over injection, ranges from **50.5 to 75.0,
median 61.3**. For `pro_top` alone: 50.5 to 75.0, median 59.7. Computing the prefix family's
slope by regressing behaviour on dose across all six surviving arms rather than reading a
single point gives a larger ratio still: **70 to 82** depending on estimator.

| estimator | judge | arm | dose Δ∥ | behaviour | beh/dose | injection beh/dose | ratio |
|---|---|---|---|---|---|---|---|
| floating | mean | `pro_top` | 0.931 | +0.890 | 0.956 | 0.01646 | 58.0 |
| floating | mean | `pro_coherent` | 0.586 | +0.605 | 1.033 | 0.01646 | 62.7 |
| within-experiment | mean | `pro_top` | 0.931 | +0.676 | 0.725 | 0.01215 | 59.7 |
| within-experiment | mean | `pro_coherent` | 0.586 | +0.431 | 0.735 | 0.01215 | 60.5 |
| 17-arm shared | mean | `pro_top` | 0.931 | +0.693 | 0.744 | 0.01175 | 63.4 |
| 17-arm shared | mean | `pro_coherent` | 0.586 | +0.448 | 0.765 | 0.01175 | 65.1 |

**On precision.** The published "58x" is the value under one estimator (`floating`, judge-mean,
`pro_top`) and it is now known that estimator is the inflated one. **Do not print 58x.** Two
significant figures are not supported. The defensible statements, in descending order of
safety:

- safest: **"more than an order of magnitude"** (the minimum over all 18 combinations is 50.5).
- recommended: **"roughly 60-fold (50 to 75 across judges, baselines and arms)"**.
- do not use: **"roughly two orders of magnitude"**. `log10(60) = 1.78`. Even the most
  favourable slope-based estimate, 82, is below 100. `compile_check.md` line 139 currently
  says "by roughly two orders of magnitude" and that phrasing should be pulled back.

**Robustness worth noting.** The ratio is essentially unchanged when computed from
`pro_coherent`, a completely different string with a different dose (0.586 versus 0.931) and a
different behavioural delta. It is not a `pro_top` peculiarity.

**FALLBACK VERSION.** Drop the ratio arithmetic entirely and state the two facts that compose
it: the prefix delivers 3.1% of the injection's on-`d` push (measurement, no judge) and is not
behaviourally weaker (6/6 point estimates). A reader can form their own ratio.

**WHAT WOULD BREAK IT.** Same as claim 1: a large on-`d` component at the 33 prefix token
positions would shrink the denominator gap. Note the accounting is insensitive to *position
count* per se, because the injection's 30.07 is also per-position over 9 prompt positions and
both sides scale together. The vulnerability is specifically the 33 positions the injection
does not have.

---

## 4. On the metric's own axis the ranking inverts: the score's maximiser is not the behaviour's maximiser

**STRONGEST DEFENSIBLE VERSION.**
The model's neutral state is essentially orthogonal to `d` (`cos(R, d) = 0.0067`). Everything
the leaderboard measures happens just above zero. The best crowd-optimised string moves the
cosine to 0.0422, a shift of **+0.0355**. Adding `α·d̂` moves it to 0.7135, a shift of
**+0.7068**, which is **19.9x larger**. The intervention that maximises the scored quantity by
20x produces **less** judged behaviour than the string that barely moves it. Behaviour per unit
of cosine shift: **36 to 39x in the prefix's favour**, stable across all three estimators
(35.8 floating, 36.8 within-experiment, 39.1 17-arm shared).

This is a Goodhart demonstration with an actual optimum in hand, not a conjecture. If the
steering vector could be submitted to the board it would beat every human-written and
crowd-optimised entry on the metric by a factor of 20 and behave worse than the best of them.
The practical consequence for the competition: **a small board score is not evidence of a small
effect, and the metric's high end is where the behaviour is not.**

**FALLBACK VERSION.** The injection's cosine shift is exact by construction (`base + α·d̂`,
`cos(diff, d) = 1.000` measured at every position) and the ordering inversion is 6/6 across
judges and estimators. Even without the 36x figure, "the top of the metric is not the top of
the behaviour" is established.

**WHAT WOULD BREAK IT.** A judge or an eval prompt set on which `+1·d` outscores `pro_top`
behaviourally. As with claim 2, the DeepSeek margin (+0.21) is not significant on its own.

**Sources.** `data/analysis/cosine_scale.md` (base 0.0067, `pro_top` 0.0422, `+1·d` 0.7135,
50 eval prompts, layer 24, α = 30.07); ratio recomputed here under all three baselines.
Caveat carried from that document: this is the 50 eval prompts, not the 16 committed board
probes; `pro_top` scores +0.10769 on probes and +0.0355 here. The ordering is preserved.

---

## 5. An injection with 16x the prefix's on-`d` push produces a dead tie

**STRONGEST DEFENSIBLE VERSION.**
`+0.5·d` delivers an on-`d` push of **15.03**, which is **16.1x** `pro_top`'s 0.93. Its
position-consistent verdict count is **8 wins, 8 losses** out of 50, `kindness_delta = +0.13`,
`p = 0.32`. `pro_top` over the same 50 prompts is +0.87 (p = 0.00026, deepseek) and +0.91
(p = 0.00013, claude). A sixteenfold larger on-`d` displacement produces a numerically exact
tie; the prefix produces a result significant at p < 0.001 under two independent judges.

This version of the claim needs no ratio, no per-unit normalisation and no cross-family curve
fitting. It is the cheapest way to state the separation.

**FALLBACK VERSION.** `+0.5·d` is null (8W/8L, p = 0.32) and `pro_top` is not. Whether the
dose gap is exactly 16x depends on the same last-token readout caveat as everything else.

**WHAT WOULD BREAK IT.** The `±0.5` generations date from the June behavioural eval and the
`±1` and prefix generations from August. If NDIF re-served OLMo-3-32B between those runs, this
comparison crosses a model-build boundary, which is exactly the condition `PROJECT_SPEC.md`
§5.4 calls a season break. `steering_dose.md` flags this itself. Regenerating `±0.5` in one run
with both judges would settle it and is the highest-value cheap experiment after the
prefix-position measurement.

**Sources.** `data/analysis/steering_dose.json` (`+0.5`: wins 8, losses 8, kindness_delta 0.13,
p 0.32257, n 50, single judge `deepseek-v4-pro`); `steering_dose.md`;
`_falsifier/recompute_result.md` FIX2a table.

---

## 6. The kindness shift is a direction effect and the coherence damage is a magnitude effect. Two independent judges, 8 norm-matched nulls, both baselines.

**STRONGEST DEFENSIBLE VERSION.**
`+1·d` clears **all 8** norm-matched random directions under **both** judges under **both**
baseline estimators. No random arm reaches p < 0.05 under either judge. Meanwhile the only arm
reaching p < 0.05 on distinct-4 is a *random* one (`rand1`, −0.068, p = 0.036), while `+1·d`
sits inside the random spread (−0.039, p = 0.29 against a random range of −0.068 to +0.029).

| | deepseek-v4-pro | claude-opus-5 |
|---|---|---|
| `+1·d`, floating | +0.450, 8/8 randoms below, p = 0.0070 | +0.540, 8/8 below, p = 0.0015 |
| `+1·d`, fixed | +0.355, 8/8 below, p = 0.0111 | +0.376, 8/8 below, p = 0.0035 |
| randoms reaching p<0.05 | 0 of 8 | 0 of 8 |

Per the established correction, placement counts are quoted rather than z, because the fixed
baseline mechanically shrinks the null sd (deepseek sd 0.100 to 0.080; claude 0.110 to 0.040)
and inflates z.

This is the highest-confidence statistic in the whole scope: pre-registered before any result
was inspected, with a measured manipulation check rather than a code reading, cross-judged, and
surviving the falsifier's re-estimation.

**FALLBACK VERSION.** None needed. This one is solid. If anything the honest framing is that it
is confirmatory best practice rather than a discovery (see Prior work).

**WHAT WOULD BREAK IT.** SteerCheck's criticism, verbatim: isotropic random draws "occupy a
narrow near-orthogonal region". Our measured `|cos(random_i, d)|` values are 0.0144, 0.0057,
0.0012, which is exactly that regime. A *sign-randomised same-construction* null (rebuild `d`
after flipping chosen/rejected within pairs) is a harder test and has not been run. If `+1·d`
failed to clear that null, the direction-specificity claim would need restating.

**Sources.** `data/analysis/steering_random_control.md`,
`steering_random_control_preregistration.md`, `steering_random_control.json`;
`_falsifier/recompute_result.md` FIX2b.

---

## 7. Prefill-only steering tolerates 10x to 40x the coefficient that published per-token steering tolerates, and its effect is still climbing there

**STRONGEST DEFENSIBLE VERSION.**
The manipulation check measured, rather than assumed, what the intervention does: the edit is
applied at **all 9 prompt positions**, at per-position norm **exactly 30.07**, with
`cos(diff, d) = 1.000` at every position, and at **none** of the generated tokens. Against
per-position baseline residual norms spanning 5.2 to 35.9, the same α is roughly 1x the local
residual at some positions and about **6x** at one. At that magnitude the model stays fluent:
distinct-4 0.872 for `+1·d` against base 0.912, random arms 0.844 to 0.940, looping 8/50 for
`+1·d` against base 8/50.

Published practice for steering that touches generated tokens is one to two orders of magnitude
smaller (see Prior work: 0.1x the normalised residual norm reported as roughly the ceiling
before degeneracy; 0.025 for one open model). The reconciliation is mechanistic and worth
stating on its own: **prefill-only steering does not compound across the generated sequence, so
it has far more magnitude headroom than steering applied at each decoding step.**

And within that headroom the effect is still increasing: +0.13 (n.s.) at α = 0.5·‖R‖ and +0.45
/ +0.54 (p = 0.007 / 0.0015) at α = 1.0·‖R‖. Half the dose gives less than a third of the
effect. The saturation hypothesis that would have been the more portable claim is falsified,
and that is worth publishing as a negative result.

**FALLBACK VERSION.** At α = 1.0·‖R‖ applied prefill-only, an OLMo-3-32B forward pass absorbs a
perturbation the size of its own residual stream without collapsing (distinct-4 0.844 to 0.940
against base 0.912, over 11 arms). That is a robustness fact about the model, independent of
`d`, and it holds regardless of the dose-response reading.

**WHAT WOULD BREAK IT.** Two doses is a line by construction. A dose placed at 2.0·‖R‖ that
came in *below* 1.0·‖R‖ would show the peak lies between 1.0 and 2.0 and would make "still
climbing" a statement about a very narrow window. Also, the 0.5 arm is single-judge and
possibly cross-build (see claim 5).

**Sources.** `steering_random_control.md` manipulation-check table and results table;
`steering_dose.md`; `steering_dose.json`.

---

## 8. Within the prefix family, on-`d` displacement orders the arms almost perfectly. This is an ordering fact, not a dose-response.

**STRONGEST DEFENSIBLE VERSION.**
Over the six arms whose behaviour is measurable, on-`d` displacement correlates with judged
behaviour at **r = +0.986 (p = 0.00031)** on the published behaviour scale and **r = +0.987
(p = 0.00027)** on the fixed-baseline scale, with Spearman ρ = +0.943 (p = 0.0048). The
ordering is monotone across the full range including both negative arms (`anti_coherent` dose
−0.22 / behaviour −0.20; `anti_hostile` dose −0.87 / behaviour −1.31).

**This is what replaces the withdrawn "one-sided compilation" claim, and it is a better
finding.** The asymmetry was carried entirely by the arm that was withdrawn as unmeasurable.
Removing it, the relationship is monotone on both sides.

**FALLBACK VERSION, and the one I recommend.** Call this an **ordering** result, not a
dose-response. The six arms differ in text content, not only in dose, so no causal reading is
licensed: this says the on-`d` displacement is a good *index* of where an arm will land
behaviourally, which is consistent with claim 1's reading of it as a marker rather than a
mechanism. `r(dose, behaviour) = +0.986` and `r(score, behaviour) = +0.836` and the two
predictors are themselves correlated at r = +0.953.

**WHAT WOULD BREAK IT.** Six points. One badly placed seventh arm moves r substantially. The
honest fix is arms placed deliberately along the dose axis, ideally the *same* text at varying
lengths or paraphrase strengths so that dose varies while content is held closer to fixed.

**Explicitly NOT claimed.** On-`d` displacement is *not* shown to be a better predictor than
the board score. At n = 7 Williams' test gives t = +1.77, p = 0.152, and the bootstrap Δr is
+0.115 [−0.05, +0.42]. At n = 6 Williams gives t = +4.32, p = 0.0228, but the paired bootstrap
Δr at n = 6 is +0.117 with a 95% lower bound of **−0.00**, touching zero. With two correlated
predictors at n = 6, that is not a claim. Report both r values side by side and let them stand
as indistinguishable.

**Sources.** `_falsifier/recompute_result.md` FIX 3 tables.

---

# Prior work

I searched for (a) whether the prefix-versus-injection comparison has been made,
(b) whether published practice applies steering at magnitudes comparable to ‖R‖ and whether
effects are still climbing there, and (c) the field's own terminology. Every URL below was
fetched. Where a number came from an HTML rendering rather than an abstract I say so.

## What already exists, and what it does to our claims

**The "compiler" goal is provably impossible in the literal sense. This is the most important
thing in this section.**
Mishra, Khashabi and Liu, *Steered LLM Activations are Non-Surjective*
(<https://arxiv.org/abs/2604.09839>, v1 10 Apr 2026, v2 7 May 2026) state in their abstract:
"Under practical assumptions, we prove that activation steering pushes the residual stream off
the manifold of states reachable from discrete prompts. Almost surely, no prompt can reproduce
the same internal behavior induced by steering." They "establish a formal separation between
white-box steerability and black-box prompting" and "caution against interpreting the ease and
success of activation steering as evidence of prompt-based interpretability or vulnerability."

Effect on our claims: **claim 1 is upgraded in confidence and downgraded in novelty.** Our
`cos(Δ, d) = 0.0381` is a clean empirical instance of a result that has been proved, on a model
and a direction they did not test, with a quantification (3.1% of α at the scored readout) that
the theorem does not supply. State it that way. It also means the maintainer's long-term goal
should be reframed now rather than after more work: compiling a vector intervention into a
token sequence that reproduces the *activation state* is ruled out. Compiling one that
reproduces the *behaviour* is not ruled out by anything, and our claim 2 is direct evidence
that it is achievable, which is the more interesting target anyway.

**Prompting beating vector steering behaviourally is established, and our ~2x is consistent
with the published magnitude.**
Wu, Arora, Geiger, Wang, Huang, Jurafsky, Manning and Potts, *AxBench: Steering LLMs? Even
Simple Baselines Outperform Sparse Autoencoders* (<https://arxiv.org/abs/2501.17148>, Jan 2025,
ICML 2025). Abstract: "For steering, we find that prompting outperforms all existing methods,
followed by finetuning." Read from the arXiv HTML v3 (<https://arxiv.org/html/2501.17148v3>),
their Table 2 mean steering scores are Prompt 0.698 versus DiffMean 0.297 on Gemma-2-2B and
Prompt 1.075 versus DiffMean 0.322 on Gemma-2-9B, a prompting advantage of 2.35x and 3.34x.
Critically for us, they select "the optimal steering factor for each method independently for
every concept based on which factor achieves the highest overall steering score."

Effect on our claims: **claim 2's numerator is corroborated and is not novel.** A ~2x prompt
advantage over difference-in-means steering is the published expectation, on a different model
family, a different direction-construction method and a different judge. That is good news for
credibility and bad news for novelty. It also disposes of the "you just under-applied the
vector" objection at the level of the field: AxBench tuned the coefficient to its own optimum
per concept and prompting still won. Cite AxBench when reporting claim 2 and present ours as
replication plus mechanism, not as discovery.

**Prompt steering and activation steering are already known to be mechanistically different.
Nobody I found has quantified the alignment.**
Heyman and Vandeputte, *Steer Like the LLM: Activation Steering that Mimics Prompting*
(<https://arxiv.org/abs/2605.03907>, 5 May 2026) formulate prompt steering as activation
steering and find existing steering methods "not faithful to the mechanics of prompt steering,
which applies strong interventions on some tokens while barely affecting others." I fetched
their HTML (<https://arxiv.org/html/2605.03907v1>): they report relative RMSE between
prompt-induced and method-induced accumulated interventions, and they estimate token-specific
coefficients, but they do **not** report cosine similarity between the prompt-induced delta and
a steering direction, and I found no per-unit-displacement efficiency figure. Kang, Liu, Ma,
Huang, Tan and Jiang, *Prompt-Activation Duality* (<https://arxiv.org/abs/2605.10664>,
11 May 2026) argue steering "becomes more reliable when interventions follow the prompt-mediated
pathways that models already use for behavioral control."

Effect on our claims: **the qualitative core of claim 3 is (b) known but unquantified.** The
specific quantity, behaviour per unit of on-direction displacement compared across the two
instruments, I could not find anywhere. That is the contribution. Note it is a *ratio of a
measured activation quantity to a judged behavioural quantity*, which the field does not
currently compute, so the framing is ours and needs defining carefully when published.

**A 2026 preprint reaches a strikingly parallel conclusion from the opposite direction.**
Mody, Agarwal, Mittal and Mahato, *Minimizing Targeted Activations: Input-Only Suppression of
Evaluation-Awareness Latents in Large Language Models* (<https://arxiv.org/abs/2607.25907>,
28 Jul 2026) run GCG-style token optimisation against an internal latent, which is our exact
setup with the sign flipped. Their abstract: "our controls tell a cautionary story about the CAA
direction: a placebo random direction is suppressed just as hard and shifts behavior just as
far, and when we hold a real eval passage in context and optimize only a prefix, suppressing the
eval-direction fails to reduce, and slightly increases, the model's behavioral eval judgment.
Activation-readability, in short, is not behavioral controllability."

Effect on our claims: this is **converging independent evidence for the marker-not-mechanism
reading**, from a different construct, different models (Llama-3.2-3B and Llama-3.1-8B) and a
different objective sign. It also raises a control we do **not** have: they optimised tokens
against a *random* direction and found it moved behaviour comparably. **We have a norm-matched
random control for the injection arm but no random-direction control for the token-optimisation
arm.** That control should be run: optimise a prefix against a random direction with the same
GCG budget and score it on the board and on kindness. If it behaves like `pro_top`, claim 8's
ordering result and much of the leaderboard's validity story weaken sharply. This is now the
second-highest-value experiment in the scope after the prefix-position measurement.

**Published steering magnitudes really are far below ours, and the reason is prefill-only.**
Via a LessWrong reproduction post
(<https://www.lesswrong.com/posts/HhF5kESdtPHku7kim/reproducing-steering-against-evaluation-awareness-in-a-large-1>)
quoting the Opus 4.6 system card *second-hand* (I did not fetch the system card itself, so treat
the attribution as unverified even though the quote is consistent across sources): Anthropic
used "a strength of 0.1 times the normalized vector magnitude", described as "roughly the
maximum steering strength that kept degeneracy or other unusual outputs at negligible levels",
with vectors scaled so that "the norm of the vector added to the middle layer is a fixed
fraction of the mean residual stream norm at that layer." The same post reports GLM-5 needing
±0.025 and ±0.0125, with strengths above 0.025 frequently rendering the model incoherent.
Panickssery, Gabrieli, Schulz, Tong, Hubinger and Turner, *Steering Llama 2 via Contrastive
Activation Addition* (<https://arxiv.org/abs/2312.06681>, Dec 2023) add their vectors "at all
token positions after the user's prompt", i.e. during generation.

Effect on our claims: **this looks like a contradiction and is not one, but it must be reported
as a tension and resolved explicitly.** Our α = 1.0·‖R‖ is 10x to 40x these figures. The
resolution is that our intervention is prefill-only and touches no generated token, which our
own manipulation check establishes independently and which CAA's "all token positions after the
user's prompt" makes the relevant contrast. This raises claim 7's value: prefill-only steering
having 10x to 40x more magnitude headroom than per-token steering is a practically useful
observation I did not find stated anywhere. It also means our "the vector is under-applied"
sentence must be qualified as **under-applied for prefill-only steering**, not as a criticism of
published per-token practice, which is calibrated to a different failure mode.

**Norm-matched random-direction controls are standard practice, not a contribution, and their
known limitation applies to us.**
Luo, Liang and Xuan, *SteerCheck: Attribution Specificity and Alignment Leakage in
Activation-Steering Audits* (<https://arxiv.org/html/2608.24335>, 25 Aug 2026): "the standard
evidence for attribution is a random-direction control", and its limitation is that "in a
high-dimensional space, isotropic draws probe the near-orthogonal region", while
sign-randomized same-construction directions "often retain substantial target alignment."
Braun, Eickhoff, Krueger, Bahrainian and Krasheninnikov, *Understanding (Un)Reliability of
Steering Vectors in Language Models* (<https://arxiv.org/abs/2505.22637>, 28 May 2025) find
steering vectors "often give an effect opposite of the desired one" across samples and that
directional coherence of the training contrasts predicts steering efficacy.

Effect on our claims: **claim 6 is correctly executed standard practice, not a novelty.**
Present it as due diligence. The `|cos(random_i, d)|` values of 0.0144, 0.0057 and 0.0012 are a
textbook instance of SteerCheck's near-orthogonality objection, so the sign-randomised
same-construction null is the honest next control.

**Framing and terminology to adopt.**
Aparin and Gaintseva, *A Geometric Account of Activation Steering through Angle-Norm
Decomposition* (<https://arxiv.org/abs/2606.06735>, 4 Jun 2026) argue "activation steering
should be parameterized by interpretable angular and radial components of the intervention,
rather than by a single additive coefficient that entangles these two effects", and find across
seven models that "concepts are represented primarily in angular structure, but that norm
remains important for the stability and downstream effects of steering."

Effect on our claims: this gives us **the field's own vocabulary for the exact decomposition
our result rests on.** Restate the headline in their terms: the injection is a pure *radial*
move along `d` of magnitude α; the prefix is a large move that is almost entirely *angularly
orthogonal* to `d`, and the small angular component along `d` tracks behaviour without
producing it. Their finding that concepts live in angular structure while norm governs
stability is directly consistent with our claim 6 dissociation (kindness shift is a direction
effect, coherence damage is a magnitude effect). Cite it for framing; it strengthens claim 6
and gives claim 3 a home in the literature.

Other useful terms the field uses, which our documents should adopt: "steering coefficient" or
"steering factor" for α, "prefill-only" versus "per-token" or "during-generation" steering,
"prompt-reachable manifold" for what claim 1 is about, "attribution specificity" for what claim
6 establishes, "difference-in-means / DiffMean steering" for the family `d` belongs to.

## Novelty verdict, plainly stated

| element | verdict |
|---|---|
| A token string optimised against `d` does not reproduce `α·d̂`'s activation state | **(c) already established**, and proved, by Mishra et al. 2026. Our value is a quantification (3.1%) on a new model and direction. |
| Prompting beats vector steering behaviourally | **(c) already established** by AxBench, at a similar magnitude (~2 to 3x). Ours replicates it on OLMo-3-32B. |
| Prefix and injection are mechanistically different instruments | **(b) known but unquantified** (Heyman & Vandeputte 2026, Kang et al. 2026). |
| **Behaviour per unit of on-direction displacement, compared across the two instruments (~60x)** | **(a) new** as far as I could find. This is the contribution. |
| Prefill-only steering has 10x to 40x more magnitude headroom than per-token steering | **(a) new** as far as I could find, and practically useful. Lower confidence in the "new" verdict: it is the kind of thing that may exist as a remark in a paper I did not surface. |
| Injection dose-response still climbing at 1.0·‖R‖ | **(b)/(a)**: my searches for a published prefill-only dose-response curve were inconclusive. Our own evidence is two points. |
| Norm-matched random control for the injection | **(c) standard practice**, per SteerCheck. |

Search limits: I did not find a paper that computes the cosine between a prompt-induced
activation delta and a steering direction. That may be because it does not exist, or because my
queries missed it. Two searches aimed directly at an efficiency-per-unit-displacement
quantification returned nothing relevant; I am recording those as **inconclusive**, not as
evidence of absence.

---

# Things I looked for and could not support. Drop these.

1. **"58x."** The number exists under exactly one estimator and that estimator is the one the
   falsifier showed is inflated. Range is 50 to 75. Two significant figures are not earned.

2. **"Roughly two orders of magnitude."** `compile_check.md` line 139 says this. 60x is 1.78
   orders and the largest defensible estimate is 82x. Pull it back to "roughly 60-fold" or
   "more than an order of magnitude."

3. **"On-`d` displacement is a better predictor of behaviour than the board score."** Already
   withdrawn and correctly so. I checked whether the n=6 result rescues it: Williams gives
   p = 0.0228 at n=6, but the paired bootstrap Δr lower bound is −0.00. It does not rescue it.

4. **"The compilation is one-sided."** Dead, as established. I looked for a replacement
   asymmetry claim within the mechanism scope and there is one available (the prefix family is
   monotone across both signs while the injection family is null on the negative side), but I do
   not recommend it: the negative prefix arms are hostile *text*, not negated `d`, so the
   comparison is not like-for-like and the causal reading is not licensed. Do not publish it.

5. **"Steering vectors are routinely applied past their useful range."** Falsified within this
   project (+0.5 gives less than a third of +1.0's effect) and, per the prior-work section,
   contradicted by published practice, which applies coefficients 10x to 40x *smaller* than
   ours. This claim is dead twice over. `steering_dose.md` already records the falsification;
   the prior-work contradiction should be added to it.

6. **A second behavioural axis confirming the mechanism separation.** I checked
   `_falsifier/honesty_result.md` for a non-kindness readout that would generalise the claim.
   It does not: `pro_top` honesty Δ = −0.108 (p = 0.55, n = 37) and `pro_coherent` = +0.000
   (p = 1.00), both null, and the document's own verdict is that 45 of 50 prompt stems put no
   honesty content at stake, so the measure is floored by construction. **The mechanism claims
   rest on a single behavioural construct measured by LLM judges.** That is the largest
   unaddressed limitation in this scope and no amount of re-analysis fixes it.

7. **A random-direction control for the token-optimisation arm.** Does not exist in this
   project. Mody et al. 2026 ran the analogous control and found a placebo direction "shifts
   behavior just as far." Until we run it, "the prefix's on-`d` component is a marker of what
   the prefix does" has an untested competitor: "GCG against *any* direction produces a
   fluent-ish prefix that shifts judged kindness." Flag this prominently.

8. **Any per-position or multi-layer evidence.** Everything here is the layer-24 last-token
   readout. `compile_check.md` names this as limit 1 and it remains true.

---

# What this evidence licenses for the vector-to-token compiler goal

Concretely, for the maintainer's stated long-term interest:

**Ruled out.** Compiling a vector intervention into a token sequence that reproduces the
*activation state* the vector produces. Mishra et al. prove no prompt admits a preimage for a
steered activation under practical assumptions, and our own measurement is a clean instance:
the best token-space optimum available reaches `cos(Δ, d) = 0.038` and 3.1% of α along `d`
while displacing the residual by 81% of α in other directions. Reframe or retire this framing
now.

**Not ruled out, and supported.** Compiling a vector intervention into a token sequence that
reproduces or exceeds the *behaviour*. Our prefix matches or beats `α·d̂` behaviourally in 6 of
6 judge-by-estimator comparisons, and AxBench independently reports prompting beating
difference-in-means steering by 2.35x and 3.34x with the steering coefficient tuned to its own
optimum. If the goal is behavioural, the evidence is encouraging and the target should be
defined behaviourally from the start.

**The objective function is the problem, and this is the actionable finding.** Optimising token
sequences against on-`d` projection is optimising against a quantity that, per claim 4, is 20x
better satisfied by the intervention that behaves worse. The on-`d` component is an index of
where an arm will land (claim 8, r = +0.986 over six arms) but the metric's high end is not
where the behaviour is. A compiler built on this objective will keep finding strings that score
well; whether it finds strings that *act* well is a separate question the objective does not
answer. Any future compiler should be scored behaviourally and validated against a
random-direction-optimised placebo.

**Two experiments, in priority order, before anything is published as a mechanism claim.**
1. Measure `cos(R_L24(t), d)` and `‖R_L24(t)‖` at each of the 33 prefix token positions. This is
   the single result that decides whether "3.1%" and "60x" survive, and it needs one forward
   pass.
2. Run GCG against a random unit direction with the same token budget, then score the resulting
   prefix on the board and on judged kindness. This is the control Mody et al. ran and we did
   not. It decides whether the on-`d` optimisation is doing anything at all.

---

# Suggested wording for the headline, at the precision the evidence supports

> A 33-token string optimised in token space against a residual direction `d` displaces the
> scored readout by 81% of a conventional steering magnitude, but only 3.1% of that
> displacement lies along `d` itself. It nonetheless produces at least as much of the target
> behaviour as injecting the full-magnitude vector does, under two independent judges and three
> baseline estimators. Per unit of on-`d` displacement the prefix is roughly 60 times more
> behaviourally effective (50 to 75 across judges, baselines and two different strings). Within
> each instrument more on-`d` displacement means more behaviour; across instruments it does not.
> Matching a direction's displacement does not reproduce its effect, and the on-`d` component of
> a prefix is better read as a marker of what the prefix is doing than as the thing doing it.

Both caveats belong in the same paragraph: this is one behavioural construct measured by LLM
judges, and it is the layer-24 last-token readout only.
