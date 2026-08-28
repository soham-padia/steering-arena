# What this project's way of working establishes

**Date:** 2026-08-27 · **Repo state:** `8c43273` (main) · **Author:** advocate pass

This is the methodological half of the record. The adversarial audit
(`_falsifier/2026-08-27-experiment-vs-hypothesis-audit.md`) asked "do the experiments test
the hypotheses" and returned only limitations, which was a function of the brief. This
document asks a different question: **for someone else running cheap behavioural
evaluations of steering interventions, what does this project's procedure establish that
they can use?**

Rules I held myself to:

- Every claim carries a number and the artifact it came from. No number, no claim.
- Where I could recompute a number from the raw records rather than quote a document, I
  did, and I say so. Recomputations are marked **[recomputed here]**.
- Each item states the strongest defensible version, a fallback if the strong version is
  contestable, and what evidence would break it.
- Each item is labelled **demonstrated** (this project's data show it) or **illustrated**
  (this project is one instance of something that needs more instances).
- Each item carries a prior-work label: **NEW**, **KNOWN-BUT-UNQUANTIFIED**, or
  **ALREADY-ESTABLISHED**. See "Prior work" at the end. One candidate contribution was
  demoted from the top of the list to the bottom by that check.

Corrections carried in from the audit and honoured throughout: `anti_top`'s behavioural
numbers are withdrawn and are never quoted here as results; fixed-baseline deltas from
`_falsifier/recompute_result.json` are preferred to the published floating-baseline ones;
on-`d` displacement is **not** claimed to beat board score as a predictor of behaviour
(Williams t = +1.77, df=4, p=0.152, `_falsifier/recompute_result.md` FIX 3).

---

## Ranked contributions

Ranked by (strength of evidence) x (usefulness to an outside practitioner), then adjusted
downward where the literature already covers the point.

---

### 1. LLM-judge contrast drift is real, measurable, directional, and fixable by call structure

**Label: KNOWN-BUT-UNQUANTIFIED. Status: demonstrated.**

**The claim.** When an LLM judge rates two texts inside one prompt, the absolute rating it
gives the *fixed* baseline text moves depending on which arm's text it is sitting next to.
The movement is not noise: it runs **opposite** to the arm's true effect, so it inflates
every effect size in both directions. It is caused by the *call structure*, not the judge
model, and it is removed by rating the baseline once per item and reusing that number.

**The numbers.**

The same 50 base continuations were re-rated inside every arm's run.

| judge | how the judge saw the pairs | base-rating drift on identical text |
|---|---|---|
| `deepseek-v4-pro` | one independent API call per pair | base mean **2.77** in the `pro_top` run vs **3.39** in the `anti_top` run: shift **+0.62**, Wilcoxon **p = 7.1e-07** (n=50, 39 non-tie); identical rating on 5/50 |
| `claude-opus-5`, 3-arm run | batches of 25 pairs in one agent context; the CSV is prompt-major so all 3 pairs sharing a base text land in the same context | shift **+0.00**, identical rating on **50/50**, p = 1.0 |
| `claude-opus-5`, 4-arm gallery run | each arm judged in a separate run | identical on **21/50**, mean within-prompt range **0.45**; `anti_coherent` vs `anti_hostile` base shift **+0.19**, Wilcoxon **p = 0.0120** **[recomputed here]** |

That third row is the load-bearing one. It is the *same judge model* drifting when the
protocol separates the contexts and not drifting when it does not, which de-confounds
"protocol" from "model" as far as these data allow.

The drift is directional. Across the seven prefix arms, the base rating an arm's run
assigns correlates **negatively** with that arm's fixed-baseline effect: **r = −0.869,
p = 0.0111** for DeepSeek and **r = −0.703, p = 0.078** for Claude **[recomputed here]**.
The base looks *less* kind next to a kind continuation and *more* kind next to a hostile
one. That is a contrast effect, and because it is signed against the effect, the floating
baseline exaggerates rather than merely blurs.

**Magnitude of the correction.** Recomputing every delta against a fixed per-prompt
baseline (`_falsifier/recompute_result.md`, FIX 2):

| arm | deepseek reported → fixed | claude reported → fixed |
|---|---|---|
| `pro_top` | +0.870 → **+0.556** (−36%) | +0.910 → **+0.796** (−13%) |
| `pro_coherent` | +0.640 → **+0.406** (−37%) | +0.570 → **+0.456** (−20%) |
| `anti_hostile` | −1.310 → **−1.114** (−15%) | −1.310 → **−1.114** (−15%) |
| `+1·d` injection | +0.45 → **+0.355** (−21%) | +0.54 → **+0.376** (−30%) |

**Sign flips: zero, on all 14 prefix rows.** Every published sign survives; the sizes shrink
by 13% to 37% on the significant arms. One arm moves the other way (`anti_top` under
Claude, −0.390 → −0.504), and that arm is withdrawn anyway.

**Strongest defensible version.** In pairwise LLM judging where the judge emits an absolute
score for each side, the baseline side's score is not a constant. Measure the drift by
pairing the baseline's ratings across arms on identical text; correct it by differencing
against a fixed per-item baseline. Here that cost zero new generations and zero new judge
calls, and it changed no conclusion while shrinking every headline by up to a third.

**Fallback if contested.** Even if you reject the causal story about call structure, the
descriptive fact stands on its own: two judges rating byte-identical text produced base
means spanning 2.77 to 3.47 across arms, and a 0.62 shift at p = 7.1e-07 is 71% of the size
of the headline effect computed from it. Any published delta that differences against a
re-rated baseline inherits that term.

**What would break it.** (a) A judge run where the arms are judged in separate contexts and
the base ratings are nonetheless statistically identical, which would show the batching
explanation is wrong. (b) Evidence that the fixed-baseline estimator is itself biased for
these data, e.g. if base continuations differ systematically in a way the grand mean
absorbs. (c) A demonstration that the drift is a per-run *level* shift rather than a
contrast effect, which the r = −0.869 result argues against but does not exclude at n=7
arms.

**Caveats I will not hide.**

- The zero-drift condition was not designed. It happened because
  `scripts/prefix_behavior_eval.py:cmd_claude_batches` chunks a prompt-major CSV into
  batches of 25, so a base text and its three arm-mates usually share one agent context.
  The fix is real and reproducible; the discovery was luck.
- The batching *mechanism* is a hypothesis, not a measured cause. 150 rows at 3 arms per
  prompt chunked into 25s means roughly 5 of 50 prompt-groups straddle a batch boundary,
  and Claude still rated the base identically on **50/50**. So either those few were
  consistent across contexts anyway, or the mechanism is coarser than "same context". What
  is measured is that the *same judge model* drifts across separately-run arms
  (21/50 identical, p = 0.0120) and not across arms judged together (50/50, p = 1.0).
- That within-judge comparison also varies the arm set (3-arm run vs 4 gallery arms), not
  only the run structure. The base texts are the same 50 in both, so the drift cannot come
  from the rated baseline changing, but "run structure" and "which arms were in the run"
  are not fully separated.
- "No drift" is a judge being self-consistent within a context, which is itself a form of
  anchoring. It is what this estimator needs; it is not evidence the rating is more
  accurate.

---

### 2. Store per-item records, not aggregate deltas, and every future question is free

**Label: ALREADY-ESTABLISHED as a principle, rarely executed. Status: demonstrated.**

**The claim.** The single highest-leverage decision in this project was writing, for every
judged pair, the raw record (verdict, both absolute ratings including the **baseline's**,
markers, comment) and caching every generation keyed by
`(model, layer, prompt, arm, alpha, max_new)`. Because of that, two of the audit's most
expensive-sounding findings were settled at zero marginal cost.

**The numbers.**

- **Honesty re-judge** (`_falsifier/honesty_result.md`): the audit's finding 1 was that the
  eval scores kindness and calls it pro-human, so `+d` might be buying agreeableness at
  the cost of honesty. Settling it required **400 blind items** and **zero new
  generations**: the already-cached continuations in `data/cache/prefix_behavioral/` were
  re-judged on an honesty-only rubric. Result: `pro_top` honesty Δ = **−0.108**, Wilcoxon
  **p = 0.546** (n=37 retained), `pro_coherent` **+0.000**, p = 1.000. The prediction is
  **not supported**, and it cost nothing to find out.
- **Fixed-baseline re-analysis** (contribution 1): zero new generations, zero judge calls,
  because `kindness_base` was stored per pair rather than folded into a delta.
- **Claims ledger** (`_falsifier/recompute.py` → `recompute_result.md`): **125 published
  numbers** re-derived from committed artifacts; **124 AGREE, 1 DISAGREE**. The single
  disagreement is in the *audit itself* (it reported "5/8 randoms below `−1·d`" under a
  fixed baseline when the recomputed count is 2/8; the 5/8 is the floating-baseline
  placement carried across estimators).

**Strongest defensible version.** For cheap behavioural evals, the durable artifact is the
per-item record, not the report. Two properties make it pay: (i) generations cached under a
key that includes every parameter that changes the text, so a new judge or a new rubric
re-reads instead of re-generates; (ii) per-item judge output that includes the *control
side's* score, so the estimator can be changed after the fact. A third practice, a script
that re-derives every number in the prose, catches the specific error class where a
quantity computed under estimator A is quoted in a table about estimator B. That error
occurred here, in the document whose job was to catch it.

**Fallback.** Even if you think re-judging cached text is weaker evidence than fresh
generation (it is: the honesty re-judge had one rater and no inter-rater estimate, and the
same base texts recur across arms so the judge could recognise them), the asymmetry in cost
is decisive. The honesty question went from "another 400 generations" to "an afternoon".

**What would break it.** A demonstration that re-judging cached text gives systematically
different answers from fresh generation under the same rubric. Nobody ran that here, and it
is the obvious next control.

**The honest limit on the honesty re-judge.** Its null is not informative about the
underlying question, and the document says so: **45 of the 50** eval stems put no honesty
content at stake, so a paired honesty delta is floored near zero by construction. The
method was cheap and correct; the prompt set could not answer the question. That is itself
the transferable lesson: a free re-analysis cannot rescue a corpus that was never built for
the axis.

---

### 3. Judge reliability is conditional on the condition being judged, not a property of the judge

**Label: KNOWN-BUT-UNQUANTIFIED. Status: demonstrated, but at small n.**

**The claim.** "This judge agrees with humans X% of the time" is the wrong summary
statistic. Agreement is high where the text is coherent and the effect is large, and at
chance where the condition damages the text. You should report agreement **per arm**, and
you should not carry a global agreement figure into an arm no human has checked.

**The numbers** (pairs where both the human and the judge returned a decided A/B verdict;
**[recomputed here]**, reproducing `_falsifier/2026-08-27-addendum-human-ratings.md` N3
exactly):

| | `pro_top` | `pro_coherent` | `anti_top` | overall |
|---|---|---|---|---|
| `claude-opus-5` | 12/15 = **80%** | 9/11 = 82% | 6/11 = 55% | 27/37 = 73% |
| `deepseek-v4-pro` | 10/12 = **83%** | 6/11 = 55% | 5/15 = **33%** | 21/38 = 55% |

DeepSeek's *overall* agreement with the human is 55%, which on a two-alternative forced
choice is chance, and would normally condemn the judge. It is 83% on `pro_top` and 33% on
`anti_top`. The judge is not bad; the judge is conditional.

For scale: the standard reference figure for LLM judges is "over 80% agreement with human
preferences, the same level of agreement between humans" (Zheng et al., MT-Bench,
[arXiv:2306.05685](https://arxiv.org/abs/2306.05685)). This project's 80-83% on the coherent
large-effect arms lands exactly there. The same judges on a degenerate arm land at chance.
A single global number would have averaged those into a misleading 55-73%.

**Strongest defensible version.** Report judge-human agreement stratified by experimental
arm, and treat an LLM-only arm as unvalidated unless a human-checked arm with *similar text
quality* exists. In this study that immediately flags the eight random-direction arms and
the ±0.5 dose arms, which no human has rated at all, as resting on an instrument validated
only where the text was fluent.

**Fallback.** The per-cell n is 11 to 15 decided pairs. Treat the *contrast* between 80-83%
and 33-55% as the finding, not the point estimates. Even as a contrast it is contestable:
`pro_coherent` gives 82% (claude) and 55% (deepseek) on the *same* pairs, so judge identity
also matters and the story is not purely "coherence explains everything".

**What would break it.** A larger human pass showing per-arm agreement is flat once n is
adequate, i.e. that the 33% cell is sampling noise. With 15 decided pairs, 5/15 has a wide
interval, and this is the weakest quantitative leg of the three top contributions.

---

### 4. A forced-choice preference judgement is undefined when one arm damages fluency, and it can invert

**Label: KNOWN-BUT-UNQUANTIFIED. Status: the inversion is demonstrated; the proposed fix is
NOT demonstrated.**

**The claim.** If your intervention degrades coherence, "which of these is kinder / better /
more X" stops being a question about X. Different raters answer it with opposite signs, and
which sign you publish is set by an arbitrary rubric convention. Blinding also fails, because
the degeneracy is the arm's fingerprint.

**The numbers** (`data/analysis/prefix_eval.md`, "Human ratings (n=54)"; verdict counts
**[recomputed here]** from `prefix_blind.csv` + `prefix_blind_key.json`):

- The `anti_top` prefix produced degenerate text: repetition flagged on **37/50** and
  incoherent on **8/50** by Claude, repetition **25/50** and incoherent **12/50** by
  DeepSeek; **39/50** degenerate overall.
- On identical blind pairs, the human preferred the **anti-prefixed** side **14 of 17
  decided pairs (82%, exact binomial p = 0.0127)**; DeepSeek preferred it on **12%**.
  Opposite signs, same pairs.
- Objective, judge-free measurement of the same arm survives untouched: distinct-4-gram
  ratio **0.805** with **12/25** continuations looping, against base **0.888** and **6/25**.

**Strongest defensible version.** Before running an A/B preference protocol over a condition
that may damage fluency, decide in advance what a non-response scores, and prefer a
judge-free structural measure (distinct-n, looping rate) for that arm. If both raters can
identify the arm on sight, the pairs are not blind regardless of what the CSV shows.
Rating more pairs does not help: the fingerprint *is* the behaviour being measured.

**The fix is proposed, not proven.** The project's response was to add an `n` ("no stance")
key to `scripts/rate_blind.py`, scoped to *either* side degenerate rather than both, and to
withdraw the arm. **Across all 96 human ratings in both blind CSVs, `n` was used exactly 0
times [recomputed here]** (`prefix_blind.csv`: 24 A, 22 B, 8 T; `behavioral_blind.csv`:
13 A, 19 B, 10 T). The no-stance rate is therefore not measured anywhere, and the
withdrawal rests on the sign inversion plus the rater's verbal report, not on recorded
no-stance counts. An outside practitioner should take the *diagnosis* from this project and
should not assume the instrument fix works, because it has never produced a datum.

**One thing the fix did do, retroactively.** The audit applied the "no stance" exclusion
rule to every arm, including the ones the project wanted to keep: `pro_top` holds at 22/31
(71%, p = 0.029) claude and 19/24 (79%, p = 0.0066) deepseek; `anti_hostile` holds at −1.24
(p = 0.0005) and −1.17 (p = 0.011); `anti_top` collapses to 3/6 and 4/9 with DeepSeek's Δ
going to +0.04. That is the correct discipline and it is separately transferable: see item 7.
(These are the audit's recomputation, not mine.)

**What would break it.** A rating pass that actually uses the `n` key and shows the arm
becomes measurable, or a demonstration that the human/judge sign inversion came from the
"neutral beats cruel" instruction alone rather than from the degeneracy, which would make
it a rubric bug rather than a structural limit. The project's own account concedes the
instruction contributed.

---

### 5. Measure the manipulation, do not read it off the code

**Label: KNOWN-BUT-UNQUANTIFIED. Status: demonstrated.**

**The claim.** Before interpreting an intervention, save the activations with and without
it and difference them. This project did that once and it changed what the whole steering
arm could be *called*.

**The numbers** (`data/analysis/steering_random_control_preregistration.md`, "Manipulation
check", repeated in `steering_random_control.md`):

| property | measured |
|---|---|
| positions edited | **all 9 prompt positions**, not last-token only |
| per-position edit norm | **30.07**, exactly α |
| direction of edit | **cos(diff, d) = 1.000** at every position |
| generated tokens edited | **none** (trace covers 9 of a 17-token sequence) |
| per-position baseline residual norms | 18.5, 21.5, 33.2, 35.1, 33.5, 32.6, 35.9, **5.2**, 31.1 |

Two facts fell out that no amount of reading the code had produced. First, the eval is
**prefill-only** steering: the vector perturbs the prompt's forward pass and is not applied
at each decoding step, so every prior description of it as "steering every token" was wrong.
Second, α is defined from the mean *last-token* residual norm and applied uniformly, so
against per-position norms of 5.2 to 35.9 the same α is about 1x the local residual at some
positions and about **6x** at others. `steering_random_control.md` prints a "corrections owed
to earlier write-ups" list as a direct result.

**Strongest defensible version.** A five-line diff of the hooked layer's output, with and
without the edit, is the cheapest validity check in the whole pipeline and it audits the
*description* of the experiment, which is the thing that propagates into papers. Report the
edit's norm, its cosine with the intended direction, the exact positions touched, and the
per-position baseline norms it is being compared against.

**Fallback.** If you think "prefill-only" is a labelling quibble, the per-position norm
spread is not: a nominally "1.0 x ‖R‖" perturbation was 6x the local residual at one
position, which is a dose the write-up did not know it was administering.

**What would break it.** Nothing about the measurement itself; it is a direct read of the
tensors. What is contestable is how much it matters, and the project's own answer is
partly reassuring: the same α was recomputed fresh two months later and returned
bit-identical at **30.07038116455078**, which is evidence the served model build did not
change.

---

### 6. A retraction that is not mechanical does not propagate

**Label: process claim, not a research claim. Status: demonstrated, with a correction to
the audit.**

**The claim.** Writing "this result is withdrawn" in the document that produced it does not
remove it from the project. The number keeps moving into new documents, including documents
written *after* the withdrawal.

**The numbers [recomputed here].**

- The withdrawal landed in commit `6ec2e20`, 2026-08-26: `anti_top`'s behavioural numbers
  (−0.86, p = 0.0005 deepseek; −0.39, p = 0.0277 claude) withdrawn as unmeasurable.
- Four downstream analysis documents (`compile_check.md`, `cosine_scale.md`,
  `steering_dose.md`, `prefix_transfer.md`) contain **zero** occurrences of "withdraw",
  "retract" or "unmeasurable".
- **Correction to the audit.** It says "four downstream documents inherited a retracted
  number". Three did, not four: `compile_check.md` (the −0.62 two-judge mean, in the
  dose-response table at line 62), `cosine_scale.md` (−0.62 at line 16), and
  `prefix_transfer.md` (−0.86, p = 0.0005 at line 33). `steering_dose.md` contains no
  mention of `anti_top` at all. The audit's separate criticism of `steering_dose.md` is a
  different error (its Limit 4).
- **The sharper fact the audit did not state.** `compile_check.md` was created in commit
  `aab84b2` and `cosine_scale.md` in `84dc1e4`, both dated **2026-08-27**, i.e. **after**
  the 2026-08-26 withdrawal. These are not stale documents that missed an update. They are
  new documents that imported a retracted number, and `compile_check.md` built its
  "the compilation is one-sided" headline on it.
- Consequence, recomputed: drop the withdrawn arm and the finding **inverts**. Overall
  dose-vs-behaviour correlation goes from r = +0.863 (p = 0.0124, n=7) to **r = +0.986
  (p = 0.00031, n=6)**, and the negative side becomes two correctly-ordered points rather
  than an absence of dose-response.

**Strongest defensible version.** A retraction needs a mechanical enforcement step, not a
prose note. The cheapest version that would have worked here: put withdrawn quantities in a
machine-readable list and have the claims-checking script (contribution 2) fail if any
document contains one. This project already had the checker; it just had no withdrawn-values
list to check against.

**Fallback.** If you regard n=3 documents as too small a sample to generalise from, the
finding still holds as an existence proof with an unusually clean timestamp: the retraction
and the re-import are 17 commits and one day apart in the same repository, by the same
author, with the withdrawal text in a file the new documents cite.

**What would break it.** Nothing empirical; this is a description of what happened. It
generalises only as far as "one project, one author, one day", which is why it is ranked
below the measurement contributions.

---

### 7. Apply a new exclusion rule to your surviving results too

**Label: KNOWN-BUT-UNQUANTIFIED. Status: demonstrated (by the audit, not by the analysis).**

**The claim.** When you invent a rule that kills an inconvenient result, the rule is only
credible if you apply it to the results you want to keep. Here, "exclude pairs where either
side takes no stance" was invented to withdraw `anti_top`. Applied symmetrically:

| arm | under the new rule | verdict |
|---|---|---|
| `pro_top` | 22/31 (71%, p = 0.029) claude; 19/24 (79%, p = 0.0066) deepseek; Δ +0.88 / +0.95 | survives |
| `anti_hostile` | Δ −1.24 (p = 0.0005) and −1.17 (p = 0.011) | survives |
| `anti_top` | 3/6 and 4/9; deepseek Δ **+0.04** | collapses |

The audit notes: "Nobody had run this." The rule is quantitatively correct and the two
surviving headlines survive their own correction.

**Strongest defensible version.** A retroactive exclusion rule is a free, zero-generation
robustness check on every arm you have. Run it on all of them and publish the table, or the
rule reads as motivated.

**Fallback.** I did not independently recompute these three rows; they are the audit's
recomputation from `data/cache/prefix_behavioral/` plus the marker fields. The
methodological point does not depend on the exact numbers.

**What would break it.** A recomputation showing `pro_top` also collapses under the rule,
which would mean the rule is not a filter but a different measurement.

---

### 8. The random-direction null: the field already knows, and the transferable lesson is about pre-registration

**Label: ALREADY-ESTABLISHED. Status: the criticism is correct and is not this project's.**

This was proposed to me as a candidate contribution. It does not survive the prior-work
check, and I am demoting it from the top of the list to the bottom.

**The criticism.** `scripts/steering_random_control.py:45-47` states its own reasoning: "In
5120 dimensions a Gaussian draw is near-orthogonal to any fixed vector, which is exactly the
point: same norm, no alignment with `d`." Measured |cos(random_i, `d`)| for the first three
draws: **0.0144, 0.0057, 0.0012**; `d` is a **5120**-dimensional vector
(`data/directions/d_olmo3_L24_logistic.npz` **[verified here]**). The audit's point is that
such a draw is near-orthogonal to `d` **and to every other concept direction**, so the
control can only test `d` against noise, never against another meaningful direction. The
project's own pre-registration named the damaging outcome (C: "activation space is organised
such that many directions move the judged construct") and then chose the one null family
structurally guaranteed not to produce it.

**Why it is not a contribution.** SteerCheck (Luo, Liang & Xuan, arXiv:2608.24335, 25 Aug
2026, [fetched](https://arxiv.org/html/2608.24335)) makes this criticism directly, two days
before the audit was written, in almost the same words: "in a high-dimensional space,
isotropic draws probe the near-orthogonal region", and notes that stronger sign-randomized
comparators "retain substantial alignment" with the target. It also supplies the better
controls, which this project's audit did not: three null families (isotropic random,
**PCA-subspace** directions that preserve dominant activation structure, and
**sign-randomized** directions that preserve paired contrasts), matched on a **KL budget**
rather than Euclidean norm, plus an alignment-leakage diagnostic. The same paper confirms
the norm-matched isotropic control is the field standard: "The standard evidence for
attribution is a random-direction control where a steering vector is compared with
directions of matched Euclidean norm."

**What still generalises, and it is not nothing.** The pre-registration
(`steering_random_control_preregistration.md`) is genuine design foresight: it was written
before any result was inspected, it names three outcomes including the one the author says
he "would be most tempted not to see", and it commits to a judge-free second measure
(distinct-4-gram) alongside the judged one. And it still failed, because pre-registering the
*threat* does not pre-register the *instrument's power against that threat*. The transferable
rule: when you write down the outcome you fear, also write down why your chosen control
could produce it. If you cannot answer that, the control is decorative. This is a case
study, not a law; it is one pre-registration.

**Also worth carrying forward, and independent of `d`.** The dissociation the control did
establish is clean and reusable as a template: perturbation at 1.0·‖R‖ damages coherence
**regardless of direction** (`+1·d` distinct-4 −0.039, inside the random spread of −0.068 to
+0.029; the only arm reaching p<0.05 on distinct-4 is a *random* one), while the kindness
shift is direction-specific (`+1·d` at +0.355 fixed-baseline, above **all 8** random draws
under both judges; **0 of 8** randoms reach p<0.05 under either judge). Running one judged
measure and one judge-free structural measure over the same arms is what let those two
effects come apart. The `−1·d` conclusion, by contrast, is estimator-dependent and should
not be carried forward: under the fixed baseline it sits **below all 8** randoms for
DeepSeek but **inside** the band for Claude.

---

## Prior work

Everything in this section was fetched and read. Where a search snippet looked relevant but
I could not verify the source, I say "unverified" or leave it out.

**Contrast drift (contribution 1).** I did not find published work testing the specific
thing measured here: whether an LLM judge's *absolute score for a fixed text* shifts as a
function of the partner it is presented with, inside a pairwise call. The nearest verified
neighbours:

- **Anchoring Bias in LLM-as-a-Judge Systems: Prior Scores Compromise Evaluation
  Independence** (Kapetanovic, Altwlkany, Mercep, Duricic, Lacic, arXiv:2608.25869,
  26 Aug 2026, [fetched](https://arxiv.org/abs/2608.25869)). Anchors on *prior scores
  supplied as metadata*, not on the comparison partner. 192,000 evaluations, 7 of 8 models
  show significant anchoring, Cohen's d up to 0.71; anchored metadata "blocks 48% of error
  corrections and flips 10.18% of correct judgments". Neither chain-of-thought nor
  disregard-instructions removed it. Different mechanism, same family.
- **Pairwise or Pointwise? Evaluating Feedback Protocols for Bias in LLM-Based Evaluation**
  (Tripathi, Wadhwa, Durrett, Niekum, arXiv:2504.14716, 20 Apr 2025,
  [fetched](https://arxiv.org/html/2504.14716v1)). Shows pairwise is far more manipulable
  than absolute scoring: introducing distractor features (assertiveness, verbosity,
  sycophancy) flips "pairwise preferences in about 35% of the cases, compared to only 9%
  for absolute scores", and finds absolute scoring "remains stable... regardless of
  distractor presence". This is the closest prior result, and note it points the *opposite*
  way to our finding: it treats absolute scores as the stable protocol, where we find the
  absolute score of a fixed text moving by 0.62 as a function of its partner. The two are
  compatible (they vary the *rated* response, we vary the *partner* of a fixed response),
  and that gap is where our measurement sits.
- **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena** (Zheng et al., arXiv:2306.05685,
  Jun 2023, [fetched](https://arxiv.org/abs/2306.05685)) established position, verbosity and
  self-enhancement bias as the canonical judge biases. Contrast against the comparison
  partner is not among them.

**Verdict: KNOWN-BUT-UNQUANTIFIED.** The bias family is thoroughly documented; the specific
estimator failure (differencing against a re-rated baseline) and its magnitude on a real
experiment are, as far as I can verify, not. Our contribution is a clean magnitude estimate
(+0.62 on a 1-5 scale, p = 7.1e-07, ~71% of the headline), a directional result
(r = −0.869 between an arm's base rating and its true effect), a protocol-level cause, and a
fix that costs nothing. That is a useful data point, not a discovery.

**Conditional judge reliability (contribution 3).** The reference agreement figure is
MT-Bench's "over 80% agreement... the same level of agreement between humans"
(arXiv:2306.05685, verified above). I found no verified source reporting agreement
stratified by whether the condition damaged text quality. Related but not the same:
**Diverging Preferences: When do Annotators Disagree and do Models Know?** (Zhang, Wang,
Hwang, Dong, Delalleau, Choi, Choi, Ren, Pyatkin, arXiv:2410.14632, ICML 2025,
[fetched](https://arxiv.org/abs/2410.14632)) builds "a taxonomy of disagreement sources
spanning ten categories across four high-level classes" and attributes most disagreement to
"task underspecification or response style". I could not confirm from the abstract whether
degeneracy/incoherence is one of the ten categories, so I record that as **inconclusive**.
**Verdict: KNOWN-BUT-UNQUANTIFIED.**

**Degeneracy and forced choice (contribution 4).** That steering damages fluency and
coherence is established. **Beyond Multiple Choice: Evaluating Steering Vectors for
Summarization** (Braun, Eickhoff, Bahrainian, Findings of EACL 2026, arXiv:2505.24859,
[fetched](https://arxiv.org/html/2505.24859v3)) documents an efficacy-quality trade-off:
"high steering magnitudes of |λ|>2 substantially degrade both intrinsic and extrinsic text
quality", with two stages of degradation ending in "degenerate repetition", and for toxicity
steering "outputs devolve into incoherent loops of toxic keywords rather than fluent
summaries". Notably that paper uses **no random-direction control**. What I could not find
published is the consequence for *preference measurement*: that two raters return opposite
signs on the same degenerate pairs so the published sign is a rubric convention. **Verdict:
KNOWN-BUT-UNQUANTIFIED**, with the caveat that our fix is undemonstrated.

**Random-direction nulls (contribution 8).** **ALREADY-ESTABLISHED.** SteerCheck
(arXiv:2608.24335, [fetched](https://arxiv.org/html/2608.24335)) states both that the
norm-matched isotropic control is the field standard and that it "probes the near-orthogonal
region", and proposes PCA-subspace and sign-randomized nulls under a matched KL budget. The
older and broader precedent for "your control must be able to fail" is **Designing and
Interpreting Probes with Control Tasks** (Hewitt & Liang, EMNLP-IJCNLP 2019,
[fetched](https://aclanthology.org/D19-1275/)), whose *selectivity* framing is the same
move: a control that the probe can only pass by memorising, so that high task accuracy alone
proves nothing. If you are designing a steering control today, those two are the sources,
not this project.

Search snippets I saw but did **not** verify, and therefore do not rely on anywhere above:
a claim that random-direction steering raises harmful compliance "from 0% to 2-27%"
(attributed in search results to "The Rogue Scalpel: Activation Steering Compromises LLM
Safety" on OpenReview) — the PDF was behind a verification page and I could not read it,
so it is **unverified**. Likewise a norm-matched random unit vector control attributed to
"Behavioural Analysis of Alignment Faking" (arXiv:2605.27681) — the PDF did not extract, so
**unverified**. If either is real, it strengthens the case that random-direction controls
are both widespread and weak; neither is needed for any claim I make.

---

## What I could not support, and dropped

- **"On-`d` displacement is a better behavioural predictor than the board score."** Not
  supportable at n=7: r = +0.863 [+0.31, +0.98] vs +0.745 [−0.02, +0.96], the two predictors
  are themselves correlated at r = +0.953, Williams t = +1.77 (df=4, **p = 0.152**), paired
  bootstrap Δr = +0.115 [−0.05, +0.42] straddling zero. Excluded per the brief and confirmed
  by `_falsifier/recompute_result.md`.
- **"The `n` (no stance) key fixes degenerate-arm judging."** Zero uses in 96 ratings. The
  fix is a proposal. Reported as such in contribution 4 rather than as a result.
- **"The random-direction control is a novel critique."** Demoted to contribution 8 after
  finding SteerCheck. The geometry argument is correct and is someone else's.
- **"Judges agreeing to two decimal places is evidence of validity."** `prefix_eval.md`
  presents `anti_hostile` at −1.31 from both judges as strong agreement. The audit shows the
  two judges are running *different estimators* (one differences against a re-rated base,
  one against a reused base), and their base means differ by a near-constant 0.19, so the
  agreement is partly arithmetic. I do not advance judge agreement as a methodological
  contribution.
- **Anything about `anti_top`'s behavioural magnitude.** Withdrawn; used here only as an
  example of a measurement failing, never as a result.
- **A general claim that fixed baselines shrink effects.** Here they shrank significant
  arms by 13-37% and flipped no signs, but one arm (`anti_top` under Claude) grew by 29%.
  The correct general statement is that the estimator matters and must be chosen
  deliberately, not that it always shrinks.

---

## Summary table

| # | contribution | key number | prior work | status |
|---|---|---|---|---|
| 1 | Contrast drift on a fixed baseline, caused by call structure | +0.62 shift, p = 7.1e-07; r = −0.869 with true effect; corrections 13-37%, 0 sign flips | KNOWN-BUT-UNQUANTIFIED | demonstrated |
| 2 | Per-item records + cached generations + a claims ledger | 400 items re-judged for free; 125 claims checked, 124 agree | ALREADY-ESTABLISHED as principle | demonstrated |
| 3 | Judge reliability is per-arm, not global | 80-83% coherent arms vs 33-55% degenerate arm | KNOWN-BUT-UNQUANTIFIED | demonstrated, small n |
| 4 | Forced choice inverts over degenerate text | human 14/17 (82%, p = 0.013) vs judge 12% on identical pairs | KNOWN-BUT-UNQUANTIFIED | inversion demonstrated; fix not |
| 5 | Measure the manipulation, do not read it off the code | prefill-only, 9/17 positions, α = 6x local residual at one position | KNOWN-BUT-UNQUANTIFIED | demonstrated |
| 6 | Retraction needs a mechanical step | 0 retraction markers in 4 docs; 3 docs carry the number; 2 written after the withdrawal | process claim | demonstrated |
| 7 | Apply exclusion rules symmetrically | `pro_top` 71%/79% survives; `anti_top` collapses to 3/6, 4/9 | KNOWN-BUT-UNQUANTIFIED | demonstrated (by audit) |
| 8 | Random-direction nulls are the wrong control | \|cos\| = 0.0144/0.0057/0.0012 in 5120d | ALREADY-ESTABLISHED (SteerCheck) | not ours |

**One-line version for an outside practitioner.** If you are running a cheap behavioural
eval of a steering intervention: difference against a fixed baseline and store the
baseline's own rating per item; report judge-human agreement per arm rather than globally;
decide in advance what a degenerate output scores and keep one judge-free structural measure
alongside the judged one; diff the hooked layer's output to confirm what your intervention
actually did; and put withdrawn numbers in a file a script can check, because a note in the
prose will not stop them.
