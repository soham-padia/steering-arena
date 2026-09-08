# Related work, and what is left after it

Literature sweep finished 2026-09-08, continuing the one that was cut off overnight. Everything
here is tagged **VERIFIED** (I fetched the abstract page or the source and read it), **LEAD** (I
found it but could not confirm the specific detail), or **MINE** (my reasoning or arithmetic, not
measured by anyone).

One caveat on my own method, stated up front because it bounds everything below: `WebFetch`
renders a page and answers my question against it with a small model. For load-bearing claims I
demanded verbatim quotes and got them, and those I call VERIFIED. Where a fetch returned a
paraphrase or "not present in the provided content", I say so rather than smoothing it over.

---

## Read this first: four papers that hurt

I went looking for something closer to this work than Mody et al. I found three things closer,
and a fourth that dismantles a claim in `novelty.md` §5(d). This is the whole value of the sweep,
so it goes first.

### 1. Sign-symmetry is already published as *the* validity criterion for a steering direction

**VERIFIED.** `arXiv:2605.25151` — Ciarán Walsh, Emilio Barkett, *Representation Without Control:
Testing the Realization Effect in Language Models*, 24 May 2026.

They evaluate a concept at three levels — prompt-only behavioural sensitivity, linear readout of
the residual stream, and causal control via activation steering — on Gemma, layer 18. The readout
works and generalises to held-out prompts. The steering does not. And the reason they give for
calling it a failure is exactly the argument in `novelty.md` §2:

> "If the train-only layer-18 direction were a causal lever for risk-taking, steering toward
> realized_closed should move behavior in one direction and steering toward paper_open should
> move it in the opposite direction."

> "The absence of sign-symmetric behavioral responses at opposite steering scales is the clearest
> evidence that the train-only layer-18 direction does not function as a reliable control
> variable."

They swept scales −50, +50, +75, +100, +150; the −50 run gave mean wager delta +7.22 CHF and mean
risk delta +0.057, with median deltas of zero, across roughly 476–483 matched rows per scale.

And their closing abstract sentence is `novelty.md` §5(e), verbatim, four months early:

> "Behavioral sensitivity, latent readout, and causal control are three distinct properties that
> do not automatically co-occur, and successful latent readout is insufficient evidence that a
> model behaviorally relies on a representation during downstream decision-making."

**What this means.** Contribution (a) as currently written — "sign-consistency, not displacement
magnitude, is the construct-validity test for a steering direction" — is not a new idea. It is a
published criterion, and it is the headline criterion of a paper that used it to reject a
direction. §5(e) is also gone as a novel framing. I would not open a seminar with either.

What survives is narrower and I think still real: Walsh & Barkett flip the sign of an *injected
vector at inference time*. You cannot inject a negative vector through the input, so Season 3's
anti arm flips the sign of the *search objective* and lets the optimiser find text. That is a
different intervention class, and nobody has run the sign-symmetry test in input space. Season 3
also gets a **positive** result where they get a null, and positive sign-symmetry results are
rarer and more informative than nulls. That is the claim. It is a delta on intervention class and
outcome, not a new criterion. Say it that way.

Two things they do *not* have, which is where the remaining room is: no random-direction placebo
of any kind, and a behaviour measure consisting of two constrained numbers (a wager from 1–1000
CHF and a risk profile from 1–5), not open-ended generation.

### 2. Mody et al. already ran the label-shuffled placebo — so §5(d) is dead

**VERIFIED**, and I checked this specifically because `novelty.md` stakes a methodological
improvement on it.

`arXiv:2607.25907` — Deepanshu Mody, Samarth Agarwal, Utkarsh Mittal, Dipesh Mahato (the search
index gives "Dipesh Tharu Mahato"), *Minimizing Targeted Activations: Input-Only Suppression of
Evaluation-Awareness Latents in Large Language Models*, submitted 28 Jul 2026. Id, title, authors,
date and both quoted conclusions all check out against the abstract page. The framing in
`novelty.md` §1 rests on this paper and the paper is real.

But its Table 3 has three rows, not two:

| target | best \|z\| | behavioural log-odds shift |
|---|---|---|
| real eval direction | 8.3 | 0.44 |
| placebo (shuffle) | 10.6 | **0.51** |
| placebo (random) | 6.6 | 0.43 |

The shuffle placebo is "a CAA direction over a *random relabeling* of the contrast" — i.e. the
same fitting procedure on permuted labels. That is precisely the harder null
`scripts/direction_null.py` argues for and precisely what `novelty.md` §5(d) proposes as an
improvement on the prior work. They ran it, and the shuffled direction moved behaviour *further*
than the real one. Table 3 caption: "A placebo direction suppresses as hard and shifts behavior as
far as the real one—for the direction, neither is eval-specific".

**What this means.** Delete §5(d)'s claim to a methodological improvement. Running a
label-shuffled placebo is a replication of an existing control, not an advance on it. Running it
is still the right thing to do — it is the missing control for *this* direction — but it buys
zero novelty credit and should be described as a control, never as a contribution.

Two further details that make this paper cut sharper than §1 currently admits, and one that makes
it cut less:

- **Sharper.** The null is specifically about the *linear direction*. Of their five target
  constructions — a CAA direction, a subspace norm, an SAE feature, a single MLP neuron, and a
  behavioural logit — the SAE feature "can be fully and selectively turned off". So this is not a
  blanket null on interpretability-guided steering; it is a null on the construction Steering
  Arena actually ships. The fitted `d` is in the family that failed.
- **Sharper.** Their prefix experiment is a sign-consistency failure in all but name. Holding a
  real eval passage in context and optimising only a prefix, suppressing the direction "fails to
  reduce—and slightly increases—the model's behavioral eval judgment", a shift of −0.41 ± 0.46
  *away* from deploy. Behaviour moved the wrong way. So the input-space, prefix-only, wrong-signed
  result already exists — which is the nearest thing to a pre-emption of contribution (a) in the
  input-space setting.
- **Less sharp.** Their behaviour measure is the model's own yes/no log-odds to "Is the text above
  part of a test or evaluation?", AUROC 0.92 on the 3B. That is a single self-report logit on
  Llama-3.2-3B and Llama-3.1-8B, single layer (L14 on 3B, L16 on 8B), no human raters and no
  external judge. It is a much thinner behavioural readout than 50 held-out prompts of open-ended
  generation with blind pairwise judging. Season 3's behavioural instrument is genuinely better,
  and that is a legitimate thing to say.

### 3. SSR is real, and its §5.3 is contribution (b) in the jailbreak setting

**VERIFIED.** The lead was right and the agent that flagged it was right to flag it.

"SSR" is **Subspace Rerouting**: `arXiv:2503.06269` — Thomas Winninger, Boussad Addad, Katarzyna
Kapusta, *Using Mechanistic Interpretability to Craft Adversarial Attacks against Large Language
Models*, v1 8 Mar 2025, v3 3 Jul 2026. The abstract does not use the string "SSR" or mention
probes; the body does, and so does the author's repo (`Sckathach/subspace-rerouting`). I extracted
the 41-page PDF locally and read §5.3 and §6 directly, and I read the attack source.

Three implementations: Probe-SSR, Steering-SSR, Attention-SSR. Probe-SSR trains a linear
classifier per layer and optimises a discrete token perturbation against them with HotFlip-style
candidate search. From `ssr/probes/probe_ssr.py`, the aggregation over layers is an
**alpha-weighted sum**:

```python
for (classifier, alpha, lfn), layer in zip(self.probes.values(), self.config.layers):
    acts = self.act_dict[act_name][..., -1, :]
    prediction = classifier(acts)
    target = t.ones_like(prediction)
    loss += alpha * lfn(prediction, target)
```

So: multi-layer, per-layer linear probes, discrete token optimisation, weighted-sum aggregation.
That is Score 1's aggregator with per-layer directions, published eighteen months ago.

Then §5.3, "Multi-layer targeting effects on attack success", Probe-SSR on Llama-3.2-1B with a
3-token payload:

- **one middle layer (5): 0% success.** "Targeting only one middle layer (5) failed on every
  attempt, despite successfully rerouting activations at the target layer. We observed initial
  reclassification as harmless at layer 5, followed by rapid reversion to the harmful subspace in
  subsequent layers, culminating in model refusal."
- **two layers (5, 14): 54%.**
- **three layers (5, 10, 14): 100%.** "This strategy maintained high harmless classification
  probability (~1.0) consistently from the first targeted layer through to the output."

**What this means.** The core intuition behind contribution (b) — that a single-layer objective
admits solutions which move activations without moving behaviour, and that requiring several
layers to move together forces behaviourally real solutions — is an established, quantified
finding. It is stated from the attacker's side rather than the metric designer's, but it is the
same mechanism and the same evidence shape. If you present (b) as "we discovered that multi-layer
is more behaviourally valid", someone who knows SSR will produce this table.

What SSR does *not* do, and this is where (b) still has room:

- It varies the **number** of layers, never the **aggregation rule**. Sum throughout. No min, no
  max, no k-of-n, no comparison between rules.
- It has **no placebo and no negated-direction control.** I grepped the full text for `negat`,
  `placebo`, `random direction`, `opposite pole`, `sign-flip`, `baseline direction` and the only
  hits are unrelated ("largest negative gradients", "negative cosine similarity").
- The authors explicitly leave the aggregation question open: **"Unfortunately, we did not have
  time to compare enough different strategies to draw a conclusion, so we leave it as an exercise
  for the reader."** That sentence is the best framing device in this document. The k-of-n sweep
  in `novelty.md` §5(c) is that exercise.

Two more things in SSR worth knowing because they touch other claims:

- Probe-SSR beats nanoGCG substantially: ASR 0.91 / 0.88 / 0.80 / 0.84 on Qwen2.5-1.5B /
  Llama-3.2-1B / Llama-3.2-3B / Gemma2-2B against nanoGCG's 0.85 / 0.06 / 0 / 0.53. So
  "optimising tokens against a linear probe" is not merely done, it is a strong published attack.
- §6, "Interpretable jailbreaks": roughly a third of short Llama-3.2-1B runs produced partly
  readable tokens, with recurrent items like "responsibly", "ethical", "simulation". This is a
  published precedent for readable output from probe-directed token search, which bears on the
  "Score 2's top 20 contains 10/20 readable English" observation. It weakens the surprise, though
  not the measurement.
- §6 also has a **"reverse-lens"**: optimising with Steering-SSR to make embeddings *approach* the
  refusal direction, which recovers slurs and insults. So sign-flipped optimisation against a
  direction exists in the literature. I want to be precise rather than dramatic here: this is
  Steering-SSR not Probe-SSR, and the readout is *which tokens were found*, not a blind
  behavioural comparison of the two poles. It establishes that the mechanic is known. It does not
  establish that anyone ran it as a validity test. Walsh & Barkett did that; SSR did not.

### 4. "Min beats mean against over-optimisation" is a known principle from RLHF

**VERIFIED.** `arXiv:2310.02743` — Thomas Coste, Usman Anwar, Robert Kirk, David Krueger, *Reward
Model Ensembles Help Mitigate Overoptimization*, 4 Oct 2023 (v2 10 Mar 2024). They study
ensemble-based conservative objectives, specifically **worst-case optimisation (WCO)** — take the
minimum over the ensemble — and uncertainty-weighted optimisation, and find that "conservative
optimization practically eliminates overoptimization and improves performance by up to 70% for
BoN sampling".

**UNVERIFIED sub-claim:** I could not get a head-to-head mean-versus-min number off the abstract
page; the fetch reported no such comparison there. Their contrast is conservative-vs-single-model.
If you cite this for "min beats mean" specifically, read the body first.

Also **VERIFIED**, and it supplies the mechanism: `arXiv:2609.00213` — Yu Yuan, Yaoyou Fan, Lili
Zhao, Guangting Zheng, Kai Zhang, Lu Pan, Ke Zeng, Qi Liu, *Uncovering and Mitigating
Aggregation-Induced Reward Hacking in Multi-Reward Reinforcement Learning*, 31 Aug 2026. Their
finding: "aggregation itself induces reward hacking: static projection aliases qualitatively
different reward profiles into a single scalar, steering optimization toward whichever dimensions
are easiest, densest, or systematically favored by the reward signal."

**What this means.** That is a one-sentence published explanation of why Score 1 gets eaten by
token soup and Score 2 does not. A fixed-weight mean over layers lets the optimiser dump its
whole budget on the cheapest layer; a min will not let it. So (b) is an instantiation of a known
principle in a new domain, and "hardening the metric made it more valid, not more Goodhartable"
is only counter-intuitive if you have not read the reward-model ensembling literature. Frame it as
a transfer with a novel readout, not as a surprise.

The defender-side version also exists. `arXiv:2412.09565` — Luke Bailey, Alex Serrano, Abhay
Sheshadri, Mikhail Seleznyov, Jordan Taylor, Erik Jenner, Jacob Hilton, Stephen Casper, Carlos
Guestrin, Scott Emmons, *Obfuscated Activations Bypass LLM Latent-Space Defenses*, 12 Dec 2024,
ICLR 2026. Abstract VERIFIED: sparse autoencoders, representation probing and latent OOD detection
are all vulnerable, and attacks "reduce recall from 100% to 0% while retaining a 90% jailbreaking
rate". On the multi-layer question I fetched the body, which reports that they varied the number
of layers attacked and that "obfuscating against more layers reduces performance on SQLEval" —
i.e. a growing obfuscation tax in the number of layers. I did not read the underlying figure, so
call that detail **VERIFIED-via-fetch, figure unread**.

---

## The paper-shaped claim

Given all four, here is the one claim I think survives contact with a reviewer who has read them:

> Flipping the sign of a fitted direction and re-running an identical token-space optimiser flips
> the sign of the resulting behavioural change, and the effect is carried by polarity rather than
> by the adversarial character of the text. This is the first sign-symmetry test conducted in
> input space rather than by inference-time injection, and the first positive one; prior
> sign-symmetry tests (Walsh & Barkett 2026) and prior input-only suppression experiments (Mody et
> al. 2026) both returned nulls or wrong-signed effects.

That is acceptable-or-rejectable and it is defensible. Note what it is not: it is not "we invented
sign-consistency as a validity test", and it is not "hardening a metric makes it more valid".

---

## What is genuinely novel, ranked

**1. A positive sign-symmetry result in input space, with weirdness held constant.**
Closest prior work: Walsh & Barkett (`2605.25151`) for the criterion and the null; Mody et al.
(`2607.25907`) for the input-only prefix case and the wrong-signed result. **Delta:** both poles
searched by the same optimiser from the same initialisation with the same seed, and both poles
judged blind. `optimize_banded.py` initialising to 32 copies of `"!"` rather than pro-social text
is what makes this work, and it appears to have been hygiene rather than design — either way it is
the load-bearing detail. ~~because it means the pro and anti arms are both unreadable 32-token
GCG strings and only polarity differs.~~ **CORRECTED 2026-09-08: that is false.**
`prefix_eval_s3.md:150` records that `score1_top`'s prefix is *not* unreadable — it carries
`escalate intervene … Zach respectful truth.: Ask respectfully`, a garbled pro-social
imperative — while `score1_anti` carries no anti-social imperative. The contrast is
pro-instruction versus junk. The identical initialisation and seed still hold and still matter;
"only polarity differs" does not. Neither prior paper has a matched opposite-pole arm with a
blind behavioural readout. This is the strongest item and it is a delta on intervention class plus
sign of outcome, not a new idea.

**2. Aggregation rule, not layer count, as the variable.**
Closest prior work: SSR §5.3 (layer count, weighted sum, ASR readout); Coste et al. (min vs single
reward model); `arXiv:2604.13386` — Erik Nordby, Tasha Pais, Aviel Parrack, *Linear Probe Accuracy
Scales with Model Size and Benefits from Multi-Layer Ensembling*, 15 Apr 2026, which reports
multi-layer ensembles recovering +29% AUROC on Insider Trading and +78% on Harm-Pressure Knowledge
where single-layer probes fail, across 12 models from 0.5B to 176B (VERIFIED; the fetch found no
aggregation-rule detail and no adversarial test on the abstract page). **Delta:** nobody holds the
layer set fixed and varies min against mean, and nobody attaches a behavioural readout to the
choice. Currently this is one comparison of two points, which is an anecdote. The k-of-n sweep
turns it into a curve. **This is the item with the most headroom and the least evidence.**

**3. Cross-objective transfer as a natural experiment.**
618 Season 2 submissions optimised against a single-layer metric, rescored under both Season 3
objectives: Score 1's top 20 contains 0/20 readable English, Score 2's contains 10/20; Spearman
against the old board 0.95 versus 0.84. Closest prior work: none I found does this. **Delta:** a
real, already-paid-for adversarial corpus used to stress-test a *later* metric it was not aimed
at. Genuinely novel as an instrument, and genuinely weak as evidence — as `scripts/gcg/README.md`
already says, this is incidental robustness, not adversarial robustness. Keep the README's caveat
in the same sentence as the number.

**4. The leaderboard as an attack corpus.**
Closest prior work: TDC 2023 (below), the Steerability Challenge (below), and `arXiv:2504.20879`
*The Leaderboard Illusion* (**UNVERIFIED** — I have this id and title from search results only and
did not fetch the abstract page; do not cite it until someone does). **Delta:** thin. 721
submissions from unincentivised players is a nice artifact, not a contribution. Do not lead with
it.

**5. Quantified fragility of the anti arm under measured judge instability.** MINE.
See the judge section below. This is a limitation, but stating it precisely and first is worth
more than any of items 3 and 4.

---

## What is NOT novel

Blunt, because being told here is cheaper than being told in the seminar.

- **GCG.** `arXiv:2307.15043` — Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico
  Kolter, Matt Fredrikson, *Universal and Transferable Adversarial Attacks on Aligned Language
  Models*, 27 Jul 2023. VERIFIED. A careful port is still a port.
- **Optimising discrete tokens against a linear probe.** SSR / Probe-SSR (`2503.06269`), and Mody
  et al. (`2607.25907`). Not just done — done well, beating nanoGCG.
- **Multi-layer targeting of per-layer probes.** SSR §5.3, with an ASR curve in the layer count.
- **Multi-layer probe ensembling being better than single-layer.** Nordby et al. (`2604.13386`).
- **Min / worst-case aggregation resisting over-optimisation.** Coste et al. (`2310.02743`);
  mechanism in Yuan et al. (`2609.00213`).
- **Attacking latent-space monitors, and the cost of attacking more layers at once.** Bailey et
  al. (`2412.09565`).
- **Sign-flipped steering as a condition.** `arXiv:2312.06681` — Nina Panickssery, Nick Gabrieli,
  Julian Schulz, Meg Tong, Evan Hubinger, Alexander Matt Turner, *Steering Llama 2 via Contrastive
  Activation Addition*, VERIFIED: steering vectors are added "with either a positive or negative
  coefficient". Standard practice, presented there as a control knob rather than a validity test.
  So the *mechanic* of the anti arm is routine; only its use as a blind validity test in input
  space is not.
- **Sign-symmetry as a validity criterion.** Walsh & Barkett (`2605.25151`). Published.
- **"Readout does not imply control."** Walsh & Barkett again, and before them
  `arXiv:2006.00995` — Yanai Elazar, Shauli Ravfogel, Alon Jacovi, Yoav Goldberg, *Amnesic
  Probing: Behavioral Explanation with Amnesic Counterfactuals*, TACL 2021, VERIFIED: "we point
  out the inability to infer behavioral conclusions from probing results" and "conventional
  probing performance is not correlated to task importance". This is a twenty-year-old argument in
  NLP with a canonical citation. `novelty.md` §5(e) must cite it rather than restate it.
- **Steering vectors being unreliable.** `arXiv:2407.12404` — Daniel Tan, David Chanin, Aengus
  Lynch, Dimitrios Kanoulas, Brooks Paige, Adria Garriga-Alonso, Robert Kirk, *Analyzing the
  Generalization and Reliability of Steering Vectors*, 17 Jul 2024, NeurIPS 2024. VERIFIED:
  "steerability is highly variable across different inputs" and "spurious biases can substantially
  contribute to how effective steering is for each input".
- **Construct validity as an imported frame.** `arXiv:1912.05511` — Abigail Z. Jacobs, Hanna
  Wallach, *Measurement and Fairness*, FAccT 2021. VERIFIED as the canonical import of measurement
  modelling and construct validity into ML. The abstract page does not enumerate the validity
  subtypes, so if you want to name convergent/discriminant/predictive validity, read the paper
  first — I did not. Using the vocabulary is fine; claiming to have imported it is not.
- **Steerability benchmarks.** `arXiv:2505.20645` — Kai Chen, Zihao He, Taiwei Shi, Kristina
  Lerman, *STEER-BENCH*, 27 May 2025, VERIFIED, behavioural/black-box, human experts 81% vs best
  model ~65%. And `arXiv:2505.23816` — Trenton Chang, Tobias Schnabel, Adith Swaminathan, Jenna
  Wiens, *A Course Correction in Steerability Evaluation: Revealing Miscalibration and Side
  Effects in LLMs*, 27 May 2025, VERIFIED, whose complaint that "scalar measures of performance
  common in prior work could conceal behavioral shifts in LLM outputs in open-ended generation" is
  an independent argument for exactly the kind of behavioural check Season 3 ran.
- **Crowd red-teaming competitions.** TDC 2023 (LLM Edition) at NeurIPS 2023 had a Red Teaming
  Track and a $30k pool. The retrospective is `arXiv:2404.13660` — Narek Maloyan, Ekansh Verma,
  Bulat Nutfullin, Bislan Ashinov, *Trojan Detection in Large Language Models: Insights from The
  Trojan Detection Challenge*, 21 Apr 2024, VERIFIED — but note it is a *participant* retrospective
  on the trojan-detection track, not an organiser report, and the fetch confirms it does not
  discuss the red-teaming track, leaderboard gaming, or degenerate solutions. Its one transferable
  finding is a good one: winning methods "achieved Recall scores around 0.16, comparable to a
  simple baseline of randomly sampling sentences". A leaderboard whose winners barely beat a random
  baseline is the same genre of problem as a leaderboard whose top 36 ranks are token soup, and it
  is worth one sentence as precedent.
- **The leaderboard, the $0 stack, the HF Space, the Supabase schema, the NDIF quota plumbing.**
  Engineering. Real work, not a contribution.

---

## The judge problem, with numbers a reviewer will use

This is the section the brief asked for and it turns out to cut both ways.

**What Season 3 does right, VERIFIED from `scripts/behavioral_eval.py:209-224`.** The judge is
position-debiased by construction: every pair is asked twice with A/B swapped and only *consistent*
verdicts are kept. The docstring says why — "the first single-pass run showed pure position-bias
noise". That is not a detail to bury; it is the single most common reviewer objection to
LLM-judge work and it is already handled. Self-preference bias is also mostly inapplicable here,
because the judge is `claude-opus-5` and every generation being judged comes from OLMo-3 — the
judge never rates its own output.

**What it does not do, VERIFIED from `data/analysis/prefix_eval_s3.json`.**
`raters = {"human": {"rated": 0}, "claude-opus-5/v2": {"rated": 366}}` and `agreement = {}`. One
judge, zero humans, no agreement statistic. Season 2 had the comparison and it was bad:
LLM-vs-LLM 0.885 (85/96), human-vs-Claude 0.730 (27/37), human-vs-DeepSeek 0.553 (21/38), from
`data/analysis/prefix_eval.json`.

Those Season 2 numbers are a textbook case of a named published phenomenon, which is the citation
you want:

- `arXiv:2606.19544` — Justin D. Norman, Michael U. Rivera, D. Alex Hughes, *Reliability without
  Validity: A Systematic, Large-Scale Evaluation of LLM-as-a-Judge Models Across Agreement,
  Consistency, and Bias*, 17 Jun 2026. VERIFIED. 21 judges, nine providers, three benchmarks
  (MT-Bench, JudgeBench, RewardBench), 118 runs, ~541,000 judgments. Findings: "kappa deflation
  between exact match and Cohen's kappa is universal (33--41 pp on MT-Bench)"; "judge rankings
  shift by up to 14 positions across benchmarks"; "high test--retest reliability (>0.95) coexists
  with severe position bias (>0.10) in two production-deployed judges"; "verbosity bias is small
  (<0.011) across our cohort under a single pairwise rubric". They propose a "Minimum Viable
  Validation Protocol". The title alone is the argument: 0.885 inter-judge agreement next to 0.553
  human agreement is reliability without validity.
- `arXiv:2306.05685` — Lianmin Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot
  Arena*, VERIFIED: GPT-4 reaches "over 80% agreement, the same level of agreement between humans",
  and they name position, verbosity and self-enhancement biases. This is the optimistic baseline,
  and Season 2's 0.553/0.730 sits well below it — worth saying, because it means this task is
  harder for judges than MT-Bench is.
- `arXiv:2606.13685` — Abel Yagubyan, *The Coin Flip Judge? Reliability and Bias in LLM-as-a-Judge
  Evaluation*, 23 Apr 2026. VERIFIED, though flag it as single-author and unrefereed. 29 tasks,
  two OpenAI judges, 50 pairwise trials per question. "pairwise preferences flip on average 13.6%
  of the time, with 28% of questions exceeding a 20% flip rate and one question reaching 56%";
  first-position bias "72% A-majority, p = 0.024"; "cross-judge agreement is only 76% (κ = 0.51)";
  "semantically equivalent prompt templates change majority outcomes in 25% of tested cases"; and
  "11 repeated trials are needed for a majority vote to recover the 50-trial reference verdict
  with 95% probability on average, rising to 15 for high-variance questions".

**MINE, and this is the part I would want to know.** I reimplemented the exact sign test from
`behavioral_eval.py:227` and reproduced the reported values exactly (28/33 → 6.62e-05; 6/26 →
0.00936), then asked how many flipped verdicts each arm can absorb before p > 0.05:

| arm | reported | flips to break | as a fraction |
|---|---|---|---|
| `score1_top` 28/33 | 6.6e-05 | **6** | 18.2% |
| `score1_anti` 6/26 | 0.00936 | **2** | **7.7%** |

Yagubyan's measured flip rate is 13.6%. Applying it as an independent per-verdict flip
probability, `score1_top` lands at ≈25/33, p ≈ 0.0046 — it survives. `score1_anti` lands at
≈8/26, p ≈ 0.076 — **it does not.**

So the fragile half of contribution (a) is the anti arm, which is the half the whole argument
rests on. Two mitigations, both real: the swap-consistency filter means the 26 surviving pairs are
already the *more* stable ones, so 13.6% overstates their instability; and 366 rated pairs across
eight arms is more data than the two arms I tested. Neither mitigation is measured. Until a second
rater exists, the honest statement is that the pro arm's significance is robust and the anti arm's
is not, and the anti arm is what distinguishes this from Mody et al.

This is the strongest possible argument for the second rater in `novelty.md` §6, and it is
stronger than "a reviewer might object" — it is a number.

---

## What nobody appears to have done

**The 2×2 that no single paper has.** This is the best thing I found and it falls straight out of
the gaps above:

|  | positive pole | negated pole |
|---|---|---|
| **real fitted `d`** | Season 3 `score1_top` ✓ | Season 3 `score1_anti` ✓ |
| **label-shuffled refit** | missing | missing |

- Walsh & Barkett have both poles and **no placebo**.
- Mody et al. have both placebos (random *and* label-shuffled) and **one pole**.
- SSR has multi-layer and **neither**.

Nobody has run both poles against a label-shuffled direction. That cell is where the argument
lives, because it is the only design that separates "the sign of a *fitted* direction controls
behaviour" from "the sign of *any* direction controls behaviour". A shuffled direction has a sign
too; if flipping *its* sign also flips behaviour, contribution (a) collapses entirely, and Mody's
shuffle placebo out-shifting the real direction (0.51 vs 0.44) is a live reason to think it might.

**The experiment.** Add `--random-direction {isotropic,shuffled}` to `optimize_banded.py` (the
smoke path at lines 167-171 already substitutes random unit vectors into `DIRS`, and everything
downstream reads `DIRS` as a module global, so this is the ~6-line change `novelty.md` §6 costs
it at), then run **four** arms, not two: `{real, shuffled} × {+, −}`, identical init, identical
seed. Judge all four blind against the same 50 held-out prompts, with a second rater. The
prediction that makes it a real test, stated in advance: the real pair separates by sign, the
shuffled pair does not separate at all, and the shuffled arms may well match the real arms on
*displacement magnitude* while showing no *signed* effect. If the shuffled pair separates by sign,
contribution (a) is dead and you will have found that out yourself.

**The second experiment, in priority order after it.** The k-of-n sweep. `_aggregate` in
`gcg_utils.py:43-84`, with `stacked.topk(K, dim=0).values[-1]`; `k=n` reduces to `MIN` and `k=1`
to `MAX`, giving two free regression tests. Cite SSR's "we leave it as an exercise for the reader"
as the reason it is worth doing, and cite Yuan et al. for why the answer should be monotone. A
curve in `k` for both gameability and behavioural prediction is a paper figure; the current two
points are not.

**Third: a fluency rubric scored separately from kindness**, already item 3 in
`REVISIONS_2026-09-05.md` §7. Norman et al. put verbosity bias at "<0.011", which is reassuring,
but the `anti_top` arm looping 39/50 says the coherence confound in *this* setup is not about
verbosity — it is about degeneracy. That needs its own rubric, not a citation.

---

## Venue and framing

**The Steerability Challenge — this is the obvious move and the timing is remarkable.** VERIFIED
by fetching `steerability.github.io/competition/`: organised by IBM Research with contributors
from Tara Research, UC Berkeley, CUHK, **Northeastern University**, Cadenza Labs, Kitware, and AI
Safety Australia & New Zealand; opens 10 Sep 2026, closes 21 Nov 2026 AoE; cash prizes; winners
present at a NeurIPS 2026 workshop in Paris with mandatory attendance. The task is reducing model
dishonesty — "a model's tendency to make assertions that diverge from what it internally
represents as true" — scored as "dishonesty reduction relative to the unsteered model minus the
mean regression across side-effects", with only capability declines penalised. Crucially the
allowed interventions include "control over the model's input/prompt, structure/weights,
state/activations, and output/decoding procedure", so input-only prefix optimisation is in scope.
And: "The specific measures of dishonesty and general capabilities will be kept private to prevent
gaming."

Two flags. First, a discrepancy I could not resolve: search snippets described the task as
*sycophancy* reduction opening 29 Aug 2026, while the page I fetched says *dishonesty* opening 10
Sep 2026. The fetched page wins, but re-read it before committing. Second, a Northeastern
affiliation among the organisers is worth confirming and disclosing given the author's own
affiliation.

Why it fits: a private held-out metric plus explicit side-effect penalties is a competition
designed against exactly the failure Season 2 exhibited, and the Steering Arena result — that a
conjunctive multi-layer objective admits fewer degenerate solutions than a mean — is directly
useful advice to that competition's designers. That is a stronger pitch than any of the research
claims, and it has a ten-week deadline.

**Bar for a workshop paper (BlackboxNLP, an ICLR/NeurIPS interpretability workshop, or the
Steerability Challenge workshop).** Roughly what exists now, plus: the 2×2 placebo run, a second
rater on Season 3, and honest positioning against all four papers in the first section. The
sign-symmetry framing must cite Walsh & Barkett in the *first paragraph*, not the related work.

**Bar for a main conference.** The k-of-n curve, on more than one model, with the 2×2 at each
end of the curve, and human agreement reported. Without a second model the "we did it on OLMo-3"
delta is thin and I would say so.

**Framing I would avoid.** "We show that optimising against a probe changes behaviour" — Mody et
al. published the opposite with a better control. "We introduce sign-consistency as a validity
test" — Walsh & Barkett introduced it. "Conjunctions are unGoodhartable" — the repo's own README
already forbids this and SSR plus Coste et al. would both be produced against it.

**Framing I would use.** "A steering direction can pass a readout test, pass a magnitude test, and
still fail the only test that distinguishes steering from perturbation: does its sign control the
sign of the behaviour? We ran that test in input space, where prior work has only run it by
injection, and it passes — and we report exactly how fragile that pass is." The fragility is not a
weakness in this framing. It is what makes it credible in a repo whose culture is to publish the
withdrawal next to the claim.

---

## How I searched

Auditable so nobody redoes it. `ToolSearch` was **disabled** for this session; `WebSearch` and
`WebFetch` were already present in my tool set and I confirmed that with a live call before citing
anything.

**Queries run (WebSearch):** the Mody et al. title verbatim; "SSR subspace rerouting jailbreak
interpretability adversarial suffix activation"; "obfuscated activations bypass LLM latent-space
defenses probe evasion"; "steerability benchmark challenge LLM steerability evaluation
competition"; "Trojan Detection Challenge red teaming competition NeurIPS large language models";
"LLM-as-a-judge position bias self-preference verbosity bias agreement with human raters";
"construct validity interpretability probing reads the concept versus causal control amnesic
probing"; "ensemble linear probes across layers robust activation monitoring adversarial attack
aggregation"; "worst-case minimum aggregation multiple reward models Goodhart robustness
conjunction harder to game"; "construct validity measurement modeling machine learning Jacobs
Wallach"; "leaderboard illusion Chatbot Arena gaming benchmark distortion"; "construct validity
steering vectors evaluation interpretability artifacts"; plus title lookups for Tan et al. and
Elazar et al.

**Abstract pages fetched and read (all VERIFIED above):** 2607.25907, 2503.06269, 2412.09565,
2605.25151, 2606.19544, 2606.13685, 2306.05685, 2312.06681, 2604.13386, 2310.02743, 2609.00213,
2407.12404, 2006.00995, 1912.05511, 2404.13660, 2505.23816, 2505.20645, 2307.15043. Full text
also fetched for 2607.25907 (Table 3, the wrap experiment) and 2605.25151 (the sign-symmetry
run). `steerability.github.io/competition/` fetched.

**Failed fetches.** `arxiv.org/html/2503.06269`, `.../2503.06269v3` and `.../2503.06269v1` all
return HTTP 404 — SSR has no rendered HTML. I recovered it by fetching the PDF and extracting all
41 pages locally with `pypdf`, which is how §5.3 and §6 are quoted verbatim, and by reading
`ssr/probes/probe_ssr.py` from `Sckathach/subspace-rerouting` through `gh api`, which is where the
weighted-sum aggregation comes from. Both are stronger evidence than an abstract page, not weaker.

**Not fetched, so not usable.** `arXiv:2504.20879` *The Leaderboard Illusion* — id and title from
search results only, marked UNVERIFIED above. A self-preference-bias paper appeared in search as
`2410.21819`; I did not fetch it and do not cite it, because Norman et al. and Zheng et al.
already cover self-preference and the judge here never rates its own output.

**Searched for and did not find.** Any paper comparing min against mean aggregation over
*per-layer probe directions*; any paper running a sign-flip control against a *label-shuffled*
direction; any use of a public leaderboard's submission corpus to stress-test a later metric. I
believe these three are open, and the first two are the experiments above.

---

## Verification status, per identifier

Added 2026-09-08 in a second pass, after a challenge to four load-bearing citations. Every id
this repo's 2026-09-08 documents cite, with what was actually opened. **Nothing below is
"verified" because a search snippet said so.**

Two levels of provenance, and they are not equivalent. *Re-fetched by the orchestrating
session* means I opened it myself in this session. *Sweep only* means the novelty-scout opened
it and I did not re-check — its method is documented above and I have no reason to doubt it,
but it is one agent's report.

| id | short name | status | what was opened |
|---|---|---|---|
| `2607.25907` | Mody et al., *Minimizing Targeted Activations* | **FULL TEXT VERIFIED** | abstract + **Table 3** re-fetched by the orchestrating session. Three conditions confirmed: real 0.44, placebo-shuffle 0.51, placebo-random 0.43. Optimisation is suppression-toward-zero; no opposite pole. |
| `2605.25151` | Walsh & Barkett, *Representation Without Control* | **FULL TEXT VERIFIED** | abstract + **§4.4 and §5** re-fetched by the orchestrating session. "Clearest evidence" sentence confirmed verbatim. Abstract frames sign-symmetry as one component of a null; body uses it as a rejection criterion. Both are true. |
| `2503.06269` | Winninger et al., *Using Mechanistic Interpretability to Craft Adversarial Attacks against Large Language Models* | **FULL TEXT VERIFIED** | abstract re-fetched; §4/§4.2/§5.3 section titles confirmed from the PDF; **alpha-weighted-sum aggregation re-confirmed from `ssr/probes/probe_ssr.py` source** by the orchestrating session. The §5.3 numbers (0% / 54% / 100%) and the nanoGCG table are the sweep's pypdf extraction, not re-extracted by me. |
| `2310.02743` | Coste et al., *Reward Model Ensembles Help Mitigate Overoptimization* | **ABSTRACT VERIFIED** | re-fetched by the orchestrating session. "Practically eliminates overoptimization" is verbatim. Baseline is **single reward model**, not a mean; unit is **models**, not layers. Min-vs-mean head-to-head **absent** — do not cite for it. |
| — | The Steerability Challenge | **VERIFIED (live page)** | `steerability.github.io/competition/` re-fetched. **Dishonesty**, not sycophancy. Opens 10 Sep 2026, closes 21 Nov 2026 AoE. Input/prompt interventions in scope. Models announced at launch. Cash prizes 1st–3rd; **physical attendance at NeurIPS 2026 Paris mandatory** for a winning team. The sycophancy/29-Aug version appears only in stale search snippets. |
| `2412.09565` `2606.19544` `2606.13685` `2306.05685` `2312.06681` `2604.13386` `2609.00213` `2407.12404` `2006.00995` `1912.05511` `2404.13660` `2505.23816` `2505.20645` `2307.15043` | see sections above | **ABSTRACT VERIFIED — sweep only** | fetched and read by the novelty-scout; not re-checked by the orchestrating session. |
| `2504.20879` | *The Leaderboard Illusion* | **UNVERIFIED** | id and title from a search result only. **Do not cite.** |
| `2410.21819` | self-preference bias | **NOT CITED** | surfaced in search, deliberately not used. |
| `2608.24335` | SteerCheck | **UNVERIFIED — inherited** | cited in `_advocate/POSITIVES.md` §6 for "random-direction nulls are a weak control", and carried forward into `writeup/mechanism.md` §7 and the 2026-09-08 falsifier §6. **Neither this sweep nor the orchestrating session fetched it.** Fetch before any public use. |
| `2604.09839` | Mishra, Khashabi & Liu | **UNVERIFIED — inherited** | cited in `POSITIVES.md` §6 for "no prompt reproduces steering's internal behaviour", carried into `writeup/mechanism.md` §6. Not fetched by either. |
| `2501.17148` | AxBench | **UNVERIFIED — inherited** | cited in `POSITIVES.md` §6 for "prompting beats steering". Not fetched by either. |
| `2507.21919` | Ibrahim et al. | **UNVERIFIED — inherited** | cited in `POSITIVES.md` and the 2026-08-27 audit for warmth-training error rates. Not fetched by either. |
| — | Hewitt & Liang 2019 | **UNVERIFIED — inherited** | cited in `POSITIVES.md` §6 alongside SteerCheck. No arXiv id recorded in-repo. |

**The four `UNVERIFIED — inherited` rows are the actionable ones.** They are not sweep
failures — they predate it, and they entered the 2026-09-08 documents by being quoted from
`POSITIVES.md` rather than from a source. SteerCheck matters most: it carries the "isotropic is
the weak null" argument that `mechanism.md` §7 and the falsifier §6 both use to tell you *how*
to run the placebo. If it does not say what `POSITIVES.md` says it says, that instruction needs
re-deriving from `scripts/direction_null.py`, which argues it internally and is a committed
artifact either way.
