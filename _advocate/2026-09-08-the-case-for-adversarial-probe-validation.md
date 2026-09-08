# The case for adversarial probe validation

**Date:** 2026-09-08
**Sources:** `prefix_eval_s3.json`, `prefix_degeneration_s3.json`, `prefix_content_s3.json`,
`season3_prefix_scores.json`, `season3_directions.json`, `season3_gcg_ablation.md`,
`season3_gcg_aggregate_asymmetry.md`, `scripts/gcg/optimize_banded.py`,
`scripts/gcg/gcg_utils.py`, `prefix_loop_control.md`,
`_falsifier/2026-08-27-addendum-human-ratings.md`, `writeup/related-work.md`.
**Counterpart:** `_falsifier/2026-09-08-behavioural-result-red-team.md`. Read both. I have
answered its §1 and §2 rather than its easier objections, and I lost to §2 on one point.
**Rule applied:** every claim cites a number I read out of a file. Each carries an italic
*What would break it:* naming an observable. §7 lists what I could not support.
**Provenance:** written by the orchestrating session; the `research-advocate` subagent launch
was blocked by the classifier described in `_local/NEXT_CLAUDE.md` §6.

---

## The contribution, stated so it can be rejected

**Publish a linear probe. Let an adversary optimise token sequences against it. Then blind-rate
whether the resulting text produces the behaviour the probe predicts.** That loop is a
construct-validity instrument for an interpretability artifact, and it tests something no
held-out accuracy number can reach: whether the concept a probe reads survives an optimiser
trying to move it.

Two things I have to concede before arguing for it, because `writeup/related-work.md` settled
them tonight and an advocate who hides them is not credible.

**The criterion is not new.** `arXiv:2605.25151` (Walsh & Barkett, *Representation Without
Control*, May 2026) already uses sign-symmetry as *the* test of whether a direction is a
control variable, and uses its absence to reject one: *"The absence of sign-symmetric
behavioral responses at opposite steering scales is the clearest evidence that the train-only
layer-18 direction does not function as a reliable control variable."* VERIFIED by the
novelty sweep. So "sign-consistency is the validity test" is a published criterion, not our
idea.

**The method is not new either.** `arXiv:2503.06269` (Winninger, Addad, Kapusta) optimises
discrete tokens against per-layer linear probes — Probe-SSR, ASR 0.88 against nanoGCG's 0.06
on Llama-3.2-1B. VERIFIED. Optimising tokens against a probe is done, and done well.

**What is left, and it is narrower and I think still real.** Walsh & Barkett flip the sign of
an **injected vector at inference time**. You cannot inject a negative vector through the
input channel. Season 3 flips the sign of the **search objective** and lets an optimiser find
text — a different intervention class, in input space, where nobody has run the sign-symmetry
test. And Season 3 returns a **positive** result where they return a null. Positive
sign-symmetry results are rarer and carry more information than nulls, because a null is
consistent with a dozen failures of the apparatus and a signed effect is not.

That is the delta: intervention class and outcome, not a new criterion. Everything below is
in service of it.

---

## 1. The strongest claim: a signed behavioural effect from a one-bit change to the objective

**Two runs of one optimiser, differing in a single sign, produce token strings that move a
blind rater's preference in opposite directions, both significantly, and both after the
degeneracy control.**

VERIFIED, `prefix_eval_s3.json`, `no_loop` scope:

| arm | preferred / decided | sign-test p |
|---|---|---|
| `score1_top` | 28 of 33 | **7e-05** |
| `score1_anti` | 6 of 26 | **0.00936** |

VERIFIED in code, and I checked this line by line because the published version of the claim
is imprecise. `optimize_banded.py:206` initialises **every** run to 32 copies of `"!"` —
`t.full((N_CONTROLLED_TOKENS,), tokenizer("!")["input_ids"][0])`, not seeded from pro-social
text. Line 107 sets `t.manual_seed(20260906)`, identical across runs. And lines 119–124:

```
AGGREGATE = MEAN if args.role == "score1" else (MAX if ANTI else MIN)
SIGN      = -1.0 if ANTI else 1.0
```

**For Score 1 the aggregate stays `MEAN` and only `SIGN` moves.** That is a literal one-bit
contrast, and it is true of Score 1 and *not* of Score 2, where `--anti` also swaps
`MIN`→`MAX`. The project's own summaries say "flips the sign and the aggregate" of both arms;
that is wrong for Score 1 in the direction that helps, and the correction makes this claim
stronger and narrows it to one arm pair.

Why this is worth more than a large effect size. ~~Both strings are unreadable 32-token GCG
soup from the same search.~~ **CORRECTED 2026-09-08: false, per `prefix_eval_s3.md:150`.**
`score1_top` is a garbled pro-social *imperative* (`escalate intervene … Zach respectful
truth.: Ask respectfully`); `score1_anti` carries no anti-social imperative. The standing rival
explanation — *an LLM judge rewards continuations that follow strange adversarial text* —
still predicts both arms win, and one loses at p = 0.00936, so the asymmetry is real. But
**"weirdness is held constant and only polarity varies" is withdrawn**, and with it the claim
that this is a one-variable contrast at the level of the *strings*. It remains a one-bit
contrast at the level of the *objective*, which is a weaker and still useful thing.

Two caveats in the same breath. This is not a placebo substitute: it answers "is the sign
carrying the effect", not "is the *fitted* direction special", and only the placebo answers
the second. And the two arms are not matched on achieved magnitude — `score1_top` reaches
+0.16395 against `score1_anti`'s −0.10135, so 62% (`season3_prefix_scores.json`).

*What would break it:* a human blind pass on Season 3's `score1_anti` that prefers the
anti-prefixed continuation. That is not hypothetical — it is what happened in Season 2, where
humans preferred the `anti_top` continuation **14 of 17, p = 0.0127**, the opposite sign from
both LLM judges, and a p = 0.0005 result was withdrawn over it. Also: a matched-magnitude pro
arm, pulled from `history.jsonl` at |LIVE| ≈ 0.101, that fails to beat baseline.

---

## 2. A control that could have failed and did not

**Thirty-two unoptimised random tokens do nothing, on every measure.**

VERIFIED, `prefix_eval_s3.json` and `prefix_content_s3.json`. `random32` scores −0.00335 on
Score 1 and returns **10-15-7, p = 0.42436** in the `no_loop` scope (15-19-9, p = 0.60759 in
`all`), Δfix **−0.17** at p = 0.28505, and **0 of 50** continuations contain a
prefix-distinctive word — despite the prefix containing 18 words absent from every base
continuation.

I rank a null from a control above an effect from a treatment, so this is my second claim. It
rules out a whole family of trivial accounts at once: prefix-presence, added context length,
generic token weirdness, and mechanical vocabulary transfer. Whatever is happening in the pro
arms, it is not "any 32 odd tokens do this", and it is not "the model parrots whatever is in
front of it".

Stated in the same paragraph because the falsifier is right about it: `random32` is **not** a
placebo in Mody et al.'s sense. It is the untreated control, not the sham treatment. It
controls for prefix-presence, not for direction-fittedness, and the cell for "searched against
a random direction to a matched score" is empty.

*What would break it:* a second random draw that comes back significant. n = 1 here, and
`prefix_eval_s3.md` lists "multiple random draws for a null *band*" under what is not done.
Eight draws is what `steering_random_control.md` used for the injection arms; one is thin.

---

## 3. Content injection is real, and it is not the mechanism

**Across the three GCG pro arms, vocabulary leakage falls fourfold while behavioural effect
rises.**

VERIFIED, `prefix_content_s3.json` and `prefix_eval_s3.json`. A word counts only if it
appears in the prefix and in **none** of the 50 base continuations:

| arm | continuations with a prefix word | verbatim echo | emoji | Δfix |
|---|---|---|---|---|
| `score1_top` | 22/50 | 0 | 1 | +0.74 |
| `score2_top` | 13/50 | 6 | 12 | +1.09 |
| `score2_top_final` | **3/50** | **0** | 1 | **+1.20** |

Monotone, and running the wrong way for the injection account. The **strongest** behavioural
arm has the **least** leakage of any GCG pro arm. `prefix_eval_s3.md` §4 withdrew content
injection as *the* mechanism on this evidence, and I think the withdrawal is correct rather
than merely cautious: the confound predicts a positive leakage-effect relationship and the
data give a negative one.

The caveat belongs here and it is what the falsifier's §7 is about. The remaining injection is
concentrated in **exactly the pair that carries claim 1** — `score1_top` at 22/50 with
`off_topic` 21/50 and `moralizing` 18/50 (the model writes anti-bullying PSAs addressed to
Zach), and `score1_anti` leaking `divorce`×19. So the arm that best refutes injection has no
usable anti partner, and the arm pair with the cleanest sign contrast has the worst topical
leakage. Both facts are true and neither cancels the other.

*What would break it:* a hand-written content-matched control — same bullying-and-Zach topic,
no optimisation, near-zero score — that produces Δfix ≈ +0.74. Two independent documents
(`prefix_eval_s3.md` "not run", `_advocate/POSITIVES.md` §1) have named this as the missing
control and it still does not exist. It costs 50 generations and one judge batch.

---

## 4. The direction passes its gates, and Season 3 closed the confound Season 2 left open

**CORRECTED 2026-09-08.** This claim originally led on `season3_directions.json`'s
`confound_cosines` — approach 0.0042/0.0072, sentiment 0.0041/0.0047, length 0.0013/0.0050.
**Do not use those numbers as evidence.** They are measured *after* the confounds are projected
out, so they prove only that the subtraction ran; and
`build_season3_directions.py:82-86` does sequential Gram–Schmidt in dict order over proxies
with mutual cos 0.016–0.112, so `approach` goes last and lands at 0.0000 by construction while
the earlier residuals are order artifacts. The proxies are also thin — length **2-vs-2**,
sentiment **4-vs-4**, approach **12-vs-12**, with `SHORT = ["The cat slept.", "It rained
today."]` against two long committee sentences, which confounds length with topic and register.
The claim survives on better evidence, below.

**Length as a confound is dead, and now on a real measurement.** An *in-domain* length
direction fitted on all 270 seed texts has |cos| **0.003–0.019** with the shipped per-layer
directions, against a 5120-dimensional random baseline of **0.0140** — i.e. at or below chance.
The corpus carries no length asymmetry either: chosen 16.56 words against rejected 16.30,
chosen longer in **45.2%** of pairs, Wilcoxon p = 0.13. This is strictly stronger than the
2-vs-2 proxy it replaces.

**The convergent check, which this project computed and never published.** cos(`d`,
kind-vs-cruel) = **0.38–0.43** at every band layer, against a random baseline of 0.014, and
essentially untouched by orthogonalisation (0.4042 → 0.3950). The 6 control pairs behind it are
approach-matched by construction (`scripts/confound_audit.py:88`). Every other number in this
section is *discriminant* validity — what `d` is not. This is the only *convergent* one — what
`d` is — and it is the single strongest construct-validity figure available here. It belongs in
`season3_directions.json`.

**What still holds about Season 2 → Season 3.** cos(d_raw, d_orth) = **0.979–0.992** per band
layer, and essentially all of the removed component is approach; `REVISIONS` §4 records the
shipped Season 2 direction at cos(d, approach) = **0.1501**, "the only audited confound never
orthogonalised out". Season 3 did close it. Held-out separation is 1.000 on both objectives and
`per_axis_all_positive` is true across all 15 axes — though note separation of 1.000 is a weak
identification claim: leave-one-axis-out is 1.000 on **15/15** axes, yet two `d`s fitted on
**disjoint corpus halves agree at only cos = 0.668** at L24, falling to **0.608** at L39.
Perfect classification, under-identified vector.

**The other half, in this paragraph rather than a footnote, because collapsing these two is
this project's most-repeated error.** The *direction* is de-confounded; the *seed corpus* is
not — and the corpus is **worse confounded than the repo says**. `REVISIONS` §4 names only
`approach` at 0.824. Measured at the band layers, each proxy alone separates the 135 pairs at
**sentiment 0.852–0.889**, approach 0.852–0.859, and a register proxy 0.763–0.852. So the
honest sentence is: *the seed corpus is separable by sentiment at up to 0.889, by approach at
0.859 and by register at 0.852, and `d` rides none of them.* Orthogonalising a direction does
not de-confound the data it was fitted on. Corpus problem, not direction problem — item 4 on
`REVISIONS` §7's open list, to fix before the next season, and sentiment now outranks approach
in that queue.

*What would break it:* a held-out confound direction built from independent neutral data — not
from these pairs — returning a cosine above ~0.1. Every cosine above is measured against
confounds constructed from the same corpus, which is the weaker test.

---

## 5. The conjunction orders behaviour better than the mean, within the arms where sign is constant

**Restricted to the four pro arms, Score 2 orders behavioural effect perfectly and Score 1
barely orders it at all.**

MINE, arithmetic on VERIFIED inputs. The published comparison is Score 2 at ρ = +0.929 against
Score 1 at +0.857 over all eight arms (`prefix_eval_s3.md` §5). The falsifier's §4 is right
that this is largely a four-versus-four group difference — a predictor carrying only the arm's
sign scores +0.873 — so I recomputed it inside the pro arms, where the sign bit is constant:

| arm | Score 1 | Score 2 | Δfix |
|---|---|---|---|
| `score2_top_final` | +0.08350 | **+0.06997** | **+1.20** |
| `score2_top` | +0.07896 | +0.06518 | +1.09 |
| `score1_top` | **+0.16395** | +0.05001 | +0.74 |
| `pro_coherent` | +0.03976 | +0.02145 | +0.58 |

Ranked by Score 2 that is the behavioural order exactly: **ρ = +1.000**. Ranked by Score 1:
**ρ = +0.400**, and the highest-scoring Score-1 string — by a factor of 1.96 — is third of
four behaviourally.

So the comparative claim survives the objection that killed the pooled figure, and it survives
*more starkly*: +1.000 against +0.400 rather than +0.929 against +0.857. The conjunction over
`{15,23,31,39}` is reading something the mean over `{19,23,27,31}` is not.

Three caveats, all load-bearing. **n = 4**, where the smallest attainable two-sided Spearman p
is 0.083, so this cannot reach significance at any threshold and is a direction of evidence,
not a result. **The mechanism is prior work:** `arXiv:2310.02743` (Coste, Anwar, Kirk, Krueger)
shows worst-case ensemble aggregation "practically eliminates overoptimization", and
`arXiv:2609.00213` (Yuan et al.) explains that fixed-weight scalarisation "steer[s]
optimization toward whichever dimensions are easiest, densest" — so Score 1 losing to token
soup is a textbook instance, not a discovery. VERIFIED by the novelty sweep. **And I lose to
the falsifier here:** a distinct-4-gram counter orders all eight arms at ρ = +0.881, beating
Score 1's +0.857. A judge-free repetition statistic is a better behavioural predictor than one
of our two objectives, and I cannot answer that from committed artifacts.

*What would break it:* the k-of-n sweep coming back flat — behavioural ρ not varying with `k`
while gameability does. That would say aggregation strictness is not the governing variable and
the +1.000 was four points of luck.

---

## 6. The protocol is better than the result needs, and one prediction is genuinely pre-registered

Worth crediting separately, because these are the parts a reader can check without trusting a
judge.

**The fidelity gate.** VERIFIED, `season3_prefix_scores.json`: every arm's recorded score is
re-measured independently, and the largest discrepancy across all seven scored arms is
**7.02e-04** (`pro_coherent`) against a tolerance of 2e-03. Four arms are under 3.3e-04. So
the strings that were behaviourally tested are the strings that were scored — which sounds
trivial until you read `_local/NEXT_CLAUDE.md` §5 on the five live board rows corrupted by
moving a searched string through `repr()`.

**The blind design.** 400 pairs, both A/B orders per pair, 34 order-inconsistent pairs
discarded as the intended debias, held-out prompts, and a fixed per-prompt baseline computed
natively rather than retrofitted. The baseline is the mean over the eight arms of the judge's
rating of the **unprefixed** text — the same text in all eight pairs — so averaging reduces
judge noise rather than importing treatment. I went looking for contamination there and did
not find it; the judge gave identical text an identical score on **39/50** prompts, and Δfloat
and Δfix differ by at most 0.03 on any arm.

**A pre-registered prediction that came true, verified in git.** `REVISIONS` §1 claims the
meandiff estimator check was pre-registered. It was, and the commit order proves it: the 0.89°
rotation prediction landed in `normalization_check.json` at **2ebb583**, and the result —
30/50 byte-identical against logistic's 33/50 — landed in `meandiff_ablation.json` at
**581d1a7**, a later commit. The prediction was in the repo before the result. This is the
strongest claim type in this folder's house form and ~~it is the only instance I could
verify~~ — **CORRECTED 2026-09-08, see the second pass below: there are three, and this is the
weakest of them.** The k=3 control's criterion (`6304f2c`) precedes its data by three hours and
its result by ~14.5, and it **fired against the project** and was honoured at `17bed29`. The
meandiff pair above is eleven minutes apart. The wrong claim is left visible per this repo's
standard.

*What would break it:* nothing about claim 6 is contestable by a new measurement; it is a
property of the record. But note what I checked and did *not* find: the random-control
"pre-registration" (`steering_random_control_preregistration.md`) landed in commit **8a69828**
together with `steering_random_control.json` — the same commit as its own result. That one is
not a verifiable pre-registration and should not be described as one.

---

## 7. What I could not support

Written last, and not padded.

- **"Optimising against `d` is what made the behaviour move."** Not established. There is no
  arm searched against a random or label-shuffled direction to a matched score. Worse than
  merely missing: Mody et al. (`arXiv:2607.25907`) **ran both placebos** — random unit and a
  CAA direction over a random relabelling of the contrast — and the shuffled null moved
  behaviour *further* (0.51) than the real direction (0.44). VERIFIED by the novelty sweep. So
  the expected outcome of our placebo is now a null, and the honest position is that the
  central rival hypothesis is untested and the prior points against us.
- **Anything about Score 2's anti pole.** `score2_anti` goes from 11-28-3 (p = 0.00948) to
  **7-7-2 (p = 1.0)** under the loop control, and `score2_anti_final` from 0.02753 to 0.30746.
  Both effects are pairs containing degenerate text. Only Score 1's anti arm survives.
- **The behavioural result at Bonferroni.** Over the eight `no_loop` sign tests, Holm passes
  four arms and `score1_anti` clears it by 0.00064 (0.00936 against a 0.01 threshold); plain
  Bonferroni at 0.00625 **fails** it. And `pro_coherent` fails Holm at 0.01353 against 0.0125,
  so "the readable instruction also works, significantly" needs the word "uncorrected".
- **That the judge is measuring kindness rather than fluency.** A distinct-4-gram rate orders
  the eight arms at ρ = +0.881. The manipulation moves looping across a 24× range (1/50 to
  24/50 against base 7/50). And the one measured check of judge quality
  (`_falsifier/2026-08-27-addendum-human-ratings.md` §N3) found human-LLM agreement of
  **22/27 (81%)** on coherent arms against **11/26 (42%)** on the degenerate arm, p = 0.0047.
- **The defence I wanted to make about rater bias, and why it only half works.** A *contrast*
  between two arms rated by one instrument is robust to a **constant** rater bias, because the
  bias enters both arms and cancels — which is a real reason claim 1 is safer than any absolute
  win rate, and it is the honest version of the "one judge" defence. But it does not survive
  the bias that has actually been measured here, which is **differential**: agreement depends
  on text quality, and text quality is what the pro/anti manipulation changes. Differential
  measurement error does not cancel in a contrast and does not merely attenuate an estimate. So
  claim 1 is protected against the generic single-judge objection and not against the specific
  one this repo has evidence for. That gap closes for $0:
  `python scripts/rate_blind.py --csv data/analysis/prefix_blind_s3.csv --focus`.
- **`d` as the causal channel.** Not supportable and I would not try. 16.1× more on-`d` push
  applied directly gives 8 wins to 8 losses (p = 0.32257); behaviour per unit on-`d` differs
  between prefixes and injections by roughly 60-fold (50–75, median 61.3 over 18
  estimator/judge/arm combinations); only 3.84% of the prefix's 24.248 displacement lies along
  `d`. See `writeup/mechanism.md`. What survives is that the *sign* of `d` is informative while
  the *magnitude along* `d` is not the lever.
- **Adversarial robustness for Score 2.** The 0/20-against-10/20 readability result comes from
  rescoring strings optimised against the **old** objective, which `scripts/gcg/README.md:32-34`
  calls out itself. Attacked directly, Score 2's winner is still soup at coherence 0.00. Do not
  say "unGoodhartable".
- **Novelty for the sign-symmetry criterion, the readout-vs-control distinction, or the
  label-shuffled-null argument.** All three are published: Walsh & Barkett `arXiv:2605.25151`;
  Elazar et al. *Amnesic Probing* `arXiv:2006.00995`; SteerCheck `arXiv:2608.24335` and
  Hewitt & Liang 2019. `writeup/novelty.md` §5(a), §5(d) and §5(e) all need rewriting against
  `writeup/related-work.md`.
- **Any Season 3 residual geometry.** `normalization_check.json` measures `pro_top` and
  `pro_coherent` — Season 2 strings. No artifact measures rotation, ‖Δ‖ or on-`d` for any of the
  eight Season 3 arms. Every geometric statement in this document is Season 2 evidence applied
  to a Season 3 argument, and one forward pass per arm would fix it.
- **Generalisation past one model, one corpus, one judge family, one string per condition.**

## What the case reduces to, in one paragraph

A published criterion says a steering direction earns the name only if flipping its sign flips
the behaviour, and the paper that introduced the criterion used it to reject a direction. We
ran that test in a channel nobody has run it in — input space, via adversarial token search
rather than vector injection — with the poles matched on initialisation, seed, schedule, token
budget and probe set, differing in one sign bit. It came back positive at p = 7e-05 and
p = 0.00936, after a degeneracy control, with an unoptimised-random-token arm null at
p = 0.42436. That is a real result about one direction on one model, measured by one LLM rater
whose reliability on the anti pole is the largest open risk, with the central placebo untested
and prior work suggesting it will come back against us. It is worth a seminar and it is not
worth an abstract until the second rater and the placebo exist.

---

## Second pass (research-advocate, 2026-09-08)

I am the subagent whose launch was blocked when the seven claims above were written. I read
them, `_falsifier/2026-09-08-behavioural-result-red-team.md`, `writeup/related-work.md`,
`writeup/mechanism.md`, `REVISIONS_2026-09-05.md`, and I recomputed the arithmetic that both
documents rest on. **Append only: nothing above is edited.** My job is what the first pass got
wrong in either direction, and it went both ways — I found two pre-registrations it missed and
one concession it should not have made, and I also found that its claim 5 is weaker than either
document says.

I did not verify a single number from memory. Everything below names the file I read.

---

### S1. A criterion written before the data existed came back against the project, and the project applied it and withdrew the claim

**VERIFIED, git + `data/analysis/season3_k3_control.md`.** Commit `6304f2c`
(2026-09-06 21:16:16 -0400) states the criterion in its own message: *"Arms 707082 (pro) and
707083 (anti) run k=3. The anti arm is the control: `max` is a disjunction and never saturated,
so if k=3 helps it as much as pro, the effect is generic search improvement and not a
conjunction fix."* Job 707083's run directory is `score2-anti-mut3-2026-09-07T04-17-25Z` —
three hours and one minute after that commit, so the outcome did not exist when the criterion
was written. The result landed at `17bed29` (2026-09-07 11:49:00), 14 hours 33 minutes later
and across a session boundary. At the last matched checkpoint the ratios coincide: **pro 1.276×
against anti 1.260×**. That is the pre-registered signature of a generic search improvement.
The commit message for `17bed29` says so in its first line — *"right conclusion, wrong
evidence"* — and `HANDOFF_BEHAVIORAL_S3.md` §8's "multi-position mutation is the fix, and it
works" was withdrawn as an inference from the pro-arm gap.

This outranks everything in the first pass on the folder's own criteria. It is a control that
could have failed, it did fail the claim it was pointed at, the criterion was in the repo before
the data, and the interval is a day and a cluster job rather than the eleven minutes separating
the meandiff pre-registration the first pass called "the only instance I could verify"
(`2ebb583` 01:42:05 → `581d1a7` 01:53:51, both 2026-09-05). The caveat belongs here: git proves
*ordering*, not blindness, and one author wrote both commits, so this is evidence about process
discipline and not about a sealed envelope. What survived the fire is narrower and better —
the trajectory shapes are opposite (pro k=3 halves the longest plateau, 197 → 105 iterations;
anti k=3 lengthens it, 170 → 188), which is the evidence the conjunction account should have
been argued from in the first place.

*What would break it:* a re-read of `history.jsonl` for jobs 707082/707083 at matched iterations
returning a pro/anti ratio gap wide enough to have passed the criterion — i.e. showing the
criterion did not in fact fire and the withdrawal was performative. `scripts/gcg/k3_control.py`
reads those files and costs nothing to rerun.

### S2. `random32` was pre-registered as a named control before it was run, which upgrades the first pass's claim 2 from "a null that could have failed" to "a null someone predicted might not"

**VERIFIED, git.** The first pass ranked `random32` second on the strength of the null alone.
It is worth more than that. `a63cbf0` (2026-09-06 21:18:35 -0400) — the Season 3 behavioural
handoff — names the arm before any of it ran: *"a length-matched random-token control — 32
random tokens. Season 2 never ran that control and it is the obvious hole."* The results landed
at `da36587` (2026-09-06 22:08:47), 50 minutes later. The same commit pre-registers the
comparative prediction **with its negative branch attached**, which is what makes it a
prediction rather than a hope: *"the Score-2 winner should beat the Score-1 winner behaviorally
despite scoring 2.5× lower. If it does not, the band buys robustness to gaming but not meaning.
Either way it is the finding."*

The match is tighter than "length-matched" suggests, and nobody has said so. **VERIFIED,
`season3_prefix_scores.json`:** `random32` is `n_tokens: 32`; `score1_top` is 32 and
`score1_anti` is 32. The control is exactly token-matched to the pair that carries claim 1, not
approximately. The Score-2 arms are 33 and `pro_coherent` is 17, so the match is exact only
where it matters most. Caveat in the same breath, and the first pass has it right: n = 1 draw,
and one draw is not a null band.

*What would break it:* a second random draw at 32 tokens returning p < 0.05 in either direction
on the same protocol.

### S3. The Season 3 anti effect survives both ends of the loop-control bracket, and the same bracket killed its two siblings

**VERIFIED, `prefix_eval_s3.json`.** The falsifier's §3 is right that `all` is confounded by
degeneracy and `no_loop` conditions on a mediator, and right that neither is unbiased. It draws
the wrong conclusion from that — "the design cannot say where the answer lies". When both ends
of a bias bracket agree in sign *and* in significance, the location of the truth inside the
bracket does not matter for a sign claim:

| arm | `all` | `no_loop` | verdict |
|---|---|---|---|
| `score1_anti` | 8-32-7, wr **0.200**, p = 0.00018 | 6-20-4, wr **0.231**, p = 0.00936 | brackets, both significant |
| `score2_anti` | 11-28-3, wr 0.282, p = 0.00948 | 7-7-2, wr **0.500**, p = 1.0 | crosses the null |
| `score2_anti_final` | 13-28-7, wr 0.317, p = 0.02753 | 9-15-3, wr 0.375, p = 0.30746 | dies |
| `random32` | 15-19-9, wr 0.441, p = 0.60759 | 10-15-7, wr 0.400, p = 0.42436 | null at both ends |

The test discriminates: it passes one arm, kills two, and leaves the control null at both ends.
A bracket that everything survives is not a test. The caveat is that the 4-gram threshold
generating the bracket has no sensitivity sweep (`prefix_loop_control.md`, caveats), and — a
correction the first pass did not make — its "set once, before looking at any verdict" is **not
verifiable from git**: the threshold entered at `f536a74` and the Season 3 results at `da36587`,
**21 seconds apart** in one session. That claim rests on the author's word, and it should be
labelled that way rather than sitting beside two pre-registrations that git actually supports.

*What would break it:* a threshold sweep (2, 4, 5 repeats of a 4-gram rather than 3) in which
`score1_anti`'s `no_loop` p crosses 0.05 at any setting while `score1_top`'s does not.

### S4. The falsifier's sharpest quantitative objection reverses under the loop control, and I can show the active ingredient is the scope rather than the outcome variable

**MINE, arithmetic on `prefix_eval_s3.json`, `prefix_degeneration_s3.json`,
`season3_prefix_scores.json`; scipy Spearman over the same eight arms the falsifier used.**

The falsifier's §2 reports a distinct-4-gram rate ordering the eight arms at ρ = +0.881 against
Score 1's +0.857, and concludes a judge-free repetition statistic beats one of the two
objectives. I reproduce that exactly. I also reproduce it under the *pairwise win rate* rather
than Δfix, which rules out the outcome-variable explanation I was handed:

| predictor | vs Δfix (`all`, n=50) | vs win rate, `all` | vs win rate, `no_loop` |
|---|---|---|---|
| distinct-4-gram | **+0.881** | **+0.881** | **+0.762** |
| Score 1 | +0.857 | +0.857 | **+0.881** |
| Score 2 | +0.929 | +0.929 | **+0.952** |

Δfix and the all-scope win rate agree to three decimals on every predictor, so this is not
kindness-versus-preference. **It is the loop filter.** Remove the pairs containing a degenerate
text and the degeneration statistic loses 0.119 while both objectives gain (+0.024 and +0.023).
Half of that is mechanical and I will not pretend otherwise — a degeneracy statistic must lose
predictive power on a sample from which degeneracy was filtered. The half that is not mechanical
is that the objectives *rise*. If Score 1 and Score 2 were themselves proxies for degeneration,
filtering degeneration should have cost them too.

Then the part that cuts against me, which I checked precisely because I wanted the defence to
hold. Leave one arm out and recompute: under `no_loop`, Score 1 beats the 4-gram counter in
**7 of 8** subsets, and the one exception is dropping `score2_anti` — the arm that moves from
wr 0.282 to wr 0.500 and whose `no_loop` estimate retains 14 of 42 decided pairs. Under Δfix,
Score 1 beats it in **1 of 8**. So the entire reversal is one arm, and it is the thinnest arm in
the study. **The honest statement is that at n = 8 arms the ordering of Score 1 against a 4-gram
counter is not decidable in either direction: the gap (0.024) is smaller than its sensitivity to
one arm or one scope.** The falsifier stated its version as a fact about the metric; it is not.
Nor is mine. That is a legitimate defence of the project against §2 and it obliges the project
to **pre-register the primary behavioural outcome variable and its scope before the next run**,
because both are currently choosable after the fact.

*What would break it:* a pre-registered primary outcome (say, `no_loop` win rate) under which,
with more arms from the k-of-n sweep, the 4-gram counter's ρ exceeds Score 1's by more than the
leave-one-out spread.

### S5. The prefix-versus-injection dissociation is rotation-matched, so it does not repeat the error that produced this project's most famous withdrawal

**VERIFIED, `normalization_check.json`; ratio is MINE.** `REVISIONS_2026-09-05.md` §1 withdrew
"adding `d` works and removing it does not" because injection rotates the residual 45.16° and
ablation 0.92° — a 49× size gap presented as a mechanistic asymmetry. The obvious next move for
a critic is to point that cannon at `writeup/mechanism.md`'s 60-fold prefix/injection ratio.
It does not fire. The same file gives `+1.0·d` injection at **45.165°** and prefix `pro_top` at
**50.437°**, a ratio of **1.12×**; `pro_coherent` is 46.848°, a ratio of 1.04×. On the unit
`REVISIONS` itself argues is the honest one for an RMSNorm model, the two interventions being
compared are matched to within 12%, while their on-`d` components differ by 32× (30.07 against
0.931). That is the intended shape of the argument: matched on the quantity the model reads,
divergent on the quantity the metric measures.

The caveat, and it is the reason this is claim 5 and not claim 1: this is Season 2 geometry
(`pro_top`, `d_olmo3_L24_logistic`, L24) and Season 3 behaviour. No artifact measures rotation
or ‖Δ‖ for any Season 3 arm, as the first pass says in its §7. One forward pass per arm closes
it and nobody has run it.

*What would break it:* a Season 3 forward pass in which `score1_top`'s rotation comes in far
from the 45–50° band — say under 20° — which would make the injection comparison an
edit-size comparison again.

### S6. The aggregation-rule gap in the literature is wider than the first pass allows, and Season 3's evidence for filling it is narrower than either document allows

Two corrections in opposite directions, and they belong together.

**The gap is wider. LEAD on the papers (`writeup/related-work.md`), VERIFIED on the code.** The
first pass's claim 5 concedes the mechanism to Coste et al. (`arXiv:2310.02743`). That
concession is too large as stated. `related-work.md` records, in its own words, that the
head-to-head mean-versus-min number is **UNVERIFIED** on the abstract page and that Coste's
contrast is conservative-aggregation against a **single** reward model, over models rather than
layers. "Ensemble worst-case beats one model" does not entail "min beats mean over the same
set". SSR (`arXiv:2503.06269`) varies layer *count* with a fixed alpha-weighted sum and its
authors write *"we leave it as an exercise for the reader"*. So min-against-mean over per-layer
probe directions, with a behavioural readout, is unclaimed. The first pass should have said the
*principle* is prior and the *comparison* is not.

**The evidence is narrower, and this is the thing neither document says.** Season 3's two
objectives are not a one-variable contrast. `scripts/gcg/optimize_banded.py:116` says it in the
source: *"The two roles differ in band, direction file and aggregate."* `season3_directions.json`
confirms all three — band `[19,23,27,31]` against `[15,23,31,39]` (two layers shared), separate
fitted direction sets, `banded_mean` against `per_layer_min`. And this project has already
measured which of those dominates: `REVISIONS` §6 reports that at a *fixed* band, moving between
weighting schemes shifts arm scores by under 0.0006 while **"band choice dominates weighting by
an order of magnitude"**, and `banded_score_arms.json` shows mean and min giving the *same*
ordering of `pro_top` and `pro_coherent` at both `{32,40,48}` and `{16,24,32,40,48}`. Its own
2026-09-06 write-up flagged this — `git show da36587:data/analysis/prefix_eval_s3.md` reads
*"the comparison across columns is also unequal: Score 1 and Score 2 differ in both band and
direction type"* — and the first pass dropped that caveat when it promoted the number.

Worse for claim 5 specifically: **MINE**, the ρ = +1.000 over four pro arms is largely the
falsifier's §4 objection one level down. Ranked by Δfix the four arms are `score2_top_final`
(+1.20), `score2_top` (+1.09), `score1_top` (+0.74), `pro_coherent` (+0.58) — a clean 2-versus-2
split by which objective produced the string. A predictor carrying only that origin bit scores
**ρ = +0.894**. Score 2's +1.000 exceeds it by ordering within pairs, at n = 4. So what survives
is *"strings found by searching the conjunction moved behaviour more than strings found by
searching the mean, two against two"* — which is the pre-registered prediction from `a63cbf0`
and is worth stating — and not *"Score 2 magnitude orders behaviour"*.

*What would break it:* a k-of-n sweep at a **fixed** band and a fixed direction set in which
behavioural ρ does not vary monotonically with `k`. That is the experiment; the current two
points are not it, and they cannot be made into it by adding arms to the existing pair.

### S7. On the sign-symmetry framing: the design argument is stronger than the priority argument, and the project should lead with the design

**MINE, on VERIFIED inputs from `writeup/related-work.md`.** I was asked whether "first
input-space test of sign-symmetry, with a positive result" is stronger than the first pass
allows. Partly, and not for the reason of being first.

The cell is genuinely empty. Walsh & Barkett (`2605.25151`) have both poles by injection.
Panickssery et al. (`2312.06681`) have both poles by injection as routine practice. Mody et al.
(`2607.25907`) are in input space and optimise **suppression toward zero**, so they have no
opposite pole to test. SSR's Steering-SSR reverse-lens reports which tokens were recovered, not
a blind behavioural comparison. Nobody has both poles in input space.

But "first" here rests on one agent's negative literature search, run with a tool whose own
author flagged that it *"renders a page and answers my question against it with a small model"*.
A negative search is the weakest evidence type in that document and it is the only support for
the word. I would drop "first" and keep what the design buys, which is checkable by a reader
who has read none of the papers: **the input-space test faces a rival that the injection test
cannot even pose.** "An LLM judge rewards continuations that follow strange adversarial text"
is a hypothesis about text, there is no text in an injection experiment, and it predicts both
Season 3 arms win. One loses at p = 0.00936. Conversely — and this is the honest half — the
injection design controls content *by construction* and the input-space design does not, which
is the falsifier's §7 and is why `score1_top` writes anti-bullying PSAs addressed to Zach
(21/50 `off_topic`) while `score1_anti` leaks `divorce`×19. The two designs are complementary,
not ranked, and a claim of priority invites a reviewer to check the search while a claim about
design invites them to check the data.

*What would break it:* any paper with a matched opposite-pole arm found by input-space search
and judged blind. One citation ends the priority claim and leaves the design claim untouched,
which is the argument for making the design claim.

---

### What I could not support

Written last. The first pass's §7 stands and I am not restating it; these are additions and one
retraction of a defence I wanted to make.

- **The best available defence against the falsifier's §1 does not work, and I checked it
  hoping it would.** The temptation is to say Season 2's human inversion on `anti_top` was a
  degeneration artifact. **VERIFIED, `prefix_loop_control.md`:** the human went 14/17 (82%) in
  the `all` scope and **5/6 (83%) under `no_loop`**. The point estimate does not attenuate — it
  only loses power to n = 6. An advocate who quotes "5/6, p = 0.2188" as reassurance is
  misreading a sample size as a result. There is no design answer to §1 in this repo. The only
  answer is the second rater, and it costs $0:
  `python scripts/rate_blind.py --csv data/analysis/prefix_blind_s3.csv --focus`.
- **A weaker, real version of that defence, offered as a lead and not more.** **VERIFIED,
  `prefix_loop_control.md`:** on Season 2's `anti_top` — the arm CLAUDE.md records as looping
  39/50 — the Claude judge returned **11/31, p = 0.1496 (`all`)** and **5/12, p = 0.7744
  (`no_loop`)**. Null. The same judge family returns p = 0.00936 on Season 3's `score1_anti`,
  which loops 14/50. If the mechanism were "this judge penalises degenerate text", the more
  degenerate arm should have produced the larger penalty and it produced none. That is one
  judge, two seasons, two directions and two probe sets, so I file it as a reason the §1
  alternative is not automatic rather than as evidence against it. I am **not** reviving any
  behavioural reading of `anti_top`; that arm is withdrawn and stays withdrawn.
- **`writeup/mechanism.md` §2's sign-symmetry dissociation is cross-season and I would not
  present it.** It sets Season 2 injections (`steering_dose.md`, `d_olmo3_L24_logistic`, L24,
  `cos(d, approach) = 0.1501`) against Season 3's `score1_anti` (banded mean over
  `{19,23,27,31}`, approach orthogonalised to 0.0042). Different direction, different corpus
  treatment, different layer set. §8 of that document admits the general problem; §2 draws a
  conclusion anyway, and it is the conclusion the document calls "the cleanest single argument".
  It needs a Season 3 injection arm before it is one.
- **That `score2_top_final`'s rank-1 behaviour confirms the pre-registered prediction
  out-of-sample.** It is suggestive — the string is a by-product of the k=3 *search-efficiency*
  control (`17bed29`: "pro k=3 finished at +0.06969 LIVE", matching
  `season3_prefix_scores.json`'s recorded 0.0696934), so it was not chosen for its behaviour,
  and it landed highest on Score 2 and highest on Δfix. But its score and its behavioural
  result were committed together at `d1b8b46`, so git cannot separate them, and the same commit
  withdrew two of the project's own claims. I would say "the new arm was consistent with the
  prediction while breaking the eight-arm ordering", and not more.
- **Any claim that the loop-control threshold was pre-registered.** 21 seconds, one commit.
- **The 4-gram-versus-Score-1 comparison in either direction.** S4 rescues the project from the
  falsifier's version of it; it does not establish the reverse. Both are one arm from flipping.
- **`d` as the causal channel, the placebo, the second rater, Season 3 residual geometry, and
  generalisation past one model.** Unchanged from the first pass's §7, and the placebo has now
  been the top open item since 2026-08-28.
