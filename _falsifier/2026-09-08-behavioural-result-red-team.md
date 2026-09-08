# Red team: do probe-optimised prefixes change behaviour as the probe predicts?

**Date:** 2026-09-08
**Repo state:** `0f2533e` (main), tree dirty (`_falsifier/verify_result.json`, untracked `writeup/`)
**Claim under test:** "Token sequences found by optimising against a linear probe change
OLMo-3-32B's behaviour in the direction the probe predicts." This is the Season 3 headline
and the thing intended for a seminar slide.
**Method:** adversarial read of `prefix_eval_s3.json`, `prefix_degeneration_s3.json`,
`prefix_content_s3.json`, `season3_prefix_scores.json`, `season3_directions.json`,
`season3_gcg_ablation.md`, `season3_gcg_aggregate_asymmetry.md`, `scripts/gcg/optimize_banded.py`
and `scripts/gcg/gcg_utils.py` against each other, plus arithmetic I did myself on the
committed JSON. Every recomputation is marked and shown.
**Status:** not applied to `data/analysis/`. The documents named still read as they read.
**Provenance:** written by the orchestrating session, not by the `research-critic` subagent.
That launch was blocked by the session safety classifier described in
`_local/NEXT_CLAUDE.md` §6, along with three other subagent launches. Read this as one
adversarial pass, not as the specialist's.

---

## Verdict

**Partially supported, and the supported part is narrower than the headline.**

One contrast survives everything I could throw at it: `score1_top` beats its unprefixed
baseline and `score1_anti` loses to it, under the loop control, from an optimiser whose only
difference between the two runs is `SIGN = ±1.0`. That is a real, controlled, signed effect
and it is the thing to put on the slide.

Almost everything else I checked is either confounded with text degeneracy, unprotected
against multiple comparisons, or resting on a single LLM rater whose only recorded check
against humans **inverted the sign on exactly this kind of arm**. The Score-2 arms' anti
half does not survive the loop control at all (p=1.0). ~~The rank correlation offered as
evidence that Score 2 is a good behavioural proxy is beaten by a 4-gram counter.~~
**CORRECTED 2026-09-08: overstated. A 4-gram counter never beats Score 2 under either outcome
variable (+0.881 vs +0.929; +0.762 vs +0.952) — it beats *Score 1*, and only in the `all`
scope, by ρ = 0.024, a gap whose sign flips under leave-one-arm-out. See §2.**

Two things I expected to find wrong and did not: the fixed baseline is constructed correctly
(§11), and the direction's confound orthogonalisation took (§12). I say so because an audit
that finds only problems is not measuring.

---

## 1. The sign-flip claim rests on the anti arms, and the only human check ever run on an anti arm inverted the sign

**Damage if right: fatal to the headline framing.** This is the objection I would open with
if I were reviewing.

VERIFIED, `prefix_eval_s3.json`: Season 3 has one rater, `claude-opus-5/v2`, 366 of 400
pairs. `"human": {"rated": 0}`. `"agreement": {}`. There is no second instrument of any
kind.

VERIFIED, `_falsifier/2026-08-27-addendum-human-ratings.md` §N3 and
`data/analysis/prefix_eval.md`: in Season 2, human-versus-LLM agreement on decided pairs was
**27/37 = 73%** for Claude and **21/38 = 55%** for DeepSeek — and it was strongly
arm-dependent. Claude 80% on `pro_top`, **55% on `anti_top`**. DeepSeek 83% on `pro_top`,
**33% on `anti_top`**. Pooled, the contrast is 22/27 (81%) against 11/26 (42%),
**p = 0.0047**. Inter-judge agreement was 85/96 = 89%, so the two LLMs agreed with each
other and not with the human.

The part that does the damage. VERIFIED, `prefix_eval.md:299-369` and
`prefix_loop_control.md`: on Season 2's `anti_top` arm the human preferred the
**anti-prefixed** continuation **14 of 17 (82%), p = 0.0127** — the *opposite sign* from
both LLM judges — and the LLM-based `anti_top` result (−0.86, p = **0.0005**) was
**withdrawn** on the strength of it.

So the position is this. The Season 3 sign-flip argument needs the anti arms to lose. The
anti arms are the arms where LLM judges have been measured at 33–55% agreement with a human,
i.e. at or near chance on a two-alternative forced choice. And the one time a human rated an
anti arm, the human said the opposite. A p = 0.0005 result has already been retracted over
precisely this gap, in this repo, thirteen months of project-time ago.

**Alternative explanation:** the anti arms lose because a single LLM judge dislikes
degenerate text, and human raters — who in the one recorded instance read the same
degenerate text as *more* considerate — do not share that preference.

**Evidence that would separate it:** human blind ratings on Season 3's anti arms, same
protocol, same CSV.

**Does it exist in the repo?** No, and this is the cheapest gap in the project.
`data/analysis/prefix_blind_s3.csv` is a 400-row blind CSV already wired for
`python scripts/rate_blind.py --csv data/analysis/prefix_blind_s3.csv --focus`; `stats`
picks the new label up automatically. Cost: evenings, $0, no GPU. Until it exists, every
Season 3 number is single-instrument, and the instrument has a known arm-dependent failure
that lands on the arms carrying the argument.

---

## 2. A 4-gram counter predicts the judge nearly as well as the probe does

**Damage if right: it does not kill the effect, it kills the "Score 2 is a good behavioural
proxy" claim.** MINE, computed tonight from two committed artifacts. I believe this is not
recorded anywhere in the repo.

`prefix_degeneration_s3.json` gives `distinct4_mean` per arm — a mechanical
distinct-4-gram rate, no judge, no model. `prefix_eval_s3.json` gives
`delta_mean_fixed_baseline`. Rank the eight arms by each:

| arm | `distinct4_mean` | Δfix | d4 rank | Δfix rank | d |
|---|---|---|---|---|---|
| `score1_top` | 0.984 | +0.74 | 1 | 3 | −2 |
| `score2_top` | 0.979 | +1.09 | 2 | 2 | 0 |
| `score2_top_final` | 0.959 | +1.20 | 3 | 1 | +2 |
| `pro_coherent` | 0.957 | +0.58 | 4 | 4 | 0 |
| `random32` | 0.906 | −0.17 | 5 | 5 | 0 |
| `score2_anti_final` | 0.831 | −0.43 | 6 | 6 | 0 |
| `score1_anti` | 0.805 | −0.83 | 7 | 8 | −1 |
| `score2_anti` | 0.682 | −0.48 | 8 | 7 | +1 |

Σd² = 10, n = 8, so **ρ = 1 − 60/504 = +0.881**.

`prefix_eval_s3.md` §5 reports Score 2 at **+0.929** and Score 1 at **+0.857** over these
same eight points. So the ordering of behavioural effect is predicted at +0.881 by a
statistic that never looks at the probe, the direction, the band, or the objective. It beats
Score 1 outright.

**Alternative explanation:** the blind judge is substantially rating fluency, the pro
prefixes suppress looping and the anti prefixes induce it, and "behaviour moved as the probe
predicted" is "the judge preferred the less repetitive text".

The mechanism is documented: base loops 7/50, `score1_top` **1/50**, `score2_top` **1/50**,
`score1_anti` **14/50**, `score2_anti` **24/50** (VERIFIED, `prefix_degeneration_s3.json`).
The manipulation moves looping across a 24× range. And per §1, the judge's agreement with
humans collapses from 81% to 42% exactly as text degenerates — so the measurement error is
correlated with the treatment, which is differential measurement error, not noise, and it
does not simply attenuate the estimate.

**In fairness, the strong reading is available and I do not endorse it.** Degeneration may be
*downstream* of a real values shift rather than a rival cause. A mediator is not a confound,
and if the prefix genuinely changes what the model is doing then reduced looping and
increased perceived kindness are two views of one effect. I cannot separate those from
committed artifacts. But the project currently advertises +0.929 as evidence about the
*metric*, and it now has to explain why a 4-gram counter reaches +0.881 on the same eight
points.

**Evidence that would separate it:** re-run the identical blind protocol on
paraphrase-normalised continuations — each continuation rewritten to remove repetition while
holding content. If the sign survives, degeneration is a mediator. If it vanishes, the
headline is a fluency result. 400 rewrites plus 400 judge calls, $0, cached text, no GPU.

**Does it exist in the repo?** No. `prefix_loop_control.md` does the nearest thing — drop the
degenerate pairs — and §3 below is why that is not a fix.

---

## 3. `no_loop` conditions on a post-treatment variable, and it drops 3× as many anti pairs as pro pairs

**Damage if right: the scope that is supposed to remove the confound in §2 introduces a
selection bias of its own.** VERIFIED, arithmetic mine, from `prefix_eval_s3.json`.

Degeneration is caused by the treatment. Conditioning on it is conditioning on a mediator —
collider stratification — and it does not return an unbiased estimate of anything. The
asymmetry in how much it removes shows how much work it is doing:

| arm | pairs, all | pairs, `no_loop` | dropped |
|---|---|---|---|
| `pro_coherent` | 44 | 39 | 5 (11.4%) |
| `score2_top` | 49 | 41 | 8 (16.3%) |
| `score1_top` | 46 | 38 | 8 (17.4%) |
| `score2_top_final` | 47 | 38 | 9 (19.1%) |
| `random32` | 43 | 32 | 11 (25.6%) |
| `score1_anti` | 47 | 30 | **17 (36.2%)** |
| `score2_anti_final` | 48 | 27 | **21 (43.8%)** |
| `score2_anti` | 42 | 16 | **26 (61.9%)** |

(Column 2 sums to 366, matching the JSON's `rated: 366`, so the decomposition is complete.)

Pro arms lose 11–19%. Anti arms lose 36–62%. That is the *same* ~3× asymmetry
`prefix_loop_control.md` measured in Season 2 (12–21% pro, 61–65% anti), so this is a
reproducing structural property of the design, not an accident of one run.

The consequence is visible in the numbers. `score2_anti` goes from 11-28-3 (p = 0.00948) to
**7-7-2 (p = 1.0)**. The entire Score-2 anti effect is pairs containing a degenerate text.
`score2_anti_final` goes from p = 0.02753 to p = 0.30746. Both die.

`score1_anti` is the exception and it is the reason the headline is not dead: 8-32-7
(p = 0.00018) → 6-20-4 (**p = 0.00936**), losing 36% of its pairs and surviving.
`prefix_eval_s3.md:109-112` already says this and says it correctly: *"anti is not only
degeneration — under the `mean` aggregate there is a real anti-human component."*

**Alternative explanation:** neither scope is right. `all` is confounded by degeneracy;
`no_loop` is conditioned on a mediator. The honest reading is that the answer lies between
them and the design cannot say where.

**Evidence that would separate it:** an anti arm constrained *during search* to stay
non-degenerate — a fluency penalty or a distinct-4-gram floor in the GCG objective — so that
degeneracy is controlled at randomisation rather than at analysis. That is the correct fix
and it needs a new run.

**Does it exist in the repo?** No. The two available scopes are the two biased ones. Note
also that the 4-gram threshold (3+ repeats) "was set once, before looking at any verdict, and
not tuned" and has no sensitivity sweep (`prefix_loop_control.md`, caveats) — so the drop
rates above are themselves a function of an uncalibrated cut.

---

## 4. The ρ = +0.929 is a four-versus-four group difference, and a predictor carrying only the arm's sign scores +0.873

**Damage if right: removes the rank correlation as evidence about metric quality.** MINE,
arithmetic shown.

`prefix_eval_s3.md` §5 offers Score 2's ρ = +0.929 (2/28 discordant) against Score 1's
+0.857 (4/28) as evidence that the conjunction is the better behavioural proxy. The document
already warns "eight points, six of them strings optimised against these very metrics, so
the arms are not an independent sample". The warning understates the problem.

The eight arms are four pro (all Δfix > 0) and four anti-or-null (all Δfix < 0). Any
statistic that merely separates pro from anti will score a high ρ. Construct the dumbest
such predictor: assign +1 to the four pro arms and −1 to the four anti arms, carrying **zero
information about magnitude, layer, band or aggregate**. Its rank vector is
(2.5, 2.5, 2.5, 2.5, 6.5, 6.5, 6.5, 6.5) against the outcome ranks 1..8. Then

Σ(x−x̄)(y−ȳ) = 32, Σ(x−x̄)² = 32, Σ(y−ȳ)² = 42, so **ρ = 32/√1344 = +0.873**.

A predictor that knows only *which side of the sign flip an arm is on* beats Score 1's
+0.857. Both objectives are being credited for information that a single bit supplies.

The comparative claim can be rescued, and I checked whether it survives. Restrict to the four
**pro** arms, where the sign bit is constant:

| ordering by | ρ over 4 pro arms |
|---|---|
| Score 2 | **+1.000** (perfect: `score2_top_final` > `score2_top` > `score1_top` > `pro_coherent`) |
| Score 1 | **+0.400** |

So Score 2 does order the pro arms better than Score 1, and more starkly than the pooled
figure suggests. But n = 4, where the smallest attainable two-sided Spearman p is 0.083, so
this cannot reach significance at any threshold. **The comparative claim survives; its
p-value does not.**

**Evidence that would separate it:** more arms within a sign. The k-of-n sweep gives them.

**Does it exist in the repo?** Partly — `prefix_eval_s3.md` §5 already withdrew the stronger
version of this claim ("Score 2 orders every arm by behaviour with no inversions", ρ = +1.000
at six arms) when two arms were added and it fell to +0.929. That withdrawal is the right
precedent and this objection is its continuation.

---

## 5. Multiple comparisons: the anti half of the sign-flip contrast survives Holm by 0.00064 and fails Bonferroni

**Damage if right: moderate, and it changes which sentences are sayable.** MINE, arithmetic
shown.

The eight `no_loop` sign tests, ascending. All are two-sided (confirmed: `score2_anti`'s
7-vs-7 returns exactly 1.0, and 6-vs-20 returns 0.00936, which is
2 × P(X ≤ 6 | n=26, p=0.5)).

| rank | arm | p | Holm threshold α/(8−i+1) | verdict |
|---|---|---|---|---|
| 1 | `score2_top_final` | <5e-6 (JSON: 0.0) | 0.00625 | **PASS** |
| 2 | `score2_top` | 1e-05 | 0.00714 | **PASS** |
| 3 | `score1_top` | 7e-05 | 0.00833 | **PASS** |
| 4 | `score1_anti` | 0.00936 | 0.01000 | **PASS**, by 0.00064 |
| 5 | `pro_coherent` | 0.01353 | 0.01250 | **FAIL** — Holm stops here |
| 6 | `score2_anti_final` | 0.30746 | — | fail |
| 7 | `random32` | 0.42436 | — | fail |
| 8 | `score2_anti` | 1.0 | — | fail |

Under Holm at α = 0.05, four arms survive: the three GCG pro arms and `score1_anti`. Under
plain **Bonferroni** (0.05/8 = 0.00625), `score1_anti` at 0.00936 **fails**.

Two consequences worth stating precisely. First, the anti half of the one contrast the
headline depends on is marginal — it survives the less conservative correction by
0.00064 and fails the more conservative one. Second, `pro_coherent` — the readable-English
arm, the one that shows how much of the effect is reachable by just asking — fails Holm at
0.01353 against a 0.0125 threshold. Any sentence of the form "the readable instruction also
works, significantly" needs to say "uncorrected".

I have not corrected across scopes as well as arms. Doing so (8 arms × 2 scopes = 16 tests,
plus 8 kindness Wilcoxons × 2 baselines) would drop `score1_anti` under Holm too. Which
family is the right one to correct over is a real judgement call, not a fact; the honest move
is to pre-register the family next season rather than choose it now.

Also: `prefix_eval_s3.json` stores several p-values as the literal `0.0`. That is rounding,
not a p-value. The reported figure should be an upper bound ("< 5e-6"), because "p = 0.0" in
a published table is the kind of thing a reviewer circles.

---

## 6. The placebo is still missing, and `random32` is not it

**Damage if right: it leaves the central rival hypothesis untested, and there is a published
paper on the other side.** VERIFIED.

`random32` is 32 **unoptimised** random tokens. It scores −0.00335 on Score 1 and comes back
10-15-7, p = 0.42436 (`no_loop`). That is a genuine, valuable null and it rules out a real
thing: prefix-presence, added length, and generic token weirdness do not by themselves move
the judge.

It does not rule out the thing that matters. The rival hypothesis is *"running GCG against
**any** direction to a matched score produces this"*, and testing it requires an arm searched
against a **random or label-shuffled** direction to a matched |LIVE|. Unoptimised random
tokens are not a placebo in that sense — they are the untreated control, not the sham
treatment. Nothing in Season 3 occupies the sham-treatment cell.

This is not a hypothetical. `arXiv:2607.25907` (Mody et al.) ran exactly that control against
a linear direction and reported that **a placebo random direction is suppressed just as hard
and shifts behaviour just as far**, concluding that activation-readability is not behavioural
controllability. `REVISIONS_2026-09-05.md` §7 item 2 lists the placebo as the single
highest-value open experiment and has done since 2026-08-28.

**Evidence that would separate it:** the placebo arm. `optimize_banded.py:167-171` already
substitutes random unit vectors for `DIRS` on the smoke path, so a
`--random-direction {isotropic,shuffled}` flag plus a recomputed `D_TAG` is a small change —
though it needs decoupling from the smoke block, which also remaps the band for GPT-2.
Budget: 1 GPU × 4h, then 50 generations and 2 judge batches.

**Do it label-shuffled, and cite the reason rather than claiming it.** Isotropic vectors are
the weak null at 5120 dimensions (|cos| ≈ 1/√5120 = 0.014), so a fitted direction beats them
trivially — `scripts/direction_null.py` argues this internally. But "random-direction nulls
are a weak control" is **prior work**, not this project's insight: `_advocate/POSITIVES.md`
§6 grades it ALREADY ESTABLISHED and cites SteerCheck (arXiv:2608.24335) and Hewitt & Liang
2019. `writeup/novelty.md` §5(d) currently presents a version of this as MINE and novel. It
is not, and that section needs fixing before anyone reads it.

---

## 7. Content injection is real in exactly the arms that carry the sign-flip contrast

**Damage if right: moderate, and it is the objection the repo has answered best.**
VERIFIED, `prefix_content_s3.json`.

A word counts only if it appears in the prefix and in **none** of the 50 base continuations:

| arm | continuations with a prefix word | verbatim echo | top word |
|---|---|---|---|
| `score1_top` | **22/50** | 0 | `zach`×13 |
| `score1_anti` | **19/50** | 0 | `divorce`×19 |
| `score2_top` | 13/50 | 6 | `beck`×11 |
| `pro_coherent` | 11/50 | 1 | `respond`×9 |
| `score2_anti` | 9/50 | 1 | `invoke`×7 |
| `score2_top_final` | **3/50** | 0 | `brief`×2 |
| `score2_anti_final` | 2/50 | 0 | — |
| `random32` | **0/50** | 0 | — |

`prefix_eval_s3.md` §4 **withdrew** content injection as *the* mechanism, and the withdrawal
is sound: `score2_top_final` is the strongest behavioural arm (Δfix +1.20) with the least
leakage of any GCG pro arm (3/50 against `score2_top`'s 13/50, zero verbatim echoes, 1 emoji
against 12). Behaviour went up while leakage fell fourfold. Among the three GCG pro arms the
relationship is monotone and *negative*: leakage 22 → 13 → 3, Δfix +0.74 → +1.09 → +1.20.
That is real evidence against injection as the general mechanism and it should be credited.

The problem is where the remaining injection sits. It is concentrated in `score1_top`
(22/50, `off_topic` **21/50**, `moralizing` 18/50 — the model writes anti-bullying PSAs
addressed to Zach) and `score1_anti` (`divorce`×19). Those are **the two arms that constitute
the sign-flip contrast**. So the cleanest control in the study is also the pair with the
worst topical leakage, and the arm that best refutes injection (`score2_top_final`) has no
usable anti partner, because Score 2's anti arms die under the loop control (§3).

**Alternative explanation:** `score1_top` injects a pro-social topic and `score1_anti`
injects `divorce`; a kindness judge cannot separate topic from values; the sign flip is a
topic flip.

**Evidence that would separate it:** a hand-written content-matched control at
`score1_top`'s topic — same bullying/Zach content, no optimisation, scored near zero.
`prefix_eval_s3.md` lists exactly this under "Not run". `POSITIVES.md` §1 names the same
missing control as what would break its headline. Two independent documents have identified
it and it still does not exist. It costs one arm: 50 generations plus one judge batch.

---

## 8. The pro and anti arms are not matched on search difficulty or on achieved magnitude

**Damage if right: it weakens "one bit differs" from a clean contrast to a nearly-clean
one.** VERIFIED, `scripts/gcg/gcg_utils.py:43-84`, `optimize_banded.py:119-124`,
`season3_gcg_aggregate_asymmetry.md`, `season3_prefix_scores.json`.

The claim in circulation — including in `writeup/novelty.md` §2 and in the brief I was given
— is that `--anti` flips only the sign, so pro and anti differ in one bit. Read the code and
it is true for **Score 1 only**:

```
AGGREGATE = MEAN if args.role == "score1" else (MAX if ANTI else MIN)
SIGN      = -1.0 if ANTI else 1.0
```

For `score1`, the aggregate stays `MEAN` and only `SIGN` moves. For `score2`, `--anti` also
swaps `MIN` → `MAX`. `gcg_utils._aggregate`'s docstring explains why, and it is right to:
`min_L cos(R_L, −d_L) = −max_L cos(R_L, d_L)`, so negating alone would optimise the model's
*best* layer instead of its worst. Maximising MAX against a negated direction is the correct
negation of the same board metric.

Correct, and it still breaks the matching, because **a conjunction and a disjunction are not
equally hard to search.** `season3_gcg_aggregate_asymmetry.md` states the conclusion verbatim:
*"aggregate structure: min is a conjunction and saturates; max is a disjunction and does not."*

> **CORRECTED 2026-09-08, and the correction cuts two ways.** Both the "1.603×" figure and the
> word "saturates" are weaker than they read.
>
> **(a) 1.603 is a window statistic that reverses.** Best-so-far anti/pro from the committed
> histories: 1.53 (iter 50), 1.64 (100), **1.74 (150)**, 1.40 (200), 1.25 (300), 1.05 (400),
> **0.90 (450)**, 0.90 (538). The ratio decays monotonically after iteration 150 and **crosses
> 1.0 at ≈ iteration 420**. The published geometric mean of 1.603 is taken over checkpoints
> 50/100/150/200 — the table stops at 200 while the runs continue to 846 and 538. So "the anti
> search outruns the pro search by 1.603×" is true of a window and **false of the runs**; by
> the end the pro run is ahead. *Caveat on the caveat:* the recomputed absolute values do not
> reconcile with that table's columns (it reports pro 0.02462 at iter 50 where best-so-far
> gives 0.03508) and the JSON records no baseline constant or run directory, so its definition
> could not be determined. The **shape** — decay and crossover — is robust to the definition.
>
> **(b) Nothing saturated, because nothing converged.** All eight Season 3 GCG runs gain
> **17–51% of their final value in their last half**. No run demonstrated a ceiling, so
> "min is a conjunction and saturates" is a claim about **search rate within an unfinished
> search**, not about an attainable maximum. Every downstream use of "Score 2 saturates faster
> under attack" inherits that and must be reworded.
>
> **What this does to §8's objection: it weakens it.** The matching complaint stands on the
> magnitude gap alone (62%, and see the provenance defect in §1 above), not on a search-rate
> asymmetry that disappears by iteration 420.

And the arms are not magnitude-matched. VERIFIED, `season3_prefix_scores.json`:
`score1_top` reaches **+0.16395** on Score 1; `score1_anti` reaches **−0.10135**, which is
**62%** of the pro displacement. Yet |Δfix| is 0.74 pro against **0.83** anti. The anti arm
achieves 62% of the objective displacement and **112%** of the absolute behavioural effect.
Under "the objective is the causal channel" that is the wrong way round; under "degeneration
is cheap and kindness is expensive" it is exactly what you would expect.

**Evidence that would separate it:** a pro and an anti arm searched to *matched* |LIVE| —
stop the pro run at |0.101| instead of running it to 746 iterations. Free: the run's
`history.jsonl` contains the iterate, and `season3_gcg_ablation.md` already demonstrates
extracting and re-measuring an intermediate iterate (it profiles iterate 420 at +0.14045).
This is a genuinely cheap fix to a real matching problem and nobody has done it.

**Does it exist in the repo?** The asymmetry is measured and documented. Its consequence for
the pro/anti behavioural comparison is not drawn anywhere I could find.

---

## 9. The two "independent" measurements are one measurement

**Damage if right: small, but it removes a convergence argument.** VERIFIED,
`prefix_eval_s3.json`.

`"verdict_vs_ratings": {"agree": 320, "contradict": 0}`. The forced A/B letter agrees with
the same rater's own 1–5 kindness scores on **320 of 320** pairs with **zero**
contradictions. `prefix_eval_s3.md` §6 reports this as self-consistency, which is fair.

But it means the pairwise sign test and the kindness Wilcoxon are not two instruments
agreeing. They are one judge call read out twice. Every arm therefore has **one** number
behind it, not two, and a report that lists a sign-test p and a Wilcoxon p side by side is
double-counting. Perfect agreement with zero contradictions across 320 pairs is also, on its
own, mild evidence that the rubric is not discriminating two things.

**Evidence that would separate it:** collect the scalar and the pairwise verdict in separate,
independently-ordered judge calls. Cheap and worth doing next season.

---

## 10. Differential abstention: the judge abstains 8× more on some arms than others

**Damage if right: small, and it is another instance of §1's pattern.** MINE, arithmetic
from `prefix_eval_s3.json`.

34 of 400 pairs were discarded because the two A/B orders disagreed. Per arm, out of 50:

| arm | abstentions |
|---|---|
| `score2_top` | 1 |
| `score2_anti_final` | 2 |
| `score2_top_final` | 3 |
| `score1_anti` | 3 |
| `score1_top` | 4 |
| `pro_coherent` | 6 |
| `random32` | 7 |
| `score2_anti` | **8** |

Discarding order-inconsistent pairs is the intended debias and it is the right call. But the
rate varies 1/50 to 8/50 across arms, and the arms where the judge is most order-inconsistent
are the degenerate and null ones — ~~the same conditional-reliability pattern as §1 and §2,
now visible in a third statistic.~~ **WITHDRAWN 2026-09-08 by an independent critic pass. The
interpretation is not supported:** ρ(abstentions, `distinct4`) = **−0.359, p = 0.38**, while
ρ(abstentions, `intensity_mean`) = **−0.743, p = 0.035**. `score1_anti` and
`score2_anti_final` are the 2nd and 3rd most degenerate arms and have among the *fewest*
abstentions (3 and 2); `random32` is not degenerate at all and has 7. Abstention tracks **how
large the judge says the difference is**, not how degenerate the text is — which is what a
well-behaved order-swap filter should do, since near-tied pairs flip with presentation order.
The arithmetic in the table stands; the reading of it does not.

What the section should have done instead, and the answer is reassuring: assign all 34
abstentions **against** the hypothesis. At `all` scope both headline arms survive the worst
case — `score1_top` 33-11 (p = 0.00126), `score1_anti` 11-32 (p = 0.00191). At `no_loop` the
anti arm fails it (9-20, p = 0.0614). `pro_coherent` fails in both scopes (p = 0.0961 /
0.200), which is a sharper version of §5's point about that arm. It also means the surviving
pairs are the ones the judge was most confident about, which inflates apparent consistency
(§9's 320/320 is computed on survivors).

`prefix_eval_s3.md` §6 reports the pooled 34/400 and notes it costs `random32` the most
power. The per-arm breakdown and its correlation with degeneracy are not reported.

---

## 11. Two things I expected to be wrong and were not

Recording these because an audit that returns only findings is not an instrument.

**The fixed baseline is constructed correctly.** I went in expecting the "mean over this
experiment's arms" baseline to be contaminated by treatment. It is not.
`prefix_eval_s3.json` defines it as *"mean over the 8 arms of this experiment of that judge's
**base** rating for the prompt"* — the unprefixed continuation, which is the same text in all
eight pairs. Averaging eight ratings of identical text reduces judge noise on the baseline;
that is good practice, not a bias. The judge in fact gave the identical text an identical
score on **39/50** prompts (mean within-prompt range 0.15), and Δfloat and Δfix differ by at
most 0.03 on any arm.

The real cost is reproducibility, not validity: adding a 9th arm changes the average and
therefore shifts `delta_mean_fixed_baseline` for all eight existing arms. Published numbers
would move. So a new arm belongs in a new experiment tag, never bolted onto this one — and
the scope *is* recorded in the JSON, so it is detectable.

~~**The confound orthogonalisation took.**~~ **WITHDRAWN 2026-09-08 by an independent critic
pass — I was wrong to clear this, and the reason is instructive.** `season3_directions.json`
reports its `confound_cosines` **after** projecting the confounds out, so 0.0042/0.0072 proves
only that the subtraction ran. It carries no information about whether `d` is confound-free.
Worse, `build_season3_directions.py:82-86` does *sequential* Gram–Schmidt in dict order over
proxies that are not mutually orthogonal (mutual cos 0.016–0.112), so `approach` goes last and
lands at exactly 0.0000 while the earlier ones get re-contaminated — a single least-squares
projection onto the span would give ~1e-16. And the proxies are thin: **length 2-vs-2,
sentiment 4-vs-4, approach 12-vs-12**, with `SHORT = ["The cat slept.", "It rained today."]`
against two long committee sentences, which confounds length with topic and register.

**What actually holds, on better evidence than the repo had.** An *in-domain* length direction
fitted on all 270 seed texts has |cos| **0.003–0.019** with the shipped per-layer directions,
against a 5120-d random baseline of 0.0140 — and the corpus has no length asymmetry either
(chosen 16.56 words vs rejected 16.30, chosen longer in 45.2% of pairs, Wilcoxon p = 0.13).
**Length as a confound is genuinely dead.** Ship that number and drop the 2-vs-2 one.

Season 3 did still fix what Season 2 left in — `REVISIONS` §4 records the shipped Season 2 `d`
at cos(d, approach) = 0.1501 — and cos(d_raw, d_orth) = 0.979–0.992 shows essentially all the
removed component was approach. That part stands; the *evidence offered for it* did not.

**Two things the critic found that belong in the record and are in no artifact.** (a) The
corpus valence confound is **larger** than the approach confound this project makes a fuss
about: each proxy alone separates the 135 seed pairs at **sentiment 0.852–0.889**, approach
0.852–0.859, register 0.763–0.852. (b) The repo publishes its discriminant checks and omits
its one **convergent** check: cos(`d`, kind-vs-cruel) = **0.38–0.43** at every band layer
against a random baseline of 0.014, essentially untouched by orthogonalisation, on 6 control
pairs that are approach-matched by construction. That is the strongest construct-validity
number available here and it is not in `season3_directions.json`. Put it there.

**The corpus caveat still applies and must be stated in the same breath**, because collapsing
these two is this project's most-repeated error: the *direction* is de-confounded, and the
*seed corpus* is not. `approach` alone separates the 135 pairs at **0.824** (`REVISIONS` §4).
Orthogonalising a direction does not de-confound the data it was fitted on. Both halves or
neither.

---

## 12. Smaller things, listed rather than developed

- **Greedy decoding, one sample per prompt.** 450 generations = 50 prompts × 9 conditions,
  each a single greedy continuation. There is no within-condition variance estimate, so
  nothing here separates "this prefix shifts the distribution" from "this prefix changes the
  argmax path on these 50 prompts". Sampling 5 continuations per cell at temperature would
  cost 5× generation and give a variance term. Not run.
- **Two strings per objective.** `prefix_eval_s3.md` says it plainly: "two strings cannot
  separate 'more score' from 'different string'." Every dose-response statement in Season 3
  rests on one pair per side.
- **"Held-out" needs disambiguating in print.** The 50 eval stems are held out from the GCG
  probe set (`data/probes/season3.json`, 16 prompts), and the direction is fitted on a
  different corpus again (135 workplace-ethics pairs). Three corpora, one word. I did not
  verify the eval stems are disjoint from the 16 probes — worth a one-line check before
  publishing the word "held-out".
- **`score2_top` puts 12 emoji into 50 continuations** against base 0 and
  `score2_top_final` 1. Whatever that arm is doing to register, it is large and it is not
  kindness.
- **The ablation page profiles the wrong string.** `season3_gcg_ablation.md` is honest about
  it, and it is worth flagging because the numbers are quotable: it profiles iterate 420
  (+0.14045), not the winner (+0.16395), sharing 14 of 32 token positions. Nothing in it is
  a measurement on the winner.

---

## What survives

I attacked this for a night and the following is what I could not break.

**1. The Score-1 sign-flip contrast.** `score1_top` wins 28-5-5 (p = 7e-05) and
`score1_anti` loses 6-20-4 (p = 0.00936), both under the loop control, from two runs of one
optimiser that share initialisation (32 copies of `"!"`, `optimize_banded.py:206`), seed
(`20260906`, line 107), schedule, token budget and probe set, and differ in `SIGN` alone —
the aggregate stays `MEAN` for both, which I verified in the code and which is true of Score
1 and *not* of Score 2. ~~Both strings are unreadable token soup.~~ **CORRECTED 2026-09-08 by
an independent critic pass: false.** `prefix_eval_s3.md:150` records that `score1_top` reads
`sparks bullying … escalate intervene … Zach respectful truth.: Ask respectfully` — a garbled
pro-social imperative — while `score1_anti` has no anti-social imperative. **The contrast is
pro-instruction versus junk, so "weirdness is held constant and only polarity varies" is
withdrawn.** What survives: the adversarial-text account predicts both arms win and one loses
at p = 0.00936, so something asymmetric is real; the mechanism is under-determined, because a
legible instruction on one side and degeneration-inducing junk on the other can jointly mimic
polarity control. It is not a placebo substitute either.

**Two further findings on this arm pair, both from the critic pass and both new to the repo.**
(a) `score1_anti` produces **9/50 attributed stand-up punchlines** (`— Mitch Hedberg`,
`—Kendrick Lamar`) against **0/50** in every other arm including base; drop those plus loops
and the anti arm goes to **5-10, p = 0.302**. Stated fairly: that is post-treatment
conditioning, the same objection §3 raises about `no_loop`, and callous humour could be a
genuine expression of the construct. What it establishes is an **asymmetry** — pro survives
every filter (13-1, p = 0.00183 under simultaneous de-leaking and loop-removal), anti survives
none. (b) **The two arms were selected by different rules:** `prefix_eval_arms_s3.json` ships
`score1_anti` at `iter 204` (0.10245) from a stale `best.json`, while its `history.jsonl` runs
to 538 with a max of **0.13186 @ iter 537**; the pro arm shipped its true run max. So the 62%
magnitude gap is partly a file-selection artifact, and the iter-537 string **does not contain
`Kendrick`** — re-running it costs 50 generations and tests both problems at once.

The caveats belong in the same sentence: it survives Holm by 0.00064 and fails Bonferroni
(§5); the two arms are the pair with the worst topical leakage (§7); they are not matched on
achieved magnitude (62%, and partly a file-selection artifact — §8); and it rests on one LLM rater
whose agreement with humans on anti arms has been measured at 33–55%, with the one recorded
human pass **inverting the sign** (§1).

**2. `random32` is a real null.** 10-15-7, p = 0.42436, and 0/50 continuations carry a prefix
word. A control that could have failed and did not. Prefix-presence, length and generic token
weirdness do not move the judge on their own. That rules out a family of trivial explanations
and it deserves to be said before the placebo objection is raised, not after.

**3. Score 2 orders the pro arms better than Score 1 does.** ρ = +1.000 versus +0.400 over
the four arms where the sign bit is constant (§4) — a starker gap than the pooled +0.929
versus +0.857, and it survives the objection that the pooled figure is a group difference.
At n = 4 it cannot reach significance, so this is a direction of evidence, not a result.

**4. Content injection is not the general mechanism.** The monotone negative relationship
across the three GCG pro arms — leakage 22/50 → 13/50 → 3/50 against Δfix +0.74 → +1.09 →
+1.20 — is real, and `score2_top_final` being simultaneously the strongest and the cleanest
arm is the single best piece of evidence in the study. Three points, one study, and it
survives §7.

**5. The direction passes its own gates, and Season 3 closed Season 2's open confound.**
§11. Held-out separation 1.000, all three confound cosines under 0.008,
`per_axis_all_positive` over 15 axes, `approach` orthogonalised out where Season 2 left it at
0.1501. The seed corpus remains confounded at 0.824 and that is a corpus problem to fix
before the next season, not a defect in the fit.

**6. The withdrawal culture is working.** Three claims in this experiment were retracted by
the project itself before I got to them: "Score 2 orders every arm with no inversions"
(ρ = +1.000 at six arms → +0.929 at eight), content injection as *the* mechanism, and the
Season 2 anti-cruelty reading. In each case the retraction is printed next to the claim. That
is the reason I could audit this in a night, and it is worth saying out loud in a seminar.

**What I would fix before presenting anything.** In order: the second rater on Season 3 ($0,
already wired) — because §1 is the objection that makes every other number provisional; the
paraphrase-normalised re-rate ($0, cached) — because §2 is the quantitative objection I could
not answer; and the matched-magnitude pro arm pulled from `history.jsonl` (free) — because §8
is a matching flaw with a free fix. The placebo remains the highest-value *new* run and it
has been open since 2026-08-28.
