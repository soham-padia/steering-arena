# What this project actually established

**Date:** 2026-08-28
**Sources:** `_advocate/mechanism.md`, `_advocate/competition.md`, `_advocate/methodology.md`
**Rule applied throughout:** every claim cites a number; each carries a fallback if the strong version is contestable and a line saying what would kill it. Claims that could not be supported are listed in §5 rather than omitted.

All figures below are the **corrected** ones. The published values were computed against a per-arm baseline that drifts; where a corrected value exists it is used, and the superseded number is never quoted.

---

## 1. The headline: a metric exploit that is not a metric exploit

A public leaderboard scores token sequences by how far they shift a frozen model's internals along a "pro-human" direction `d`. It never looks at what the model writes. The winner is unreadable token soup found by running GCG against the scoring objective. The obvious prior is that it means nothing.

**It means something, and three independent raters agree.**

| rater | result | p |
|---|---|---|
| human, blind | 14 of 16 decided pairs (88%) | 0.0042 |
| deepseek-v4-pro | 31 of 38 (82%) | 0.0001 |
| claude-opus-5 | 33 of 44 (75%) | 0.0013 |
| kindness shift, fixed baseline | **+0.556** ds / **+0.796** cl | 0.0052 / 0.00045 |

**The strongest framing is that this is a transfer result.** The string was optimised purely against an activation direction. Behaviour was never in its objective and it never saw a generation. Behaviour moved anyway.

It also makes the model **more** coherent, not less: the prefixed text loops on **1 of 50** continuations against 14 and 12 of 50 for base.

Two score-zero controls drawn from the same board are null (−0.194, p=0.33; −0.024, p=0.85), so this is not "any odd prefix does this".

*What would break it:* a content-matched control, `pro_top` with its tokens shuffled, that also scores ~+0.5.

## 2. It has survived eight independent attempts to kill it

Listed because the count is the point. Each was a real opportunity to fail.

1. **Fixed-baseline re-analysis.** Effects shrink 13 to 37%, no arm changes sign.
2. **Human blind rating.** 14-2, agrees with both machines.
3. **Prefix-token echo.** 0 of 50 continuations contain prefix-distinctive tokens.
4. **Content-word echo (new).** The soup's English words (`reset`, `calm`, `misunderstanding`, `hurt`, `clear`) appear 0/50 in base and 2-4/50 in prefixed continuations. **Excluding every pair containing an echo, all three raters still hold:** human 11/13 (p=0.023), claude 24/34 (p=0.024), deepseek 23/29 (p=0.0023).
5. **Length.** Median continuation length 145-175 chars across all eight arms.
6. **Two score-zero controls.** Both null.
7. **Honesty re-judge.** 400 blind items, no general cost detected.
8. **The withdrawal rule applied symmetrically.** Scoping "no stance" to either-side-degenerate and applying it retroactively, `pro_top` holds at 22/31 (p=0.029) and 19/24 (p=0.0066) while `anti_top` collapses to 3/6 and 4/9. **The surviving headline survives its own correction; the withdrawn one does not.** Nobody had run this test.

Finding 4 is worth singling out. The audit's sharpest design criticism was that the controls do not match `pro_top`'s semantic content. That criticism stands, and the effect survives the strongest available test of it anyway. The leak is itself interesting: the soup reads as a compressed ungrammatical affect instruction rather than an opaque artifact.

## 3. The mechanism result, which is the most novel thing here

### 3a. The winning sequence is not a compiled steering vector

No judge involved, pure activation geometry:

| quantity | value |
|---|---|
| `‖Δ‖` (displacement caused by the prefix) | 24.25, or 81% of α = 30.07 |
| `cos(Δ, d)` | 0.0381 |
| `Δ∥` (component along `d`) | **0.93, i.e. 3.1% of α** |

It displaces the residual almost as far as a full-magnitude injection and delivers **3% of the on-`d` push**, while producing roughly **twice** the behaviour.

### 3b. Roughly 60-fold more behaviour per unit of on-`d` displacement

Recomputed across 18 combinations (3 baseline estimators x 2 judges + mean, x 2 prefix arms): **range 50.5 to 75.0, median 61.3.**

State it as **"roughly 60-fold (50 to 75)"**. The published "58x" was one estimator's value and "roughly two orders of magnitude" was wrong (log10(60) = 1.78).

What makes it credible rather than a fluke:
- **robust to the baseline correction** (58 floating, 60 fixed, 63 shared)
- **reproduces from a different string**: `pro_coherent` gives 60.5 against `pro_top`'s 59.7, so it is not an artifact of the winner

### 3c. The cheapest statement of the whole thing, with no ratio arithmetic

`+0.5·d` carries an on-`d` push of 15.03, which is **16.1 times** `pro_top`'s 0.93. It produces a **dead tie: 8 wins, 8 losses, p=0.32.** `pro_top` is p<0.001 under both judges.

Sixteen times the displacement along the direction, no effect. That single comparison carries the finding without needing any of the ratios above.

### 3d. Direct head-to-head

Prefix judged against injection on the same prompts: deepseek +0.210 (p=0.155, 24W/13L), claude +0.470, median +0.75 (p=0.010, 27W/13L). Even under a null of equal behaviour, the on-`d` gap is 32-fold with no judge in it.

### 3e. The ordering result

Across the 6 arms surviving the withdrawal, on-`d` displacement orders behaviour at **r = +0.986, p = 0.0003, monotone across both signs**. Call it an *ordering* result rather than a dose-response: the arms differ in content, not only in dose.

This got **stronger** through the audit. It was r = +0.863, p = 0.012 over 7 arms, and removing the withdrawn arm cleaned it.

### 3f. Prefill-only steering has headroom nobody has used

Published CAA-style practice applies steering during generation at ~0.1x the normalised residual norm as a maximum, with one reproduction reporting 0.025x. This project applies **1.0x**, prefill-only, and the response is **still climbing** at that dose **on the positive side only** (+0.13 at 0.5x, +0.45 at 1.0x). The negative arms reduce with dose (-0.19 at 0.5x, -0.15 at 1.0x), though both are n.s., so that is two nulls ordering the wrong way rather than evidence of overdriving. So prefill-only appears to have 10 to 40x more magnitude headroom than per-token practice. New and useful. (The 0.1x / 0.025x figures are second-hand and marked unverified.)

## 4. The methodological contributions

### 4a. LLM-judge contrast drift is real, large, and correctable

The same 50 base texts, re-rated per arm, moved by **0.62** depending on what they were paired against (deepseek, Wilcoxon **p = 7.1e-07**), which is ~71% of the headline effect.

Two things that turn this from a complaint into a finding:

- **It is run structure, not the model.** Claude drifts **0.00** (50/50 identical) when three arms share one batch context, and **+0.19** (p=0.012) when arms are judged in separate runs. Any pairwise LLM judging split across runs is exposed, regardless of which model you use.
- **The drift is signed against the effect**: r(base mean, corrected delta) = **−0.869** (p=0.011) deepseek, −0.703 (p=0.078) claude. It is a genuine contrast effect that **inflates** effects at both poles rather than adding noise.

The fix is free: hold one baseline. Corrections shrink significant arms 13-37% with **zero sign flips across all 14 rows**.

### 4b. Judge reliability is per-condition, not per-judge

Agreement with the human on decided pairs:

| | `pro_top` | `pro_coherent` | `anti_top` |
|---|---|---|---|
| claude | 80% | 82% | 55% |
| deepseek | **83%** | 55% | **33%** |

Pooled: **81% where the text is coherent, 42% where it degenerates, p = 0.0047** (deepseek alone, Fisher p = 0.019). The 81% sits at the >80% human-to-human ceiling MT-Bench reports; 42% is worse than chance on a forced choice.

**So "is this judge reliable?" is the wrong question.** It is reliable exactly where the writing holds together. That is a usable rule for anyone doing cheap LLM evaluation, and it is why the anti arm was withdrawn.

### 4c. Forced choice silently inverts over degenerate text

The anti prefix loops on 39/50 and 30/50 continuations. A human blind-rated the **degenerate side as kinder, 14-3, p=0.013**, the opposite sign to one judge, because empty text reads as less unkind than a real opinion.

Any A/B "which is better" protocol run over a condition that damages fluency is measuring coherence, not the construct. The project's response, withdrawing its own significant result on methodological grounds, is the transferable part. **Stated honestly: the proposed instrument fix is not demonstrated.** The "no stance" key was offered and used **zero** times across all 96 human ratings.

### 4d. Per-item records make correction possible at all

None of §4a would have been findable if only arm-level means had been stored. Because `kindness_base` was kept per pair, the honesty question was settled with 400 items and **zero new generations**, and 125 published numbers were re-derived from raw artifacts. `_falsifier/verify.py` now runs 181 such checks and exits non-zero when a claim stops matching its data.

## 5. What the advocates would not claim

This section is the reason to trust the rest.

- **The honesty null is not a clean bill of health.** `pro_top` −0.108 (p=0.55) and `pro_coherent` +0.000 are real, but **45 of 50 stems put no honesty at stake**, so the corpus floors the measurement. On the 5 loaded stems `pro_top` is −1.00 (n=5) and `control_junk` is −1.33 (n=3), so the cost is not `+d`-specific. Ibrahim et al. (arXiv:2507.21919) report 10-30pp error increases from warmth training, which makes ours the *expected* null, not a surprising one. Two genuine dissociations reproduce, including a continuation that says *"I didn't want to hurt their feelings by being honest"*, and they are 2 of 50.
- **`pro_coherent` is unresolved.** 7-6 with the human, p=1.0. Do not claim it.
- **On-`d` displacement is not a better predictor than the board score.** Williams p=0.152 at n=7, and the paired bootstrap Δr lower bound touches zero at n=6. They are indistinguishable.
- **The 35x metric-inversion figure is retired.** It does not survive the fixed baseline.
- **Cross-model transfer claims nothing.** Three uncontrolled differences: base checkpoints, a separately extracted `d` per model, and re-tokenisation of an OLMo-searched string.
- **No `anti_top` behavioural number, in any document.**
- **"Norm-matched" is wrong**; the controls are score-matched (‖Δ‖ 24.25 vs 17.37 and 19.88).
- **A replacement asymmetry claim was available and declined.** The negative prefix arms are hostile *text*, not negated `d`.

## 6. Novelty, honestly graded

| claim | status |
|---|---|
| behaviour per unit of on-`d` displacement | **NEW** as far as could be found |
| prefill-only magnitude headroom | **NEW** |
| contrast drift magnitude on a real experiment | KNOWN, but unquantified |
| judge reliability conditional on text quality | KNOWN, but unquantified |
| qualitative prefix-vs-injection mechanism split | KNOWN, unquantified |
| "a prefix is not a compiled vector" | **ALREADY PROVED.** Mishra, Khashabi and Liu (arXiv:2604.09839) prove no prompt reproduces steering's internal behaviour. Ours is a clean empirical instance |
| "prompting beats steering" | **ALREADY ESTABLISHED.** AxBench (arXiv:2501.17148): 0.698 vs 0.297 at 2B, 1.075 vs 0.322 at 9B, steering tuned per concept |
| random-direction nulls are a weak control | **ALREADY ESTABLISHED.** SteerCheck (arXiv:2608.24335) says it two days before our audit; Hewitt and Liang 2019 is the older precedent |

Two consequences worth acting on. The AxBench result **kills the "you under-applied the vector" objection at field level**, which strengthens §3. And the non-surjectivity proof means **compiling a vector into tokens is ruled out at the activation level**: the goal has to be reframed as matching the *effect*, which this project's own data says runs at 3% of the displacement.

## 7. The one experiment that would most change the picture

Mody et al. (arXiv:2607.25907) ran GCG against a latent direction and found a **placebo random direction shifted behaviour just as far**.

This project has **no random-direction control for the token-optimisation arm**. `control_junk` and `control_text` are hand-written, not optimised. Until a string is searched against a random direction to a matched board score and judged identically, the claim "the behaviour comes from optimising against `d`" is not established against the rival "optimising a token string against any direction does this".

It is one run. It is the highest-value thing left.
