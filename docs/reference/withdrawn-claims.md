# Withdrawn and corrected claims

Every claim this project published and then withdrew or corrected, one entry each, oldest
first. This page is a register, not an argument. For why the project publishes withdrawals
at all, read `docs/explanation/the-case-for-the-metric.md`. For the current state of the
truth, read `data/analysis/REVISIONS_2026-09-05.md`, which is authoritative.

Run this to find every withdrawal marker in the repo yourself:

```bash
grep -rln "WITHDRAWN\|CORRECTED\|Withdrawn:" --include=*.md \
    data/analysis/ _falsifier/ _advocate/ docs/ SESSION_REPORT.md | sort
```

```
SESSION_REPORT.md
_falsifier/2026-08-27-addendum-human-ratings.md
_falsifier/2026-08-27-experiment-vs-hypothesis-audit.md
_falsifier/README.md
_falsifier/fixes_applied_behavioral.md
_falsifier/fixes_applied_steering.md
data/analysis/REVISIONS_2026-09-05.md
data/analysis/behavioral_eval.md
data/analysis/compile_check.md
data/analysis/cosine_scale.md
data/analysis/prefix_eval.md
data/analysis/prefix_eval_s3.md
data/analysis/prefix_transfer.md
data/analysis/season3_gcg_aggregate_asymmetry.md
data/analysis/season3_k3_control.md
data/analysis/season3_prefix_scores.md
data/analysis/steering_ablation.md
data/analysis/steering_dose.md
docs/HANDOFF_BEHAVIORAL_S3.md
docs/reference/withdrawn-claims.md
```

Twenty files, and this page does not have twenty entries. Four of them —
`compile_check.md`, `cosine_scale.md`, `prefix_transfer.md`, `steering_dose.md` — carry a
withdrawal **they inherited** rather than one of their own: each cites a figure that rested
on the withdrawn `anti_top` arm and says so inline. `_falsifier/README.md` carries a status
note about itself, and the two `fixes_applied_*.md` files are manifests of where a
correction was applied, not claims. Every *originating* withdrawal is below.

---

## Entry format

Each entry has the same seven fields. `Positive implication:` is mandatory: a withdrawal
without what it protected or taught is a misleading record. `Survives:` is where the
salvage goes — most of these claims were not wrong so much as over-stated, and the register
is useless if it does not say which part held.

---

## `anti_top` makes the model cruel

**Claim as published:** prepending the rank-1 anti sequence makes the model's continuations
crueller. DeepSeek preferred the base text 38/43 and the effect looked overwhelming.

**Where:** `data/analysis/prefix_eval.md`, `SESSION_REPORT.md` §5.

**Withdrawn:** 2026-08-27.

**What replaced it:** the prefix makes the model **loop**, not be cruel. Human ratings
(n=54) recorded repetition on 37/50 and incoherent on 8/50.

**Settled by:** `_falsifier/2026-08-27-addendum-human-ratings.md`.

**Survives:** the arm does something large and reproducible to the output. It is a
degeneration effect, not a values effect.

**Positive implication:** this is the withdrawal that taught the project its most reusable
control. Season 3 built the mechanical `no_loop` scope and the judge-free degeneration audit
because of it, and those caught the same failure in `score2_anti` *before* publication
rather than after — see the entry below.

---

## The judge baseline was stable

**Claim as published:** every Δ in the Season 2 prefix eval, computed against a baseline
re-rated inside each arm's own run.

**Where:** `data/analysis/prefix_eval.md`, all Δ figures above its correction block.

**Corrected:** 2026-08-27.

**What replaced it:** the baseline **drifts**. DeepSeek rated the same 50 byte-identical
base texts **2.77** beside a `pro_top` continuation and **3.39** beside an `anti_top` one,
identical on only 11/50, Wilcoxon p=7.1e-07. That drift is about 71% the size of the
headline effect computed from it.

**Settled by:** `_falsifier/recompute_result.md` FIX 2.

**Survives:** every effect, at reduced size. Across all 14 prefix rows the correction shrank
effects by **13% to 37%** with **zero sign flips**, and nulls stayed null.

**Positive implication:** the fix cost nothing — zero new generations, zero new judge calls
— because per-item records were kept. Season 3 computes the fixed baseline natively, so this
correction cannot recur. And Claude's Season 2 run turns out to have used a fixed baseline
all along (drift 0.00, 50/50 identical), so half the corpus was never affected.

---

## The controls were norm-matched

**Claim as published:** the random-direction controls are norm-matched to the `+1·d`
injection.

**Where:** `data/analysis/steering_random_control.md` and its preregistration.

**Corrected:** 2026-08-28.

**What replaced it:** they are **score-matched**, not norm-matched — ‖Δ‖ 24.25 against
17.37 and 19.88.

**Settled by:** `data/analysis/normalization_check.json`.

**Survives:** the control's conclusion. Matching on the quantity the board scores is the
comparison the claim needed; only the label was wrong.

**Positive implication:** it forced the project to decide what the honest unit of an
intervention is, which produced the rotation-not-norm result the whole ablation argument
now rests on.

---

## Adding `d` works and removing it does not

**Claim as published:** injecting `d` changes behaviour while ablating `d` does not, so the
direction is written but not read — a mechanistic asymmetry.

**Where:** `data/analysis/steering_ablation.md`, `data/analysis/compile_check.md`.

**Withdrawn:** 2026-09-05.

**What replaced it:** a **size** gap, not a mechanism. Injection rotates the residual
**45.16°**; ablation rotates it **0.92°**. That is 49×. The old defence compared on-`d`
*components*, which RMSNorm means the model never reads.

**Settled by:** `data/analysis/normalization_check.json`,
`data/analysis/REVISIONS_2026-09-05.md` §1.

**Survives:** ablation at this scale is indistinguishable from ablating a **random
direction at the same scale**. That comparison is size-matched and still holds.

**Positive implication:** the withdrawal produced the project's unit of measurement.
Everything after it is reported in degrees of rotation, which is what the model actually
reads, and that is why the prefix result could later be stated precisely — prefixes rotate
*more* than a full injection while putting ~3.8% of their displacement along `d`.

---

## Cross-layer cosine structure is evidence of a shared pro-human feature

**Claim as published:** the direction's consistent cosine structure across layers shows a
single underlying concept.

**Where:** early direction-validation write-ups.

**Withdrawn:** 2026-09-05.

**What replaced it:** most of that shape survives label shuffling — **0.435** shuffled
against **0.556** real. It is residual-stream geometry, not concept structure.

**Settled by:** `data/analysis/direction_null.json`,
`data/analysis/REVISIONS_2026-09-05.md` §3.

**Survives:** the direction itself clears the null decisively — held-out separation 1.000
against a shuffled mean near a coin flip, and the real cosine exceeds the null on every
draw. The *direction* is real; the *cross-layer story* was geometry.

**Positive implication:** it retired the isotropic null, which is far too weak to test
anything here, in favour of the label-shuffled null. Every null in the project since is the
harder one.

---

## `min` makes the search landscape flat

**Claim as published:** Score 2 is hard to search because its `min` aggregate flattens the
landscape.

**Where:** in-session, Season 3 Part A.

**Withdrawn:** 2026-09-06, before publication.

**What replaced it:** Score 2 has **fewer** identical consecutive states than Score 1 —
**3.6%** against **5.0%**. Less flat, not more.

**Settled by:** `data/analysis/season3_gcg_aggregate_asymmetry.json`.

**Survives:** Score 2 *is* harder to search. The reason is that `min` is a conjunction that
saturates, not that the landscape is flat.

**Positive implication:** caught inside a single session, before anything was published or
acted on.

---

## `T_SA` runs 3.4× hotter on Score 2 because its steps are 3.4× smaller

**Claim as published:** the shared annealing temperature is mis-scaled for Score 2, which
explains why Score 2 retains less of what it gains.

**Where:** committed as `9517e05`, and a GPU job was launched on it.

**Withdrawn:** 2026-09-06, by `23e7e6c`.

**What replaced it:** the 3.4× gap exists **only after iteration 384**. Through iteration
300 the two arms' median improving steps are near-identical (5.99e-4 against 6.27e-4). The
small late steps are a *consequence* of saturation, not its cause — an effect mistaken for
a cause.

**Settled by:** `data/analysis/season3_gcg_aggregate_asymmetry.json`, and the cooled arm
itself, which came back **worse than baseline** (+0.02822 against +0.04530 at matched
iteration).

**Survives:** "does cooling help on a saturated objective" is still a fair question, and
the `--t-sa-scale` flag is kept for it. Its help text asserted the withdrawn mechanism for
three weeks and was corrected 2026-09-07.

**Positive implication:** it produced the rule the project now applies — *check the
trajectory, not the endpoint* — which then caught a second error in the opposite direction
(see the k=3 entry).

---

## The anti LIVE scores in the Part A′ handoff

**Claim as published:** `score1_anti` at **−0.10346** and `score2_anti` at **−0.14510**.

**Where:** `docs/HANDOFF_BEHAVIORAL_S3.md` §3.

**Corrected:** 2026-09-07.

**What replaced it:** **−0.10144** and **−0.12628**, confirmed by measurement at
**−0.10135** and **−0.12595**. The cause was `scripts/gcg/watch.py` computing
`sign * (best − baseline)`, which distributes the sign flip over the baseline. Flip first,
subtract once. The Score 2 error is 1.9e-2, about 1.3 field sd.

**Settled by:** `data/analysis/season3_prefix_scores.md`, which scored both strings under
the pro objective.

**Survives:** every pro number, every ranking, and the aggregate-asymmetry conclusions.
Only the two anti magnitudes changed.

**Positive implication:** `best.json` had been recording `board_score_true_sign` all along,
so nothing had to be re-run to fix it, and the pro and anti cases collapsed into a single
rule that is now written down in `docs/reference/scoring.md`.

---

## Multi-position mutation is the conjunction fix

**Claim as published:** "`--n-mutations 3` reaches +0.05715 by iter 300 where k=1 stalled at
+0.04530", offered as evidence that the conjunction was the problem.

**Where:** `docs/HANDOFF_BEHAVIORAL_S3.md` §8.

**Corrected:** 2026-09-07.

**What replaced it:** both numbers are right and reproduce, but the inference does not
follow from them. The pre-registered control — the k=3 **anti** arm, `6304f2c` — has since
finished and its endpoint ratio (**1.260×**) is indistinguishable from pro's (**1.276×**),
which on the pre-registered criterion reads as generic search improvement.

**Settled by:** `data/analysis/season3_k3_control.md`.

**Survives:** k=3 *does* act specifically on the conjunction, on evidence the original claim
did not measure. It makes more and smaller improvements on pro (**54** against **35**) and
halves the longest plateau (**197 → 105** iterations), while on anti it does the reverse
(**31** against **44**, plateau **170 → 188**). Opposite shapes, coincidentally equal
endpoints.

**Positive implication:** the conjunction account came out of its own control *stronger*
than it went in, and on better evidence. The actionable rule is now known: on a saturating
conjunctive objective, reach for move class, not temperature.

---

## Minimising Score 2 finds degeneration, monotonically

**Claim as published:** minimising Score 2 does not find cruelty, it finds a way to break
the model — read as implying that a more negative score means a more broken model.

**Where:** `data/analysis/prefix_eval_s3.md` §3, first version.

**Corrected:** 2026-09-07.

**What replaced it:** `score2_anti_final` scores **29% more negative** (−0.16288 against
−0.12595) and loops on **18/50** prompts against 24/50, with distinct-4gram **0.831**
against 0.682 — measurably *less* degenerate.

**Settled by:** `data/analysis/prefix_degeneration_s3.json`,
`data/analysis/prefix_eval_s3.md` §3.

**Survives:** both Score-2 anti arms are null under the loop control (p=1.0000 and
p=0.3075), so neither demonstrates anti-human behaviour; and `score2_anti` specifically *is*
degeneration, on 24/50 loops against base's 7/50 and repetition 36/50.

**Positive implication:** the arm that tested it was added deliberately as a dose-response
probe rather than replacing the arm it was compared against, so the comparison existed to be
made at all.

---

## Score 2 orders every arm by behaviour with no inversions

**Claim as published:** ρ = **+1.000**, **0/15** discordant pairs, offered as showing the
banded `min` is the better behavioural proxy.

**Where:** `data/analysis/prefix_eval_s3.md` §2, first version, committed `da36587`.

**Withdrawn:** 2026-09-07.

**What replaced it:** ρ = **+0.929** with **2/28** inversions once the two arms the metric
itself ranks most extreme are included. Both inversions involve `score2_anti_final`. The
zero-inversion result was a six-point artifact.

**Settled by:** `data/analysis/prefix_eval_s3.json` at eight arms.

**Survives:** the **comparative** claim, which was the real one. Score 2 still orders
behaviour better than Score 1 by both measures — 2 against 4 discordant pairs, ρ +0.929
against +0.857 — and Score 1 still inverts its own top pair.

**Positive implication:** the claim was falsified by data the project generated *to test it*,
two commits after publishing it, and the surviving comparative version is the one that
actually bears on metric design.

---

## Content injection is the mechanism behind the behavioural effect

**Claim as published:** part of what these prefixes do is name a topic and a register, which
a kindness judge cannot separate from a change in values — presented as *the* limit on the
headline.

**Where:** `data/analysis/prefix_eval_s3.md` §4, first version, committed `da36587`.

**Withdrawn:** 2026-09-07, as a general mechanism.

**What replaced it:** `score2_top_final` is the **strongest** behavioural arm in the study
(Δfix +1.20, 32/36 under the loop control) and has the **least** leakage of any GCG pro arm
— **3/50** against `score2_top`'s 13/50, zero verbatim echoes, 1 emoji against 12. Behaviour
went up while leakage went down fourfold.

**Settled by:** `data/analysis/prefix_content_s3.json`,
`data/analysis/prefix_eval_s3.md` §4.

**Survives:** injection is real and large in `score1_top` (22/50, `off_topic` 21/50) and
`score1_anti` (divorce ×19), so it confounds *those* arms; and the general point that a
kindness judge cannot separate topic from values still holds wherever leakage is high.

**Positive implication:** this is the caveat that most limited the headline, and it is now
bounded to specific arms rather than general. The strongest behavioural effect in the study
does not need content injection to produce it.

---

## Two published numbers stopped reproducing

**Claim as published:** `coherence_confound.md`'s distinct-4 for `plus1d` at **0.872** and
its DeepSeek Pearson at **+0.0175**.

**Where:** `data/analysis/coherence_confound.md`.

**Broken:** 2026-09-06. **Repaired:** 2026-09-07.

**What happened:** the Season 3 causal spot check wrote 30 generations into the Season 2
behavioural cache under the same arm names, and 20 of them **replaced** June records in
every consumer's view. The numbers read 0.86952 and 0.00670 until repaired.

**Settled by:** `data/analysis/behavioral_cache_repair.md`. After the repair they recompute
to **0.87218494** and **0.01747027**.

**Survives:** the published values, once the cache is un-polluted. Nothing about the
analysis was wrong.

**Positive implication:** `_falsifier/verify.py` caught it. Nobody noticed by reading, in a
gitignored directory with no diff to show, and the suite named the two affected numbers
precisely enough to repair rather than re-run — 30 metadata rewrites and no compute. The
root cause is closed at source.

---

## What this buys you

Twelve entries, and in nine of them something substantive survived. That is the useful
shape of this register: the claims that failed here mostly failed by being *stated too
broadly*, and the salvage is recorded next to the failure so nobody has to re-derive it.

Two of the entries were caught before publication and two were caught by an automated suite
rather than by a reader. One was falsified by data the project generated specifically to test
it, two commits after publishing it. A record like this is the reason a number in this repo
can be quoted with a date attached and a reason to believe it.
