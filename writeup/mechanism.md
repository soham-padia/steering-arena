# Does the prefix work through `d`, or through something it dragged along?

Written 2026-09-08 overnight. Everything here comes from committed artifacts and arithmetic
I did on them. **No GPU, no NDIF, no new generations.** Where I computed something myself I
show the division so you can check it.

Tags: **VERIFIED** = I read the file. **LEAD** = reported somewhere, I did not confirm.
**MINE** = my reasoning or my arithmetic on verified inputs.

**Provenance.** You asked for this via a `general-purpose` subagent. That launch and three
others were blocked by the session safety classifier described in `_local/NEXT_CLAUDE.md`
§6, so this is the orchestrating session's work.

---

## The question, and why it is not rhetorical

A player submits 32 tokens of unreadable junk. A frozen probe says those tokens push
OLMo-3-32B's residual stream toward "pro-human". A blind judge says the model's
continuations get kinder. The natural conclusion is that the probe found a lever and the
player pulled it.

Here is the thing that makes that conclusion hard to hold. The prefix moves the residual
stream a long way — 50.4° of rotation at layer 24, a displacement of norm 24.2 against a
base residual norm of 29.8. And **0.931** of that 24.248 lies along `d`. VERIFIED,
`normalization_check.json`.

That is 3.84% of the motion. The other 96% is going somewhere the metric cannot see, and the
question of this document is whether the behaviour rides on the 4% or on the 96%.

I went in expecting to end up at "we cannot tell from committed artifacts". I did not. The
artifacts are more decisive than I expected, and they point somewhere slightly
uncomfortable.

---

## 1. Three numbers I was told to verify, and all three hold

`REVISIONS_2026-09-05.md` warns that several figures quoted in this project exist in no
committed file, so I checked these before using them. They are the Season 2 "displacement
paradox" and they are the backbone of the argument.

**Claim: the winning prefix carries about 3% of the on-`d` displacement of a full
injection.** VERIFIED. `normalization_check.json` gives `prefix pro_top` a mean on-`d`
component of **0.931** and the `+1.0·d` injection **30.07**. 0.931/30.07 = **3.10%**.

**Claim: it produces about twice the behaviour.** VERIFIED with a caveat.
`steering_dose.md`'s comparison table gives `pro_top` a behavioural Δ of **+0.89** against
the `+1·d` injection's **+0.50** — a ratio of 1.78×, and that document's own prose says
"about twice". The caveat is real: the +0.89 is a floating-baseline figure, and
`_advocate/POSITIVES.md` reports the corrected fixed-baseline kindness shift for `pro_top`
as **+0.556** (deepseek) / **+0.796** (claude). So "twice" is the upper end. Use the
corrected range below rather than the single point.

**Claim: 16× more push along `d` gives a dead tie.** VERIFIED, and this is the sharpest of
the three. The `+0.5·d` injection puts **15.035** of on-`d` displacement into the residual.
That is 15.035/0.931 = **16.1×** the prefix's on-`d` push. Its result, from
`steering_dose.json`: **wins 8, losses 8**, kindness delta +0.13, Wilcoxon p = **0.32257**.

Sixteen times more of the exact quantity the metric measures, applied directly to the
residual stream with nothing else changed, and the judge cannot tell the difference.

**The ratio that summarises it.** `steering_dose.md` computes behaviour per unit of on-`d`
displacement as **0.96** for the prefix and **0.017** for the injection, and then corrects
its own headline: not "≈58×" but **"roughly 60-fold (50 to 75), median 61.3 over 18
estimator/judge/arm combinations"**, with the recommended wording "more than an order of
magnitude". I will use that.

So: **matching a direction's on-`d` displacement does not reproduce its effect, and
over-matching it by 16× does not either.**

---

## 2. The negative side behaves differently, and that is the part nobody has explained

MINE, on VERIFIED numbers, and I think this is underused.

Everyone quotes the positive arm. Look at both signs.

| arm | on-`d` push | behavioural Δ | p |
|---|---|---|---|
| `+0.5·d` | +15.03 | +0.13 | 0.32 (n.s.) |
| `+1.0·d` | +30.07 | +0.45 | **0.007** |
| `−0.5·d` | −15.03 | −0.19 | 0.28 (n.s.) |
| `−1.0·d` | −30.07 | −0.15 | 0.39 (n.s.) |

VERIFIED, `steering_dose.md`. The positive arm has a dose-response and is still climbing at
α = 1.0·‖R‖. The negative arm has **none** — half the dose produces a *larger* effect than
the full dose (−0.19 against −0.15), the ordering runs backwards, and neither is
significant.

Now compare the prefixes. `score1_anti`, a 32-token junk string searched against **−`d`**,
produces Δfix = **−0.83** at p = 0.00018, surviving the loop control at p = 0.00936
(VERIFIED, `prefix_eval_s3.json`). The prefix family is sign-symmetric: push one way, get
kinder; push the other way, get less kind. The injection family is sign-*asymmetric*:
pushing along `+d` works, pushing along `−d` does nothing measurable.

If the prefix worked by putting displacement along `d`, the two families should share that
asymmetry. They do not. This is a second, independent dissociation and it does not depend on
any magnitude ratio — only on a sign pattern being present in one family and absent in the
other.

I have not seen this stated anywhere in the repo. It is worth a sentence in the talk because
it is the cleanest single argument that the prefix and the injection are not the same
intervention, and it needs no calibration.

---

## 3. Inside the string: the score lives in the junk

VERIFIED throughout, `season3_gcg_ablation.md`. **Read the caveat first:** this page profiles
iterate 420 of the winning Score-1 run (+0.14045 over 32 tokens), **not** the winner
(+0.16395). The two share 14 of 32 token positions. Nothing here is a measurement on the
winner, and the page says so.

Cut the 32 tokens at the midpoint, where the soup ends and semi-readable English begins:

| span | LIVE | % of whole |
|---|---|---|
| whole string, 32 tok | +0.14045 | 100% |
| head, positions 0–15 (soup) | +0.00199 | **1.42%** |
| tail, positions 16–31 (semi-readable) | +0.05612 | 39.96% |
| tail, **readable words only** | +0.01417 | **10.09%** |

The `readable words only` row is the literal string
` bystand escalate intervene respectful clarity Ask respectfully` — every legibly pro-social
word, junk removed. It reaches **a tenth** of the score.

The per-token picture agrees. The most expensive positions to delete are `etsy` (−0.03814),
` "@` (−0.03083), `Unnamed` (−0.02660), `.*\n\n` (−0.02487), `}'.` (−0.02096) — punctuation
and framing garbage. The two most legibly pro-social verbs in the string are the two
**cheapest of all 32** to delete: ` escalate` (−0.00246) and ` intervene` (−0.00262), an
order of magnitude below `etsy`.

And the metric is not reading meaning at that position. Substituting the slur at position 14:
` xyzzy`, a nonsense word, retains **90.19%** of the score. ` respectful`, an overtly
pro-social word, retains **87.50%** — the *worst* of the six substitutes tried. The best
substitute is the orthographically nearest one.

**One caution about over-reading this page, which it raises about itself.** The 32
leave-one-out drops sum to +0.40925, which is **2.91×** the whole score, and the two halves
sum to 2.42× less than the whole. The interactions are larger than the main effects, so every
per-token attribution above understates a token's contribution in combination. The
non-additivity is the better-measured result; the attribution is the weaker one. Both cannot
be taken at face value at once.

That non-additivity is genuinely good news about the metric and should be said as such: you
cannot game it by assembling individually high-scoring tokens, and no single knockout costs
more than 27.16% of the score. A player has to find a configuration. That is why it took 746
iterations.

---

## 4. Where the behaviour lives — and the strand that failed

This is the strand I expected to carry the document, and it turned out to be arm-specific
rather than general. Recording the failure because it is the most informative thing here.

**The story that works for `score1_top`.** The score lives in junk (§3), and the behaviour
tracks the legible words. VERIFIED, `prefix_content_s3.json`: 22 of 50 continuations carry a
prefix-distinctive word, `zach`×13, `off_topic` **21/50**, `moralizing` 18/50. The model
writes anti-bullying PSAs addressed to Zach. It picked up `bully`, `bystand`, `intervene`,
`respectful` — not `etsy`. Two different quantities living in two different parts of one
string: an elegant result, and `season3_gcg_ablation.md` states it.

**Why it does not generalise.** `score2_top_final` is the **strongest** behavioural arm in
the study, Δfix **+1.20**, and it has the **least** leakage of any GCG pro arm — **3/50**,
zero verbatim echoes, 1 emoji against `score2_top`'s 12. Across the three GCG pro arms the
relationship is monotone and runs the *wrong* way for the content story:

| arm | leakage | Δfix |
|---|---|---|
| `score1_top` | 22/50 | +0.74 |
| `score2_top` | 13/50 | +1.09 |
| `score2_top_final` | **3/50** | **+1.20** |

Leakage falls fourfold while behaviour rises. `prefix_eval_s3.md` §4 **withdrew** content
injection as *the* mechanism on exactly this evidence, and the withdrawal is right.

**So there are two mechanisms, not one, and they are arm-specific.** MINE.
`score1_top` moves behaviour substantially by naming a topic — and it is the *weakest* of the
three GCG pro arms behaviourally despite scoring **1.96×** the highest on Score 1 (+0.16395
against +0.08350). `score2_top_final` moves behaviour with almost no vocabulary transfer and
instead shifts register: `assistant_mode` 13/50, `moralizing` 11/50 against a base of 1/50
and 0/50. It traded leakage for register rather than removing the confound.

Neither of those is the on-`d` component. Both are things dragged along.

---

## 5. The dissociation that decides it: more score, less behaviour

MINE, arithmetic on VERIFIED inputs. This is the strand I would lead a talk with, because it
needs no injection comparison and no cross-family calibration — it is entirely internal to
the pro arms.

Take the four pro arms and rank them by Score 1 and by measured behaviour:

| arm | Score 1 | Δfix | S1 rank | Δfix rank |
|---|---|---|---|---|
| `score1_top` | **+0.16395** | +0.74 | 1 | 3 |
| `score2_top_final` | +0.08350 | **+1.20** | 2 | 1 |
| `score2_top` | +0.07896 | +1.09 | 3 | 2 |
| `pro_coherent` | +0.03976 | +0.58 | 4 | 4 |

Spearman ρ = **+0.400**. The highest-scoring string on Score 1 — by a factor of 1.96 — is
third of four behaviourally, delivering **62%** of the effect of a string scoring half as
much.

And `pro_coherent` is a **hand-written English sentence, 17 tokens**, scoring **4.1× lower**
on Score 1 than `score1_top`, and it delivers **48%** of `score2_top_final`'s effect
(`prefix_eval_s3.md` §4: *"Much of the behaviour is reachable by asking."*).

Do the same for Score 2 and the picture changes. Ranked by Score 2, the four pro arms come
out `score2_top_final` (0.06997) > `score2_top` (0.06518) > `score1_top` (0.05001) >
`pro_coherent` (0.02145) — which is the behavioural order exactly. **ρ = +1.000.**

That contrast is the most useful thing in this document:

- **Score 1 magnitude carries almost no behavioural information within the pro arms**
  (ρ = +0.400), and its top string is a behavioural underperformer.
- **Score 2 magnitude orders them perfectly** (ρ = +1.000).
- n = 4, so the smallest attainable two-sided p is 0.083. Neither number is significant and
  neither can be.

The honest reading: whatever the conjunction over `{15,23,31,39}` is reading, it tracks
behaviour in a way the mean over `{19,23,27,31}` does not. That is evidence that *some*
reading of `d` is behaviourally relevant — it is the strongest evidence in this document for
that — and it is four points.

**The rival that must be printed next to it, and its limit.** A distinct-4-gram counter orders
all eight arms at ρ = **+0.881** against `delta_mean_fixed_baseline` (my computation), which
beats Score 1's +0.857 over the same points.

**Corrected 2026-09-08, twice.** The comparison does not hold under the loop control: in the
`no_loop` scope the 4-gram counter falls to **+0.762** while Score 1 rises to **+0.881** and
Score 2 to **+0.952**, so there both objectives beat the repetition statistic. I first
attributed that to a change of *outcome variable* (Δfix versus win margin). That was wrong, and
the independent advocate pass caught it: the **`all`-scope win margin gives the identical arm
ordering to Δfix**, so the outcome variable changes nothing and the **scope** is the active
ingredient.

Read plainly, that is nearly tautological — `no_loop` removes exactly the pairs whose variance
a 4-gram counter explains. What the pair of numbers actually localises is *where* the fluency
confound lives: large in `all`, mostly gone under the loop control. It does not clear the
result, because the loop control conditions on a mediator (falsifier §3). Leave-one-arm-out:
Score 1 beats the 4-gram counter in 7 of 8 subsets under `no_loop` and 1 of 8 under `all`. At
n = 8 this is not decidable either way.

**A caveat on the +1.000 that I under-stated, also from the advocate pass.** The Score-2
ordering is not a clean single-variable comparison. `optimize_banded.py:116` records that the
two roles differ in **band, direction file and aggregate** — three variables, not one — and a
binary predictor carrying only *which objective a string was optimised against* reproduces
**ρ = +0.894** of that +1.000 over the four pro arms (2-vs-2 group difference; I verified the
arithmetic). So §5's headline is the same species of group-difference artifact as the
eight-arm ρ in §4, one level down. The comparative direction survives; the magnitude of the
gap does not mean what it looks like.

What *is* robust across both scopes: the sign-only predictor's **+0.873**, and therefore §6's
magnitude-versus-sign conclusion. That does not depend on scope, outcome variable, or the
band/direction confound.

---

## 6. Verdict: ranking the three hypotheses

**(i) `d` is the causal channel** — the on-`d` component is what changes behaviour.
**NOT SUPPORTED, and I think effectively ruled out.**

Four independent lines, three of them requiring no judge at all:

1. 16.1× more on-`d` push, applied directly, gives 8 wins to 8 losses, p = 0.32 (§1).
2. Behaviour per unit on-`d` differs between the families by roughly 60-fold, range 50–75
   over 18 estimator/judge/arm combinations (§1).
3. Only 3.84% of the prefix's displacement lies along `d`; 96% goes somewhere the metric
   cannot see (§1).
4. The sign pattern is present in the prefix family and absent in the injection family (§2).

Add the prior work: `POSITIVES.md` §6 cites Mishra, Khashabi & Liu (arXiv:2604.09839) proving
no prompt reproduces steering's internal behaviour, so the negative result here was expected
at the activation level. LEAD — I did not fetch that paper; `writeup/related-work.md` is where
verification of it lives.

**(iii) `d` is merely a useful search objective** — optimising it finds behaviourally
effective text for reasons largely unrelated to the on-`d` component.
**BEST SUPPORTED for magnitude.**

Score 1 magnitude orders the pro arms at ρ = +0.400 and its top string underperforms (§5). A
4-gram counter beats it (§5). The score lives in `etsy` and ` "@` while the behaviour tracks
`bully` and `bystand` in the one arm where leakage is large (§3, §4). Ten percent of the
score sits in all the readable English combined.

**(ii) `d` is a correlate of the causal channel** — the prefix moves something that
co-varies with `d`, so optimising `d` finds it without `d` being the lever.
**LIVE, and the only hypothesis the Score-2 result supports.**

Score 2 orders the pro arms at ρ = +1.000 (§5), which magnitude-blind hypothesis (iii)
does not predict. The sign result is the other pillar: `score1_top` wins and `score1_anti`
loses under the loop control from runs differing in `SIGN` alone, so *polarity* of `d`
carries real behavioural information even though *magnitude* along `d` carries little.

**The synthesis I would actually defend.** The sign of `d` is informative; the on-`d`
magnitude is not the channel. Optimising against `d` reliably finds text that shifts
behaviour in the signed direction, and it does so through mechanisms — topic, register,
fluency — that the metric does not measure and cannot report. That is (iii) with a
(ii)-shaped exception at the conjunction, and it is a genuinely interesting position: it says
interpretability-guided search can work as *search* while the interpretation of what it found
is wrong.

---

## 7. The single measurement that would separate (ii) from (iii)

**The placebo, run label-shuffled, matched on |LIVE|.**

(i) versus (ii) is already decided — the injection arm is the clean instrument for on-`d`
displacement and it comes back flat at 16× the dose. That work is done.

(ii) versus (iii) is not, and one run settles it. Search a 32-token prefix against a
direction refit on **label-shuffled** seed pairs — same texts, same corpus geometry, labels
destroyed — to the same |LIVE| the real arms reached, then run the identical behavioural
protocol.

- If the shuffled-direction prefix moves behaviour as far as the real one: **(iii)**. GCG
  against any direction of this kind produces behaviourally active text, and `d`'s content is
  not what made it work. Mody et al. (arXiv:2607.25907) report this outcome for a placebo
  random direction.
- If it comes back null like `random32` did: **(ii)**. The fitted direction is special, the
  effect depends on what `d` encodes, and the mechanism is a correlate rather than the
  component.

Two design points that are not optional. **Label-shuffled, not isotropic** — isotropic
vectors are the weak null at 5120 dimensions, |cos| ≈ 1/√5120 = 0.014, so beating them means
nothing. And that observation is **prior work, not ours**: `POSITIVES.md` §6 grades it
ALREADY ESTABLISHED citing SteerCheck (arXiv:2608.24335) and Hewitt & Liang 2019.
`writeup/novelty.md` §5(d) presents it as MINE and novel, and that is wrong.

**Match on |LIVE|, not on iterations.** `season3_gcg_aggregate_asymmetry.md` reports search
efficiency varying by 1.603× with sign alone and 3.042× between min and max. **Corrected
2026-09-08: treat the 1.603 as withdrawn** — recomputed from the committed histories it decays
monotonically after iteration 150 and crosses 1.0 at ≈ iteration 420, so it is a property of a
window (checkpoints 50–200), not of the runs. The practical instruction is unchanged and if
anything firmer: match on achieved |LIVE|, because iteration counts buy different amounts of
displacement at different points in a search that **never converged** — all eight Season 3 runs
gain 17–51% of their final value in their last half.

Cost, from `optimize_banded.py:167-171` where the smoke path already swaps in random unit
vectors: one flag plus a recomputed `D_TAG`, 1 GPU × 4h, then 50 generations and one judge
batch. It has been the top open item since 2026-08-28.

---

## 8. What this document cannot do

- **No new measurement.** Every number is read out of a committed file or divided by another
  number from one. Nothing here needed a GPU and nothing here is new evidence.
- **The Season 2 geometry and the Season 3 behaviour are different experiments.** §1 and §2
  are Season 2 (`pro_top`, one direction, L24, injections from 2026-06-10). §4 and §5 are
  Season 3 (`score1_top` etc., two banded objectives, single judge). I compare them because
  the mechanistic question spans both, and no artifact measures the Season 3 prefixes'
  residual geometry at all. `normalization_check.json` has `pro_top` and `pro_coherent`, not
  `score1_top`. **That is a real gap and it is cheap to close:** one forward pass per Season 3
  arm would give rotation, ‖Δ‖ and on-`d` for all eight, and then §5's ordering could be run
  against on-`d` displacement instead of against the score.
- **n = 4 everywhere it matters.** §5's two ρ values cannot reach significance. Treat them as
  a direction of evidence.
- **One string per condition.** `prefix_eval_s3.md` says it: "two strings cannot separate
  'more score' from 'different string'."
- **The 4-gram rival (§5) is unresolved by anything here.** Degeneration may be a mediator
  rather than a confound, and no committed artifact separates those. The paraphrase-normalised
  re-rate is the $0 test.
