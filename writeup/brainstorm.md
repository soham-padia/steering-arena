# Ideas you did not ask for, ranked by what they cost

Written overnight 2026-09-08, while Soham slept. This is a **menu**, not a claim register.
Almost everything below is tagged MINE, which means I reasoned it and did not measure it.
Where I did measure something I say so and name the two files I multiplied together.

**Process note, and it matters for how you read this.** You asked for this file via the
`research-brainstormer` subagent. That launch was blocked by a session-level safety
classifier — the same one `_local/NEXT_CLAUDE.md` §6 warns about, which fires on earlier
conversation content and keeps firing. Four of my six subagent launches died that way. So
this is me, not the specialist, and it is missing whatever divergence a fresh context would
have brought. The `novelty-scout` launch got through before the classifier started firing;
`writeup/related-work.md` is its work, not mine.

The rule I held myself to: **every idea names the cheapest observation that would kill it.**
Anything I could not attach a test to is in §CUT with the reason.

---

## The one finding I would put above every idea in this file

**§0. A 4-gram statistic predicts your blind judge almost as well as your probe does.**
VERIFIED — I computed this tonight from two committed artifacts, and I believe it is new to
the project.

Take `distinct4_mean` from `prefix_degeneration_s3.json` — the fraction of distinct
4-grams in each arm's continuations, a mechanical text statistic with no judge and no model
involved. Rank the eight arms by it. Then rank the same eight arms by
`delta_mean_fixed_baseline` from `prefix_eval_s3.json`, the blind judge's kindness shift.

| arm | `distinct4_mean` | Δfix | d4 rank | Δfix rank |
|---|---|---|---|---|
| `score1_top` | 0.984 | +0.74 | 1 | 3 |
| `score2_top` | 0.979 | +1.09 | 2 | 2 |
| `score2_top_final` | 0.959 | **+1.20** | 3 | 1 |
| `pro_coherent` | 0.957 | +0.58 | 4 | 4 |
| `random32` | 0.906 | −0.17 | 5 | 5 |
| `score2_anti_final` | 0.831 | −0.43 | 6 | 6 |
| `score1_anti` | 0.805 | **−0.83** | 7 | 8 |
| `score2_anti` | 0.682 | −0.48 | 8 | 7 |

Σd² = 10 over 8 arms, so **ρ = 1 − 60/504 = +0.881**.

Compare that against the numbers `prefix_eval_s3.md` §5 reports for the objectives
themselves: Score 2 orders behaviour at **+0.929**, Score 1 at **+0.857**.

So a zero-cost repetition statistic beats Score 1 and comes within 0.048 of Score 2 at
predicting what the judge said. Every one of the three orderings is computed over the same
eight points, so they are directly comparable and equally underpowered.

I do not think this kills the behavioural result, and I want to be careful here, because
the obvious over-reading is available and wrong. Degeneration is plausibly *downstream* of
the intervention rather than a rival cause — the prefix could genuinely shift values and
mechanically reduce looping, and then both statistics would track the same real thing. A
mediator is not a confound. But the project currently offers ρ = +0.929 as evidence that
Score 2 is a good behavioural proxy, and it now has to explain why a 4-gram counter is
almost as good a proxy, using the same eight points. That is a question about what the
*judge* is sensitive to, and it is answerable for $0 from cached text.

What makes this worse than a generic fluency worry: `_falsifier/2026-08-27-addendum-human-ratings.md`
§N3 measured exactly this failure mode in Season 2. Human-vs-LLM judge agreement was
**22/27 (81%)** on the coherent arms and **11/26 (42%)** on the degenerate arm,
p=0.0047 for the contrast. The judge is reliable where the text is clean and at chance
where it degenerates — and degeneration is precisely what the pro/anti manipulation
changes (`score1_top` loops 1/50, `score2_anti` 24/50, base 7/50). The measurement error is
not noise around the effect; it is **correlated with the treatment**. In econometrics that
is differential measurement error, and unlike classical error it does not merely attenuate
an estimate, it can move it in either direction by an unbounded amount.

*Cheapest observation that would kill it:* rate the 400 cached pairs with the continuations
**paraphrase-normalised** — pass each through a rewrite that fixes repetition and holds
content, then re-run the identical blind protocol. If the sign survives on the paraphrased
text, fluency is not carrying it and §0 collapses to "degeneration is a mediator". If the
sign vanishes, the behavioural headline is a fluency result. Cost: 400 rewrites + 400
judge calls, $0 on the existing rater, no GPU.

---

## STRONG — I would defend these

### 1. Score the initialisation. It may explain a published asymmetry for free.

MINE, on VERIFIED code. `optimize_banded.py:206` starts **every** run at 32 copies of
`"!"`. `season3_gcg_aggregate_asymmetry.md` reports that with the aggregate held fixed, the
anti search outruns the pro search by a geometric-mean factor of **1.603** at matched
iterations, and attributes the whole family of asymmetries to aggregate structure
("min is a conjunction and saturates; max is a disjunction and does not").

> **CORRECTED 2026-09-08 — the premise of this idea is gone; see §15 below, which measured it.**
> The 1.603 covers checkpoints 50–200 only; recomputed over the full histories the anti/pro
> ratio decays after iteration 150 and **crosses 1.0 at ≈ iteration 420**, ending at 0.90. And
> no run converged, so "saturates" is about rate, not a ceiling. There is no standing asymmetry
> for a non-neutral initialisation to explain. The experiment is still cheap and still worth
> doing for measurement hygiene — if `"!"`×32 is not score-neutral, every Season 3 |LIVE| is
> quoted against a baseline the search did not start from — but the motivation below is void.

That explanation cannot cover the score1 column, because score1 pro and anti use the same
MEAN aggregate. Something else makes the anti search 1.6× faster there. The obvious
candidate nobody has checked: **the starting point is not neutral.** `"!"` is an emphatic
token; 32 of them may already sit on the negative side of `d`. If LIVE for `"!"`×32 is
negative, the anti run starts downhill and the pro run starts uphill, and 1.6× is partly a
head start rather than a fact about objectives.

*Cheapest observation that would kill it:* score the literal string `"!"`×32 —
`python scripts/score_banded_local.py` on one arm file, one forward pass per probe, minutes,
$0. If it comes back within ±0.005 of baseline, the initialisation is neutral and this idea
is dead. If it is meaningfully negative, every pro-versus-anti magnitude comparison in
Season 3 inherits an unreported offset.

I like this one because it is the cheapest test in this file and it bears on a number you
have already published.

### 2. Ask whether the metric reads the prefix or reads the last token.

MINE. The score is `cos(R_L(seq ⊕ probe)[-1], d)` — the residual at the **last token** of
the composed string. The prefix sits 16-plus tokens upstream. So the objective never asks
"is the model in a pro-human state"; it asks "does this prefix change the residual of the
final token of a different sentence". Those two come apart, and a 32-token junk prefix has
an easy way to do the second one: reshape attention at that one position.

`season3_gcg_ablation.md` is consistent with this and does not test it. The expensive
positions are `etsy` (−0.03814), ` "@` (−0.03083), `Unnamed` (−0.02660), `.*\n\n`
(−0.02487) — punctuation and framing junk, the kind of token that restructures attention.
The two most legibly pro-social verbs are the two *cheapest* of all 32 to delete:
` escalate` (−0.00246) and ` intervene` (−0.00262). VERIFIED, that page.

*Cheapest observation that would kill it:* compute the on-`d` cosine at **every** probe
position, not just the last, for `score1_top` and for `pro_coherent`. If the pro shift is
distributed across the probe, the metric is reading a state and this idea dies. If it is
concentrated at the final position and near zero at positions 1..n−1, the metric is reading
a positional artifact — and `pro_coherent`, a readable English sentence, should show the
distributed pattern while the soup shows the spike. Cost: one cached forward pass per probe
per arm, no NDIF. **This is the single most diagnostic cheap experiment I found and I do not
think anyone has run it.**

### 3. Use the 15 axes you already have. This is the construct-validity test.

MINE, on VERIFIED data. `season3_directions.json` gives a per-axis margin for all 15 axes
under both objectives — `privacy` is the weakest (0.20306 / 0.22089), `empathy` the
strongest (0.29551 / 0.33151). The behavioural eval uses 50 interpersonal stems and reports
one pooled number per arm. Nobody has crossed the two.

If `d` measures pro-human values, a prefix that moves it should move behaviour **more on
the axes where its margin is larger**. If it is a general affect or fluency shift, it should
move all axes about equally. That is a predictive-specificity test, and specificity is the
part of construct validity this project has never touched — it has done convergent validity
(three raters agree) and discriminant validity (confound cosines near zero) and skipped the
one in the middle.

*Cheapest observation that would kill it:* label the 50 held-out stems by dominant axis
(an afternoon by hand, or one judge pass, $0), then correlate per-axis margin against
per-axis Δfix. A flat line kills it and is itself a result — it would say `d` is a valence
axis wearing 15 axis labels. A positive slope is the strongest construct-validity evidence
this project could obtain without new generations. Watch the power: 15 axes with ~3 stems
each is thin, so pre-commit to reporting the slope and its CI rather than a p-value.

### 4. Reframe: you measured a Goodhart curve, you just have two points on it.

MINE. The framing that travels furthest is not about pro-human directions at all. It is:
**when does optimising a proxy produce the target, as a function of how hard the proxy is
to satisfy?** You have two proxies of different hardness over the same model, the same
search, and the same behavioural instrument — MEAN over 4 layers, and MIN over 4 layers.
The strict one predicts behaviour better (+0.929 vs +0.857) and ~~saturates faster under
attack~~ **gains |LIVE| more slowly per iteration under attack** (corrected 2026-09-08: §15
below shows no run converged, so nothing saturated). That is two points on a curve nobody has
drawn.

`writeup/novelty.md` §5(c) already proposes the k-of-n sweep that turns this into a curve,
and prices it at about 20 lines. I am adding the framing rather than the experiment: with
`k` running 1..4, `k=1` is MAX and `k=4` is MIN, so you get a monotone family with two free
regression tests at the ends. Report gameability (|LIVE| reached per 100 iterations) and
validity (ρ against behaviour) on the same x-axis. If both move monotonically in `k`, the
figure is the paper.

*Cheapest observation that would kill it:* run `k=2` and `k=3` and find them non-monotone,
or find that ρ against behaviour is flat in `k` while gameability changes. Either says
aggregation strictness is not the governing variable. Cost: 2 GCG runs + 2 behavioural arms
— and mind the trap that adding arms shifts `delta_mean_fixed_baseline` for all existing
ones, so this wants its own experiment tag, not a 9th arm bolted onto Season 3's eval.

### 5. The judge is a probe. Treat it as one and build the 2×2.

MINE, and this is the psychometrics import done properly. Campbell and Fiske's
multitrait-multimethod matrix asks: do two *methods* measuring the same *trait* agree more
than one method measuring two traits? You have exactly two methods (a linear probe on
activations, an LLM judge on text) and two traits (pro-human, anti-human). Fill the 2×2.

The reason to bother is that the project keeps treating the judge as ground truth and the
probe as the thing on trial, and the agreement data says the judge is the less reliable
instrument on the arms that carry the argument. An MTMM analysis makes the asymmetry
explicit: if the **method** factor explains more variance than the **trait** factor, then
neither instrument is measuring the construct and the correlation between them is method
covariance.

*Cheapest observation that would kill it:* if judge-versus-probe agreement within an arm
exceeds probe-versus-probe agreement across the two objectives, the method factor is small
and this framing adds nothing. Both quantities are computable from committed files, $0.

---

## WORTH TESTING — plausible, cheap, I would not bet on them

### 6. Ask the judge to guess the condition. This is the embarrassing one.

MINE. You have 400 blind pairs and a judge that returned a decided verdict on 366 of them.
Nobody has asked it the one question that checks the blinding: **which side had a prefix?**
If it can tell above chance, the pairs are not blind on the dimension that matters, and
"prefers the prefixed continuation" may be "prefers the continuation it can tell is the
treatment".

The harder version is better. Show the judge only the continuation — no prefix, no pair —
and ask it to guess whether the prefix that produced it was **pro or anti**. If it scores
well above chance, the behavioural effect is a visible signature rather than a subtle value
shift, and you can report exactly how visible in bits.

*Cheapest observation that would kill it:* it comes back at chance, and you have a clean
blinding check to put in the methods section. Cost: 366 judge calls on cached text, $0,
no GPU. Either outcome is publishable, which is the mark of a good cheap experiment.

### 7. Within-axis label shuffling — a stronger null than the one you have.

MINE, and this is the ecology import. `scripts/direction_null.py` already argues that
isotropic random vectors are the weak null at 5120 dimensions (|cos| ≈ 1/√5120 = 0.014) and
prefers a **label-shuffled refit**. Good. But global label shuffling destroys the axis
structure along with the valence label, so the shuffled direction has nothing left to read.
Ecology's answer to this is the constrained null: randomise *within* strata so the
structure you are not testing survives.

Here that means shuffling chosen/rejected **within each of the 15 axes**. The refit then
has to find valence without being able to use topic. `REVISIONS_2026-09-05.md` §3 reports
global shuffling drops held-out separation from 1.000 to 0.519 (0.265–0.794), i.e. to a coin
flip. If within-axis shuffling lands materially above 0.519, some of what the real fit
achieves is axis identity rather than valence.

*Cheapest observation that would kill it:* within-axis shuffled separation comes back at
0.52 as well, and the null was already tight enough. Cost: a refit on cached activations,
minutes, $0. I rate this WORTH TESTING rather than STRONG because I expect it to come back
null — the confound cosines are already 0.004 to 0.007 — but it is nearly free and it
closes §5 of `REVISIONS`' open list.

### 8. Instrumental variables: your injection arm is the clean instrument.

MINE, on VERIFIED numbers. You want the causal effect of on-`d` displacement on behaviour.
The prefix is a terrible instrument for it — it moves `d`, and it also injects vocabulary
(`score1_top`: 22/50 continuations carry a prefix word, `zach`×13), changes length, and
changes fluency. That is a textbook exclusion-restriction violation. The **injection** is a
clean instrument: it moves `d` and nothing else.

Run the numbers as an IV and the dissociation gets sharp. `steering_dose.md` gives
behaviour per unit of on-`d` push as **0.017** for the `+1·d` injection and **0.96** for the
`pro_top` prefix — corrected in that same document to "roughly 60-fold (50 to 75), median
61.3 over 18 estimator/judge/arm combinations". If the injection is the valid instrument,
then the IV estimate of `d`'s causal effect on behaviour is 0.017 per unit, and the prefix's
0.96 is ~60× larger, so **at most about 2% of the prefix's effect runs through `d`**. That
is a quantitative version of the claim `writeup/mechanism.md` argues at length, stated in a
frame an econometrician would accept.

*Cheapest observation that would kill it:* a threshold or strongly convex injection
dose-response, which would invalidate extrapolating from the injection to the prefix's much
smaller push. Test it with doses at 0.25 and 0.75 — and note `steering_dose.md`'s own
finding that the positive arm is still **climbing** at α = 1.0, which is the condition
under which linear extrapolation is least safe. This idea depends on a linearity assumption
that the existing data mildly contradicts, which is why it is here and not in STRONG.

### 9. Cross-objective transfer, and a framing trap I walked into.

I want to show this one as a mess, because I got it backwards first.

VERIFIED from `season3_prefix_scores.json`: `score1_top` scores +0.16395 on Score 1 and
+0.05001 on Score 2. `score2_top` scores +0.06518 on Score 2 and +0.07896 on Score 1 —
*higher on the objective it was not optimised for*. My first reading was "the conjunction
generalises to the mean for free, and not the reverse; another point for Score 2."

That reading is an artifact of comparing raw numbers across two objectives with different
natural scales (a min is bounded above by a mean, and the baselines differ: −0.00101 versus
−0.00941). Normalise each string by the **best value anyone has reached on the target
objective** and it reverses: `score1_top` reaches 0.05001/0.06997 = **71.5%** of the best
known Score-2 value, while `score2_top_final` reaches 0.08350/0.16395 = **50.9%** of the
best known Score-1 value and `score2_top` reaches 48.2%. On that normalisation the Score-1
winner is the better all-rounder, which cuts *against* the conjunction story.

I do not think either normalisation settles it, and that is the point: the comparison is
confounded by how hard each objective is to move at all, and two strings per objective
cannot separate that. Recording it so nobody quotes the raw ratio as a result.

*Cheapest observation that would kill the whole question:* a third string per objective at
an intermediate score. `prefix_eval_s3.md` already flags the same gap for the anti
dose-response ("a third anti arm at an intermediate score would discriminate; it does not
exist").

---

## LONG SHOT — probably wrong, cheap, interesting if right

### 10. The pro prefixes are anti-degeneration, and nobody asked why.

MINE, on VERIFIED data. Everyone reads the loop table as "anti makes it loop, that is a
confound". Invert it. Base loops on **7/50**. `score1_top` loops on **1/50** and
`score2_top` on **1/50** (`prefix_degeneration_s3.json`). A 32-token junk prefix makes
OLMo-3 *seven times less* likely to fall into a loop than no prefix at all.

That is a mechanistic effect on decoding dynamics with no obvious connection to
pro-human values, and it is the sort of thing that would be a finding on its own if it
replicated. Candidate mechanism: the prefix raises next-token entropy and breaks the
attractor that produces 4-gram loops. If so, "pro-human direction" is partly an
anti-attractor direction, and the kindness judge is reading fluency, as §0 suggests.

*Cheapest observation that would kill it:* measure mean next-token entropy over the
generated continuations per arm from the cached generations. If entropy is flat across arms
while looping varies 1/50 to 24/50, the entropy story is wrong and something else explains
the loop suppression. $0, cached, no GPU — though check whether logprobs were saved; if only
text was cached this needs a regeneration pass and stops being free.

### 11. Feed the prefix to a model that has never seen this direction.

MINE. `prefix_transfer.md` and `transfer_report.md` already tried cross-model transfer and
`_falsifier/2026-08-27-addendum-human-ratings.md` §N5 shows why the result is
uninterpretable: model, tokenisation, **and** direction all varied at once, because the
REGISTRY gives each model its own extracted `d`. The addendum also kills the tidy
explanation — OLMo-3-32B is itself a base checkpoint, so "the Llamas are base models" cannot
be why `pro_coherent` works on OLMo (+0.64, p=0.0051) and not there.

The version worth running holds the direction fixed and varies only the model, which you
cannot do across architectures — but you *can* across **checkpoints or precisions of the
same model**. Run `score1_top` on OLMo-3-32B in fp16 versus bf16, or against the model's
own instruct sibling if one exists, with the identical `d`. If a prefix that was searched
to 32-token precision against bf16 activations collapses under a precision change, it is a
brittle adversarial artifact rather than a semantic intervention, and that is worth knowing
before anyone calls it a steering method.

*Cheapest observation that would kill it:* the score reproduces within the calibration
tolerance you already established (`|gap|` max 3.71e-4 local-vs-NDIF), in which case the
prefix is robust to numerics and this is a dead end. 1 GPU-hour.

---

## CUT — generated and rejected, so nobody regenerates them

- **"Random-direction nulls are a weak control" as a contribution.** I had this as an idea
  and then found `_advocate/POSITIVES.md` §6 grading it **ALREADY ESTABLISHED**, citing
  SteerCheck (arXiv:2608.24335) and Hewitt & Liang 2019. This matters beyond my CUT list,
  because `writeup/novelty.md` §5(d) currently offers a version of the same point as MINE and
  novel. It is not novel. The *application* — run your placebo label-shuffled rather than
  isotropic — is still the right call; the observation behind it is prior work and must be
  cited, not claimed.
- **"Compile the prefix into a steering vector and compare."** Ruled out at the activation
  level: `POSITIVES.md` §6 cites Mishra, Khashabi & Liu (arXiv:2604.09839) proving no prompt
  reproduces steering's internal behaviour. Your own data agrees at 3.1% of the on-`d`
  displacement. Do not spend a run on it.
- **"Show that prompting beats steering."** AxBench (arXiv:2501.17148) established it —
  0.698 vs 0.297 at 2B, 1.075 vs 0.322 at 9B. Your `pro_coherent` delivering 48% of the best
  GCG arm's effect at 4.1× lower score is a nice instance of a known result, not a result.
- **A cruelty-focused anti rubric.** The anti pole is degeneration, not cruelty — withdrawn
  twice now (`prefix_eval.md`, and CLAUDE.md's withdrawn list: 39/50 loops in Season 2).
  Rebuilding the rubric to find cruelty would be searching for a thing the artifacts say is
  not there. What would have to be different: an anti arm that does *not* degenerate, which
  is a search-constraint problem, not a rubric problem.
- **More layers in the band.** `REVISIONS` §2 already ran the causal curve at five layers
  and found a uniform null *for the expected reason* — the edit is 1.1% to 2.8% of the norm
  at every depth. More layers will not help. §7 item 1 says the same: the missing sweep is
  on `k` at one layer, not on depth.
- **Re-scoring everything on NDIF to make it more canonical.** The calibration already
  closed this: `|gap|` max 3.71e-4, Spearman 1.0, 0 rank inversions of 1225. Spending NDIF
  quota to re-confirm agreement is spending the scarce resource on the settled question.
- **An adversarial-robustness claim for Score 2.** Tempting and unsupported: attacked
  directly, Score 2's winner is still soup at coherence 0.00, and `scripts/gcg/README.md:32-34`
  says so itself. The 0/20-versus-10/20 readability result is incidental robustness against
  strings aimed at the old objective. Do not upgrade it.
- **A second GCG seed to test search variance.** I wanted this and cut it on cost honesty:
  it answers "is one string representative" with one more string. The k-of-n sweep (§4)
  gives you search variance *and* a curve for a similar budget. Prefer that.

---

## What I would do first, if it were my week

Three of these are effectively free and bear on numbers you have already published: score
the `"!"` initialisation (§1), read the score at every probe position rather than the last
(§2), and re-rate the cached pairs on paraphrase-normalised text (§0). None needs a GPU
allocation, none touches NDIF quota, and each has an outcome that is interesting in both
directions. §3, the per-axis specificity test, is the one I would actually build a figure
around.

---
---

# Second pass (research-brainstormer, 2026-09-08)

I am the specialist whose launch was blocked overnight. I read `CLAUDE.md`,
`REVISIONS_2026-09-05.md`, everything above this line, `related-work.md` and
`prefix_eval_s3.md` before generating anything, and I have deliberately not regenerated
§0–§11. Where I touch the same territory I say which item I am extending and what is new.

**One thing makes this section read differently from the one above it, and you should know
it before the ranking makes sense.** The brief said a brainstorm is a menu, not a claim
register — agreed. But six of the items below turned out to be cheap enough to *run* while
I was writing them, from committed artifacts plus one CPU refit off the `/work` activation
cache. So they are tagged **VERIFIED** and they are not speculation: they are measurements
that revise published numbers. The speculative items are still tagged MINE and are still a
menu. Do not let the two blur.

Reproduction: `second_pass_repro.py` in this session's scratchpad
(`/tmp/claude-20371/-home-padia-so-neu-steering-arena/f026d66c-.../scratchpad/`), which is
**ephemeral** — every number below also names the committed artifact it comes from, and the
arithmetic is small enough to redo by hand. Nothing here touched NDIF, the network, or a
GPU. Blocks A–D and F–G run in about four seconds on the login node; block E needs sklearn
and the seed-pair activation cache and takes about six minutes on CPU.

---

## The three measured revisions, before any ideas

### R1. Your behavioural null is 0.400, not 0.500 — and the anti arm does not clear it

**VERIFIED**, from `prefix_eval_s3.json` alone, no new data.

`random32` is described in `prefix_eval_s3.md` §1 as "the load-bearing control for the whole
enterprise". I agree. But it is used only as a *pass/fail* — it comes back null, so the
metric passes — and never as what it actually is, which is an **estimate of the judge's
criterion when shown a meaningless 32-token prefix**. That estimate is not 0.5. Under the
loop control `random32` wins **10/25 = 0.400**. The judge prefers the *unprefixed*
continuation three times in five when the prefix is noise.

Every arm's significance is then tested against the wrong reference. Re-test against 0.400:

| arm | no_loop | p vs 0.500 | p vs 0.400 |
|---|---|---|---|
| `score1_top` | 28/33 | 0.0001 | **0.0000** |
| `score2_top` | 32/37 | <1e-4 | **0.0000** |
| `score2_top_final` | 32/36 | <1e-4 | **0.0000** |
| `pro_coherent` | 24/33 | 0.0135 | **0.0002** |
| `random32` | 10/25 | 0.4244 | 1.0000 |
| `score1_anti` | 6/26 | **0.0094** | **0.1076** |
| `score2_anti` | 7/14 | 1.0000 | 0.5868 |
| `score2_anti_final` | 9/24 | 0.3075 | 1.0000 |

The pro arms get *more* significant. `score1_anti` — the one anti arm that survives the loop
control, and therefore the entire evidential basis for the sign-symmetry claim
`related-work.md` calls "the paper-shaped claim" — goes from **p = 0.0094 to p = 0.108**.

The head-to-head is the same story and is the test I would actually report, because it does
not need the criterion to be estimated at all:

| Fisher 2×2, `no_loop` scope | | p |
|---|---|---|
| `score1_top` vs `random32` | 28/33 vs 10/25 | **0.00065** |
| `pro_coherent` vs `random32` | 24/33 vs 10/25 | **0.0165** |
| `score1_anti` vs `random32` | 6/26 vs 10/25 | **0.237** |
| `score1_top` vs `score1_anti` | 28/33 vs 6/26 | **3.0e-06** |

**The pro pole is distinguishable from a meaningless prefix. The anti pole is not.** The
huge pro-vs-anti contrast (OR 18.7) is real but is carried almost entirely by the pro side.
This is an independent second reason to worry about the anti arm — `related-work.md` got
there via judge flip-rates, I got there via choice of null, and the two do not share an
assumption.

I want to state the counter-argument as strongly as I can, because it is good. `random32`
is **one string** and **25 decided pairs**; the Wilson 95% CI on 10/25 is **[0.234, 0.593]**,
which comfortably contains 0.5. So the criterion is not *established* at 0.400 — it is
merely the best estimate available, and the honest conclusion is that **the anti arm's
significance is not robust to a defensible choice of null, and this dataset cannot decide
between the two nulls.** That sentence belongs in the paper regardless of which way it
resolves.

*Cheapest observation that would kill it:* draw four more random 32-token prefixes and run
them through the existing pipeline. If the pooled random-control rate lands at 0.50 ± 0.05,
R1 evaporates and the published p-values stand. `prefix_eval_s3.md` already lists "multiple
random draws for a null *band* (n=1 here)" as not-run; this is that experiment, and it is now
the highest-value one in the project. Cost: 4 × 50 greedy generations (~5 GPU-min, the cache
already holds the base side) plus 400 judge calls, $0.
*Power, so nobody runs it too small:* separating 0.400 from `score1_anti`'s 0.231 at 80%
power needs **~117 decided no-loop pairs per side**. You have ~25. Four more random draws
gets the control there; the anti side needs more eval prompts, not more strings.

### R2. "More score buys more behaviour" on the pro side is a null when tested pairwise

**VERIFIED**, from `prefix_judge_claude_s3.json` + `prefix_blind_key_s3.json`.

`prefix_eval_s3.md` §2 is titled "More score buys more behaviour — on the pro side only" and
says "**Pro: confirmed.**" The evidence is `score2_top_final` (Δfix +1.20) against
`score2_top` (+1.09), plus four other measures moving the same way. Both arms were rated on
the same 50 prompts by the same protocol, so the comparison can be made *paired*, and nobody
made it paired.

Paired, per prompt, on the judge's own 1–5 ratings:

```
score2_top_final − score2_top:  n = 50   mean = +0.110   sd = 0.600   se = 0.085   t = +1.30
                                signs: +12 / −8 / 30 ties      sign-test p = 0.503
```

Thirty of the fifty prompts got the **identical** rating under both arms. The effect is
+0.11 on a 5-point scale with a standard error of 0.085. This is not a confirmed
dose-response; it is a null with a positive point estimate.

I do not think this is fatal to §2 and I want to be careful about that. The four other
measures moving the same way is genuine corroboration, and a consistent sign across five
correlated measures is worth something even when no single one is significant. But "Pro:
confirmed" is stronger than the data, and the document's own caveat — "two strings cannot
separate 'more score' from 'different string'" — understates the problem: at this effect
size **two strings cannot separate 'more score' from 'nothing'**.

*Cheapest observation that would kill it:* re-rate only these two arms with 32 fresh judge
contexts (200 calls, $0) and see whether the +0.11 reproduces in sign. If it does twice, the
sign test over 100 prompts starts to have teeth. If it flips, §2's pro half goes the way of
its anti half.

### R3. `d` classifies at 1.000 and is only 0.66-reproducible from its own corpus

**VERIFIED**, block E, logistic probe at each layer on the cached seed-pair activations
(`/work/.../cache/seedpair_acts_all64_695903.npz`, 135×64×5120, the same activations the
shipped extraction used).

I went in expecting to find that `d` fails to generalise across the 15 axes — that
`extract_direction.py`'s held-out split is a random split over *pairs*, so every axis appears
in both train and val, and the famous "held-out separation 1.000" is a within-axis number.
That hypothesis is **dead**: leave-one-axis-out separation is 1.000 on 15 of 15 axes at L24,
and 0.993–1.000 across L15–L39. `d` generalises to axes it never saw. Good news, and I would
publish it as a clean positive.

What I found instead is a dissociation. Fit `d` twice on **disjoint halves of the corpus**
and the two vectors agree at:

| layer | LOAO accuracy | cos(d_A, d_B), disjoint axis halves | cos(d_A, d_B), disjoint pair halves |
|---|---|---|---|
| 15 | 1.000 | 0.6612 | 0.7062 |
| 19 | 0.993 | 0.6528 | 0.6982 |
| 23 | 1.000 | 0.6732 | 0.7168 |
| **24 (shipped)** | 1.000 | **0.6677** | **0.7120** |
| 27 | 1.000 | 0.6543 | 0.6964 |
| 31 | 1.000 | 0.6279 | 0.6733 |
| 39 | 1.000 | 0.6075 | 0.6604 |

Two directions 48 degrees apart both separate held-out pairs perfectly. **The decision
boundary is identified; the vector is not.** That is a control-theory identifiability
problem, not a statistics problem — the corpus is not persistently exciting enough to pin
down a direction in 5120 dimensions, only enough to pin down which side of a hyperplane
things fall on. Accuracy is the wrong diagnostic and `REVISIONS` §5 already half-knew this
("probe accuracy cannot rank layers here"); the margin fix it adopted does not address
direction identifiability at all.

This matters because the leaderboard does not score submissions with a classifier. It scores
them with `cos(R, d)` to five decimal places, against one draw of a vector that a resample of
its own defining corpus would have moved by 48 degrees. And it gets **worse with depth**
(0.712 at L24 → 0.660 at L39), right across Season 3's Score-2 band {15, 23, 31, 39}, where
every one of the four per-layer directions is an independent draw.

Two honest caveats. First, 0.66 is enormously above the isotropic null of 0.013 that
`direction_null.py` establishes, so this is not "`d` is noise". Second, each half is fit on
~67 pairs, so some of the disagreement is just small-n variance and a bigger corpus would
shrink it — which is itself the actionable finding.

*Cheapest observation that would kill it:* rescore the eight Season-3 arms under `d_A` and
under `d_B` and compare the orderings. If the arm ranking is identical under two directions
48 degrees apart, the instability is real but behaviourally irrelevant and R3 becomes a
methods footnote. If the ranking scrambles, then §5 of `prefix_eval_s3.md` — "Score 2 orders
behaviour better than Score 1" — is a statement about one draw of a random variable. Cost:
one GPU-hour, 8 arms × 16 probes × 2 directions, no NDIF. **This is the single experiment I
would run first in this whole section.**

---

## STRONG — I would defend these

### 12. The arms are not compared on a common denominator, and fixing that strengthens the result

**VERIFIED.** The `no_loop` scope drops pairs *per arm*, so `score2_anti` is judged on 14
prompts and `score1_top` on 33 — and they are not the same prompts. Any between-arm
comparison then confounds "which arm" with "which prompts survived", and the surviving
prompts are not exchangeable: **loop propensity is substantially a property of the prompt.**
Prompts on which the base model loops average 2.86 looping arms out of 9; the rest average
1.40. Ten of fifty prompts carry 41% of all loop events. (`data/cache/prefix_behavioral_s3/`,
the same 4-gram rule `_degeneration` uses.)

The fix is the standard one from any repeated-measures design: fix the prompt set first, then
compare. Drop every prompt on which **more than one of the nine arms** loops — a rule chosen
without looking at any verdict — and 25 prompts survive. Re-run the sign tests on that common
set:

| arm | balanced (25 prompts) | p | published `no_loop` | p |
|---|---|---|---|---|
| `score1_top` | 17/19 | 0.0007 | 28/33 | 0.0001 |
| `score2_top` | 20/22 | 0.0001 | 32/37 | <1e-4 |
| `score2_top_final` | 19/21 | 0.0002 | 32/36 | <1e-4 |
| `pro_coherent` | **18/21** | 0.0015 | 24/33 | 0.0135 |
| `random32` | 8/18 | 0.8145 | 10/25 | 0.4244 |
| `score1_anti` | 3/19 | 0.0044 | 6/26 | 0.0094 |
| `score2_anti` | 7/18 | 0.4807 | 7/14 | 1.0000 |
| `score2_anti_final` | 9/20 | 0.8238 | 9/24 | 0.3075 |

Everything survives, `random32` moves toward 0.5 (8/18 = 0.444, which softens R1 a little and
I am flagging that against my own interest), and the anti/pro asymmetry is unchanged
(`score1_anti` vs `random32`, Fisher p = 0.079; `score1_top` vs `random32`, p = 0.0051).

The result I did not expect: on the balanced set **`pro_coherent` (18/21 = 0.857) is
statistically indistinguishable from `score1_top` (17/19 = 0.895), Fisher p = 1.000.** Once
you evaluate on prompts where nothing degenerates, a 17-token hand-written English sentence
matches 32 tokens of GCG soup that scores 4.1× higher on Score 1. `prefix_eval_s3.md` §4
already says "much of the behaviour is reachable by asking"; on the clean prompt set it is
*all* of it, within the resolution of this instrument.

*Cheapest observation that would kill it:* it already ran. What would make me withdraw it is
a demonstration that the ≤1-arm rule is doing work — so run the whole table again at ≤0 and
≤2 and report all three. I did ≤0 (n=8 prompts, everything underpowered, pro arms 5/5 to 7/7,
anti arms 1/6 to 3/7 — same signs, no power) and it agrees. Cost: seconds.

### 13. The outcome-variable problem has a principled answer, and the project already built it

**VERIFIED**, and this is my answer to the question the brief flagged: the distinct-4-gram
statistic beats Score 1 on Δfix but loses to it on the no-loop win margin, so which outcome
should you believe?

I built the whole specification matrix — 7 predictors × 8 behavioural outcomes, all Spearman
over the same 8 arms — rather than arguing about two cells of it. The result:

| predictor | Δfix | Δfloat | prefixed mean | intensity | margin all | **margin no_loop** | margin no_leak | −cruelty markers |
|---|---|---|---|---|---|---|---|---|
| Score 1 | +0.857 | +0.857 | +0.857 | +0.048 | +0.857 | **+0.881** | +0.857 | +0.786 |
| **Score 2** | **+0.929** | **+0.929** | **+0.929** | **+0.310** | **+0.929** | **+0.952** | **+0.929** | +0.786 |
| distinct4 | +0.881 | +0.881 | +0.881 | +0.119 | +0.881 | +0.762 | +0.881 | **+0.908** |
| distinct-words | +0.833 | +0.833 | +0.833 | +0.143 | +0.833 | +0.738 | +0.833 | +0.822 |
| −max 4-gram repeats | +0.833 | +0.833 | +0.833 | +0.143 | +0.833 | +0.738 | +0.833 | +0.822 |
| −loop count | +0.778 | +0.778 | +0.778 | +0.108 | +0.778 | +0.683 | +0.778 | +0.766 |
| word count | −0.310 | −0.310 | −0.310 | −0.405 | −0.310 | −0.333 | −0.310 | −0.491 |

Score 2 wins **7 of 8** specifications. The one it loses is the cruelty-marker count, where a
repetition statistic wins — which is exactly what you would predict if the marker count is
dominated by degenerate text. Nothing here is arbitrary.

And the ordering that flips is the one whose outcome variable *contains the confound*. Under
the no-loop margin — the scope built specifically to remove degeneration — distinct4 drops to
+0.762 and Score 1 rises to +0.881. That is not a coin flip between two equally good
outcomes; it is a proxy losing its advantage the moment you take away the thing it proxies.
**So the principled rule is: prefer the outcome variable that has had the known confound
removed, and say so in advance.** The project already committed to `no_loop` as "the only one
worth reading" for the anti arms (`prefix_behavior_eval.py:806`); it should say the same for
every ρ it reports.

The jackknife then decides which claim is fragile. Under Δfix, dropping **any one of the three
pro GCG arms** flips the winner from Score 2 to distinct4. Under the no-loop margin, Score 2
wins in 7 of 8 leave-one-out subsets and ties in the eighth, and distinct4 never wins. So
§0's finding above is correct and important, and the right conclusion is narrower than it
looks: *distinct4 rivals the objectives on the outcome that has not been de-confounded, and
loses on the one that has.*

*Cheapest observation that would kill it:* pre-register a ninth outcome variable — a fluency
rubric scored separately from kindness, which is item 3 on `REVISIONS` §7's open list — and
find that distinct4 still beats Score 2 on the de-confounded margin. Cost: 366 judge calls,
$0. If the two predictors also swap under a *different* de-confounding, my rule is doing no
work and this reduces to §0.
**Trap:** with n = 8 arms, a ρ difference of 0.09 (0.952 vs 0.881) is not a significant
difference and I am not claiming it is. The jackknife stability is the load-bearing part, not
the point estimates.

### 14. Run a negative-control *outcome*, not just a negative-control exposure

MINE. This is the epidemiology import and it is a different experiment from §6 above.

`random32` is a negative-control **exposure**: a treatment that should not move the outcome.
Lipsitch, Tchetgen Tchetgen & Cohen (2010) pair it with a negative-control **outcome**: a
measurement that the treatment cannot plausibly affect through the causal path you care
about, but which the suspected confounder *would* affect. If the treatment moves the
negative-control outcome, you have caught the confounder in the act.

Concretely, take the 400 cached blind pairs and re-run the identical protocol with the
kindness rubric replaced by two other questions:

1. **"Which continuation is more grammatically fluent?"** — the suspected confounder,
   measured directly. If arm ordering on fluency matches arm ordering on kindness with
   ρ near 1, the judge is running one comparison and reporting it twice.
2. **"Which continuation describes events happening later in the day?"** — a genuinely
   irrelevant construct that no prefix should touch and that halo effects will happily
   contaminate. Pro arms winning this above chance is a pure halo signature.

This is sharper than the fluency rubric on `REVISIONS` §7's list, which asks the judge to
score fluency *alongside* kindness in the same pass and so cannot separate halo from
correlation. Separate passes with separate contexts can.

*Cheapest observation that would kill it:* the pro and anti arms come out at chance on the
time-of-day question and ordered on fluency in the same direction as kindness. Then you have
a clean halo negative and a *measured* fluency confound to report a partial correlation
against, which is strictly better than the current situation of arguing about it. Cost:
800 judge calls on cached text, $0, no GPU. Both outcomes are publishable.

### 15. No GCG run in Season 3 converged, so every saturation claim is about search *rate*

**VERIFIED**, from the eight `history.jsonl` files under `/work/.../steering-arena/gcg/`.
Running-max of the objective, per run:

| run | iters | final | gain in the *second half* of iterations | iter to reach 90% of final |
|---|---|---|---|---|
| `score1` | 847 | +0.16294 | +16.7% | 543 / 847 |
| `score1-anti` | 539 | +0.13186 | +23.3% | 386 / 539 |
| `score2` (18:53) | 753 | +0.05149 | +22.4% | 403 / 753 |
| `score2` (19:20) | 754 | +0.04607 | +32.7% | 571 / 754 |
| `score2` (00:17) | 502 | +0.06888 | **+50.6%** | 433 / 502 |
| `score2-anti` | 502 | +0.16531 | +23.1% | 372 / 502 |
| `score2-anti-mut3` | 502 | +0.21773 | **+49.4%** | 389 / 502 |
| `score2-mut3` | 502 | +0.06009 | +24.5% | 332 / 502 |

Not one run is flat at the end. Every one gains 17–51% of its final value in its last half,
and none hits 90% of final before 66% of its budget. **The eight arm scores are "what the
search reached before I stopped it", not "what the metric can be pushed to".**

Three published things inherit this. (a) `season3_gcg_aggregate_asymmetry.md`'s "min is a
conjunction and saturates; max is a disjunction and does not" is a statement about *rates
within an unfinished search*, and no run has demonstrated a ceiling. (b) The 1.603 anti/pro
geometric-mean gap at matched iterations is measured in the steepest part of both curves,
where it is most sensitive to where you stop — which stacks on top of §1's `"!"`-init
concern rather than competing with it. (c) The dose-response question in R2 has a cheaper
answer than a third string: **let one run keep going.** A `score2` arm at +0.10 instead of
+0.070 would be a real dose step rather than a 7% one.

*Cheapest observation that would kill it:* restart the `score2` run from its final string
for 500 more iterations. If it gains under 2%, the plateau is real, my reading is wrong, and
you have established a genuine ceiling — which is a better result than the one I am
predicting. Cost: one GPU-run, no NDIF. Log it, and note that a higher-scoring arm is a
**ninth arm** and therefore must not be bolted onto the Season-3 eval; it needs its own
experiment tag, since adding it shifts `delta_mean_fixed_baseline` for all eight existing
arms.

---

## WORTH TESTING — plausible, cheap, I would not bet on them

### 16. Build the cheap distinguisher at the pair level, where you have 366 points instead of 8

MINE, and this is the cryptography import: an adversary that distinguishes two distributions
has an *advantage*, and a construction is only as good as the best distinguisher against it.

Every "the metric predicts behaviour" argument in this project — §5 of `prefix_eval_s3.md`,
§0 above, my own §13 — is a rank correlation over **eight arms**. Eight. The judge produced
**366 decided verdicts**, and the interesting question is answerable at that resolution:
fit a logistic model on per-pair surface features only (distinct-4gram of each side, length
difference, type-token ratio, punctuation and emoji counts, prefix-word leak flag) predicting
which side the judge preferred, and report its AUROC. Then add arm identity and report the
*increment*. That increment is the part of the behavioural result that surface statistics
cannot reach, measured on 366 points with a standard error you can actually quote.

If surface features get AUROC 0.85 and arm identity adds 0.02, §0's worry is correct and the
behavioural claim is mostly a fluency claim. If surface features get 0.60 and arm identity
adds 0.20, §0's worry is bounded and quantified — which is worth more than the current
situation where it is neither confirmed nor refuted.

*Cheapest observation that would kill it:* the surface-feature model comes in near AUROC 0.5,
in which case there was nothing to distinguish and the n=8 correlations were a red herring.
Cost: ~40 lines of sklearn on committed JSON, minutes, $0. This should be run **before** §0's
paraphrase-normalisation experiment, because it costs a hundredth as much and tells you
whether the expensive one is worth doing.

### 17. Run the blind protocol on text that is identical on both sides. This is the embarrassing one.

MINE. Nobody has measured the **instrument's own floor**. Take 50 pairs where side A and side
B are the *same string* — a base continuation duplicated — and push them through
`prefix_behavior_eval.py`'s rating path unchanged, 32 fresh contexts, both orders. The
correct answer is 50 ties and a verdict-vs-ratings agreement of 50/50.

This is embarrassing precisely because everyone assumes the answer. It is also not what §6
above proposes: §6 asks whether the judge can *detect* the condition, which is a blinding
check on a real manipulation. This asks what the protocol does when there is nothing there at
all, and it is the only way to distinguish "the rubric has a floor of 0 spurious verdicts"
from "the rubric has a floor of 6 and everything within 6 of a tie is noise". The related
work has the number that makes it worth doing: Yagubyan's judges flip 13.6% of pairwise
preferences on *repeat presentations of the same pair*, so a nonzero floor here is a live
possibility, not a paranoid one.

*Cheapest observation that would kill it:* 50/50 ties, in which case you have a one-line
methods sentence that pre-empts the most obvious reviewer objection and you spent 100 judge
calls. If it is not 50/50, the number goes in the caveats and every margin under it gets
withdrawn — including, I suspect, R2's +0.11.

### 18. Ask the deployed server for a score you already know. This is the other embarrassing one.

MINE, and I flag that I did not verify the current deployment state — memory says Season 3
went live 2026-09-06 as DB row id 5, and CLAUDE.md still says the Space is on Season 2, so
**check which before doing anything**.

`season3_prefix_scores.json` records offline LIVE scores for all eight arms with a max gap of
7.0e-4 against the recorded values, and `calibration/local_vs_ndif_tf5.10.2_sdpa_695054.json`
establishes local-vs-NDIF agreement at |gap| ≤ 3.71e-4. Both are *offline*. What has never
been checked end to end is whether **the running web server, through its own queue, cache and
`norm_key` path, returns the number the analysis says it should.** Two submissions. If they
agree, you have an integrity check to cite and it cost two NDIF calls. If they disagree, every
board number is suspect and you found out for two calls.

It is embarrassing because it is the most basic conceivable check and because the
`(season_id, norm_key)` cache is exactly the kind of component that silently serves a stale
season's score after a season flip.

*Cheapest observation that would kill it:* the returned score matches to 1e-4 and this is a
dead end. Cost: 2 NDIF calls off the maintainer's quota — the only item in this section that
spends any.

### 19. Ask whether `d` is partly a repetition direction, in activation space, with no judge

MINE. This is the mechanistic version of §0 and of §10, and it is the one question that
settles both without a single judge call.

§0 asks whether the *judge* reads fluency. §10 asks why pro prefixes suppress looping and
proposes entropy. Neither asks the prior question: **is the fitted direction itself partly a
degeneration axis?** It is directly measurable. Take the 450 cached Season-3 continuations,
label each by the mechanical `looping` flag (116 pairs have a looping side), read the L24
residual at the last token, fit `d_loop` by the same logistic recipe, and report
`cos(d, d_loop)`.

If `|cos|` is near the isotropic null of 0.013, then `d` is not a repetition direction, §0's
worry is confined to the judge and cannot be about the metric, and §10's mechanism has to be
downstream. If `|cos|` is 0.3 or more, then "pro-human direction" is partly "anti-degeneration
direction" *at the level of the vector*, which is a far stronger version of §0 than anything
the judge can tell you.

I checked the cheap proxy first and it argues against me, which is why this is here and not
in STRONG: in the seed corpus the **rejected** texts are marginally *more* lexically diverse
than the chosen ones (type-token 0.957 vs 0.944, t = −2.22, signs 40/−54). If a repetition
axis had leaked in from the corpus it would have leaked with that sign, and the anti pole
would be the *less* repetitive one. It is not.

*Cheapest observation that would kill it:* `|cos(d, d_loop)| < 0.05`. Cost: one forward pass
over 450 short strings, ~10 GPU-minutes, no NDIF.

### 20. Sweep the prefix *length*, not the aggregation rule

MINE, and deliberately orthogonal to §4's k-of-n sweep — that varies the objective, this
varies the input budget. The control-theory framing is input-constrained reachability: what
is the set of score values a prefix of `n` tokens can reach?

You have two accidental points. `pro_coherent` is 17 tokens and reaches Score 1 = +0.0398;
`score1_top` is 32 tokens and reaches +0.1640 — 4.1× the score for 1.9× the tokens, which is
*superlinear* and, if it holds, means the leaderboard is a token-budget contest with a
32-token cap rather than a measurement of anything. Run GCG at 8, 16, 32 and 64 tokens
against Score 1 and plot reachable |score| against `n`. And take the behavioural readings at
each length, because §12's finding — 17 hand-written tokens matching 32 optimised ones
behaviourally while scoring 4.1× lower — predicts that the *score* curve and the *behaviour*
curve have different shapes. If they do, that gap is the Goodhart curve §4 wants, obtained on
an axis nobody has to redefine a season for.

*Cheapest observation that would kill it:* reachable score is flat in `n` above 16 tokens,
which would say the metric saturates on input budget and the 32-token cap is not load-bearing.
Cost: 4 GCG runs, GPU only. Note the constraint: nothing here changes the scoring config, so
no new season is forced — but the behavioural arms are again *new* arms and need their own
tag.

---

## LONG SHOT — probably wrong, cheap, interesting if right

### 21. The seed corpus is three characters longer on the chosen side

**VERIFIED** and I rate it a long shot anyway, because the effect is small and partly handled.

Over all 135 pairs: chosen texts average **89.13 characters**, rejected **86.10**, a paired
difference of **+3.02, se 0.740, t = +4.08**, signs 77/−49. Word counts differ by only +0.24
(t = 1.41, n.s.), so this is characters, not words — longer words on the kind side.

`extract_direction.py` does orthogonalise a length direction out, but that direction is built
from **four toy sentences** (`LONG`/`SHORT`: "The committee reviewed the quarterly
schedule…" versus "The cat slept."), a contrast of roughly 100 characters against 15. A
3-character, same-word-count asymmetry is not the thing that basis was built to remove.
`REVISIONS` §4 says approach is the "only audited confound never orthogonalised out"; on my
reading, character length is a *second* one — nominally orthogonalised, but against a basis
too crude to catch it.

*Cheapest observation that would kill it:* build the length direction **from the corpus
itself** — regress the per-pair character-count difference onto the per-pair activation
difference at L24 — project it out of `d`, and re-check held-out separation and the kind>cruel
gap the way `direction_purity.py` did for approach. If they move by less than approach's
0.0008, this is noise and I withdraw it. Cost: minutes on the cached activations, CPU, $0.
Bundle it with R3, which reads the same file.

### 22. Season 3's per-layer directions are four independent draws of R3's unstable vector

MINE, extrapolating R3. Score 2 uses a *different* direction at each of {15, 23, 31, 39}, and
R3 says each is only 0.61–0.67 reproducible. A `min` over four independently unstable
directions is a conjunction of four noisy constraints, and the noise is not shared — which
could be *why* min is harder to game. Not because conjunctions are hard, but because
**min over four independent draws is an implicit ensemble**, and Coste et al.'s worst-case
optimisation result is about ensembles, not about depth.

If that is right, it reframes contribution (b) in `related-work.md`: the thing that makes
Score 2 better is not multi-layer targeting (SSR owns that) and not conjunction (Coste owns
that), it is that per-layer refitting **decorrelates the estimation error**, and you would get
the same benefit from four directions at the *same* layer fit on four bootstrap resamples of
the corpus. That version is testable without touching depth at all, and if it works it is a
cheaper and more general recipe than a layer band.

*Cheapest observation that would kill it:* build Score 2′ = min over four bootstrap-resampled
directions all at L24, run GCG against it, and compare gameability against Score 2. If Score 2′
is as easy to game as Score 1, decorrelation is not the mechanism and depth is doing the work.
Cost: 1 GCG run + one CPU refit. **This forces a new season if it ever ships**, so run it as a
research objective, not a board objective.

---

## CUT — generated, tested where cheap, and rejected

- **"The loop control is under-inclusive, and that is what carries the anti arm."** I
  generated this, and it is the most natural extension of §0, so I tested it. The judge's own
  free-text comments cite looping or repetition in **31% of `score1_anti`'s surviving pairs**
  and **29% of `score2_anti`'s**, against **3%** for `random32` — so the mechanical 4-gram
  filter *is* under-inclusive on exactly the arms that matter. But excluding those pairs makes
  `score1_anti` **stronger, not weaker**: 6/26 (p = 0.0094) becomes **2/19 (p = 0.0007)**, and
  every pro arm survives too. Among the judge-flagged pairs it dropped, the prefixed side had
  won 4 of 7. **The fluency story does not explain the anti arm.** This is a falsification
  attempt that failed, it is the strongest single defence of the behavioural result I
  produced all session, and it belongs in `_advocate/` rather than here. (Caveat, stated
  because it is real: conditioning on a judge-generated variable that is downstream of the
  verdict is a post-hoc filter, so read it as a sensitivity analysis, not an estimate.)
- **Leave-one-axis-out as a construct-generalisation test.** Ran it. 1.000 on 15 of 15 axes at
  L24, 0.993–1.000 across L15–L39. `d` generalises across axes cleanly and there is no finding
  here. This is *not* §3 above, which asks whether the *behavioural* effect is axis-specific —
  §3 survives untouched and is still the best figure in this file. What would have to be
  different to revive my version: an axis that is not in the 15 at all.
- **The E-value.** I computed it (`score1_top` RR = 5.6, E = 10.7; `score1_anti` RR = 3.3,
  E = 6.1) and I am cutting it. VanderWeele & Ding's E-value bounds an unmeasured confounder
  of *exposure assignment*, and here the exposure is assigned by construction — the confounder
  in play is on the measurement side, not the assignment side, so the number would be quoted
  against the wrong causal diagram. §14's negative-control outcome answers the same worry with
  the right design. Recording the numbers so nobody recomputes them thinking they mean
  something.
- **A repetition-optimised placebo arm** (search a prefix that maximises continuation
  distinct-4gram, with no reference to `d`, then judge it). Right question, wrong instrument:
  the objective needs generation inside the search loop, which is orders of magnitude more
  expensive than GCG's single forward pass. §19 answers the same question in activation space
  for 10 GPU-minutes. If §19 comes back with `|cos| > 0.3`, revive this.
- **A per-prompt random-effects model of the verdicts.** Correct, and I cut it on
  cost-per-insight: it needs a mixed-model dependency the repo does not have, and §12's
  balanced prompt set gets the same protection from prompt-level confounding with arithmetic
  the reader can check by hand. Prefer the simpler design.
- **"Readout does not imply control", in any wrapping.** Foreclosed twice over (Elazar et al.
  amnesic probing; Walsh & Barkett). I tried to rescue it via control theory's
  observability/controllability duality and the rescue is cosmetic — same claim, new
  vocabulary. What *did* survive from that import is R3 (identifiability) and §20
  (input-constrained reachability), which are different questions.
- **Rating the eight prefix strings themselves for kindness and correlating with Δfix.** Cheap
  and I cannot make it mean anything: the raters never saw the prefixes, so a correlation
  would not implicate the protocol, and 8 points again. If the leak analysis in
  `prefix_content_s3.json` had shown high verbatim echo it would be worth it — it shows 0–6
  echoes per arm.
- **Re-deriving the Season-2 human-vs-LLM agreement contrast on Season 3.** `raters` in
  `prefix_eval_s3.json` is `{"human": {"rated": 0}}`. There is nothing to compute; it needs new
  human ratings, which is a resourcing decision the maintainer already made ($0), not an idea.

---

## If it were my week

R3 first — rescore the eight arms under two half-corpus directions, one GPU-hour, and find
out whether the behavioural ordering is a property of the model or of a corpus draw. Then §16,
the pair-level distinguisher, because it costs forty lines and it tells you whether §0's
expensive paraphrase experiment is worth running. Then the four extra `random32` draws from
R1, because they are the difference between "the anti pole works" and "the anti pole is not
distinguishable from noise", and that sentence is the paper.

R2 and §12 need no new data at all. They are corrections to `prefix_eval_s3.md` and I would
make them before anything is submitted anywhere: §2's "Pro: confirmed" should read "consistent
in sign across five measures, not significant paired (t = 1.30, sign-test p = 0.50)", and §5's
Spearman table should say which outcome variable it is conditioned on and why.
