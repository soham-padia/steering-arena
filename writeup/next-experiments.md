# What to run next, ranked by information per GPU-hour

Written 2026-09-08 overnight. Ranked by **information per GPU-hour**, not by how interesting
the result would be. That metric puts nine experiments that need no GPU at all above
everything else, so the file is organised in tiers and ranked within them.

Each entry gives: the question, **the hypothesis it could kill**, a design with a
pre-registered decision rule, the cost, which existing script it extends, and how you would
know it went wrong. An experiment that cannot come back negative is not on this list.

**Constraints applied throughout.** $0 forever — no paid API, no paid compute. AICR cluster,
2-GPU cap, B200s. NDIF is canonical for any published number; local runs are for search and
sweeps. **Any scoring change forces a new season.** Never `set -u` in an sbatch script. Log
every job to `/work/neu/p2026_0037_neu/runlog/commands.jsonl` via
`~/trauma-experiment-gpt/bin/runlog.py` — `add --why` before, `note` after.

**One trap that applies to five of these.** `stats` computes its fixed baseline as the mean
over *the arms of this experiment*, so adding a 9th arm to the Season 3 eval shifts
`delta_mean_fixed_baseline` for all eight existing arms and every published number moves.
Any new arm goes in a **new experiment tag**, never bolted onto `s3`. The scope is recorded
in the JSON, so it is detectable — but detect it in advance.

**A naming collision to avoid.** `scripts/gcg/k3_control.py` and `season3_k3_control.md`
already exist and are about **k = 3 token positions mutated per GCG step**. They are *not*
about k-of-n layer aggregation. Experiment 11 below is the layer one and it is unrun. Do not
let the two be conflated in a slide or a commit message.

---

## Tier 0 — no GPU, no NDIF, $0. Do these first.

By the ranking metric these are undefined-over-zero. Four of them bear on numbers already
published.

### 1. A second rater on Season 3

**Question.** Does a human agree with `claude-opus-5/v2` about the Season 3 arms, and
specifically about the anti pole?

**What it could kill.** The entire Season 3 headline. This is the top item because of a
number the novelty sweep found: `arXiv:2606.13685` measures a judge verdict flip rate of
**13.6%**. Applied to the surviving sign-flip contrast, `score1_top` absorbs **6** flipped
verdicts before losing significance and `score1_anti` absorbs **2**. The half of the argument
that carries the sign claim does not survive the flip rate the literature measures. Add the
repo's own precedent: in Season 2, human-LLM agreement was 22/27 (81%) on coherent arms
against 11/26 (42%) on the degenerate arm (p = 0.0047), the human **inverted the sign** on
`anti_top` (14 of 17, p = 0.0127), and a p = 0.0005 result was withdrawn over it.

**Design and pre-registered decision rule.** Rate the existing 400-row blind CSV. Pre-commit
before rating: *`score1_anti` must remain below 0.05 on the human sign test in the `no_loop`
scope for the sign-flip claim to stand. If the human's modal preference on `score1_anti`
inverts — as it did in Season 2 — the claim is withdrawn, not caveated.* Write that sentence
into a file and commit it **before** the first rating. The project has done this correctly
three times already — see §11's note on `season3_k3_control.md` — and the discipline is worth
keeping precisely because the last criterion fired against the project. Focus the sample on the
anti arms; they need the power and `--focus` exists for this.

**Cost.** $0, zero GPU. Your evenings. 366 pairs is a lot; 100 concentrated on
`score1_anti` and `score1_top` is enough to answer the question that matters.

**Extends.** `python scripts/rate_blind.py --csv data/analysis/prefix_blind_s3.csv --focus`.
Already wired; `scripts/prefix_behavior_eval.py stats` picks up the `human` label
automatically and computes agreement for free.

**How you would know it went wrong.** You start recognising strings. Rate in one sitting per
arm-blind block, do not look at `prefix_blind_key_s3.json`, and stop if you can predict the
condition. Also: 0 uses of the `n` (no stance) key across all 96 human ratings in Season 2
(`_falsifier/2026-08-27-addendum-human-ratings.md` §N4) means that key produced no data last
time — decide in advance what it means and use it.

### 2. Re-rate on paraphrase-normalised continuations

**Question.** Is the blind judge rating kindness or fluency?

**What it could kill.** The behavioural headline, and it is the objection I could not answer
from committed artifacts. A distinct-4-gram rate orders all eight arms at **ρ = +0.881**
against Δfix, beating Score 1's +0.857
(`_falsifier/2026-09-08-behavioural-result-red-team.md` §2). The manipulation moves looping
across a 24× range: `score1_top` 1/50, base 7/50, `score2_anti` 24/50.

**Corrected 2026-09-08, and it sharpens this experiment rather than removing it.** In the
`no_loop` scope the 4-gram rate falls to **+0.762** while Score 1 rises to **+0.881** and
Score 2 to **+0.952**. I first attributed that to the outcome variable; that was wrong — the
`all`-scope win margin orders the arms identically to Δfix, so the **scope** is the active
ingredient, which is close to tautological because `no_loop` removes exactly the pairs a 4-gram
counter explains. The fluency confound is therefore large in `all` and mostly removed by a
control that itself conditions on a mediator (experiment 3). **Add a step:** pre-register both
the primary scope and the primary outcome variable *before* re-rating, or the paraphrase result
will be readable two ways as well.

**Design and pre-registered decision rule.** Rewrite each of the 450 cached continuations to
remove repetition while holding content, then run the identical blind protocol on the
rewritten text. Pre-commit: *if `score1_top` stays above 0.6 win-rate and `score1_anti` below
0.4 on paraphrased text, degeneration is a mediator and the claim stands. If both move inside
[0.4, 0.6], the headline is a fluency result and is withdrawn.*

**Cost.** $0, zero GPU. 450 rewrites plus 400 judge calls on the existing rater.

**Extends.** `scripts/prefix_behavior_eval.py` (the `stats` and blind-CSV path) plus a new
normalisation step. New experiment tag — the paraphrased corpus is not the `s3` corpus.

**How you would know it went wrong.** The rewriter changes content, not just repetition.
Gate it: reject any rewrite whose content words differ from the original by more than a fixed
Jaccard threshold, and hand-check 20. If the rewriter "improves" the anti continuations into
kindness, you have measured the rewriter.

### 3. Score the `"!"` initialisation

**Question.** Is 32 copies of `"!"` a neutral starting point?

**What it could kill.** Part of the one-bit-contrast claim. **Note the original motivation for
this experiment is gone:** it was pitched against the published geometric-mean **1.603×**
pro/anti search-efficiency gap, and that figure is now withdrawn — recomputed from the
committed histories the anti/pro ratio decays after iteration 150 and crosses 1.0 at
≈ iteration 420, so 1.603 describes checkpoints 50–200 rather than the runs. There is no
standing asymmetry left for a head start to explain.

**It is still worth the 0.2 GPU-hours**, for a different and simpler reason: if `"!"`×32 is not
score-neutral, then *every* Season 3 |LIVE| figure is reported against a baseline the search did
not actually start from, and the pro and anti runs started at different distances from their
targets. That is a measurement-hygiene question independent of any asymmetry claim.

**Design and pre-registered decision rule.** Score the literal string `"!"`×32 on both
objectives. Pre-commit: *within ±0.005 of baseline on both = neutral, this is closed. Outside
that on either = report the offset alongside every |LIVE| figure in Season 3 and re-derive
the pro/anti magnitude comparison against the initialisation rather than against baseline.*

**Cost.** One forward pass per probe. Call it **0.2 GPU-hours**, and it is the cheapest test
in this file relative to what it bears on.

**Extends.** `scripts/score_banded_local.py --arms <file>` with a one-line arms file. Gate
the string through that script rather than pasting it — never move a searched string through
`repr()`.

**How you would know it went wrong.** The tokeniser gives you something other than 32 tokens.
Assert `n_tokens == 32` before reading the score; `"!"` may merge with neighbours.

### 4. Read the score at every probe position, not just the last

**Question.** Does the metric measure a model state, or the residual of one token?

**What it could kill.** The interpretation of the whole objective. The score is
`cos(R_L(seq ⊕ probe)[-1], d)` — the **last** token of the composed string, with the prefix
16-plus tokens upstream. A junk prefix has an easy route to that one position: reshape
attention there. `season3_gcg_ablation.md` is consistent with this and does not test it — the
expensive positions to delete are `etsy` (−0.03814), ` "@` (−0.03083), `.*\n\n` (−0.02487),
while ` escalate` (−0.00246) and ` intervene` (−0.00262) are the two cheapest of all 32.

**Design and pre-registered decision rule.** For `score1_top` and `pro_coherent`, compute the
on-`d` cosine at every probe position, not just the final one. Pre-commit: *if the soup arm's
shift is concentrated in the last 2 positions and near zero at positions 1..n−2 while
`pro_coherent`'s is distributed, the metric is reading a positional artifact for optimised
strings and a state for written ones — and that is a headline. If both are distributed, this
is closed.*

**Cost.** **0.5 GPU-hours.** 16 probes × 2 arms, one forward pass each, activations already
hookable.

**Extends.** `scripts/gcg/ablate_prompt.py` (it already decodes, re-tokenises and measures
variants) or `scripts/layer_sweep_prefix.py`.

**How you would know it went wrong.** Off-by-one on the layer index. It is
`hidden_states[L + 1]`, not `[L]` — index 0 is the embedding output, and this is the single
most common reproduction bug in the project. Sanity-check against a known arm score first.

### 5. Ask the judge to guess the condition

**Question.** Was the blinding real?

**What it could kill.** Every pairwise result, by showing the judge can identify the
treatment. Note the judging is **not** naive — `scripts/behavioral_eval.py:209-224` already
asks twice with A/B swapped and keeps only consistent verdicts, which is a genuine position
debias, and 34 of 400 pairs were discarded that way. So this tests validity, not position
bias.

**Design and pre-registered decision rule.** Two passes on cached text. (a) Given a pair, ask
which side had a prefix. (b) Harder and more interesting: given a single continuation with no
pair and no prefix, ask whether the prefix that produced it was pro or anti. Pre-commit:
*(a) above 60% = blinding compromised, report it as a limitation on every arm. (b) above 70% =
the behavioural effect is a visible signature rather than a subtle value shift, and should be
described that way.*

**Cost.** $0, zero GPU, 366 + 450 judge calls on cached text.

**Extends.** `scripts/behavioral_eval.py`'s judge path with a new rubric.

**How you would know it went wrong.** Both come back at exactly chance because the judge
refuses to guess and defaults. Force a choice and check the answer distribution is not
degenerate before reading the accuracy.

### 6. Per-axis specificity — the construct-validity test that does not exist yet

**Question.** Does the prefix move behaviour more on the axes where `d`'s margin is larger?

**What it could kill.** "`d` measures pro-human values" as opposed to "`d` measures general
valence". This is *predictive specificity*, and it is the part of construct validity the
project has never touched — it has done convergent (three raters agree) and discriminant
(confound cosines near zero) and skipped the middle one.

**Design and pre-registered decision rule.** `season3_directions.json` gives a per-axis margin
for all 15 axes (`privacy` weakest at 0.20306/0.22089, `empathy` strongest at
0.29551/0.33151). Label the 50 held-out stems by dominant axis, then regress per-axis Δfix on
per-axis margin. Pre-commit: *report the slope and its bootstrap CI, not a p-value — 15 axes
with ~3 stems each cannot support a p-value. A CI excluding zero is specificity evidence; a
flat slope says `d` is a valence axis wearing 15 axis labels, and that is a publishable
negative.*

**Cost.** $0, zero GPU. An afternoon of hand-labelling, or one judge pass.

**Extends.** `scripts/direction_purity.py` (it already computes per-axis structure) plus the
existing Δfix table.

**How you would know it went wrong.** The 50 stems do not span the 15 axes — they are
interpersonal first-person stems and may all load on 3 or 4 axes. **Check coverage before
labelling**, because if 12 axes have zero stems the experiment is not runnable and you should
find that out in ten minutes rather than after the labelling.

### 7. Season 3 residual geometry

**Question.** What do the Season 3 prefixes actually do to the residual stream?

**What it could kill.** Nothing directly — it closes a gap that makes every geometric
statement in `writeup/mechanism.md` an extrapolation. `normalization_check.json` measures
`pro_top` and `pro_coherent`, which are **Season 2** strings. No artifact measures rotation,
‖Δ‖ or on-`d` for any of the eight Season 3 arms.

**Design.** One forward pass per arm at the band layers; record mean rotation, ‖Δ‖ and on-`d`
as `normalization_check.json` does. Then re-run the §5 ordering of `writeup/mechanism.md`
against **on-`d` displacement** instead of against the score. Pre-commit: *if on-`d`
displacement orders the four pro arms better than Score 2's ρ = +1.000, the causal-channel
hypothesis revives; if it orders them worse, hypothesis (i) is closed on Season 3 data too and
not only on Season 2's.*

**Cost.** **0.5 GPU-hours.**

**Extends.** `scripts/normalization_check.py` directly — new arms, same measurement.

**How you would know it went wrong.** `pro_top`'s numbers do not reproduce (50.437°, ‖Δ‖
24.248, on-`d` 0.931). Re-measure it as a positive control in the same run.

### 8. Within-axis label shuffling — a stronger null than the current one

**Question.** Is the fitted direction reading valence, or axis identity?

**What it could kill.** The direction's held-out separation as evidence about valence.
`scripts/direction_null.py` shuffles labels globally, which destroys the axis structure along
with the label, so the shuffled refit has nothing left to read — separation falls to 0.519
(0.265–0.794). Shuffling **within each of the 15 axes** preserves topic and destroys only
valence, which is the ecology-style constrained null and the harder test.

**Design and pre-registered decision rule.** Refit on within-axis-shuffled labels, n = 40
draws to match the existing null. Pre-commit: *within-axis separation at or below 0.55 = the
direction reads valence, closed. Materially above 0.55 = some of the real fit's separation is
axis identity, and §5 of `REVISIONS`' open list gets a real answer.*

**Cost.** $0 if the activations are cached (`scripts/capture_all_layers.py` output);
minutes of CPU. **0 GPU-hours** if cached, ~1 if not.

**Extends.** `scripts/direction_null.py` — add a stratified-shuffle mode alongside the global
one.

**How you would know it went wrong.** 9 pairs per axis means within-axis shuffling has few
distinct permutations and the null becomes degenerate. Count the achievable permutations
first; if it is under ~50 per axis, report the null as exact rather than sampled.

### 9. A matched-magnitude pro arm, pulled from the run log

**Question.** Does `score1_top` still beat baseline when matched to `score1_anti`'s achieved
displacement?

**What it could kill.** The claim that the sign-flip contrast is well-controlled.
`score1_top` reaches +0.16395 and `score1_anti` −0.10135 — **62%, and that gap is real.**

> A "corrected" version of this paragraph briefly claimed the gap was a file-selection
> artifact, on the grounds that `score1_anti` shipped a stale `iter 204` snapshot while the run
> reached 0.13186 at iter 537. **Retracted the same day.** That compared `history.jsonl`'s
> `score` column against a `board_score`: at iter 537 `board_score` is **0.07105**, against
> iterate 204's **0.10245**. Iterate 204 is the run's board maximum and `best.json` is correct.
> The "anti search is 1.603× more efficient" figure remains withdrawn for the separate reason
> given in `_falsifier/2026-09-08-...` §8.

**What is real and still unexplained:** the anti run's board score peaked at iterate 204 of 538
and never improved over the remaining 334 iterations, while the pro run peaked at 746 of 846.
The searches plateaued at very different points, which is worth a look before any
matched-magnitude claim.

**Design.** Extract the iterate at |LIVE| ≈ 0.101 from the winning run's `history.jsonl` —
`season3_gcg_ablation.md` already demonstrates exactly this, recovering iterate 420 at
+0.14045 and independently re-measuring it to within 1.4e-4. Generate 50 continuations, judge
against the same fixed baseline. Pre-commit: *magnitude-matched `score1_top` must stay above
0.65 win-rate in `no_loop` for the contrast to be called matched.*

**Cost.** Extraction free. 50 generations ≈ **0.5 GPU-hours**, plus one judge batch.

**Extends.** `scripts/gcg/ablate_prompt.py` for extraction, `scripts/prefix_behavior_eval.py`
for the arm. New experiment tag.

**How you would know it went wrong.** You gate the string through `repr()` and corrupt its
newlines — five live board rows are already broken this way. Run
`scripts/score_banded_local.py --arms <file>` and confirm the score matches the history entry
before generating anything.

---

## Tier 1 — needs a real allocation

### 10. The 2×2: both poles × {fitted, label-shuffled} direction

**This is the experiment.** It is the placebo that has been open since 2026-08-28, upgraded by
what the novelty sweep found tonight into something no prior work has run.

**Question.** Is the *fitted* direction special, or does GCG against any direction of this
kind produce signed behavioural change?

**What it could kill.** The central claim, outright. And the literature says to expect it to:
Mody et al. (`arXiv:2607.25907`) Table 3 ran **both** placebos — random unit (behavioural
shift 0.43) and a CAA direction over a random relabelling of the contrast (**0.51**) —
against the real direction (0.44). **The shuffled null moved behaviour further than the real
direction.** VERIFIED by the sweep. So the honest prior is that this comes back against us,
which is exactly why it is worth the GPU-hours.

**Why it is a 2×2 and not a placebo.** The sweep mapped the cells:

| | pro pole | anti pole |
|---|---|---|
| **fitted `d`** | done (`score1_top`) | done (`score1_anti`) |
| **label-shuffled `d`** | **empty** | **empty** |

Walsh & Barkett (`arXiv:2605.25151`) have both poles and no placebo. Mody et al. have both
placebos and one pole. SSR (`arXiv:2503.06269`) has neither. **Nobody has run both poles
against a shuffled direction.** That fills the two empty cells and it is the design worth
pre-registering publicly.

**Design and pre-registered decision rule.** Refit `d` on label-shuffled seed pairs, then run
GCG at both poles to |LIVE| **matched to the fitted arms' achieved values** — 0.164 and 0.101
— not to matched iterations, because no Season 3 run converged (all eight gain 17–51% of their
final value in their last half), so equal iteration counts buy unequal displacement. Then the
identical behavioural protocol, new experiment tag. Pre-commit:

> *If the shuffled-pro and shuffled-anti arms show the same signed separation as the fitted
> arms, the fitted direction's content is not what made it work, and the claim is withdrawn in
> favour of "GCG against a linear direction in this geometry produces signed behavioural
> change" — which is a different and smaller result. If the shuffled arms come back null like
> `random32` (p = 0.42436), the fitted direction is special and the claim stands. If only the
> shuffled-pro arm works, the effect is a general "optimised prefixes read as kinder" artifact
> with no sign structure.*

**Label-shuffled, not isotropic**, and cite rather than claim the reason: isotropic vectors are
the weak null at 5120 dimensions (|cos| ≈ 1/√5120 = 0.014), and that observation is prior work
— SteerCheck (`arXiv:2608.24335`) and Hewitt & Liang 2019, per `_advocate/POSITIVES.md` §6.
Running isotropic as a *third* condition is cheap and makes the comparison to Mody et al.
direct, since they ran both.

**Cost.** 2 GCG runs at ~750 iterations ≈ **8 GPU-hours** (1 GPU × 4h each, per the costing in
`writeup/novelty.md` §6), plus 100 generations ≈ 1 GPU-hour, plus 2 judge batches at $0. Call
it **9–10 GPU-hours**, one `b200-batch` allocation. Add a third isotropic pair and it is ~18.

**Extends.** `scripts/gcg/optimize_banded.py` — lines 167–171 already substitute random unit
vectors for `DIRS` on the smoke path, so this is a `--random-direction {isotropic,shuffled}`
flag plus a recomputed `D_TAG`; everything downstream reads `DIRS` as a module global. **The
one real gotcha:** that substitution lives inside `if SMOKE:`, which also remaps the band for
GPT-2's 12 layers. Decouple the two or you will run the real model against a remapped band.
Direction refit extends `scripts/build_season3_directions.py` + `scripts/direction_null.py`.

**How you would know it went wrong.** Three ways, all checkable. (a) The shuffled direction
has non-trivial cosine with the real one — assert |cos| < 0.05 before searching. (b) The
shuffled arms fail to reach matched |LIVE| in the iteration budget, in which case you have an
unmatched comparison and must report the achieved values rather than claiming a match. (c) The
shuffled refit's held-out separation comes back near 1.000, which would mean the shuffle did
not take.

**Does this need a new season?** No. It changes no scoring config on the live board — it is an
offline search against a non-shipped direction. Only shipping a new `d`, layer, or scoring mode
forces a season break.

### 11. The k-of-n layer sweep

**Question.** Does behavioural validity vary monotonically with how strict the cross-layer
aggregation is?

**What it could kill.** The conjunction story, which currently rests on 4 points at n = 4.
And it should be run because it is the **most defensible open contribution left**, which is a
correction to what this file said on 2026-09-08. `arXiv:2310.02743` (Coste et al.) establishes
that worst-case aggregation "practically eliminates overoptimization" — but against a
**single** reward model, not a mean, and over an ensemble of separate **models**, not layers.
`arXiv:2503.06269` (SSR) varies the **number** of layers and never the **rule**, aggregating
with an alpha-weighted sum throughout. So min-versus-mean over *per-layer probe directions* is
unclaimed, the sweep searched for it and found nothing, and SSR's authors leave it open in
print: *"we leave it as an exercise for the reader."* The contribution is the curve **and** the
rule comparison, in a setting neither paper covers.

**Design and pre-registered decision rule.** `_aggregate` in `scripts/gcg/gcg_utils.py:43-84`
is one function; k-of-n is `stacked.topk(K, dim=0).values[-1]`, subdifferentiable, routing
gradient to one layer — the same argument that justifies `min`. `k = n` reduces to `MIN` and
`k = 1` to `MAX`, giving two free regression tests. Run `k = 2` and `k = 3` pro arms, plot
gameability (|LIVE| per 100 iterations) and validity (ρ against behaviour) on one x-axis.
Pre-commit the direction of the predicted monotonicity **before** running.

**Write the criterion down and honour it — there is a good precedent, and I misread it at
first.** `season3_k3_control.md` rests on a criterion stated in commit `6304f2c` at
2026-09-06 21:16:16, three hours before the anti arm's run directory
(`2026-09-07T04-17-25Z`) existed and ~14.5 hours before the result. The pro and anti ratios
coincided at 1.276× and 1.260×, which by the stated rule meant *generic search improvement*.
The criterion **fired against the project and was honoured**: commit `17bed29` is titled
"the k=3 control fired: right conclusion, wrong evidence" and withdraws the evidence it
undercut. An earlier version of this file said the document "argues around" the result. That
was wrong and unfair to the record. Do the same thing here: state the rule with its negative
branch, timestamp it, and report whatever it says.

**Cost.** 2 GCG runs ≈ **8 GPU-hours**, plus 2 behavioural arms ≈ 1 GPU-hour. New experiment
tag, because 2 more arms would shift all eight existing Δfix values.

**Extends.** `scripts/gcg/gcg_utils.py` (`_aggregate`), `scripts/gcg/optimize_banded.py`
(a `--k` argument).

**How you would know it went wrong.** `k = 4` does not reproduce the existing `MIN` run's
trajectory, or `k = 1` does not reproduce `MAX`. Those are the free regression tests — run
them first and stop if either fails.

### 12. A fluency-constrained anti arm

**Question.** Is there an anti-human effect that is not degeneration?

**What it could kill.** The `no_loop` scope's legitimacy, by making it unnecessary. Right now
both available scopes are biased: `all` is confounded by degeneracy, and `no_loop` conditions
on a mediator while dropping 36–62% of anti pairs against 11–19% of pro pairs. The fix is to
control degeneracy **at search time** rather than at analysis time.

**Design and pre-registered decision rule.** Add a distinct-4-gram floor to the GCG objective
— reject candidates whose induced continuation degenerates, or penalise the aggregate. Search
an anti arm to matched |LIVE|. Pre-commit: *a fluency-constrained anti arm that still loses
significantly on the full unrestricted sample establishes an anti-human effect independent of
degeneration. One that cannot reach matched |LIVE| under the constraint establishes that the
anti pole IS degeneration, which is equally publishable and closes a claim withdrawn twice
already.*

**Cost.** ~**6 GPU-hours** — the constraint needs generation inside the search loop, which is
expensive, so consider a cheap proxy (repetition of the prefix's own tokens) first.

**Extends.** `scripts/gcg/optimize_banded.py` objective, `scripts/gcg/gcg_utils.py`.

**How you would know it went wrong.** The constraint makes the search fail to move at all, and
you cannot distinguish "no non-degenerate anti direction exists" from "the constraint is too
tight". Sweep the threshold rather than picking one, and report the frontier.

---

## What I would not run, and why

- **More layers in the causal ablation curve.** `REVISIONS` §2 already ran five and got a
  uniform null *for the expected reason* — the edit is 1.1% to 2.8% of the norm at every
  depth, about a degree everywhere. The missing sweep is on `k` at one layer (`REVISIONS` §7
  item 1), not on depth.
- **Re-scoring settled things on NDIF.** The calibration closed it: |gap| max 3.71e-4,
  Spearman 1.0, 0 rank inversions of 1225. Spend the scarce resource on the open question.
  Do re-score on NDIF anything that becomes a *published* number.
- **Cross-model transfer, as currently designed.**
  `_falsifier/2026-08-27-addendum-human-ratings.md` §N5: model, tokenisation and direction all
  vary at once, because the REGISTRY gives each model its own `d`. And the tidy explanation is
  already dead — OLMo-3-32B is itself a base checkpoint, so "the Llamas are base models"
  cannot explain why `pro_coherent` works here (+0.64, p = 0.0051) and not there. Redesign
  before rerunning.
- **A cruelty-focused anti rubric.** The anti pole is degeneration, not cruelty — withdrawn
  twice. A new rubric would be searching for something the artifacts say is absent. Experiment
  12 is the version of this question that can come back either way.
- **A second GCG seed for search variance.** It answers "is one string representative" with
  one more string, for a similar budget to experiment 11, which answers more.

---

## The order I would actually go in

Tier 0 items 1, 2 and 3 this week — none needs an allocation, all three bear on numbers
already published, and item 1 is the one that decides whether anything else is worth doing.
Then item 4, because it is 0.5 GPU-hours and could reframe what the objective is. Then
**experiment 10**, which is the only item here that a reviewer would call the experiment, and
which is now a 2×2 that no published work has filled.

Item 6 is the one I would build a figure around if it turns out the stems cover the axes.
Check that first; it costs ten minutes and decides whether the afternoon of labelling is worth
starting.
