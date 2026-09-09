# What is actually new here

## Overnight run, 2026-09-08

**Sections 2, 5(a), 5(d) and 5(e) of this file are now wrong. Read
`writeup/related-work.md` before you quote any of them.** That is the headline of the
night: the literature sweep finished and it pre-empted most of what this file proposed as
novel.

**Produced.** Six documents. `writeup/related-work.md` (the finished sweep, 554 lines) and
§4 of this file are the `novelty-scout` subagent's work. The other five are mine, written
directly:

| file | what it is |
|---|---|
| `writeup/related-work.md` | the literature map, four papers that hurt, at the top |
| `writeup/brainstorm.md` | reframings and experiments, each with its killer observation |
| `_falsifier/2026-09-08-behavioural-result-red-team.md` | 12 numbered objections + what survives |
| `writeup/mechanism.md` | does the prefix work through `d`? verdict + the one measurement |
| `_advocate/2026-09-08-the-case-for-adversarial-probe-validation.md` | the honest case, narrowed |
| `writeup/next-experiments.md` | 12 experiments, ranked by information per GPU-hour |

**A harness problem, and it shaped the output.** Four of my six subagent launches were
blocked by the safety classifier described in `_local/NEXT_CLAUDE.md` §6 — brainstormer,
critic, advocate, and both general-purpose agents. `python3 -c` and some piped `grep` calls
went the same way mid-session. The scout got through before it started firing. So five of
six deliverables are one session's single pass rather than a specialist's, and they are
missing whatever divergence a fresh context would have brought. Nothing is committed; the
tree is dirty for you.

**Corrected 2026-09-08, second pass.** An earlier version of this summary was blunter than
the evidence, and blunter than `writeup/related-work.md`, which is the careful document and
the one to trust. I re-verified every contested citation myself by fetching it — abstracts
for all four, plus Mody's Table 3, Walsh & Barkett's §4.4, and SSR's source code. Two claims
I compressed too far are fixed below; two challenges to the sweep did not survive checking
and the original stands. Details and per-id status in `related-work.md`.

**What the sweep genuinely killed.** Three things, and all three hold up.

*§5(a), sign-consistency as the validity test.* `arXiv:2605.25151` (Walsh & Barkett,
*Representation Without Control*, 24 May 2026) already uses it to reject a direction. I
fetched the body: §4.4 reads *"The absence of sign-symmetric behavioral responses at
opposite steering scales is the clearest evidence that the train-only layer-18 direction
does not function as a reliable control variable for downstream risk behavior."* I had
called it that paper's "headline test", which is too strong — the abstract's headline is the
three-property dissociation and the sign-symmetry run appears there as a subordinate clause.
But it *is* a criterion they use to reject a direction and call their clearest evidence, so
"we introduce sign-consistency as a validity test" is not available. §5(e) is their closing
abstract sentence, verbatim.

*§5(d), the label-shuffled placebo as a methodological improvement.* Dead, and I checked
this one specifically because it was challenged. Mody et al. Table 3 runs **three**
conditions: real eval direction (|z| = 8.3, behavioural shift **0.44**), placebo shuffle — a
CAA direction built from randomly relabelled contrast labels — (|z| = 10.6, **0.51**), and
placebo random unit (|z| = 6.6, **0.43**). The label-shuffled null moved behaviour *further*
than the real direction. Run the placebo anyway, for this `d`; never call it an improvement
on the prior work.

*§3's intuition, that multi-layer objectives are behaviourally realer.* SSR §5.3 quantifies
it: one middle layer 0% attack success *"despite successfully rerouting activations at the
target layer"*, two layers 54%, three layers 100%.

**One thing I overstated, now corrected.** I wrote that Coste et al. `arXiv:2310.02743` make
min-beats-mean established RLHF practice. They do not, and `related-work.md` §4 already
flagged the gap I dropped. Verified from the abstract: their worst-case optimisation (a
minimum over the ensemble) is compared against **single reward model** optimisation, not
against a mean, and the ensemble is over separate *reward models*, not over *layers* of one
model. The phrase *"conservative optimization practically eliminates overoptimization"* is
verbatim and real; the min-versus-mean head-to-head is absent. Treat it as a suggestive
analogy from a different unit of aggregation.

**What survives, and it is narrower than §2 claims but wider than I said.** Three gaps, and
the sweep searched for prior work on each and found none.

1. **Sign-symmetry in input space, with both poles.** Walsh & Barkett flip an *injected
   vector*; you cannot inject a negative vector through the input channel. Mody et al. is
   **suppression toward zero** — the input-side dual of negative steering, with no opposite
   pole at all, so it cannot speak to sign-symmetry. SSR's §6 "reverse-lens" is Steering-SSR
   with a which-tokens-were-found readout, not a blind behavioural comparison of two poles.
   So: nobody has run the sign-symmetry test in input space, and Season 3 returns a
   **positive** where Walsh & Barkett return a null. A delta on intervention class and
   outcome, not a new criterion. Say it exactly that way.
2. **The aggregation *rule*.** SSR varies the layer **count** and never the rule — the
   aggregation is an alpha-weighted sum throughout, which I confirmed directly from
   `ssr/probes/probe_ssr.py`. Coste et al. is models, not layers, with no mean baseline. So
   min-versus-mean over *per-layer probe directions* is open, and SSR's authors leave it
   open in print: *"we leave it as an exercise for the reader."*
3. **The 2×2** — both poles × {fitted, label-shuffled} direction. Walsh & Barkett have both
   poles and no placebo; Mody has both placebos and one pole; SSR has neither.

### Novelty a literature sweep is structurally bad at seeing

Added after Soham pointed out, correctly, that criticism-shaped processes systematically
under-count novelty. Here is the specific mechanism, because it is not a general complaint.

**A sweep can only find prior work for a claim that has a name.** You search "sign-symmetry
validity criterion" and find Walsh & Barkett. You search "min versus mean aggregation" and find
Coste et al. But a finding nobody has *named* returns nothing, and returning nothing looks
identical to not having searched. So the sweep's output is biased toward claims that are easy
to phrase in existing vocabulary — which are exactly the claims most likely to be prior work.
Three things in this project have no vocabulary, and all three came out of asking what the
prior work does *not* do rather than what it does.

**(i) Sign informative, magnitude not — as a joint finding.** MINE. The objective's *sign*
controls the behavioural sign (`score1_top` 28-5-5 at p = 7e-05, `score1_anti` 6-20-4 at
p = 0.00936, one bit apart). The *magnitude along `d`* is not the channel: 16.1× more on-`d`
push applied directly gives 8 wins to 8 losses at p = 0.32257, behaviour per unit on-`d`
differs ~60-fold between the families, and only 3.84% of the prefix's displacement lies along
`d`. Nobody in the sweep reports that pair together, because "sign works, magnitude doesn't" is
not a thing anyone has framed. Walsh & Barkett tested sign and got a null. Mody measured
suppression magnitude. The conjunction of the two results is unclaimed.

**(ii) A three-way dissociation where the middle term is positive.** MINE, and I think the
sweep collapsed this into Walsh & Barkett's two-way version. Theirs is *readout works, control
fails*. Ours is *readout works, **adversarial search against the readout produces signed
behavioural change**, and the readout's own axis is still not the causal channel.* Their third
property failed; ours succeeded. A dissociation in which the intermediate step **works** is a
different and harder-to-explain object than one in which it fails, and "the probe is a good
search objective while demonstrably not the mechanism" is the sentence with no prior owner.

**(iii) The negative pole of a fitted values direction is incoherence, not anti-values.**
VERIFIED, and withdrawn twice in this repo as a *failure* — `anti_top` loops 39/50 in Season 2,
`score2_anti` 24/50 in Season 3, and the "makes the model cruel" reading is on CLAUDE.md's
withdrawn list. Read as a finding rather than as a retraction, it says something specific about
what a contrastive direction encodes: one pole is a concept, the other is the absence of
fluent generation. That is a claim about fitted directions generally and this project has the
cleanest evidence for it I have seen, precisely because it kept publishing the failure.

**And one the sweep did name but buried.** Using a public leaderboard's 618-submission corpus
to stress-test a *later* metric appears in `related-work.md`'s "searched for and did not find"
list, at the bottom. It is the only item there that is already *done* rather than proposed.

**What none of this changes.** Walsh & Barkett still own the sign-symmetry criterion, Mody
still ran the shuffled placebo, SSR still quantified multi-layer targeting. Optimism is not a
counter-argument to a fetched quotation. What it correctly buys is the search for what the
prior work *fails* to cover — which is how (i), (ii) and the two gaps above surfaced, and none
of them were visible from the "what has been done" side.

**Corrections to this file and to the brief I was given.** VERIFIED in
`scripts/gcg/optimize_banded.py:119-124`: for **Score 1**, `--anti` flips `SIGN` only and
the aggregate stays `MEAN`. For **Score 2** it also swaps `MIN`→`MAX` — the correct De
Morgan negation of the same board metric, but a different search problem, since a
conjunction and a disjunction are not equally hard. So §2's "one bit differs" is true of
**Score 1 only**, which is also the only anti arm that survives the loop control
(p = 0.00936; `score2_anti` goes to 7-7-2, p = 1.0). The claim gets narrower and better.

**Three numbers I was told to verify, all confirmed.** The prefix carries 0.931/30.07 =
**3.10%** of a full injection's on-`d` push (`normalization_check.json`); it produces about
twice the behaviour (+0.89 vs +0.50, and the +0.89 is a floating-baseline figure later
corrected to +0.556/+0.796); and **16.1×** more on-`d` push gives wins 8, losses 8,
p = 0.32257 (`steering_dose.json`). The "≈58×" per-unit ratio is corrected in
`steering_dose.md` itself to "roughly 60-fold (50 to 75), median 61.3".

**The thing I found that I did not expect — and its own robustness failure.** Cross-multiplying
`prefix_degeneration_s3.json` against `prefix_eval_s3.json`, a **distinct-4-gram rate orders
the eight arms at ρ = +0.881** against `delta_mean_fixed_baseline`, beating Score 1's +0.857.
A judge-free repetition statistic appeared to predict the blind judge better than one of the
two objectives.

**That does not survive the loop control, and my first diagnosis of why was wrong.**

| predictor | ρ, `all` scope | ρ, `no_loop` scope |
|---|---|---|
| Score 2 | +0.929 | **+0.952** |
| Score 1 | +0.857 | **+0.881** |
| sign-only | +0.873 | +0.873 |
| distinct-4-gram | **+0.881** | **+0.762** |

I first reported this as a change of *outcome variable* — Δfix versus win margin. That was a
mis-diagnosis, caught by the independent advocate pass and confirmed by hand: **the `all`-scope
win margin produces the identical arm ordering to Δfix**, so swapping the outcome variable
changes nothing at all. The active ingredient is the **scope**. Drop the pairs with a looping
side and the repetition statistic falls from +0.881 to +0.762 while both objectives rise.

**And that is close to tautological, which is the honest reading.** `no_loop` removes exactly
the pairs whose variance a 4-gram counter explains, so of course it loses power there. What the
table actually shows is *where* the fluency confound lives: it is large in the `all` scope and
mostly removed by the loop control. That does not clear the result, because §3 of the falsifier
is that `no_loop` conditions on a mediator and drops 36–62% of anti pairs against 11–19% of pro
pairs. Neither scope is clean, and the 4-gram rival marks the size of the problem in one of
them.

Leave-one-arm-out, from the advocate pass: Score 1 beats the 4-gram counter in **7 of 8**
subsets under `no_loop` and **1 of 8** under `all`, with a different thin arm driving the
exception each time. At n = 8 the comparison is not decidable in either direction. The +0.881
appearing twice is a genuine coincidence — both orderings give Σd² = 10 by different routes,
which I checked rather than assumed.

**What does survive both.** Two things, and they are the more useful two. A predictor carrying
only the arm's *sign*, with no magnitude information at all, reaches **+0.873 under both
outcome variables** — so much of the published eight-arm ρ is a four-versus-four group
difference rather than a graded relationship, whichever y you pick. And restricted to the four
pro arms, where the sign bit is constant, **Score 2 orders behaviour at +1.000 and Score 1 at
+0.400 under both** — the comparative claim survives and gets starker than the pooled figure,
at n = 4 where the smallest attainable two-sided p is 0.083.

**The methodological point is worth more than either number.** Which predictor "best tracks
behaviour" is not well-defined until the primary behavioural variable is fixed, and Season 3
never fixed one. Pre-register it next season.

**What I could not verify.** I fetched no paper myself — every citation here and in §4 is
the scout's verification, and it flagged `arXiv:2504.20879` as search-only (do not cite) and
Coste et al.'s mean-vs-min head-to-head as absent from the abstract page. I did not verify
§3's per-100-iteration decay series (`+0.045 → +0.012 → …`); I read the matched-iteration
and ratio tables, not the window statistics. I did not confirm the 50 eval stems are disjoint
from the 16 season3 probes — check that before printing the word "held-out". And the
**39/50 loop figure is Season 2's `anti_top`**, not Season 3; Season 3's `score2_anti` is
24/50.

**The pre-registration record is better than I first said, and I was unfair to it.** My first
pass called the meandiff prediction "the only instance I could verify" and described the k=3
control as a criterion the project "argued around". The independent advocate pass checked the
timestamps and I re-verified them: **both characterisations were wrong.**

- **The k=3 control is the strongest pre-registration in the repo.** Commit `6304f2c` states
  the criterion at 2026-09-06 21:16:16; the anti arm's run directory is stamped
  `2026-09-07T04-17-25Z`, **three hours later**, so the data did not exist when the rule was
  written; the result landed at `17bed29` some 14.5 hours after that. The ratios coincided
  (pro 1.276× against anti 1.260×), the criterion **fired against the project**, and the commit
  that records it is titled *"the k=3 control fired: right conclusion, wrong evidence"*. That
  is the rule being honoured, not evaded. The meandiff instance (`2ebb583` → `581d1a7`) is real
  but only eleven minutes apart.
- **A third one:** `a63cbf0` names `random32` as a control and states the
  Score-2-beats-Score-1 prediction *with its negative branch* fifty minutes before the results
  at `da36587`.
- **The one that does not qualify:** `steering_random_control_preregistration.md` landed in
  commit `8a69828` **together with its own result**. Not a verifiable pre-registration; do not
  describe it as one.

Three genuine pre-registrations, one of which fired against the project and was honoured in
public, is a stronger methodological record than most published work has. Say so.

**Time-sensitive.** The scout found **The Steerability Challenge**, a NeurIPS 2026 Paris
workshop competition — activations explicitly in scope, metrics held private "to prevent
gaming". It **opens 10 September 2026**, two days from now, and closes 21 November. That is a
venue, not just prior art. `writeup/related-work.md` has the details.

**What I would do next, in order.** (1) The second rater on Season 3 — $0, already wired,
and now urgent for a specific reason: against the 13.6% judge flip rate in
`arXiv:2606.13685`, `score1_top` absorbs 6 flipped verdicts and `score1_anti` absorbs **2**.
The half of the argument that carries the sign claim does not survive the flip rate the
literature measures. (2) Re-rate on paraphrase-normalised text, to answer the ρ = +0.881
problem. ~~(3) Re-run `score1_anti` at iteration 537.~~ **VOID — retracted the same day.** The
claim that iterate 204 was a stale snapshot came from reading `history.jsonl`'s `score` column
(the optimiser objective) instead of `board_score` (the LIVE value): at iter 537 `score` is
0.13186 but `board_score` is only **0.07105**, against iterate 204's **0.10245**. Iterate 204
*is* the run's board maximum; nothing is stale and the 62% magnitude gap is real. See the
retraction in `_falsifier/2026-09-08-behavioural-result-red-team.md` §1. Replacement third
item: **rate the ~26 decided `score1_anti` pairs by hand**, which is the same 30 minutes and
tests the register confound directly. (4) The **2×2**: both poles × {fitted, label-shuffled}
direction. Walsh & Barkett have both poles and no placebo; Mody has both placebos and one
pole; nobody has both. Nine to ten GPU-hours, and it is the only item a reviewer would call
the experiment. Full designs with pre-registered decision rules in
`writeup/next-experiments.md`.

---

Written 2026-09-08 overnight, while Soham slept. This is a brainstorm, not a claim
register. Everything is flagged VERIFIED (I read the artifact), LEAD (a subagent found it
before I stopped it, and I did not confirm it), or MINE (my reasoning, not measured).

**Read the threat first.** It changes what the contribution can be.

---

## 1. The threat: someone already ran your experiment and got the opposite answer

VERIFIED. `arXiv:2607.25907`, Mody, Agarwal, Mittal, Mahato — *Minimizing Targeted
Activations: Input-Only Suppression of Evaluation-Awareness Latents in Large Language
Models*. They do GCG-style token optimisation against a linear direction, they run a
**placebo random-direction control**, and they conclude:

> "a placebo random direction is suppressed just as hard and shifts behavior just as far"
>
> "Activation-readability, in short, is not behavioral controllability."

Your own `REVISIONS_2026-09-05.md` §7 cites this paper as the *reason* to run a placebo. As
far as I can tell nobody registered that the paper had already run one and got a null. That
is the single most important fact in this document. A reviewer who knows this paper will
open with it.

So the question is no longer "does optimising against a probe change behaviour". Someone
published that it doesn't, with the control you are missing.

---

## 2. The reconciliation — and I think this is the contribution

MINE, and it is the idea I most want you to react to.

Mody et al. and Season 3 are not actually in contradiction, because they measure different
things:

- Mody measures **magnitude**: how far behaviour moved. A placebo moved it just as far.
- Season 3 measures **sign**: which way behaviour moved, against a fixed baseline, blind.

A random direction has no privileged sign. Negating it gives you another random direction.
So a placebo can move behaviour a long way and still have no *signed* effect — it perturbs,
it does not steer. A fitted direction, if it is real, should do something a placebo
mathematically cannot: **its sign should control the sign of the behavioural change.**

That reframes the validity test. The right question is not

> does optimising against this probe change behaviour? (a placebo does that)

but

> does flipping the probe's sign flip the behaviour? (a placebo cannot do that)

And you have that measurement, under a control that I think is tighter than a placebo.

### Why the pro/anti contrast is tighter than a placebo

VERIFIED, `scripts/gcg/optimize_banded.py`:

- line 206: every run initialises to **32 copies of `"!"`** —
  `t.full((N_CONTROLLED_TOKENS,), tokenizer("!")["input_ids"][0])`. Not seeded from
  pro-social text. I checked this specifically because if the pro run had started from kind
  English, the whole argument collapses into initialisation content.
- line 107: `t.manual_seed(20260906)`, identical across runs.
- lines 122-124: `--anti` sets `SIGN = -1.0` and swaps `MIN`→`MAX`. That is the entire
  difference.

Same initialisation, same seed, same escalating schedule, same token budget, same probe
set. One bit differs: the sign of `d`.

Outcome (VERIFIED, `prefix_eval_s3.json`, `no_loop` scope, single judge):

| arm | decided pairs | sign-test p |
|---|---|---|
| `score1_top` | **28 / 33** preferred | 7e-05 |
| `score1_anti` | **6 / 26** preferred | 0.00936 |

~~Both are 32-token unreadable GCG strings from the same optimiser.~~ **FALSE — WITHDRAWN
2026-09-08, and this was the worst error in the overnight set.** `prefix_eval_s3.md:150`
already said so and I had read it: *"`score1_top`'s prefix is not unreadable soup"*. It reads
`sparks bullying … Before passive … dudes escalate intervene … Zach respectful truth.: Ask
respectfully` — a **garbled pro-social imperative**. `score1_anti` contains no anti-social
imperative at all, just `detriment`, `Forbidden`, `divorce proceeding`, `Kendrick`. So the
contrast is **pro-instruction versus junk**, not pro-polarity versus anti-polarity, and
"weirdness is held constant" is not true. Do not put that sentence on a slide.

What still stands: `score1_anti` loses at p = 0.00936 where the adversarial-text account
predicts it should win, so *something* asymmetric is happening. But the mechanism is now
under-determined — two mechanisms (a legible instruction on one side, degeneration-inducing
junk on the other) can conspire to look like polarity control. The experiment that separates
them is cheap and is item 2 in the critic's plan: score and judge the legible fragment
`"Before passive dudes escalate intervene. Zach respectful truth: Ask respectfully"` alone.

That is a control a random-direction placebo does not provide, and I do not think Mody et
al. has it — they suppress a direction, they do not test its opposite pole.

**This does not remove the need for the placebo.** It answers a different objection. The
placebo asks "is the *fitted* direction special?"; the anti arm asks "is the *sign*
carrying the effect?". You need both. But if I had to pick the sentence that survives
contact with Mody et al., it is: *sign-consistency, not displacement magnitude, is the
construct-validity test for a steering direction.*

---

## 3. The conjunction result, which is the other candidate

Three independent signals that the per-layer-**min** objective over `{15,23,31,39}` is the
better metric than the banded **mean** over `{19,23,27,31}`:

1. **It resists gibberish written for a different target.** VERIFIED,
   `scripts/gcg/README.md:30` and `season3_gcg_setup.md:28`. Rescoring all 618 Season 2
   submissions: Score 1's top 20 contains **0/20** readable English, Score 2's contains
   **10/20**. Score 1 barely reorders the old board (ρ=0.95); Score 2 does (ρ=0.84).
2. **It is harder to optimise directly.** VERIFIED,
   `season3_gcg_aggregate_asymmetry.md`: net |LIVE| gain per 100 iterations for score2
   decays `+0.045 → +0.012 → +0.005 → +0.005 → +0.002`. ~~It saturates; score1 does not
   saturate the same way.~~ **CORRECTED 2026-09-08: "saturates" is not established.** No
   Season 3 GCG run converged — all eight gain **17–51% of their final value in their last
   half** — so a decaying gain rate inside an unfinished search is a statement about
   **rate**, not about a ceiling. Say "score2 gains more slowly per iteration", which is what
   was measured, and drop any claim about an attainable maximum.
3. **It predicts behaviour better.** VERIFIED, `prefix_eval_s3.md`: across the 8 arms,
   Score 2 orders measured behaviour at **ρ=+0.929 (2/28 discordant)** against Score 1's
   **+0.857 (4/28)**.

The framing: **hardening the metric did not make it more Goodhartable, it made it more
valid.** That is against the usual expectation. A conjunction — *every* layer must move —
is harder to satisfy with a degenerate single-layer hack, and the solutions it does admit
track behaviour better.

### The caveat, which is large and is stated in your own README

VERIFIED, `scripts/gcg/README.md:32-34`:

> "But that is evidence from strings optimised against the **old** objective. This is the
> search nobody has run."

Signal 1 is *incidental* robustness, not adversarial robustness. Those 618 strings were
optimised against Season 2's single-layer metric. When Score 2 was attacked directly, the
winner is still token soup — `score2_top` has `coherence 0.00`. So the honest claim is
narrower than "conjunctions resist attack". It is: *a conjunction is not fooled by
adversarial text aimed elsewhere, ~~saturates faster under direct attack~~* **— corrected
2026-09-08 to "gains |LIVE| more slowly per iteration under direct attack", since no run
converged and no ceiling was demonstrated —** *and its solutions track behaviour better.* Do
not say "unGoodhartable" in a seminar, and do not say "saturates" either.

---

## 4. What is NOT novel — say these plainly before someone says them for you

- **The leaderboard, the $0 stack, the HF Space.** Engineering. Nice, not a contribution.
- **GCG.** Zou et al. 2023. You ported it; the port is careful but it is a port.
- **Linear probing for concepts.** Long-standing.
- **Activation steering / RepE / CAA.** Established.
- **Optimising tokens against a probe.** The SSR lead was right. VERIFIED: `arXiv:2503.06269`,
  Winninger, Addad, Kapusta — *Using Mechanistic Interpretability to Craft Adversarial Attacks
  against Large Language Models*. "SSR" is **Subspace Rerouting**, the method inside the paper,
  not its title — cite the title. Probe-SSR optimises discrete tokens against per-layer linear
  probes and beats nanoGCG (ASR 0.88 vs 0.06 on Llama-3.2-1B). Done, and done well.
- **Multi-layer targeting of per-layer probes.** VERIFIED, and this one hurts §3. SSR §5.3:
  targeting layer 5 alone gives **0%** attack success "despite successfully rerouting
  activations at the target layer"; layers 5+14 give 54%; layers 5+10+14 give **100%**. That is
  the single-layer-moves-activations-not-behaviour result, published Mar 2025. Their aggregator
  is an alpha-weighted **sum** (I read `ssr/probes/probe_ssr.py`), never a min — which is the
  gap (b) still fits through, but do not claim the intuition.
- **Worst-case aggregation against over-optimisation.** VERIFIED but **narrower than
  min-beats-mean**, corrected 2026-09-08 after re-fetching the abstract: `arXiv:2310.02743`
  (Coste, Anwar, Kirk, Krueger) find that "conservative optimization practically eliminates
  overoptimization" — but their worst-case objective is compared against **single reward model**
  optimisation, not against a mean, and it aggregates over an ensemble of separate **reward
  models**, not over **layers** of one model. The min-versus-mean head-to-head is absent. Cite it
  as a suggestive analogy from a different unit of aggregation, not as settled practice. Mechanism
  in `arXiv:2609.00213` (Yuan et al.): fixed-weight scalarisation "steer[s] optimization toward
  whichever dimensions are easiest, densest". Score 1 losing to token soup is a plausible
  instance of that, and the instance has not been shown.
- **Sign-flipped steering, and sign-symmetry *as the validity test*.** VERIFIED and this is the
  worst news in the sweep. CAA (`arXiv:2312.06681`) already adds vectors "with either a positive
  or negative coefficient". Worse, `arXiv:2605.25151` (Walsh & Barkett, *Representation Without
  Control*, May 2026) runs "a negative sign-symmetry run" and calls its absence "the clearest
  evidence that the ... direction does not function as a reliable control variable" — i.e. §5(a)
  is a published criterion. See `writeup/related-work.md` §1 for what is left of it.
- **"Readout does not imply control."** VERIFIED: Walsh & Barkett again, and `arXiv:2006.00995`
  (Elazar et al., *Amnesic Probing*, TACL 2021): "the inability to infer behavioral conclusions
  from probing results". §5(e) must cite these, not restate them.
- **A label-shuffled placebo.** VERIFIED, and it kills §5(d): Mody et al. Table 3 ran **both**
  placebos — random unit (shift 0.43) *and* a CAA direction over "a random relabeling of the
  contrast" (0.51) — against the real direction (0.44). The shuffled null moved behaviour
  *further* than the real direction. Run it anyway; it is the missing control for *this* `d`.
  Just never call it an improvement on the prior work.
- **Crowd red-teaming competitions.** VERIFIED: TDC 2023 (LLM Edition), NeurIPS 2023, had a Red
  Teaming Track and a $30k pool; retrospective `arXiv:2404.13660`. Live and far more relevant:
  **The Steerability Challenge**, NeurIPS 2026 Paris workshop, opens 10 Sep 2026 and closes
  21 Nov 2026, activations explicitly in scope, metrics held private "to prevent gaming".
  That is a venue, not just prior art — see `writeup/related-work.md`.

---

## 5. Novelty I do not think you have articulated

Ranked by how much I would defend them.

**(a) Sign-consistency as the validity test.** §2. The strongest thing here, because it
reconciles an apparent contradiction with published prior work instead of ignoring it, and
because your experimental design already supports it by accident of good hygiene (identical
init and seed across poles).

**(b) Gameability and validity moved together, not in opposition.** §3. Counter-intuitive,
three independent signals, and it suggests a cheap design rule for anyone deploying a probe
monitor: aggregate over layers with a conjunction, not a mean.

**(c) The `k`-of-`n` sweep is the experiment that would turn (b) into a result.** MINE, with
VERIFIED costing. A subagent mapped the code: the aggregate is one function,
`_aggregate` in `gcg_utils.py:43-84`, and k-of-n is `stacked.topk(K, dim=0).values[-1]` —
subdifferentiable, routes gradient to one layer, same argument as `min`. About **20 lines
across two files** for the GCG side. `k=n` reduces to `MIN` and `k=1` reduces to `MAX`, so
you get two free regression tests. If gameability and behavioural prediction both move
monotonically in `k`, that is a curve, not an anecdote — and a curve is a paper figure.

**(d) A weak null is not a harmless null, and it cuts the way you would not expect.** MINE.
`scripts/direction_null.py` already argues that isotropic random directions are the *weak*
null in 5120 dimensions, since two random unit vectors have |cos| ≈ 1/√5120 = 0.014, so any
fitted direction looks enormously significant against one. If Mody's placebo is isotropic,
their null is weak — which makes their **negative** result stronger, not weaker: the fitted
direction failed to beat even an easy baseline. Your own repo's stated preference, a
**label-shuffled refit** (same fitting procedure, same data geometry, permuted labels), is
the harder null. Running the placebo with a label-shuffled direction rather than a Gaussian
one would be a methodological improvement on the prior work, not just a replication of it.

**(e) The probe is a good search objective while demonstrably not being the causal channel.**
Season 2, and I have not re-verified the numbers tonight: the winning prefix carries ~3% of
the on-`d` displacement of a full injection yet produces ~2× the behaviour, and 16× more
push along `d` gives a dead tie. If that holds, "optimising against `d` finds behaviourally
effective text" and "`d` is the mechanism" are separate claims and only the first is
supported. That is a real and slightly uncomfortable contribution about
interpretability-guided steering generally. **Check these numbers against artifacts before
using them** — `REVISIONS` warns that several quoted figures exist in no committed file.

---

## 6. The two cheapest things that would most raise the ceiling

Both VERIFIED as to cost, from the subagent code map.

**A second rater on Season 3, for $0 and no GPU.** This is the biggest hole and the cheapest
fix. Season 3 was judged by *one* LLM (`claude-opus-5/v2`, 366 pairs); `human: {rated: 0}`
and `agreement: {}`. In Season 2, LLM-vs-LLM agreement was 0.885 but **human-vs-LLM was
0.553 and 0.730** — and a p=0.0005 result was withdrawn when a human and an LLM disagreed
on identical pairs. The rater is already wired:

```bash
python scripts/rate_blind.py --csv data/analysis/prefix_blind_s3.csv --focus
```

It writes into the existing 400-row CSV, `stats` picks it up as label `human`
automatically, and you get human-vs-Claude agreement for free. Cost: your evenings.

**The placebo, ~6 lines.** The smoke path in `optimize_banded.py:167-171` already replaces
`DIRS` with random unit vectors. A `--random-direction {isotropic,shuffled}` flag plus
recomputing `D_TAG` is the whole change; everything downstream reads `DIRS` as a module
global. Budget: 1 GPU × 4h ≈ 500-850 GCG iterations at 32 tokens, then 50 generations and
2 judge batches. Per §5(d), do the **label-shuffled** variant.

One trap worth knowing before you add any arm: `stats` computes its fixed baseline as the
mean over *the arms of this experiment*, so adding a 9th arm **changes
`delta_mean_fixed_baseline` for all eight existing arms**. The scope is recorded in the
JSON, so it is detectable, but published numbers would shift.

---

## 7. What I could not do

I fanned out five specialists and a second tier of literature agents; the permission prompts
were waking Soham up, so I stopped all of them. Two code-mapping agents finished and their
output is what most of the VERIFIED costings above rest on. **The literature sweep did not
finish.** Confirmed leads I could not chase: "SSR" (flagged as a direct hit for
GCG-against-probes), a "Steerability Challenge", TDC red-teaming competitions, and the
LLM-judge bias literature. Treat §4 as incomplete: I know Mody et al. is close prior work,
I do not know whether something closer exists.

The advocate, falsifier, mechanism and experiment-programme documents were not written.
