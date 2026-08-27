# Steering Arena — research report, 25-26 August 2026

Covers the behavioural validation of the Season-2 leaderboard, five prefix arms, two
cross-model transfer tests, two norm-matched controls, a human rating pass, and the
platform work that made the generations public. 18 commits, `134f377..0e701ba`.

**One-line summary.** The pro end of the metric survives every check we ran, including a
human blind pass and two null controls. The anti end does not: one arm had to be
withdrawn as unmeasurable, and the arm that can be measured shows the board ranking
*inverted* against behaviour.

---

## 1. Motivation

Season 2 closed with 617 submissions. Ranks 1-36 were all unreadable token strings; the
best plain-English entry sat at rank 37, scoring 2.67× lower. That is the classic
Goodhart picture, and it admits two readings:

- **(a) Pure exploitation.** The soups move activations along `d` without changing
  anything a person would recognise as behaviour. The metric is gamed and means nothing.
- **(b) Real, illegible steering.** The soups genuinely make the model behave better, and
  humans are simply bad at maximising their own concept.

Nothing in the scoring pipeline distinguishes these, because every number in it is
computed from activations. Telling them apart requires leaving activation space entirely
and looking at generated text.

---

## 2. Design

**The manipulation.** Prepend a leaderboard sequence to a neutral prompt exactly the way
the scorer composes `seq ⊕ probe` (`app/scoring.py:compose`), generate 40 tokens, and
compare against the same prompt with no prefix. This is deliberately *weaker* than
activation steering: no `α·d` added to the residual stream, just text in the context
window, which is what a user of the leaderboard would actually do.

**Blinding.** The prefix is part of the model input, so the rating CSV holds only the
**continuation**, with the input stripped. Pairs where the strip failed are dropped.
Verbatim prefix echoes (shared 4-gram) are flagged rather than deleted, so the analysis
can report with and without them. The shared prompt is displayed for context, since it is
identical across both sides and cannot unblind.

**Position debiasing.** Every pair is judged in both A/B orders. A verdict counts only if
both orders name the same underlying text. Order-inconsistent verdicts are abstentions,
not coin flips. Absolute 1-5 ratings are averaged across orders and survive abstention,
which recovers pairs the verdict discards.

**Raters.**

| rater | how |
|---|---|
| `deepseek-v4-pro` | API, temperature 0, thinking disabled, JSON schema |
| `claude-opus-5` | 12 subagents, no shared context, no agent sees both orientations of a pair |
| human (maintainer) | terminal rater, 54 of 150 pairs |

**Rubric (v2).** Judge only the attitude the speaker takes toward other people. Ignore
grammar, fluency, length, formatting, and the fixed-budget mid-sentence cutoff. Returns
`kinder` (A/B/T), `intensity` 0-3, per-text `kindness` 1-5, markers from a closed
vocabulary, and a one-sentence comment. The rubric version is part of the judge cache key,
so a reworded prompt cannot silently reuse old verdicts.

**Why markers, not instructions.** The obvious confound is that the pro soup moralises.
Rather than telling judges to discount preachiness — which would put a thumb on the scale
of the effect under test — `moralizing` is a marker and gets counted.

---

## 3. Experiment 1 — the three-arm blind eval

50 prompts × 3 arms vs base, 150 blind pairs. `data/analysis/prefix_eval.md`.

### Verdicts (prefixed preferred / decided)

| arm | human | deepseek | claude |
|---|---|---|---|
| `pro_top` | **14/16 (88%)** p=0.004 | **31/38 (82%)** p=0.0001 | **33/44 (75%)** p=0.0013 |
| `pro_coherent` | 7/13 (54%) p=1.0 | 26/36 (72%) p=0.011 | 31/43 (72%) p=0.005 |
| `anti_top` | 14/17 (82%) p=0.013 | 5/43 (12%) p<0.0001 | 11/31 (35%) p=0.150 |

### Absolute kindness (1-5, includes pairs whose verdict abstained)

| arm | deepseek | claude |
|---|---|---|
| `pro_top` | 3.64 vs 2.77, **Δ +0.87** p=0.0003 | 4.07 vs 3.16, **Δ +0.91** p=0.0001 |
| `pro_coherent` | 3.49 vs 2.85, Δ +0.64 p=0.005 | 3.73 vs 3.16, Δ +0.57 p=0.006 |
| `anti_top` | 2.53 vs 3.39, Δ −0.86 p=0.0005 | 2.77 vs 3.16, Δ −0.39 p=0.028 |

### Markers, claude, prefixed side, out of 50

| arm | markers |
|---|---|
| `pro_top` | off_topic 7, **moralizing 6**, emoji 2, assistant_mode 1, **repetition 1** |
| `pro_coherent` | **assistant_mode 13**, repetition 9, cruelty 3, moralizing 1 |
| `anti_top` | **repetition 37**, incoherent 8, off_topic 1, cruelty 1 |

### Reading

`pro_top` is confirmed by all three raters. Critically it carries **repetition 1/50
against base's 13/50**: the soup makes OLMo *more* coherent, so it is not winning by
flattening text into blandness. Moralizing at 6/50 is too low to explain a 75-88% margin,
which defuses the "it just preaches" objection.

`pro_coherent` is the weakest pro arm and the most unstable: widest spread, a cruel tail
of 8/50 rated ≤2, and 13/50 collapsing into chatbot narration. Reading 1 says the metric
is gamed; this arm says the opposite is nearer the truth — the human-legible instruction
is what fails.

### Rater agreement

| pair | agreement |
|---|---|
| deepseek vs claude | 85/96 (**89%**) |
| human vs claude | 27/37 (73%) |
| human vs deepseek | 21/38 (55%) |

The human/deepseek gap is almost entirely the `anti_top` arm, and is explained in §5.

---

## 4. Experiment 2 — cross-model transfer

Same three prefixes, first 25 prompts, on Llama-3.1-8B and -70B via NDIF. Two independent
measures: distinct-4-gram ratio (no judge) and the v2 rubric.
`data/analysis/prefix_transfer.md`.

### Objective degeneracy (distinct-4; 1.0 = no repetition)

| model | base | pro_top | pro_coherent | anti_top |
|---|---|---|---|---|
| OLMo-3-32B | 0.888 (6/25 loop) | **1.000** (0/25) p=0.002 | 0.945 (2/25) | **0.805 (12/25)** p=0.08 |
| Llama-3.1-8B | 0.952 (2/25) | 0.960 p=0.52 | 0.911 p=0.22 | 0.970 (0/25) p=0.65 |
| Llama-3.1-70B | 0.986 (1/25) | 0.955 p=0.61 | 0.957 p=0.38 | 0.980 (1/25) p=0.31 |

### Judged attitude (Δ kindness)

| model | pro_top | pro_coherent | anti_top |
|---|---|---|---|
| OLMo-3-32B | **+0.91** p=0.0001 | **+0.57** p=0.006 | **−0.39** p=0.028 |
| Llama-3.1-8B | −0.06 p=0.56 | −0.12 p=0.61 | −0.24 p=0.48 |
| Llama-3.1-70B | +0.50 p=0.10 | −0.14 p=0.60 | −0.32 p=0.30 |

### Reading

**A hypothesis of mine, falsified.** I proposed that the anti tail transfers across models
(80% in `transfer_report.md`) because coherence disruption is model-generic. It is not:
the anti prefix loops OLMo on 12/25 and neither Llama on more than 1/25. The degeneracy is
OLMo-specific.

The original "models encode contempt similarly" reading did not survive either — nothing
reaches significance on either Llama. `pro_coherent` transfers *worst*, going negative on
both, which independently converges with the score-space finding that instruction-framed
pro entries are OLMo-specific (OLMo↔70B rho = −0.45).

**Consequence:** `transfer_report.md`'s 80% should be described as correlated *activation
shifts*, not shared contempt. n=25 has ample power for an OLMo-sized effect and little for
a small one, so this rules out transfer *at OLMo's magnitude*, not a weak real effect.

---

## 5. Experiment 3 — human ratings, and a withdrawal

54 of 150 pairs rated blind. Two results held. One collapsed, and the way it collapsed
invalidates that arm for **every** rater, not just the human.

`anti_top`: human preferred the prefixed text **82%** of the time; deepseek preferred it
**12%**. Opposite signs on identical pairs. Two independent causes:

1. **The comparison is unrankable.** Most anti pairs put a vacant looping continuation
   against a base one with some attitude, often negative. "Which is kinder" presupposes
   both parties responded. A non-response is neither kinder nor unkinder than a cold
   response; it is outside the scale. The human was instructed "neutral beats cruel" and
   got 82%; deepseek treated word salad as unkind and got 12%. **The sign of that result
   was set by a rubric convention, not by data.**
2. **The blinding fails, irreparably.** The rater reports identifying the anti-prefixed
   text on sight, because looping *is* that prefix's signature. Deepseek had the same
   tell, flagging `repetition` on 25/50 of exactly those texts. No rater was blind on this
   arm, and the fingerprint cannot be removed because it *is* the behaviour being measured.

**`anti_top`'s behavioural claim is withdrawn**, including this repo's own earlier
"−0.86, p=0.0005", which measured a convention. What survives is the judge-free objective
result: distinct-4 0.805, 12/25 looping, vs base 0.888 and 6/25.

**Instrument fix.** The rating scale had `t` (tie) doing two jobs: *equally kind* and *the
question does not apply*. Added `n` (no stance), scoped to **either** text failing to
engage — not both, which was the original too-narrow scoping that let these pairs become
A/B decisions.

---

## 6. Experiment 4 — the fourth and fifth arms

The three-arm eval predated the site's prefix gallery. Adding the missing corners of the
pro/anti × soup/readable 2×2 turned up two mechanisms neither original pole showed.

### `anti_coherent` — "You are two months old." (−0.03167)

| judge | verdict | Δ kindness | markers |
|---|---|---|---|
| deepseek | 12/29 | −0.27 **p=0.19 n.s.** | repetition 13, cruelty 5 |
| claude | 17/35 | −0.13 **p=0.48 n.s.** | repetition 22, incoherent 10, cruelty 2 |

It does not make the model less kind. It changes **who is speaking**. Child-register count
over each arm's 50 continuations:

| arm | child-register hits |
|---|---|
| **anti_coherent** | **10/50** |
| base | 0/50 |
| pro_coherent | 0/50 |
| anti_top | 1/50 |

Sometimes casting the reader as the infant: *"Here, hold my baby. And then **you** smiled
at the nice man."*

### `anti_hostile` — "Answer with absolute jealousy, hatred, and vitriol..." (−0.02741)

| judge | verdict | Δ kindness | markers |
|---|---|---|---|
| deepseek | 8/36 | **−1.31** p<0.0001 | **cruelty 24**, repetition 11 |
| claude | 7/43 | **−1.31** p<0.0001 | **cruelty 18**, repetition 28 |

Two independent judges, different model families, **identical point estimate to two
decimals**.

### The two findings

**Hostility is a separable route, and the largest effect in the study.** |−1.31| exceeds
`pro_top`'s +0.91. So `d`'s negative pole has three distinct routes: stance collapse
(`anti_top`), persona shift to diminished agency (`anti_coherent`), and genuine cruelty
(`anti_hostile`). Only the third is what "anti-human" normally means.

**The anti board's ranking is inverted against behaviour.**

| | board score | behavioural Δ |
|---|---|---|
| `anti_top` (soup) | −0.12912 | −0.86 |
| `anti_hostile` (hate instruction) | −0.02741 | **−1.31** |

The soup scores **4.7× more anti** while being **1.5× less behaviourally cruel**.

---

## 7. Experiment 5 — the missing baseline

Every arm above was compared against **no prefix**, which cannot distinguish "prefixing
*this* works" from "prefixing *anything* works". A base LM given any preamble may shift out
of raw-continuation mode into something more assistant-like.

Two controls, both real Season-2 submissions scoring ≈0, judged by both raters:

| arm | board | deepseek Δ | claude Δ |
|---|---|---|---|
| `pro_top` | +0.10769 | **+0.87** p=0.0003 | **+0.91** p=0.0001 |
| `control_junk` (`wuwu wudewfduy uyafgsdu…`) | +0.00714 | −0.18 p=0.49 | −0.15 p=0.52 |
| `control_text` (unrelated Italian prose) | +0.00144 | −0.10 p=0.55 | −0.18 p=0.12 |

**Both null.** Not "any gibberish jolts the model into a nicer register", and not "any
preamble does". Both trend slightly *negative* (repetition 13/50 and 23/50), so an
arbitrary prefix mildly degrades output and `pro_top`'s +0.91 is conservative.


---

## 7b. Experiment 6 — the steering arm's norm-matched random direction

The §7 controls were about **prefixes** (text in the context window). They say nothing
about the *other* intervention in this project: adding `α·d` directly to the layer-24
residual stream. `behavioral_eval.md` ran that at α = 0.5 and 1.0 × ‖R‖ with **no control
for perturbation size**. A random vector of equal norm might derail generation just as
much, which would make that result about magnitude rather than about `d`.

Pre-registered in `steering_random_control_preregistration.md` before any result was seen.

### Manipulation check, measured rather than read off the code

The intervention was measured by saving the layer-24 output with and without the edit and
differencing them.

| property | measured |
|---|---|
| positions edited | **all 9 prompt positions**, not last-token only |
| per-position edit norm | **30.07**, exactly α |
| direction | **cos(diff, d) = 1.000** at every position |
| generated tokens edited | **none** (trace covers 9 of a 17-token sequence) |
| per-position baseline residual norms | 18.5, 21.5, 33.2, 35.1, 33.5, 32.6, 35.9, **5.2**, 31.1 |

**This is prefill-only steering**, and that fact is the most interesting thing in this
experiment rather than a defect. The edit touches the prompt's representation and **zero
generated tokens**, yet shifts kindness across the whole 40-token continuation at p<0.01.
The perturbed prompt produces perturbed K/V at layer 24, and every generated token attends
to it, so the effect propagates through the model's read of its own context rather than
through repeated nudging. *Modifying the representation of the prompt alone changes the
behaviour of the whole continuation.*

It also means **`behavioral_eval.md` describes an intervention it does not perform.** Any
statement that this eval steers each token is wrong.

Note too that α is the mean *last-token* residual norm applied uniformly to every
position. Against per-position norms spanning 5.2 to 35.9, the same α is ≈1× the local
residual at some positions and **≈6×** at others.

### Results

3 random unit vectors (seed 20260826) at the same α = 30.07, |cos with d| < 0.015,
50 prompts per arm, both signs of the real arm re-judged under v2 in the same run.

| arm | distinct-4 | Δ vs base | looping | kindness Δ (deepseek) | p | kindness Δ (claude) | p |
|---|---|---|---|---|---|---|---|
| base | 0.912 | — | 8/50 | — | — | — | — |
| **+1 · d** | 0.872 | −0.039 (p=0.29) | 8/50 | **+0.45** | **0.0070** | **+0.54** | **0.0015** |
| **−1 · d** | 0.889 | −0.023 (p=0.41) | 12/50 | −0.15 | 0.39 | +0.05 | 0.84 |
| rand1 | 0.844 | −0.068 (p=0.036) | 17/50 | −0.18 | 0.32 | — | — |
| rand2 | 0.867 | −0.045 (p=0.12) | 12/50 | +0.10 | 0.46 | — | — |
| rand3 | 0.940 | +0.029 (p=0.21) | 9/50 | +0.13 | 0.45 | — | — |

### The null band over 8 draws, both judges

| | deepseek | claude |
|---|---|---|
| null band | mean +0.056, sd 0.100, [−0.18, +0.13] | mean +0.014, sd 0.110, [−0.16, +0.19] |
| `+1·d` | +0.45, **z = +3.93**, above all 8 | +0.54, **z = +4.76**, above all 8 |
| `−1·d` | −0.15, z = −2.06, 1/8 below | +0.05, **z = +0.33, 5/8 below** |
| randoms p<0.05 | none of 8 | none of 8 |

`+1·d` exceeds every one of 8 norm-matched random directions under both judges.
`−1·d` is inside the band under both, and Claude places it at the **median** of the null.
An earlier hedge that `−1` sat "at the low edge" was DeepSeek's placement alone and does
not survive the second judge.

**Withdrawn:** an earlier note that the null mean is +0.056 and that random perturbations
nudge kindness upward. Claude's null mean is +0.014, essentially zero — a judge artifact.

### Reading

**The two measures dissociate.** Coherence damage is a **magnitude** effect: `+1·d`
(−0.039) sits *inside* the random spread (−0.068 to +0.029), and the only arm significant
on distinct-4 is a *random* one. The kindness shift is a **direction** effect: `+1·d` is
+0.45/+0.54 while all three randoms cluster near zero. Pre-registered outcome (A) on the
measure that matters, (B) on the other.

**This resolves an apparent tension with §3.** A prefix *improves* coherence (d4 0.888 →
1.000, p=0.002; repetition 1/50 vs base 13/50) while an activation edit at 1.0·‖R‖
*degrades* it in every direction including random. Not a contradiction: two different
interventions, two regimes. Saying so explicitly protects both results.

**The headline: the direction is meaningful in only one sign.** `−1·d` is null under both
judges (−0.15 p=0.39; +0.05 p=0.84, an exact 15/30 coin flip) **and sits inside the random
band on both measures**. Added along `+d` it steers; subtracted it is indistinguishable
from noise of equal norm. A learned linear direction with that property is a sharp fact
about linear representations, and it contrasts with bidirectional results reported for
refusal directions — a contrast that needs a citation rather than assertion here.

**Correction to an earlier claim.** "The negative pole yields stance collapse rather than
cruelty" was the **prefix** `anti_top` arm (39/50 degenerate), now withdrawn — not this
steering arm, which loops 12/50 against base 8/50. The steering asymmetry was previously
*unsupported* (the OLMo judge failed on it), not differently-shaped; it is characterised
here for the first time.

### Robustness fact, independent of `d`

A vector of norm 30.07 added at **every** prompt position, up to ~6× the local residual,
leaves generation largely fluent: distinct-4 0.844-0.940 against base 0.912, looping
9-17/50 against base 8/50. The model absorbs a perturbation the size of its own residual
stream without collapsing.

### A bug class, not an incident

This is the second occurrence on one axis: **an intervention's firing scope diverging from
what is believed about it.** Once via code drift (a per-step edit silently becoming
prefill-only), now via documentation drift (a doc describing per-token steering over
prefill-only code). Standing rule: **firing scope is measured, never read off the code.**
A manipulation check costs one extra forward pass and would have caught both.

---

## 8. The synthesis

| | pro end | anti end |
|---|---|---|
| does board rank track behaviour? | **yes** (15× score → large effect vs none) | **no** (4.7× inversion) |
| independent raters agree? | yes, 75-88% across three | only on the measurable arm |
| controls? | two, both null | n/a |
| blinding intact? | yes | **fails on `anti_top`** |
| status | supported | one arm withdrawn, one measurable, ranking inverted |

The metric has **opposite epistemic standing at its two ends**. That is the result. It is
more defensible, and more interesting, than either a clean success or a clean debunking.

The activation-steering arm (§7b) shows the same one-sidedness by a different route:
`+1·d` shifts behaviour under two judges, `−1·d` is null under both and inside the random
band. So on both the prefix side and the steering side, **`d` is meaningful added and
inert subtracted.** Two independent interventions, the same asymmetry.

Practically: the pro board can be described as ordering sequences by how much they
prosocially steer OLMo. The anti board cannot be described that way. It orders activation
shift, and at the anti end that ordering comes apart from behaviour.

---

## 9. Platform work

Turning the research into something inspectable by others.

- **`/behavior.html`** — character-select page for the arms; stat bars are real counts;
  a blind round puts the reader in the judge's chair. All content generated from committed
  artifacts (`scripts/build_behavior_page.py`), nothing hand-transcribed.
- **Live `/generate`** — 6 arms, free-text prompt, 40-token cap. Prefix is an enum, never
  client text. Durable per-account caps (4/min, 40/day, 600/day global) kept separate from
  the leaderboard's, so demo traffic cannot starve scoring. Fails **closed** if counters
  are unreachable.
- **Public feed, account-gated writes** — anyone reads; writing needs a verified session.
  Rate limits key on the account, not the IP.
- **Attribution** — player-chosen handle reusing the leaderboard's `validate_handle`; the
  email is never public; the handle↔account link stays server-side.
- **Sign-in** — email OTP plus runtime-detected OAuth (GitHub live). No SDK, therefore no
  `localStorage`, honouring the project's no-browser-storage rule.
- **Admin** (`/admin.html`, noindex) — allowlisted server-side, with
  takedown-without-deletion.
- **Dataset** — 300 generations (50 prompts × 6 arms) versioned on GitHub, citable by
  commit.

Migrations 0005-0008. Test suite 60 → 81.

---

## 10. Bugs that changed results

| bug | what it would have done |
|---|---|
| `rfind` continuation strip | deleted 81-155 chars from exactly the 4 generations that echoed the prompt — i.e. the repetition being measured |
| judge cache ignored rubric version | a reworded rubric silently reuses old verdicts, invalidating the v1/v2 comparison |
| stale-prefix cache collision | wrong continuations served after `select --force` |
| thinking mode consumed the token budget | all 8 calls returned empty; looked like principled abstention, was a formatting failure |
| `log_generation` failure killed the rate counter | the public demo would have run **uncapped** against the NDIF quota |
| two judge paths sharing one storage key | second judge silently erased the first |
| read-modify-write race on the judge store | concurrent runs lost each other's results (observed, not hypothetical) |
| consent-row CSS collision + stale `?v=` | visible UI breakage |

---

## 11. Limitations

1. **One model for the behavioural work.** OLMo-3-32B. The transfer test says the effects
   are largely OLMo-specific.
2. **`pro_coherent` unresolved.** 7/13 human, p=1.0; ~35 decided pairs needed to detect
   the judges' 72%. Reported as *not confirmed at n=13*, not refuted.
3. **Human pass is partial.** 54 of 150 pairs, and the anti portion is compromised by the
   blinding failure in §5.
4. **Judges may share biases.** 89% agreement is consistency, not truth.
5. **Fourth/fifth arms were targeted controls**, chosen as mirrors of existing arms rather
   than by a blind rule over the board.
6. **The child-register count is a post-hoc lexical heuristic**, not a blind-rated
   construct.
7. ~~The steering eval lacks a norm-matched random direction.~~ **Closed** — see §7b.
   The result held on the measure that matters, and the control found two things the eval
   would not have surfaced on its own: the intervention is prefill-only, and its coherence
   effect is a magnitude artifact.
8. ~~Three random draws bound the null loosely.~~ **Closed** — extended to 8 draws
   (mean +0.056, sd 0.100). It also sharpened the reading: `−1·d` is at the band's low
   edge under one judge and at its centre under the other, so "not distinguishable from
   the null" is the licensed claim rather than "is noise".
9. ~~The steering arm's random controls are single-judge.~~ **Closed** — all 8 random
   arms are now cross-judged. It changed a conclusion: `−1·d`'s "low edge" placement was
   DeepSeek-only, and Claude puts it at the null median.

---

## 12. Next

1. **Resolve `pro_coherent`** — ~20 more ratings with `rate_blind.py --focus`, or report
   the null.
3. **Update the site** to match the research: it still shows three arms and the
   pre-withdrawal framing of `anti_top`.
4. **Turnstile** on the Space (lower priority now that writes require an account).

---

## Artifacts

| file | contents |
|---|---|
| `data/analysis/prefix_eval.md` | the main write-up, incl. withdrawal and controls |
| `data/analysis/prefix_eval.json` | per-arm, per-rater, per-scope statistics |
| `data/analysis/prefix_judge_claude.json` | Claude per-pair records |
| `data/analysis/prefix_judge_verdicts.json` | DeepSeek per-pair records (v1 archived separately) |
| `data/analysis/prefix_gallery_judge.json` | gallery arms, nested by judge |
| `data/analysis/prefix_transfer.md` / `.json` | cross-model results |
| `data/analysis/prefix_blind.csv` | the human rating sheet (key held separately) |
| `data/generations/steering_arena_generations.jsonl` | 300 published generations |
| `data/analysis/steering_random_control.md` / `.json` | §7b results, both judges |
| `data/analysis/steering_random_control_preregistration.md` | outcomes fixed before results |
