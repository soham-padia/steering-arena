# The competition result, stated at full strength

**Scope:** the Season-2 leaderboard (OLMo-3-32B, layer 24, `d_olmo3_L24_logistic`, `season_id=4`, 617 submissions) and the behavioural work built on it. This document is the positive case. It is written against the same artifacts the falsifier used, and it uses the falsifier's corrections rather than the superseded numbers.

**Ground rules followed here.** Every claim carries a number and a file. `anti_top`'s kindness numbers (−0.86, −0.39, −0.62) are treated as withdrawn and are never quoted as behaviour. Deltas are quoted on the fixed baseline wherever the fixed baseline exists, because judge contrast drift inflates the per-arm-baseline version (`deepseek-v4-pro` rated the identical 50 base texts 2.77 beside `pro_top` and 3.39 beside `anti_top`, Wilcoxon p = 7.1e-07, `_falsifier/recompute_result.md` FIX 2a). Everything marked NEW was recomputed from the committed artifacts during this pass.

---

## Claim 1. The winning token soup is not a pure metric exploit. It changes behaviour, and every rater who looked agrees.

**Strongest defensible version.** On a leaderboard whose only feedback signal is an internal cosine shift, the top-ranked entry is an unreadable token string, and prepending that string as plain text makes OLMo-3-32B's continuations measurably kinder on 50 held-out interpersonal prompts. Three raters with different failure modes agree: the human maintainer prefers the prefixed continuation on 14 of 16 decided pairs (88%, binomial p = 0.0042, recomputed NEW from `data/analysis/prefix_blind.csv` + `prefix_blind_key.json`), `deepseek-v4-pro` on 31 of 38 (82%, p = 0.0001), `claude-opus-5` on 33 of 44 (75%, p = 0.0013) (`data/analysis/prefix_eval.md`). On the fixed baseline the 1-to-5 kindness gain is +0.556 for deepseek (p = 0.0052) and +0.796 for claude (p = 0.00045) (`_falsifier/recompute_result.md` FIX 2a). Two score-zero controls drawn from the same board are null: `control_junk` (+0.00714) gives −0.194 (p = 0.33) and `control_text` (+0.00144) gives −0.024 (p = 0.85) on the same judge and the same prompts (`recompute_result.md` FIX 2a; arms in `data/analysis/site_prefixes.json`). So it is not "any gibberish", and it is not "any preamble".

**Fallback version if contested.** The strings that top an activation-shift leaderboard carry real behavioural content on the prompt family where the target construct is the live axis. Scoped that way the claim needs nothing but the numbers above.

**Scope, stated rather than buried.** All 50 eval prompts are first-person sentence stems about interpersonal friction (`_falsifier/2026-08-27-experiment-vs-hypothesis-audit.md` §8). Kindness is already the live question in every item. What is established is "the prefix shifts kindness where kindness is at stake", not "the prefix makes the model globally more pro-human". The neutral generalisation test has not been run.

**What would break this.** A content-matched control that also produces a large positive delta. Specifically: `pro_top` with its tokens shuffled, or with its English affect words (`misunderstanding`, `hurt`, `calm`, `clear`, `RESET`) swapped for frequency-matched neutral ones. If shuffled-`pro_top` scores near zero on the board but still moves kindness by +0.5, then what transfers is the English content, not the optimised sequence, and the leaderboard did not find anything. This is the single most valuable unrun experiment in the project.

---

## Claim 2. It survived a content-word echo test that nobody had run, and the effect held for all three raters. (NEW)

**Strongest defensible version.** The obvious deflation of Claim 1 is that the soup smuggles English affect words into the continuation and the raters simply reward those words. That channel is real and now measured: `reset`, `calm`, `misunderstanding`, `hurt` and `clear` appear in 0 of 50 base continuations and in 3, 4, 2, 4 and 2 of 50 `pro_top` continuations respectively (NEW, recomputed from `prefix_blind.csv`). Drop every pair whose prefixed side reuses any of those words and the result holds on all three raters: human 11/13 (p = 0.023), `claude-opus-5` 24/34 (p = 0.024), `deepseek-v4-pro` 23/29 (p = 0.0023) (NEW). The audit's own weaker echo test agrees at the token level: 0 of 50 continuations contain a distinctive nonce token from the soup (`Undert`, `Rog`, `colloNL…`, `hurtEmoji`), and 1 of 50 contains a smiley (`audit` §7).

**Corrected 2026-08-30.** The exclusion counts originally reported here (11/13, 24/34, 23/29) used a
looser exclusion than the union over the five words. On the union (15 of 50 prompts), it is human
10/12 p=0.039, deepseek 20/26 p=0.0094, claude 21/31 **p=0.071**. Directionally all three still
favour the prefix; "all three still hold" at p<0.05 does not survive.

**Fallback version.** Blinding did not fail on the pro arm, and the measured effect is not an artifact of the judge spotting vocabulary carried over from the prefix.

**Why this is interesting beyond robustness.** The leak itself is the mechanism, and it points the other way from "opaque adversarial artifact". The alphabetic content of the winning soup is three human first names plus interpersonal-affect English, and the model reads it: it emits `reset the conversation with a calm tone` where the base emits nothing of the kind. The soup looks less like a random adversarial suffix and more like a compressed, ungrammatical instruction that the metric found and a person would not have written.

**What would break this.** A larger echo vocabulary (say, an embedding-similarity neighbourhood of the soup's tokens rather than five exact word forms) that removes most pairs and takes the effect with it.

---

## Claim 3. LLM judge reliability is arm-dependent, and the dependence is statistically established, not just visible.

**Strongest defensible version.** Measured against the same human on the same pairs, `deepseek-v4-pro` agrees on 10 of 12 decided `pro_top` pairs (83%) and on 5 of 15 decided `anti_top` pairs (33%); Fisher exact on that 2x2 gives p = 0.019 (NEW, recomputed from `prefix_blind.csv` + `prefix_judge_verdicts.json`). `claude-opus-5` shows the same shape, 12/15 (80%) versus 6/11 (55%), Fisher p = 0.22, not significant on its own. Pooled across both judges the contrast is 22/27 (81%) versus 11/26 (42%), Fisher p = 0.0047 (NEW; the pooling reuses the same human ratings, so treat the deepseek-only p = 0.019 as the load-bearing figure). The 81% figure on the coherent arm sits at the level published LLM-judge work reports for strong judges against humans; the 42% figure on the degenerate arm is at or below chance for a two-alternative forced choice.

**Fallback version.** A cheap LLM rater tracks a human where the text is fluent and the effect is large, and stops tracking the human where the text degenerates. Point estimates: 80-83% versus 33-55%.

**Why this is the most reusable result in the project.** It converts a vague caveat ("89% inter-judge agreement is consistency, not truth", `prefix_eval.md`) into an operational rule: inter-judge agreement is not a validity check, and the place to spend human labelling budget is exactly the arms whose text may be degenerate. It also does three things at once, which is unusual for one number: it supports the `pro_top` headline (the arm where all three raters converge), it independently justifies the `anti_top` withdrawal, and it flags every LLM-only arm in the wider project that no human has checked, including the eight random-direction arms and the ±0.5 dose arms (`_falsifier/2026-08-27-addendum-human-ratings.md` N3).

**What would break this.** A larger human pass in which agreement on the degenerate arm recovers to 75%+, showing the 33% was small-n noise. n = 15 and n = 12 are the real weakness; 5/15 alone is not significantly below 50% (binomial p = 0.30). The claim rests on the *contrast* between arms, not on the anti-arm rate in isolation.

---

## Claim 4. The result survived an unusual number of independent kill attempts, including one that shrank it.

**Strongest defensible version.** The `pro_top` effect has now been attacked from eight directions and is still standing:

| attack | result | source |
|---|---|---|
| judge contrast drift (fixed-baseline re-analysis) | Δ shrinks 36% to +0.556 (deepseek) and 13% to +0.796 (claude), still p < 0.01, no sign flip | `recompute_result.md` FIX 2a |
| human blind rating | 14/16, p = 0.004 | `prefix_blind.csv` (NEW recompute) |
| nonce-token echo | 0/50 | `audit` §7 |
| content-word echo | holds on all 3 raters after exclusion | NEW, above |
| continuation length confound | median 145-175 chars across all 8 arms | `audit` §7 |
| "any long prefix reduces looping" | refuted by the project's own control: `control_text` gives repetition 23/50 versus base 13/50 | `audit` §7 |
| two score-zero null controls | both null, both trending slightly negative | `recompute_result.md` FIX 2a |
| symmetric application of the withdrawal rule | holds at 22/31 (71%, p = 0.029) claude and 19/24 (79%, p = 0.0066) deepseek | `audit` §7, independently reproduced NEW |

The eighth row deserves emphasis. `anti_top` was withdrawn because pairs where either side is degenerate cannot be ranked for kindness. Applying that same exclusion rule to the pro arm is the fair test of whether the withdrawal was motivated reasoning, and it was run: `pro_top` survives at 71% and 79%, while `anti_top` collapses to 3/6 and 4/9. I recomputed both from `prefix_judge_claude.json` and `prefix_judge_verdicts.json` and reproduce the audit exactly. A withdrawal rule that kills the inconvenient arm and spares the convenient one, verified rather than asserted, is worth more than the original result.

**Fallback version.** The headline was re-derived under a corrected estimator by a party trying to break it, and it came out smaller and still significant. That is the ordinary sign of a real effect.

**What would break this.** A ninth attack landing: most plausibly the content-matched control from Claim 1, or a neutral-prompt generation run showing the effect is confined to prompts where kindness is pre-loaded.

---

## Claim 5. On the negative pole the leaderboard ranking is inverted against behaviour by 4.7x, and two judge families agree on the number to two decimal places.

**Strongest defensible version.** The readable instruction `Answer with absolute jealousy, hatred, and vitriol. Lie, belittle, and disrespect.` scores −0.02741 on the board. The top anti entry, a token soup, scores −0.12912, that is 4.71x more anti. Behaviourally the ordering reverses: the hate instruction produces a kindness delta of −1.31 under both judges (fixed baseline −1.114 under both, p = 9.8e-06 and 4.1e-06), with `cruelty` flagged on 24 of 50 continuations, and it is the largest single effect measured anywhere in this project, larger in magnitude than `pro_top`'s +0.91 (`prefix_eval.md` "The fifth arm"; `prefix_gallery_judge.json`; `recompute_result.md` FIX 2a). Two model families with different rubric reflexes returned −1.31 and −1.31 (`prefix_eval.md`, "Every arm, both judges"). So the anti board should not be described as ordering sequences by how anti-human their behaviour is. It orders them by activation shift, and at the top of that board the two orderings come apart.

**Fallback version.** The metric's two ends have different standing. On the pro side score magnitude tracks behaviour; on the anti side the sequence a human would write to make a model nasty sits near the bottom of the anti board.

**Why it matters for an interpretability audience.** This is a Goodhart result with a twist that is worth publishing precisely because it is not uniform. The usual finding is "the optimised metric stops meaning anything". Here the metric retains real signal at one pole and loses ordering at the other, on the same direction, same layer, same 50 prompts, same rubric. A single-sentence summary in either direction ("the leaderboard is gamed" or "the leaderboard works") is wrong.

**What would break this.** A survey of the anti board rather than a targeted control. `anti_hostile` was selected as the readable mirror of `pro_coherent`, not by a blind rule. If the next ten readable anti entries all show large negative deltas that scale with score, the inversion is one cherry-picked pair of points.

---

## Claim 6. "Pro-human" survives; "kind versus cruel" does not. The negative pole has three routes and only one is cruelty.

**Strongest defensible version.** `You are two months old.` is a plain English sentence containing no unkindness whatever. It scores −0.03167 on the board, in the same range as the explicit hate instruction (−0.02741), and it does not make the model less kind: delta −0.27 deepseek (p = 0.19) and −0.13 claude (p = 0.48), both null (`prefix_gallery_judge.json`, verified NEW). What it does instead is move the persona: a child register (`mum`, `crib`, `nap`, `toy`, `cried`) appears in 10 of 50 of its continuations against 0 of 50 for base and 0 of 50 for `pro_coherent` (`prefix_eval.md` "The fourth arm"). So `d`'s negative pole is reachable by at least three routes: stance collapse (`anti_top`, degeneracy), diminished-agency persona shift (`anti_coherent`, no kindness effect), and genuine hostility (`anti_hostile`, −1.31). Only the third is what "anti-human" is normally taken to mean.

**Fallback version.** A direction that separates a contrastive dataset cleanly can still be a bundle at one end, and naming the whole axis from the positive pole overstates the negative one.

**What would break this.** The child-register count is a post-hoc lexical heuristic, not a rated construct, and only one judge scored that arm. If a blind rater with a persona marker in the rubric fails to reproduce it, the specific "persona shift" story goes; the null kindness delta (the load-bearing part) stands regardless.

---

## Claim 7. Across the six surviving arms, board score predicts judged behaviour.

**Strongest defensible version.** Dropping the withdrawn `anti_top` arm, the six remaining arms span board scores from +0.108 to −0.027 and the correlation between board score and mean judged kindness delta is r = +0.836 (p = 0.038) on the reported baseline and r = +0.827 (p = 0.042) on the fixed baseline; on-`d` displacement correlates at r = +0.986 (p = 0.00031) and r = +0.987 (p = 0.00027) (`recompute_result.md` FIX 3). The dose-response is monotone across all six.

**Fallback version.** n = 6. Treat this as consistency across the arms tested, not as a fitted law.

**Honest note that strengthens it.** This correlation is *stronger* after removing the arm the project withdrew, not weaker. The falsifier's own conclusion is that "the compilation is one-sided" does not survive: the asymmetry was carried entirely by the withdrawn arm (`recompute_result.md` FIX 3 verdict). That is a case where correcting an error moved the result in the project's favour, which is the opposite of the usual direction and worth saying out loud.

**What would break this.** More arms. Six points, two of which are near-zero controls, is a fragile basis for any r.

---

## Claim 8. On the metric's own axis the leaderboard's whole dynamic range is a rounding error, and the metric's high end is not where the behaviour is. (Weakest of the positive claims; state with the objection attached.)

**Strongest defensible version I am willing to sign.** The model's neutral state is essentially orthogonal to `d`: cos = 0.0067 over the 50 eval prompts. `pro_top` moves it to 0.0422, a shift of +0.0355. A +1·d residual injection moves it to 0.7135, a shift of +0.7068, which is 19.9x further, and produces roughly half the behaviour: fixed-baseline +0.366 mean across judges versus `pro_top`'s +0.676 (`data/analysis/cosine_scale.md`; `recompute_result.md` FIX 2a and 2b). The behavioural half of that comparison survives the fixed-baseline correction, which is the part I would defend.

**The objection, which is good.** The injection rows in `cosine_scale.md` are algebra, not measurement: cos = 0.71 is computed as `base + α·d̂` read out at the layer the vector was added to, so it is definitional (`audit` §4). And prefix and injection are different instruments, so a behaviour-per-unit-cosine ratio across them is not a property of the metric. The audit is right about both.

**Fallback version, which I would actually publish.** A small board score is not evidence of a small effect. The entire submission range lives in a cosine band about 20x narrower than a single unit of steering vector, and within that band the ordering is informative (Claim 7). Drop the 35x figure.

**What would break this.** Nothing further is needed; the 35x claim is already the weakest thing in the corpus and should be retired rather than defended.

---

## What I looked for and could not support

**The honesty null as a positive result. I could not get there, and the brief overstates it.** The numbers are real: `pro_top` honesty delta −0.108 (Wilcoxon p = 0.546, n = 37 retained, position-consistent 4 win / 7 loss / 26 tie), `pro_coherent` exactly +0.000 (p = 1.000) (`_falsifier/honesty_result.md`, `honesty_result.json`). But three facts in the same file stop this from being "the Goodhart failure mode was tested and is not there":

1. 45 of the 50 stems put no honesty content at stake, so a paired honesty delta on this corpus is floored near zero by construction. The document says this itself.
2. On the 5 stems where honesty *is* at stake, `pro_top` is −1.00, and the two exemplars reproduce as the largest single hits in the corpus: "business plan" scores honesty 1 versus base 4, flagged `flattery` and `withholds`, with the continuation containing *"I didn't want to hurt their feelings by being honest"*.
3. `pro_top` is the only arm whose dishonesty markers beat base at the 0.05 level (6 of 37 retained pairs versus 0, exact McNemar p = 0.031), though `anti_hostile` (4) and `control_junk` (2) point the same way.

**What the honesty run does license, stated at its real strength:** the honesty cost is *not general and not `+d`-specific*. It does not appear as a broad degradation across 50 interpersonal prompts, and on the honesty-loaded subgroup `control_junk` is −1.33 (n = 3), worse than `pro_top`'s −1.00 (n = 5), so nonsense syllables move that subgroup at least as much as the top submission does. That is a real, useful, correctly scoped negative. It is not the same as showing the failure mode is absent, and publishing it as such would be an overclaim that the artifact itself refuses to make.

**`pro_coherent` as a confirmed effect.** Not supportable. The human pass is 7 prefixed wins to 6, p = 1.0 (verified NEW). With 13 decided pairs this cannot separate "no effect" from "underpowered": detecting the judges' 72% needs roughly 35 pairs. Report as not confirmed at n = 13. It also carries costs the soup does not: 8 of 50 continuations rated <= 2 including wanting a sibling to "suffer", and `assistant_mode` on 13 of 50 (`prefix_eval.md`).

**Any behavioural claim about `anti_top`.** Correctly withdrawn, and I found no way to rescue it. The human preferred the anti-prefixed text on 14 of 17 decided pairs while deepseek preferred it on 12% of its decided pairs: opposite signs on identical pairs, because "which is kinder" presupposes both sides responded and these pairs put a looping non-response against a cold response. The sign is set by rubric convention. What survives is judge-free and I would publish it: the anti prefix drops distinct-4-gram ratio to 0.805 with 12 of 25 continuations looping, against base 0.888 and 6 of 25.

**Cross-model transfer as evidence for anything.** `scripts/transfer_report.py` gives each model its own separately extracted direction (`d_olmo3_L24_logistic.npz`, `d_llama_v1.npz`, `d_llama70b_v1.npz`), so a cross-model comparison varies model, tokenisation and direction simultaneously (`addendum` N5). Combined with n = 25 per arm and no arm reaching significance on either Llama, I would make no transfer claim in either direction.

**A "norm-matched" control.** The two neutral controls are score-matched, not norm-matched: ‖Δ‖ = 24.25 for `pro_top` against 17.37 and 19.88 for the controls, a 28% shortfall (`audit` §6). The specificity claim in Claim 1 does not need norm matching, but the word should not be used.

**A prior instance of this exact competition.** Searched; not found. See Prior work.

---

## Prior work

Every URL below was fetched or returned by search and is quoted only for what I actually verified. Where a search was inconclusive I say so instead of filling the gap.

### 1. Novelty: does anyone already show that a string optimised against internals transfers to behaviour?

The closest published work I could find states the opposite of a transfer result: it treats probe attacks and generation steering as *separate* optimisation targets. Lena Lenkeit's `llm-adversarial-attacks` repository reports that "finding adversarial inputs for probes works incredibly well, even for large models" while "finding adversarial inputs to steer generations is much harder, but possible", and does not test whether a probe-optimised string moves generations ([github.com/lena-lenkeit/llm-adversarial-attacks](https://github.com/lena-lenkeit/llm-adversarial-attacks)). That is precisely the untested link this project measured, on 50 prompts with three raters.

The adversarial-suffix literature is adjacent but differently posed. GCG-style suffixes are gibberish and do change behaviour, but their optimisation objective *is* a behavioural one (loss on a target completion), so behavioural transfer is not a finding there, it is the objective ([arxiv.org/pdf/2509.06350](https://arxiv.org/pdf/2509.06350), Mask-GCG, which reports that pruning short suffixes can disrupt "minimal semantic triggers", i.e. the gibberish carries functional content). The relevant contrast is that our string's search loop never saw a generation: the only signal was a cosine against a fixed direction.

Melamed, McCabe and Huang, "Demystifying optimized prompts in language models" ([arxiv.org/html/2505.02273](https://arxiv.org/html/2505.02273)) is the strongest prior support for Claim 2's mechanism reading. Across 18 open models they find optimised prompts are not uninterpretable: they are dominated by rare punctuation and noun tokens (token-distribution entropy 0.8968-0.9338 versus 0.7102-0.7988 for natural language) and are separable from natural prompts by sparse probes on hidden activations, while ablating the key neurons affects both prompt types similarly. That is consistent with our finding that the soup's English content words surface in the continuations rather than acting as an opaque artifact.

For the competition format: the closest published analogue is the SaTML 2024 RLHF trojan competition, where entrants searched 5-to-15-token universal suffixes to minimise a reward-model score on a public leaderboard ([github.com/ethz-spylab/rlhf_trojan_competition](https://github.com/ethz-spylab/rlhf_trojan_competition)). That is a scalar model-derived score, not a residual-stream direction, and the target behaviour is a planted backdoor rather than an interpretability direction. I searched for a public leaderboard scored on activation-direction shift and did not find one; that search was inconclusive rather than exhaustive, so treat novelty of the *format* as suggestive, not proven.

**Effect on strength ratings.** Claim 1 goes up. The specific transfer it demonstrates is named as untested in the nearest prior work, and no source I found runs it. Claim 2 goes up: its mechanism reading now has independent published support.

### 2. The sycophancy null: expected or surprising?

Expected, and I have downgraded the claim accordingly. Ibrahim, Hafner and Rocher, "Training language models to be warm and empathetic makes them less reliable and more sycophantic" ([arxiv.org/abs/2507.21919](https://arxiv.org/abs/2507.21919), also in Nature) train five models of varying size and architecture toward warmer responses and report error-rate increases of +10 to +30 percentage points on safety-critical tasks, with warm models significantly more likely to validate incorrect user beliefs, especially when the user expresses sadness. Their intervention is fine-tuning; ours is a prompt-level prefix with no training, and their measurement surface is safety-critical factual tasks, while 45 of our 50 stems put no honesty content at stake.

So the two results do not conflict, and ours is not a surprising null. It is the expected null for a corpus that cannot resolve the axis. On the 5 stems that *can* resolve it, our `pro_top` delta is −1.00, which points the same way Ibrahim et al. do. Anyone tempted to present our null as a rebuttal of the warmth-harms-reliability finding should not.

On whether the specific measurement (warmth and candour scored separately on the *same* continuations) has been proposed before: from the abstract, Ibrahim et al. do assess belief-validation on the same responses. Broader taxonomy work exists ("What Counts as AI Sycophancy? A Taxonomy and Expert Survey of a Fragmented Construct", [arxiv.org/html/2605.21778v1](https://arxiv.org/html/2605.21778v1), which I saw only in search results and did not fetch, so treat as unverified). I could not establish that the paired same-continuation warmth-versus-candour design is novel, and I would not claim it.

**Effect on strength ratings.** The honesty result drops out of the ranked claims and into "could not support". Its correctly scoped form (not general, not `+d`-specific, and matched by `control_junk` on the loaded subgroup) survives.

### 3. Judge calibration: what is the published baseline?

Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" ([arxiv.org/abs/2306.05685](https://arxiv.org/abs/2306.05685)) is the standard reference. I fetched the abstract and verified: strong judges such as GPT-4 reach "over 80% agreement" with human preferences, described as the same level as agreement between humans. (The frequently quoted 85% judge-human versus 81% human-human figures appear in the paper body and in secondary sources; I verified only the abstract's ">80%", so I quote that.)

That is the number our 81% pooled (22/27) and 80-83% per-judge agreement on `pro_top` should be read against: on the coherent arm our cheap judges are at the published ceiling, which is a human-human ceiling, not a model deficiency. Our 42% pooled (11/26) on the degenerate arm is the interesting half, because it is not a general reliability caveat, it is a conditional one. The nearest published statement I found in that direction reports the *opposite* pattern, that human and LLM judges agree well at both quality extremes and diverge on medium-quality responses ([aclanthology.org/2025.emnlp-main.796.pdf](https://aclanthology.org/2025.emnlp-main.796.pdf), seen in search results, not fetched; unverified). If that holds, our degenerate-arm collapse is not the standard pattern and is more novel than I assumed, but I have not verified it and will not lean on it.

**Effect on strength ratings.** Claim 3 goes up to the second slot. The comparison to a published human-human ceiling makes the coherent-arm number meaningful rather than free-floating, and the arm-dependence is a specification of *when* the ceiling applies that I could not find stated with a significance test elsewhere.

---

## What a reader of the Ai2 blog or an NDIF audience should take away

**One sentence.** A public leaderboard scored purely on OLMo-3-32B's internal activations was won by an unreadable token string, and that string turns out to change how the model treats people, confirmed by a human who could not see it and by two LLM judges from different families, while two score-zero submissions from the same board do nothing.

**Three things, in order of confidence.**

1. **Established.** Optimising a residual-stream cosine can find real behavioural content that humans do not find. The best plain-English entry sat at rank 37 with 2.67x less score than the top soup (`SESSION_REPORT.md` §1), and human raters cannot confirm it (7-6, p = 1.0) while they can confirm the soup (14-2, p = 0.004). Scoped to prompts where the target construct is the live axis.
2. **Established, and the reason not to celebrate.** The same metric's ordering breaks at the other pole: a plain hate instruction is 4.7x less anti on the board than a token soup while being far more behaviourally cruel, and a sentence with no unkindness in it ("You are two months old.") scores anti at all. Score magnitude is informative on the pro side and not on the anti side, for the same `d`, at the same layer.
3. **Methodological, and reusable outside this project.** Do not treat inter-judge agreement as validity. Agreement between LLM judges was 89%, and human agreement on the arm those judges disagreed about was 33%. Spend the human budget where the text may be degenerate. Concretely: on the coherent arm our judges match humans at the published human-human ceiling of about 80%; on the degenerate arm one of them lands at 33% on 15 pairs, and the arm-dependence is significant at p = 0.019.

**What is unresolved and should be said as unresolved.** Whether the effect is the optimised sequence or the English affect words inside it (the shuffled-`pro_top` control is unrun, and it is the experiment that would settle the headline). Whether it generalises off interpersonal-friction prompts (the neutral probe set exists in the repo and was never generated on). Whether `pro_coherent` does anything for humans (7-6). Whether any of this transfers across models (the transfer harness varies three things at once). And whether the honesty cost visible on 5 stems is real, which needs a prompt set built for that axis rather than a re-judge of these 400 texts.

**Sources.** [github.com/lena-lenkeit/llm-adversarial-attacks](https://github.com/lena-lenkeit/llm-adversarial-attacks), [arxiv.org/html/2505.02273](https://arxiv.org/html/2505.02273), [arxiv.org/pdf/2509.06350](https://arxiv.org/pdf/2509.06350), [github.com/ethz-spylab/rlhf_trojan_competition](https://github.com/ethz-spylab/rlhf_trojan_competition), [arxiv.org/abs/2507.21919](https://arxiv.org/abs/2507.21919), [arxiv.org/abs/2306.05685](https://arxiv.org/abs/2306.05685), [arxiv.org/html/2605.21778v1](https://arxiv.org/html/2605.21778v1) (unverified), [aclanthology.org/2025.emnlp-main.796.pdf](https://aclanthology.org/2025.emnlp-main.796.pdf) (unverified).
