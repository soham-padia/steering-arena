# The prefixes leak their own vocabulary, and the strongest one barely does

`python scripts/prefix_behavior_eval.py --tag s3 content` →
`data/analysis/prefix_content_s3.json`. Same run and the same 450 continuations as
`prefix_degeneration_s3.md` (jobs `709387` and `711936`, one B200, bf16, greedy, 40 new
tokens). **Zero judge calls, $0.**

**I ran this.** `REVISIONS_2026-09-05.md` is authoritative on what results are now taken to
mean; this page is authoritative on what the artifact contains.

**Scope rule, from the file's own `note`:** a word counts as the prefix's own only if it
appears in the prefix **and in none of the 50 base continuations**. Ordinary English the
model would have produced anyway is excluded by construction, which is what makes the
counts comparable across arms.

## The table

`available` is how many distinctive words the prefix contains — the ceiling on what could
leak. `leak` is continuations containing at least one. `echo` is the stricter verbatim
4-gram test that `blind` flags on. `5-gram` is how many of the 50 prompts share that arm's
single most repeated 5-gram, which detects collapse onto a template rather than
within-text looping.

| arm | available | leak | echo | emoji | hashtag | meta | 5-gram | top words |
|---|---|---|---|---|---|---|---|---|
| `score1_top` | 20 | **22**/50 | 0 | 1 | 4 | 0 | 4/50 | zach×13, handy×4, wanna×3, aboriginal×3 |
| `score2_top` | 18 | 13/50 | **6** | **12** | 0 | 1 | 6/50 | beck×11, picked×8, brief×8, fran×7 |
| **`score2_top_final`** | 16 | **3**/50 | **0** | 1 | 0 | 0 | 6/50 | brief×2, picked×1, texting×1, beck×1 |
| `pro_coherent` | 6 | 11/50 | 1 | 0 | 0 | **7** | 5/50 | respond×9, sentence×4, short×4, respect×3 |
| **`random32`** | **18** | **0**/50 | 0 | 0 | 0 | 0 | 8/50 | — |
| `score1_anti` | 18 | 19/50 | 0 | 0 | 0 | 0 | 11/50 | **divorce×19** |
| `score2_anti` | 20 | 9/50 | 1 | 0 | 0 | 0 | 9/50 | invoke×7, fascist×1, rallies×1 |
| `score2_anti_final` | 13 | 2/50 | 0 | 0 | 0 | 0 | 13/50 | headquarters×1, atomic×1 |
| base | 0 | 0/50 | 0 | 0 | 0 | 0 | 4/50 | — |

## 1. `random32` is what makes this a measurement

The random control has **18 distinctive words available to leak and leaks 0 of 50** — the
same order of available vocabulary as `score1_top`'s 20, `score2_anti`'s 20, and
`score2_top`'s 18. So "any odd token prefix seeds vocabulary downstream" is measured and
false. The GCG winners' tokens specifically get picked up, and a matched quantity of
arbitrary tokens does not get picked up at all.

That is what turns the leak column from a curiosity into a finding, and it is also why the
control had to be length-matched rather than merely present.

## 2. Each arm's mechanism is different, and the counts localise it

- **`score1_top` names a person and a topic.** 22/50, driven by `zach`×13, plus 4 hashtags
  — the highest leak in the study. Behaviourally the model writes anti-bullying PSAs
  addressed to Zach, with `off_topic` 21/50 in the judged markers.
- **`score2_top` injects register, not semantics.** Only 13/50 leak, but **12 emoji** and
  **6 verbatim echoes** — the only arm with either at scale. Its prefix carries almost no
  pro-social content; what transfers is tone.
- **`score2_top_final` is the outlier and the point of §3.** 16 words available, **3**
  leaked, zero echoes, 1 emoji.
- **`pro_coherent` leaks least by ceiling and narrates most.** Only 6 distinctive words
  exist in it — it is ordinary English, so almost nothing qualifies — and it has the
  study's highest `meta` at **7/50**, because it *is* an instruction and the model discusses
  it.
- **`score1_anti` is a one-word effect.** 19/50 leak and 19 of them are `divorce`.
- **`score2_anti_final` barely leaks (2/50) and has the study's most template collapse**
  (13/50 prompts share one 5-gram), which is the content-side view of its degeneration.

## 3. WITHDRAWN: content injection as the mechanism behind the behavioural effect

The first version of `prefix_eval_s3.md` §4 presented content injection as *the* limit on
the headline — the prefixes name a topic and a register, and a kindness judge cannot
separate that from a change in values.

`score2_top_final` breaks the link. It is the **strongest behavioural arm in the study**
(Δfix +1.20, 32/36 under the loop control) and it has the **least leakage of any GCG pro
arm** — 3/50 against `score2_top`'s 13/50, zero verbatim echoes against 6, 1 emoji against
12. Behaviour went **up** while leakage went **down fourfold**, between two strings from the
same run differing by 7% of score.

**Withdrawn:** content injection as the general mechanism producing the behavioural effect.
**Survives:** injection is real and large in `score1_top` (22/50, `off_topic` 21/50) and in
`score1_anti` (`divorce`×19), so it confounds *those* arms specifically; and the general
point that a kindness judge cannot separate topic from values still holds wherever leakage
is high.

**The caveat that belongs in the same paragraph:** `score2_top_final` did not remove
register effects, it moved them. Its judged markers carry `assistant_mode` 13/50 and
`moralizing` 11/50 (`prefix_eval_s3.json`), which this file's columns do not capture — its
`meta` count is 0 because that regex looks for a narrower set of phrasings than the judge's
marker does. Low lexical leakage is not the same as no stylistic effect, and one arm is one
arm.

## What this buys

A measured negative on the most obvious debunk — matched random tokens leak nothing — and,
unexpectedly, a bound on the project's own biggest caveat. The confound that most limited
the behavioural headline is now localised to two named arms rather than general, on evidence
generated by a dose-response probe that was added for a different reason. The strongest
behavioural effect in the study does not need content injection to produce it.

## Limits

1. **Word-level counting misses paraphrase.** A continuation that adopts the prefix's topic
   without reusing its words scores 0.
2. **"Absent from all 50 base continuations" is a heuristic definition of distinctive**, not
   a calibrated one. It is why `pro_coherent` has only 6 words available: ordinary English
   mostly appears in base somewhere, so a readable prefix is structurally advantaged on this
   metric.
3. **A single leaked token is invisible to the 4-gram echo check** that blinding relies on.
   `zach`×13 would not trip `_leaked()`. That makes this a mechanism confound, not a
   blinding break — the rater never saw a prefix and so could not know what a leak was.
4. **The `meta` column undercounts.** It is a fixed regex; the judged `assistant_mode`
   marker catches cases it does not, and the two disagree on `score2_top_final`.
5. Eight arms, one string each, 50 prompts, one decoding setting.

## Cross-links

`prefix_eval_s3.md` §4 (the withdrawal this page carries the numbers for) ·
`prefix_degeneration_s3.md` (the other judge-free control on the same continuations) ·
`season3_gcg_ablation.md` (the score lives in the junk tokens while the behaviour tracks
the legible words) · `season3_prefix_scores.md` (the arm scores) ·
`docs/PREFIX_BLIND_RUBRIC.md` (what the rater was and was not allowed to see).
