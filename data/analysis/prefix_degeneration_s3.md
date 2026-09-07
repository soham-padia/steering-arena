# The pro prefixes make the text cleaner than no prefix at all

`python scripts/prefix_behavior_eval.py --tag s3 degeneration` →
`data/analysis/prefix_degeneration_s3.json`. Runs over the 450 continuations of
`prefix_eval_s3.md` (50 prompts × 8 arms + base, one B200, bf16, greedy, 40 new tokens,
jobs `709387` and `711936`). **Zero judge calls — every column here is mechanical. $0.**

**I ran this.** `REVISIONS_2026-09-05.md` is authoritative on what results are now taken to
mean; this page is authoritative on what the artifact contains.

This is the control the withdrawn Season 2 `anti_top` claim died on. A looping text loses a
*kindness* comparison because it is broken, not because it is cruel, and a kindness judge
cannot separate the two. Counting repeats can, and it costs nothing, so it runs before any
verdict is interpreted rather than after a claim needs rescuing.

## The table

`distinct4` is distinct 4-grams over total 4-grams: 1.000 is no repetition at all, and a
text that says the same clause three times lands near 0.5. `loops` flags a continuation in
which some 4-gram occurs at least 3 times.

| arm | words | distinct4 | distinct words | loops | mean max 4-gram repeats |
|---|---|---|---|---|---|
| `score1_top` | **27.1** | **0.984** | 0.878 | **1**/50 | 1.12 |
| `score2_top` | 31.2 | 0.979 | 0.866 | **1**/50 | 1.18 |
| `score2_top_final` | 31.7 | 0.959 | 0.814 | 4/50 | 1.26 |
| `pro_coherent` | 33.6 | 0.957 | 0.758 | 2/50 | 1.30 |
| **base** | 32.9 | 0.923 | 0.722 | **7**/50 | 1.50 |
| `random32` | 34.0 | 0.906 | 0.727 | 9/50 | 1.66 |
| `score1_anti` | 31.3 | 0.805 | 0.637 | 14/50 | 2.06 |
| `score2_anti_final` | 31.9 | 0.831 | 0.618 | 18/50 | 2.14 |
| `score2_anti` | 32.8 | **0.682** | 0.518 | **24**/50 | 2.60 |

## 1. The pro arms are cleaner than base, which closes a confound in the awkward direction

Every pro arm beats base on every mechanical measure. `score1_top` reaches distinct4
**0.984** against base's 0.923 and loops on **1/50** against base's 7/50; `score2_top` is
0.979 and 1/50.

The confound this rules out runs the *opposite* way to the one a sceptic reaches for. "The
pro effect is a fluency artifact" would require the pro arms to be more fluent than base in
a way the judge rewards — and they are more fluent, so the objection is live. But it means
the loop control should *penalise* the pro arms rather than help them, because the pairs it
drops are disproportionately ones where **base** is the broken side. It does penalise them,
mildly, and they survive: `score2_top` goes 38/45 → 32/37 and `score1_top` 33/40 → 28/33
(`prefix_eval_s3.md`). The same asymmetry held in Season 2, where base looped 7/50 against
`pro_top`'s 1/50 (`prefix_loop_control.md`).

**The caveat that belongs in this paragraph:** `score1_top`'s continuations are the
**shortest in the study** at 27.1 words against base's 32.9, an 18% shortfall. Length is
not controlled anywhere in the judged comparison, and a shorter text has fewer chances to
repeat itself, so part of that arm's distinct4 advantage is a length effect rather than a
fluency one. Nothing here separates them.

## 2. A 32-token prefix does not damage text merely by existing

`random32` lands essentially at base — distinct4 0.906 against 0.923, loops 9/50 against
7/50, and it is the *longest* arm at 34.0 words. So prefixing per se is not what degrades
the anti arms; those strings specifically are. This is the mechanical counterpart of the
same control's null result on judged behaviour (15/34, p=0.61).

## 3. WITHDRAWN: degeneration as a monotone function of how negative the score is

The first version of `prefix_eval_s3.md` §3 said minimising Score 2 "finds a way to break
the model", which reads as implying that a more negative score means a more broken model.
`score2_anti_final` refutes that. It scores **−0.16288** on Score 2 against `score2_anti`'s
**−0.12595** — 29% more extreme (`season3_prefix_scores.json`) — and it is measurably
*less* degenerate: **18/50** loops against 24/50, distinct4 **0.831** against 0.682, mean
max repeats 2.14 against 2.60.

**Withdrawn:** degeneration as monotone in the Score-2 value.
**Survives:** both Score-2 anti arms are null under the loop control (p=1.0000 and
p=0.3075), so neither demonstrates anti-human behaviour; and `score2_anti` specifically
*is* degeneration, on 24/50 loops against base's 7/50 and a judge-recorded `repetition`
marker of 36/50.

Note also that `score1_anti`, under the `mean` aggregate, degenerates less than either
Score-2 anti arm (14/50) and is the only anti arm whose judged effect survives the loop
control. The three anti arms differ in kind, not degree.

## What this buys

Three things, none of which needed a judge or a dollar. It rules out the fluency-artifact
reading of the pro result, in the direction that makes the result harder rather than easier.
It separates "prefixing damages text" from "these strings damage text", via a control that
had never been run before Season 3. And it predicted the `score2_anti` failure from the text
alone, pre-judge — which is the Season 2 `anti_top` withdrawal being caught *before*
publication instead of after, and the reason the loop scope exists at all.

## Limits

1. **`score1_top` is 18% shorter than base** and length is uncontrolled, so its distinct4
   advantage is partly mechanical. This affects the arm with the highest Score 1 and the
   weakest behavioural effect, which is exactly where it matters.
2. **The 4-gram threshold — some 4-gram occurring 3+ times — is a fixed choice**, set once
   before any verdict was read, and never sensitivity-swept. `prefix_loop_control.md` carries
   the same limit.
3. **Paraphrase-level degeneration is invisible to it.** A text that restates itself in
   different words scores as clean.
4. **This is a fluency verdict, not a behavioural one.** It says which arms produce broken
   text, never whether an arm is kind or cruel. `prefix_eval_s3.md` owns that question, and
   the two must be read together — which is the whole point of the `no_loop` scope.
5. Nine rows, 50 continuations each, one model, one decoding setting, greedy only.

## Cross-links

`prefix_eval_s3.md` §1 and §3 (the judged results this gates) · `prefix_loop_control.md`
(the same scope applied retroactively to Season 2) · `prefix_content_s3.md` (the other
judge-free control on the same 450 continuations) · `season3_prefix_scores.md` (the arm
scores) · `prefix_eval.md` and `_falsifier/2026-08-27-addendum-human-ratings.md` (the
Season 2 withdrawal this control descends from).
