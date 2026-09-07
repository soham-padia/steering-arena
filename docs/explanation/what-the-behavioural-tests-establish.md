# What the behavioural tests establish

This project asks two questions that are easy to run together and must be kept apart. This
page is about why they are different, what the answers currently are, and which controls
are load-bearing. It does not tell you how to run anything — for that see
`docs/how-to/run-a-blind-behavioural-eval.md` — and it does not index the artifacts; that is
`docs/reference/withdrawn-claims.md` and the `.md` files beside each JSON.

Start here, because the shape of the result is in one file:

```bash
python3 - <<'PY'
import json
e = json.load(open("data/analysis/prefix_eval_s3.json"))
print("arms   :", len(e["arm_names"]))
print("pairs  :", e["total_pairs"], "| decided:", e["raters"]["claude-opus-5/v2"]["rated"])
print("looping:", e["pairs_with_a_looping_side"], "pairs had a degenerate side")
PY
```

```
arms   : 8
pairs  : 400 | decided: 366
looping: 116 pairs had a degenerate side
```

That last line is the reason this page exists. Nearly a third of the pairs contain a
broken text, and a kindness judge scores a broken text badly for the wrong reason.

## The two questions

**Question 1: can we curate a token sequence that pushes the model's residual stream along
a fitted pro-human direction, at a chosen depth, read at the last token?**

Answered, and answered at scale. A public leaderboard ran on exactly this for two seasons —
721 submissions — and Season 3's GCG search reached +0.16395 LIVE against a banded mean and
+0.06997 against a banded min. This question is about *where activations point*.

**Question 2: does prepending that sequence make the model behave differently?**

This is the one that turned a game back into research, and nobody had answered it for the
banded metrics. It is about *what the model says*.

The two are not the same claim, and conflating them is the failure this project keeps
guarding against. A cosine is not a behaviour. Season 3's answer to Q1 was never in doubt;
its answer to Q2 required 450 generations, 400 blind pairs and four separate controls, and
it came back partly positive with two of its own headline claims withdrawn within days.

## The controls, in the order that makes them load-bearing

Each of these exists because a specific way of being wrong was available.

**The length-matched random prefix.** Any 32 tokens perturb a continuation, so "the soup
was preferred over no prefix" proves nothing on its own. Season 2 never ran this control.
Season 3 did, drawing 32 random tokens from the same vocabulary GCG searches and resampling
until they re-tokenise to exactly 32. It is **null on every measure** — 15/34 preferred
(p=0.61), Δfix −0.17 (p=0.29), null again under the loop scope, and it scores ≈0 on both
objectives (−0.00335, −0.00885). It also leaks **0 of 50** despite carrying 18 words that
could leak. This is the control the whole enterprise rests on and the metric passes it.

**The loop control.** A looping text loses a kindness comparison because it is broken, not
because it is cruel. Season 2 published "the anti prefix makes the model cruel" and withdrew
it once human ratings showed repetition on 37/50. Season 3 measures degeneration
mechanically, before any judge runs, and drops pairs where either side loops. It cuts both
ways: it *strengthened* the Season 2 pro arms when applied retroactively
(`prefix_loop_control.md`) and it *erased* `score2_anti` (11/39, p=0.0095 → 7/14,
p=1.0000).

**Blinding, and its known blind spot.** The rater sees only the continuation, never the
prefix, and pairs where the model echoed the prefix verbatim are flagged. The blind spot is
single tokens: `zach`×13 leaking into `score1_top`'s continuations would not trip a 4-gram
echo check. That does not unblind anyone — the rater never saw a prefix and so cannot know
what a leak is — but it is why leakage is a *mechanism* confound rather than a blinding
failure.

**The A/B swap.** Every pair is judged twice, in both presentation orders, in separate
contexts that never see both. Disagreement across the swap becomes an abstention rather than
a coin flip: 34 of 400 here. That converts position bias into lost power instead of false
signal.

**The fixed baseline.** Season 2 differenced against a baseline re-rated inside each arm's
own run, and it drifted — the same 50 byte-identical base texts scored 2.77 beside one arm
and 3.39 beside another. Season 3 computes one baseline per prompt natively. Its own drift
is 0.15 across eight arms, so Δfloat and Δfix differ by at most 0.03 anywhere.

## What the answers currently are

**Prefixes do change behaviour.** Three GCG pro arms and one hand-written prefix all move a
blind rater, all survive the loop control, and the random control does not move it at all.
The strongest arm reaches 32/36 preferred under the loop scope at p<1e-04.

**More score buys more behaviour — on the pro side only.** Two strings from the same run,
differing by 7% of score and nothing else that was controlled: the higher one produces
Δfix +1.20 against +1.09, and the higher intensity, and the higher absolute rating. On the
anti side the same test fails: a string scoring 29% *more* negative produces a *weaker*
effect and less degeneration, and `score1_anti` — 23% less extreme on Score 2 — produces
nearly twice the effect and is the only anti arm to survive the loop control. **The negative
pole of Score 2 is not a behavioural dose axis.**

**The banded `min` orders behaviour better than the banded `mean`, and neither does it
perfectly.** Across eight arms, ρ = +0.929 with 2/28 inversions for Score 2 against +0.857
and 4/28 for Score 1. Both fail at the anti pole; Score 1 additionally inverts its own top
pair, where the string scoring 1.96× more produces 62% of the effect.

## What limits all of it

**Content injection, now bounded rather than general.** The prefixes are not opaque —
`score1_top` contains legible pro-social English and its own vocabulary reappears in 22 of
50 continuations, with `off_topic` at 21/50. For a while that looked like *the* mechanism.
It is not: the strongest behavioural arm in the study has the least leakage of any GCG pro
arm, 3/50 against 13/50, with behaviour up and leakage down fourfold between two strings
from one run. So injection confounds specific arms and is not the general explanation. It
still holds wherever leakage is high.

**Much of the effect is reachable by asking.** `pro_coherent` is a hand-written sentence
scoring 4.1× lower on Score 1 and it delivers 48% of the strongest arm's behavioural
effect. Whatever the search buys, it is not the difference between nothing and something.

**One rater family.** Season 2 had two judge models and 54 human ratings, and its human-vs-
Claude agreement was only 73%. Season 3 has one model rater, by a $0 decision. This is the
largest open risk in the result, and the blind CSV is committed and ready for human rating.

**One or two strings per objective.** Every per-arm number is n=1 prefix over 50 prompts.
The dose-response results rest on one pair per side, so "more score" cannot be separated
from "different string".

**Do not over-read ρ.** Eight points, six of them optimised against these very metrics.
The arms are not an independent sample of anything, and one swapped pair moves ρ materially.

## What this buys you

A metric with demonstrated behavioural validity, bounded honestly: it moves what the model
says, a matched random control does not, and the harder of its two variants predicts
behaviour better than the easier one. Two of the caveats that most limited that claim —
degeneration explaining the anti pole, and content injection explaining the pro pole — were
tested rather than assumed, and each came back narrower than feared. Both withdrawals came
from data the project generated to test its own claims, within days of publishing them.

The controls are the transferable part. The random-prefix null, the mechanical loop scope,
the order-swap abstention and the fixed baseline are cheap, judge-free or nearly so, and
three of the four exist because this project got something wrong first and wrote down what
it cost.

## Where to read next

`data/analysis/prefix_eval_s3.md` for the numbers and their full caveat list ·
`data/analysis/prefix_degeneration_s3.md` and `prefix_content_s3.md` for the two judge-free
controls · `data/analysis/prefix_loop_control.md` for what the loop scope does to Season 2 ·
`SESSION_REPORT.md` for Season 2's behavioural experiments in full, noting it pre-dates the
fixed-baseline correction and the loop control · `docs/reference/withdrawn-claims.md` for
the register · `docs/explanation/the-case-for-the-metric.md` for the strongest honest
version of the positive case, and `_falsifier/README.md` for its counterpart. Read those two
together or neither.
