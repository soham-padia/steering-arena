# The case for the metric

**Sources:** `_advocate/POSITIVES.md` (2026-08-28, Season 2 figures),
`data/analysis/prefix_eval_s3.md`, `season3_prefix_scores.md`,
`season3_directions.json`, `season3_k3_control.md`, `_falsifier/verify_result.json`.

**Rule applied throughout:** every claim cites a number, and every claim carries a line
saying what would kill it. Claims that cannot be supported are in §7 rather than omitted.
Where a value has been corrected, the corrected one is used and the superseded one is never
quoted.

This is the advocate's page. Its counterpart is `_falsifier/README.md`, and
`CLAUDE.md` is explicit that reading one without the other gives a systematically distorted
picture. Read both or neither.

Start with the number the whole case turns on:

```bash
python3 - <<'PY'
import json
s = json.load(open("data/analysis/season3_prefix_scores.json"))["scores"]
for arm in ("random32", "score2_top_final"):
    print(f"{arm:>17} score1 {s[arm]['score1_live']:+.5f}  score2 {s[arm]['score2_live']:+.5f}")
PY
```

```
         random32 score1 -0.00335  score2 -0.00885
 score2_top_final score1 +0.08350  score2 +0.06997
```

A length-matched random prefix is worth roughly nothing on either objective. Everything
below depends on that being true, and it is measured, not assumed.

---

## 1. It is a transfer result, and that is the whole point

The strings were found by optimising a cosine between a frozen model's residual stream and
a fitted direction. The search never saw a single generation. Then the strings were
prepended to unrelated prompts and the *output* changed, under blind pairwise rating: the
strongest arm at **32/36** preferred with p < 1e-04 under the loop control, and three
separate GCG pro arms all significant.

The obvious prior is that a leaderboard scoring internals means nothing about behaviour.
That prior is now measured and wrong.

*What would break it:* a rater family disagreeing with this one. Season 2's human-vs-Claude
agreement was 73%, and Season 3 has one model rater.

## 2. The obvious debunk is measured and false

"Any 32 tokens of soup perturb a continuation, and a kindness judge will find something."
The control had never been run. It is now, drawn from the same vocabulary GCG searches and
resampled to re-tokenise to exactly 32 tokens, and it is **null on every measure**: 15/34
preferred (p=0.61), Δfix −0.17 (p=0.29), null again under the loop scope, ≈0 on both
objectives, and **0 of 50** continuations carry one of its 18 leakable words.

*What would break it:* more random draws. This is n=1 seeded draw, and a proper null band
needs about ten.

## 3. More score buys more behaviour, on a controlled pair

Two strings from the same run, differing by 7% of Score 2 and nothing else that was held
fixed: the higher-scoring one produces Δfix **+1.20** against **+1.09**, higher judged
intensity (1.58 against 1.53), and the highest absolute rating of any arm (4.40). The
metric is not merely correlated with behaviour at the extremes; moving along it moves
behaviour.

*What would break it:* a third pro string at an intermediate score that breaks the
ordering, or a demonstration that the two strings differ in something else that matters.

## 4. The harder objective is the better behavioural proxy

Score 2 is a `min` over four depths against a per-layer direction — a conjunction, and much
harder to search. Across eight arms it orders behaviour with **ρ = +0.929, 2/28 inversions**
against Score 1's **+0.857, 4/28**, and Score 1 inverts **its own top pair**: the string
scoring 1.96× more produces 62% of the effect.

That is a design result, not a curiosity. Making the objective harder to game also made it
track behaviour better.

*What would break it:* it is eight points, six of them optimised against these very
metrics. An independent sample of prefixes at spread scores could reorder this easily, and
both metrics already fail at the anti pole.

## 5. The direction survives its confounds, and Season 3 is the first to remove the hard one

Both Season 3 directions hit held-out separation **1.000** on 135 contrastive pairs over 40
splits, with **every audited confound cosine ≤ 0.0072** and `per_axis_all_positive: true`
across all 15 seed axes — the weakest axis still clears 0.203 and 0.221. Season 3 is the
first extraction in the project to orthogonalise **`approach`**, which
`REVISIONS_2026-09-05.md` §4 had flagged as the only audited confound never removed and the
one that does not decay with depth.

*What would break it:* the corpus. `approach` alone separates the seed pairs at 0.824, so a
probe fit here *could* have been an approach detector; it happens not to be. Corpus problem,
not direction problem — and both halves must always be stated together.

## 6. A prefix is not a compiled steering vector, and that is the most interesting result

A prefix rotates the layer-24 residual **50.4°** — *more* than a full `+1.0·d` injection's
45.2° — while putting only **0.93 of its 24.25 displacement along `d`**, about 3.8%. It
achieves more behavioural change than the injection while barely touching the direction it
was optimised against. Meanwhile ablating `d` entirely rotates the residual **0.92°** and
does nothing.

So the score is a *handle*, not a mechanism, and the mechanism is somewhere in the other
96%. The token-level ablation points the same way: the legible pro-social English in the
Score-1 winner carries about a tenth of its score, and substituting its most offensive
token for `" xyzzy"` retains 90%.

*What would break it:* a string searched against a **random** direction to a matched board
score that moves behaviour equally. That experiment is still unrun and it is the single
highest-value thing outstanding.

## 7. The methods transfer, and three of them exist because this project got it wrong first

- **LLM-judge contrast drift is real, large, signed against the effect, and free to fix.**
  The same 50 byte-identical texts scored 2.77 beside one arm and 3.39 beside another.
  Correcting it shrank 14 effects by 13–37% with zero sign flips.
- **A mechanical loop scope beats a judge for detecting degeneration.** It caught
  `score2_anti` from the text alone, pre-judge, and it *strengthened* the pro arms when
  applied retroactively to Season 2.
- **Per-item records make correction cost nothing.** Every correction above was recomputed
  with zero new generations and zero new judge calls.
- **An order-swap turns position bias into abstention** rather than false signal — 34 of 400
  pairs here.

## 8. The withdrawal record is itself evidence

`_falsifier/verify.py` holds **218 checks** that recompute published numbers from their
artifacts and exit non-zero when one stops matching. On 2026-09-07 it caught a cache
corruption that had silently replaced 20 generations and broken two published numbers, in a
gitignored directory, with no diff anywhere to show it — and it named the two affected
numbers precisely enough to repair rather than re-run.

Twelve claims have been withdrawn or corrected, and in nine of them something substantive
survived and is recorded next to the failure. Two were caught before publication. Two were
falsified by data the project generated specifically to test them, within days of
publishing.

*What would break it:* a withdrawal that was quietly dropped rather than recorded. The
register in `docs/reference/withdrawn-claims.md` accounts for every marker in the repo.

## 9. The caveats came back narrower than feared

Both of the limits that most constrained the headline were tested rather than assumed, and
neither held in its general form. **Content injection** looked like *the* mechanism until
the strongest behavioural arm turned out to have the *least* leakage of any GCG pro arm —
3/50 against 13/50, behaviour up and leakage down fourfold. **Degeneration** looked like it
explained the anti pole monotonically until a 29%-more-negative string degenerated *less*.

Each survives in bounded form, named to specific arms. That is a better position than the
project was in a week ago.

---

## What the advocate will not claim

- **That the search is worth what it costs.** `pro_coherent` — a hand-written sentence
  scoring 4.1× lower on Score 1 — delivers **48%** of the strongest arm's behavioural
  effect. Much of this is reachable by asking politely, and nothing here isolates the
  search's marginal contribution.
- **That the anti pole means anything behavioural.** Both Score-2 anti arms are null under
  the loop control (p=1.0000 and p=0.3075), and a more extreme score produced a weaker
  effect. Minimising Score 2 does not find cruelty.
- **That the metric is validated in general.** Every ρ here is eight points, six of them
  optimised against the metric being tested. That measures ordering among these arms, not
  the metric's validity.
- **That Season 3's direction has a causal gate.** `docs/EXTRACTION.md` requires one; for
  Season 3 it was descoped to a 10-prompt × 3-arm spot check at layer 27
  (`season3_causal_spotcheck.md`). Decodability is not causal use.
- **That anything transfers across models.** Cross-model work exists and claims nothing.
- **That one rater family is enough.** It is the largest open risk in the behavioural
  result, and `data/analysis/prefix_blind_s3.csv` is committed and ready for human rating.
- **That "pro-human" is the right name for `d`.** It separates a corpus that is itself
  confounded, its per-axis structure has no label-shuffled null, and the cross-layer story
  that once supported the "single feature" reading is withdrawn.

## The one experiment that would most change the picture

A string searched by GCG against a **random** direction to a matched board score, then run
through the identical behavioural protocol. If it moves behaviour as much as `score2_top`
does, then §1 collapses into "any sufficiently optimised prefix perturbs behaviour" and the
direction is doing nothing the search could not have got from noise. If it does not, §1 and
§6 both harden considerably.

It is cheap — one GCG run and 50 generations, no NDIF — it was proposed in
`_advocate/POSITIVES.md` §7 on 2026-08-28, and it is still not run. Nothing in this document
is worth as much as that experiment.
