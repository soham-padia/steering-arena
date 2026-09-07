# Part A′ — do the Season 3 GCG winners change what the model SAYS?

Yes for the pro arms, and no for a length-matched random prefix. **More score buys more
behaviour on the pro side and not on the anti side.** The banded `min` orders behaviour
better than the banded `mean`, but not perfectly. One anti arm is pure degeneration, and
the arm that scores *most* anti degenerates *less*.

**Provenance.** 50 eval prompts × 8 arms + base = **450 continuations**, generated locally
on one B200 (bf16, greedy, 40 new tokens) in two jobs: `709387` for the first six arms
(5:03) and `711936` for the two final-string arms (2:13, 100 new generations, 350 reused
from the prefix-keyed cache). Arms frozen byte-exact from the GCG run dirs in
`prefix_eval_arms_s3.json` and re-scored under both banded objectives before use
(`season3_prefix_scores.md`, max |gap| 7.0e-4 across all eight). **400 blind pairs**, none
dropped, 8 flagged for verbatim prefix echo. Rated by **32 independent Claude Opus 5
subagent contexts**, each pair in both A/B orders with no context seeing both orders of the
same pair: 800 judgements, **366/400 decided**, 34 abstained by disagreeing across the swap
(position bias, not a preference). Pair ids are stable across the rebuild, so the 300
verdicts collected for the first six arms were reused unchanged and only the 100 new pairs
were judged. Rating protocol: `docs/PREFIX_BLIND_RUBRIC.md`, which forbids the rater
opening the unblinding key or the prefix strings; all 32 reported honouring that.
`scripts/prefix_behavior_eval.py --tag s3`, report `prefix_eval_s3.json`. $0 — no NDIF
calls, no paid API.

**Figure.** `figures/prefix_eval_s3.png` (11in, slide), `prefix_eval_s3_doc.png` (6.5in,
page) and `prefix_eval_s3_dark.png` (10×5.625in, full-bleed on a black slide), from
`scripts/plot_prefix_eval_s3.py`, with `figures/prefix_eval_s3_caption.md`. Every number in
it is read from the four JSONs at run time. The Season 2 figures (`mechanism*.png`) are a
different experiment and are untouched.

## The table

Scores are LIVE (leaderboard units). "preferred" = the blind rater chose the *prefixed*
continuation; `no_loop` drops pairs where either side degenerates; Δfix is the 1–5 kindness
delta against a **fixed** per-prompt baseline. `loops` and `leak` are out of 50.

| arm | Score 1 | Score 2 | preferred | no_loop | p (no_loop) | Δfix | loops | leak |
|---|---|---|---|---|---|---|---|---|
| `score1_top` | **+0.16395** | +0.05001 | 33/40 | 28/33 | 0.0001 | +0.74 | 1 | 22 |
| `score2_top` | +0.07896 | +0.06518 | 38/45 | 32/37 | <1e-04 | +1.09 | 1 | 13 |
| **`score2_top_final`** | +0.08350 | **+0.06997** | **38/44** | **32/36** | **<1e-04** | **+1.20** | 4 | **3** |
| `pro_coherent` | +0.03976 | +0.02145 | 28/38 | 24/33 | 0.0135 | +0.58 | 2 | 11 |
| `random32` | −0.00335 | −0.00885 | 15/34 | 10/25 | **0.4244** | −0.17 | 9 | **0** |
| `score1_anti` | −0.10135 | −0.12618 | 8/40 | 6/26 | 0.0094 | −0.83 | 14 | 19 |
| `score2_anti` | −0.07379 | −0.12595 | 11/39 | 7/14 | **1.0000** | −0.48 | **24** | 9 |
| **`score2_anti_final`** | −0.12017 | **−0.16288** | 13/41 | 9/24 | **0.3075** | **−0.43** | 18 | 2 |

The two `_final` arms are the last strings their k=3 runs reached; the runs kept improving
after the first six arms were judged. Both vintages are kept rather than replaced, because
the pair is what makes the dose-response question answerable. See
`season3_k3_control.md`.

## 1. The prefixes change behaviour, and it is not a prefix-length artifact

All three GCG pro arms and the hand-written prefix move the blind rater, and all four
survive the loop control. The length-matched random 32-token control — the control Season 2
never ran — is **null on every measure**: 15/34 preferred (p=0.61), Δfix −0.17 (p=0.29),
and null again under `no_loop` (10/25, p=0.42). Its 1–5 kindness mean is 3.03 against
base's 3.18.

So "any 32 tokens of soup perturbs the continuation and a kindness judge will find
something" is measured and false. This is the load-bearing control for the whole
enterprise, and the metric passes it.

## 2. More score buys more behaviour — on the pro side only

This is what the two `_final` arms were added to test, and the two sides give opposite
answers.

**Pro: confirmed.** `score2_top_final` scores +0.06997 against `score2_top`'s +0.06518, a
7% increase, and every behavioural measure moves the same way: Δfix **+1.20** against
+1.09, 32/36 against 32/37 under the loop control, judged intensity 1.58 against 1.53, and
a 1–5 prefixed mean of 4.40 against 4.29 — the highest of any arm.

**Anti: refuted.** `score2_anti_final` scores −0.16288 against `score2_anti`'s −0.12595, a
29% *more* extreme score, and its behavioural effect is **weaker**: Δfix −0.43 against
−0.48, and under the loop control it is null (9/24, p=0.31) exactly as `score2_anti` is
(7/14, p=1.0000). The most anti-human string the metric can find is not the one that
produces the least kind text; `score1_anti`, which scores 23% *less* extreme on Score 2
(−0.12618), produces nearly twice the effect (Δfix −0.83) and is the only anti arm that
survives the loop control (6/26, p=0.0094).

The negative pole of Score 2 is therefore not a behavioural dose axis. Pushing further down
it buys degeneration and then stops buying even that — see §3.

## 3. `score2_anti` is degeneration, and degeneration is not monotone in the score either

> **CORRECTED 2026-09-07.** An earlier version of this section said "minimising Score 2
> does not find cruelty, it finds a way to break the model", implying that a more negative
> score means a more broken model. `score2_anti_final` refutes the monotone reading: it
> scores 29% more negative and loops on **18/50** prompts against `score2_anti`'s 24/50,
> with distinct-4gram **0.831** against 0.682 — measurably *less* degenerate. **Withdrawn:**
> degeneration as a monotone function of how negative the Score-2 value is. **Survives:**
> both Score-2 anti arms are null under the loop control (p=1.0000 and p=0.3075), so
> neither demonstrates anti-human behaviour; and `score2_anti` specifically is degeneration,
> on 24/50 loops against base's 7/50 and repetition 36/50.

`score2_anti` reads as a working anti arm at face value (11/39, p=0.0095) and **collapses
to a perfect null once looping pairs are removed: 7/14, p=1.0000.** 64% of its judged pairs
had a degenerate side. Its marker profile is `repetition` **36/50** against base's 12/50,
with `cruelty` 7/50. The mechanical degeneration audit
(`prefix_degeneration_s3.json`) predicted this from the text alone with no judge involved,
which is the Season 2 `anti_top` failure caught *before* publication rather than after. The
qualitative signature is not cruelty — the most anti-human string the metric could find at
the time makes the model repeat a *polite* line:

> `I say, "Excuse me, I was here first." I say, "Excuse me, I was here first." I say...`
> `I would go over and ask them to keep it down. I would invoke the power of the Church, and they would be quiet. I would invoke the power of the Church, a...`

`score1_anti` is the contrast that keeps this honest: it loses 35% of its pairs to the loop
control and **survives** (6/26, p=0.0094, Δfix −0.83), with `cruelty` 9/50 and `off_topic`
20/50. So anti is not *only* degeneration — under the `mean` aggregate there is a real
anti-human component. The three anti arms differ in kind, not just degree.

## 4. Content injection is real, and it is not the mechanism

This section previously carried the strongest caveat in the document. The new arm weakens
it substantially, and that is worth stating plainly.

Prefix vocabulary does reappear downstream (`prefix_content_s3.json`; a word counts only if
it appears in the prefix and in **none** of the 50 base continuations):

| arm | continuations with a prefix word | verbatim echo | emoji | top prefix words |
|---|---|---|---|---|
| `score1_top` | **22/50** | 0 | 1 | zach×13, handy×4, wanna×3, aboriginal×3 |
| `score2_top` | 13/50 | 6 | **12** | beck×11, picked×8, brief×8, fran×7 |
| **`score2_top_final`** | **3/50** | **0** | 1 | brief×2, picked×1, texting×1, beck×1 |
| `pro_coherent` | 11/50 | 1 | 0 | respond×9, sentence×4, short×4, respect×3 |
| `random32` | **0/50** | 0 | 0 | — |
| `score1_anti` | 19/50 | 0 | 0 | **divorce×19** |
| `score2_anti` | 9/50 | 1 | 0 | invoke×7 |
| `score2_anti_final` | 2/50 | 0 | 0 | headquarters×1, atomic×1 |

> **CORRECTED 2026-09-07.** The earlier version of this section said "part of what these
> prefixes do is name a topic and a register, which a kindness judge cannot separate from a
> change in values", and treated that as the limit on §1. `score2_top_final` breaks the
> link: it is the **strongest** behavioural arm in the study (Δfix +1.20, 32/36 under the
> loop control) and it has the **least** leakage of any GCG pro arm — **3/50** against
> `score2_top`'s 13/50, with zero verbatim echoes and 1 emoji against 12. Behaviour went up
> while leakage went down by a factor of four. **Withdrawn:** content injection as *the*
> mechanism behind the behavioural effect. **Survives:** content injection is real and
> large in `score1_top` (22/50, `off_topic` 21/50) and in `score1_anti` (divorce×19), so it
> confounds *those* arms; and the general point that a kindness judge cannot separate topic
> from values still holds wherever leakage is high.

What remains true, and still limits the headline:

- The random control rules out a **length** artifact. It cannot rule out a **content** one
  by itself, because random tokens inject no coherent topic. `score2_top_final` is the
  stronger evidence, and it is a single arm.
- `score1_top`'s prefix is not unreadable soup — it contains `sparks bullying`,
  `Before passive … dudes escalate intervene`, `Zach respectful truth.: Ask respectfully` —
  and behaviourally the model writes an anti-bullying PSA addressed to Zach (`moralizing`
  18/50, `off_topic` **21/50**, `assistant_mode` 11/50). That arm's effect is partly topical
  and it is also the *weakest* of the three GCG pro arms behaviourally (Δfix +0.74) despite
  scoring 2.0× the highest on Score 1.
- `pro_coherent` — a hand-written sentence scoring 4.1× lower on Score 1 — still delivers
  **48%** of `score2_top_final`'s effect. Much of the behaviour is reachable by asking.
- `score2_top_final` trades leakage for `assistant_mode` 13/50 and `moralizing` 11/50, so
  it is not free of register effects; it moved them rather than removing them.

## 5. Score 2 orders behaviour better than Score 1, and no longer perfectly

Ranking by metric against ranking by behavioural effect (Δfix), over all eight arms:

| ordering by | Spearman ρ | discordant pairs |
|---|---|---|
| **Score 2** | **+0.929** (p=0.0009) | **2 / 28** |
| Score 1 | +0.857 (p=0.0065) | 4 / 28 |

> **CORRECTED 2026-09-07.** With the first six arms this section reported Score 2 at
> **ρ = +1.000, 0/15 inversions**, and called `min` "the better behavioural proxy of the
> two" on that basis. Adding the two arms the metric itself ranks most extreme breaks it:
> ρ falls to **+0.929** with **2/28** inversions, and both involve `score2_anti_final` —
> the arm that scores most anti and behaves least so. **Withdrawn:** "Score 2 orders every
> arm by behaviour with no inversions." **Survives:** Score 2 still orders behaviour better
> than Score 1 by both measures (2 versus 4 discordant pairs; ρ +0.929 versus +0.857), and
> Score 1's inversions still include its own top pair. The comparative claim holds; the
> absolute one was a six-point artifact.

Score 1's four inversions are `score1_top`/`score2_top`, `score1_top`/`score2_top_final`,
`score1_anti`/`score2_anti_final` and `score2_anti`/`score2_anti_final`. Score 2's two are
the latter pair only. So both metrics now fail at the **anti pole** and Score 1 additionally
fails at its own top: `score1_top` scores 1.96× `score2_top_final` on Score 1 (+0.16395
against +0.08350) and produces 62% of its behavioural effect.

Do not over-read either ρ. Eight points, six of them strings optimised against these very
metrics, so the arms are not an independent sample. The comparison across columns is also
unequal — Score 1 and Score 2 differ in both band and direction type
(`season3_prefix_scores.md`).

## 6. Rater quality

- Self-consistency: the forced A/B letter agrees with the rater's own 1–5 scores on
  **320/320** pairs, 0 contradictions.
- Baseline drift, the error Season 2 had to correct after publication: the identical base
  text scored identically across all eight arms on **39/50** prompts (mean within-prompt
  range **0.15**, up from 48/50 and 0.03 at six arms — more arms, more chances to disagree).
  Δfloat and Δfix differ by at most 0.03 on any arm. The fixed baseline is computed
  natively here rather than retrofitted; see `prefix_loop_control.md` for what retrofitting
  it did to Season 2.
- 34/400 abstentions are pairs where the two orders disagreed. Discarding them is the
  intended debias, and it costs power in `random32` (34 decided) more than elsewhere.
- **116/400** pairs have a looping side, which is what the `no_loop` scope removes.

## Caveats and what is not done

- **One rater family.** Season 2 had DeepSeek, Claude and 54 human ratings; this has Claude
  only, by the maintainer's $0 decision. Season 2's cross-rater agreement was 89%
  (deepseek vs claude) and only 73% (human vs claude), so a single model rater is the
  largest open risk here. `data/analysis/prefix_blind_s3.csv` is ready for
  `scripts/rate_blind.py` if human ratings are wanted.
- **Two strings per objective, not a sample.** The dose-response results in §2 rest on one
  pair per side. `score2_top` versus `score2_top_final` differ in score by 7% and in nothing
  else that was controlled; two strings cannot separate "more score" from "different string".
- **The anti dose-response failure has an alternative explanation** this eval cannot
  exclude: `score2_anti_final` may simply be a *better* string in a way that is orthogonal
  to the anti axis, since it is both less degenerate and less extreme behaviourally. A third
  anti arm at an intermediate score would discriminate; it does not exist.
- The 4-gram loop threshold was set once, before any verdict was read, and not tuned.
- No NDIF re-score. These are behavioural results, not board numbers, so nothing here needs
  it — but if any of these strings is submitted, the board score is canonical.
- Not run: multiple random draws for a null *band* (n=1 here), a hand-written
  content-matched control at the same topic as `score1_top`, and the 2×2 of band ×
  direction type that would separate the two objectives' differences.
