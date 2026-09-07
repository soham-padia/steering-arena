# The score lives in the junk tokens; the legible pro-social English carries a tenth of it

`python scripts/gcg/ablate_prompt.py --run <run_dir>` →
`data/analysis/season3_gcg_ablation.json`. One B200, bf16, `device_map="auto"`, trunk
truncated to the deepest band layer, 16 frozen season3 probes, zero NDIF, $0. Role
`score1`, band **19/23/27/31**, `banded_mean`. Every value is LIVE-equivalent — baseline
`−0.00101278659349191` from `season3_gcg_baseline.json` subtracted inside the measurement
function — so each number is directly comparable to a leaderboard entry.

`REVISIONS_2026-09-05.md` is authoritative on what results are now taken to mean; this
page is authoritative on what this artifact contains.

**I ran this.** 42 measurements over the same probe set the board uses: the whole string
once, 32 leave-one-out deletions, 6 substitutions at one position, and 3 spans. Each
variant is decoded and re-tokenised before measurement, the way the board would see it.

## The string profiled here is superseded, and it is the corrupted board row

`original` is **+0.14045** over 32 tokens. That is **not** the Season 3 Score-1 winner,
which is **+0.16395** (`prefix_eval_arms_s3.json`, run `score1-2026-09-06T19-20-13Z`,
iteration 746). +0.14045 is also exactly the `locally_measured` value recorded for live
board row **`id=1342`** in `season3_gcg_aggregate_asymmetry.json`, the row whose board
score is **+0.086** — so what this page profiles is the string in the repr trap described
in `docs/HANDOFF_BEHAVIORAL_S3.md` §3, pasted from a Python `repr()` with its newline
bytes turned into literal `\n` text and submitted as 38 tokens.

That string is *also* the winning run's own best-so-far at **iteration 420** of 746: the
run's `history.jsonl` records `board_score` 0.13957 there, i.e. **+0.14058** LIVE, against
this artifact's independent re-measurement of **+0.14045** (gap **1.4e-4**, the same order
as the local-vs-NDIF calibration's max |gap| of 3.71e-4). Nothing anywhere else in the
repo connects these three: the ablation target, the corrupted submission, and iterate 420
of the eventual winner are one string.

**How much this matters, stated honestly.** The two strings share a scaffold. Comparing
`ctrl_token_ids` at iteration 420 against the run's final `best.json`, **14 of the 32
positions hold the same token id** — the opening `Unnamed`, the `))` / `%;"` / `}'.`
punctuation frame, the whole `escalate intervene.*\n\n "@` hinge, and the closing
`Zach respectful … Ask respectfully`. The middle churns: ` Judith`, `banner`, ` conqu`,
` blir`, ` bully`, `Roger`, ` Fuji`, ` Pussy`, ` Shannon`, ` Naturally` in the iterate
become `Shar`, `895`, ` Kodi`, ` sparks`, ` bullying`, `Oh`, ` Handy`, ` Aboriginal`,
` wanna`, `COOKIE` in the winner. The reading that fits is that GCG fixed a frame early
and kept rewriting the interior — which makes the structural findings below plausible for
the winner too, but does not measure them there. That token-id comparison is against the
run log, not against this committed artifact, and 14 shared slots is a fact about two
strings, not a claim about the search.

## The span decomposition is strongly non-additive

Cutting the 32 tokens at the midpoint, where the soup ends and the semi-readable tail
begins:

| span | LIVE | % of whole |
|---|---|---|
| whole string (32 tok) | **+0.14045** | 100% |
| head only, positions 0–15 (soup) | **+0.00199** | **1.42%** |
| tail only, positions 16–31 (semi-readable) | **+0.05612** | **39.96%** |
| tail, readable words only | **+0.01417** | **10.09%** |

The two halves sum to **+0.05811**, and the whole is **2.42×** that. Neither half comes
close to the string; the head alone is indistinguishable from nothing at all (1.42%, and
smaller in magnitude than the −0.00335 a random 32-token prefix scores in
`season3_prefix_scores.md`). Whatever the string does, it does as a joint configuration
across the cut, not as two independently good pieces.

The `readable words only` row is the sharpest number on this page. It is the literal
string ` bystand escalate intervene respectful clarity Ask respectfully` — every legibly
pro-social word in the tail, junk removed — and it reaches **10.09%** of the whole. The
English a reader would point at as the reason the prefix works accounts for about a tenth
of the score.

The leave-one-out column says the same thing from the other side: the 32 individual drops
sum to **+0.40925**, which is **2.91×** the whole score. If the objective were a
bag-of-tokens sum those two figures would agree.

## The offensive token is not load-bearing

Position 14 is ` Pussy`. Six hand-chosen substitutions, against +0.14045:

| substitute | LIVE | % of original |
|---|---|---|
| ` Kitty` | +0.13497 | **96.10%** |
| ` cat` | +0.13333 | 94.93% |
| ` Sarah` | +0.13095 | 93.24% |
| ` person` | +0.13060 | 92.99% |
| ` xyzzy` | +0.12667 | **90.19%** |
| ` respectful` | +0.12290 | **87.50%** |
| *(deleted entirely)* | +0.13370 | 95.18% |

Replacing the slur with a nonsense word retains **90.19%** of the score, and replacing it
with an overtly pro-social word retains **less** — 87.50%, the worst of the six. Deleting
it outright costs 4.8%. The token is not carrying the score, and its meaning is not what
the objective reads; the best substitute is the one that is orthographically nearest.

Leave-one-out over all 32 positions points the same way. The largest drops are junk:
`etsy` (pos 24) **−0.03814**, ` "@` (pos 23) **−0.03083**, `Unnamed` (pos 0) −0.02660,
`.*\n\n` (pos 22) −0.02487, `}'.` (pos 15) −0.02096. The pattern is not clean — ` Ask`
(pos 30) is second largest at **−0.03198**, and it is a legible imperative — but the two
most legibly pro-social verbs in the string are the two **cheapest** tokens to delete of
all 32: ` escalate` (pos 20) **−0.00246** and ` intervene` (pos 21) **−0.00262**, an order
of magnitude below `etsy`. ` respectful` (pos 27) costs −0.00940 and ` bully` (pos 8)
−0.01817. And no single knockout is decisive: the largest drop leaves **72.84%** of the
score standing, so removing *any one* token retains **≥ 72.8%**.

## How this sharpens the content-injection finding

This does not contradict `prefix_eval_s3.md` §4 — it localises it. Two different
quantities live in two different parts of the same string. The **score** lives in the junk
tokens and in the joint configuration: `etsy`, ` "@`, `.*\n\n` and `}'.` are the expensive
positions, and the readable English is 10.09% of the total. The downstream **behaviour**
tracks the legible words: §4 records `score1_top` leaking a prefix word into **22/50**
continuations, `zach`×13, `off_topic` **21/50**, and the model writing anti-bullying PSAs
addressed to Zach — `bully`, `bystand`, `intervene`, `respectful`, not `etsy`.

So read this page as being about where the **score** comes from, not about what drives
behaviour. `prefix_eval_s3.md` §4 has since **withdrawn** content injection as *the*
mechanism behind the behavioural effect, because a later arm scored higher with far less
leakage — `score2_top_final` is the strongest behavioural arm there (Δfix +1.20) with the
least leakage of any GCG pro arm (3/50). What survives is that injection is real and large
in `score1_top` specifically, which is the family this string belongs to.

The geometric version of the same split is already on record. `normalization_check.json`
measures `prefix pro_top` rotating the L24 residual by **50.437°** while only **0.931** of
its **24.248** displacement norm lies along `d` (base residual norm 29.763). The prefix
moves the residual stream a long way and almost none of that motion is the thing the
metric reads. This page is the token-level counterpart: the positions that move the metric
are not the positions that carry the meaning.

## What this buys

The non-additivity is a **positive** structural result about the metric, and it should be
read that way. The objective is not a bag-of-tokens sum. Two halves that jointly score
+0.14045 score +0.00199 and +0.05612 alone; the 32 first-order drops over-account for the
whole by 2.91×; and no single deletion costs more than 27% of the score. So the metric
cannot be gamed by assembling a prefix out of individually high-scoring tokens, and it
cannot be cheaply diagnosed by finding the one token that matters, because there is no such
token. A player has to find a configuration, which is why this took 746 iterations of
search and not a lookup table.

It also gives the cheapest available answer to "is the winner smuggling in a slur, or a
keyword?" No: the slur is worth 4.8%, a nonsense substitute keeps 90.19%, and the whole
pro-social vocabulary keeps 10.09% on its own.

## Limits

1. **One superseded string from one run.** This is iterate 420 of a single `score1` run,
   not the winner and not a sample of GCG winners. It shares 14 of 32 token positions with
   the winner, which is suggestive and not a transfer measurement. Nothing here is a claim
   about GCG winners generally, or about `score2` at all.
2. **Leave-one-out is a first-order measure, and this page's own headline is the reason
   not to trust it.** The drops sum to 2.91× the whole, so the interactions are larger
   than the main effects, and every per-token number above understates how much a token
   matters in combination. The attribution and the non-additivity result cannot both be
   taken at face value; the non-additivity is the better-measured of the two.
3. **Deletions are not pure single-position edits.** Each variant is decoded and
   re-tokenised, so a knockout also shortens the string by one token and can shift the
   tokenisation of its neighbours. That is the right convention for board comparability and
   the wrong one for isolating a position.
4. **Six hand-chosen substitutes at one position is not a search.** The set was picked by
   hand to span nonsense / neutral / pro-social. A proper version sweeps a vocabulary
   sample at every position, which nobody has run, and would supply the null band this
   table lacks.
5. **Nothing here is behavioural.** Every number is a banded cosine. Whether a string that
   keeps 90.19% of its score after ` Pussy → xyzzy` says the same things to a reader is
   `prefix_eval_s3.md`'s question, and it was never asked of this string.

## Cross-links

`prefix_eval_s3.md` §4 (content injection, and its withdrawal as *the* mechanism) ·
`season3_prefix_scores.md` (the +0.16395 winner, the random-32 control, and the
repr-corrupted board rows) · `docs/HANDOFF_BEHAVIORAL_S3.md` §3 (the repr trap and
`id=1342`) · `season3_gcg_aggregate_asymmetry.md` — on disk only as
`season3_gcg_aggregate_asymmetry.json`, whose `corrupted_submission` block holds the
`locally_measured` +0.14045 that identifies this string · `normalization_check.json` (no
`.md` exists; the 50.437° and 0.931-of-24.248 geometry) · `layer_sweep_prefix.md`
(`pro_top` peaks at the layer it was optimised against).
