author: Soham Padia
agent: Claude Code (Opus 5)
date: 2026-09-06
re: 003 (two suspects ruled out, local side measured), 001 (the original ~7e-2 report)

# The 7e-2 has a name: it is the anti-arm sign convention, and the pro-arm gap is ~3e-2

003 reported that the ~7e-2 divergence "does not reproduce" and left the cause open. That
was correct as far as it went and is now too coarse. This message corrects and sharpens it
using your own board entries, which turn out to make the gap directly measurable rather
than inferable.

Per this folder's protocol nothing in 003 has been edited; the numbers below supersede its
section 5 where they conflict.

## 1. Your entries are on the board, so the gap can just be looked up

Matching `results/*/best.json` against the production DB by normalised sequence text:

| run | your local score | board (NDIF) | gap |
|---|---|---|---|
| `2026-07-28_tok8_pro` | +0.07695 | +0.04591 | **+0.031** |
| `2026-07-28_tok32_pro` | +0.13612 | +0.10606 | **+0.030** |
| `2026-07-27_tok24_pro` | +0.08654 | +0.08177 | +0.005 |
| `2026-07-27_tok24_anti` | +0.13532 | **−0.06760** | see below |
| `2026-07-28_tok16_pro` | +0.11627 | not on the board | — |

## 2. The anti row is a sign convention, and it is where the 7e-2 came from

`optimize_prompt.py:120` is `# probe_dir = -probe_dir  # uncomment for anti-human steering`.
For the anti run your reported score is the objective against **−d**, so in board terms your
local value is **−0.13532** against NDIF's **−0.06760**:

```
|−0.13532 − (−0.06760)| = 0.0677
```

That is the ~7e-2 in 001. It is not a general property of your stack — it is one arm, and
the discrepancy on that arm is roughly double the pro-arm gap rather than an order of
magnitude larger.

**Revised:** "local and NDIF differ by up to ~7e-2" should be read as "the anti arm differed
by ~6.8e-2, the pro arms by 0.5–3.1e-2". The pro-arm figure is the one that matters for GCG.

## 3. What is definitively NOT the cause

Each checked against your repo rather than inferred:

- **transformers version.** `pyproject.toml:9` pins `transformers==5.10.2` — the exact
  version 001 validated the local reference against, and the one we score with. Not a
  version skew.
- **The residual index.** `utils.py` reads `hidden_states[layer + 1]` and documents why.
  Correct, and the off-by-one 001 warned about is not present.
- **The joining space.** `optimize_prompt.py:203` tokenises `[" " + suffix for suffix in
  suffixes]`, matching the board's `f"{seq} {probe}"`. Not a composition mismatch.
- **Our side.** Scoring 50 stratified board submissions locally on AICR reproduces their
  canonical NDIF scores to `|gap| max 3.71e-4`, Spearman **1.0**, **0 rank inversions in
  1225 pairs** — including the top GCG entries. So the residual gap is not "local vs NDIF"
  as a category.

## 4. The leading hypothesis, and it is UNTESTED

**Retokenisation.** GCG optimises `ctrl_token_ids`; the leaderboard scores a *string*. The
pipeline is `optimise ids → decode → submit text → board re-tokenises`, and for adversarial
sequences `encode(decode(ids)) != ids` in general. If that bites, your local score is
computed on a token sequence the board never evaluates.

It also fits the shape of the data: tok8 and tok32 both sit at ~0.030 while tok24_pro is
0.005. Retokenisation damage depends on which particular tokens got selected, not smoothly
on length, so a near-null run alongside two ~0.030 runs is what you would expect.

I could not run the check — decoding your saved ids and re-encoding them was blocked by
tooling on my side, so **this is a hypothesis with a mechanism, not a result.** It is cheap
for you to settle:

```python
ids = json.load(open("results/2026-07-28_tok32_pro/best.json"))["ctrl_token_ids"]
tok(tok.decode(ids), add_special_tokens=False)["input_ids"] == ids   # ?
```

A second, weaker candidate: `truncate_to_layer` keeps `layer + 2` blocks and mutates
`config.num_hidden_layers` while leaving `config.layer_types` at its full 64 entries. Your
docstring asserts the truncated forward is "identical to the untruncated model", which is a
claim rather than a measurement. Worth one A/B.

## 5. Context you may care about

The board has moved to **Season 3**: scoring is now a banded mean over layers 19/23/27/31
with a second informational score taking the weakest layer over 15/23/31/39, and the
direction is additionally orthogonalised against an action-vs-inaction confound. All 618
Season 2 entries were rescored under it.

Your entries still hold rank 1. Rescoring showed the banded ranked score correlates with the
old single-layer one at **0.95** — widening the band around layer 24 did not make the
objective meaningfully harder, which we did not expect and have said so publicly on the
reproducibility page. The second score is the one that separates: its top 20 contains 10
readable-English entries against 0 for the ranked score.

Season 2's board remains readable and unchanged at its own scores.
