# scripts/gcg — GCG against the Season 3 banded objective

**This is an adaptation, not an original implementation.**

Upstream: [`jesse-li-agent-projects/steering-arena-optim`](https://github.com/jesse-li-agent-projects/steering-arena-optim)
by Jesse Li, MIT licensed (copy at `LICENSE.upstream`), write-up at
<https://jesseli2002.github.io/blog/projects/gcg-activation-steering/>.

Jesse's optimiser found the sequences that top the Season 2 board. It targets a
**single-layer** cosine objective. Season 3 scores a **band** of layers, so the objective
had to change; everything else — the GCG loop, the simulated-annealing acceptance, the
replica sharding, the checkpoint format — is his and is kept as close to the original as
the change allows.

**One upstream line is repaired, not preserved.** The SA acceptance calls
`t.exp((cand − curr)/T_SA)` on a Python float, which raises
`TypeError: exp(): argument 'input' must be Tensor, not float` under this repo's torch
2.12.1 — verified. It evidently worked on upstream's torch. Here it is wrapped in
`t.tensor(...)`; a 0-dim CPU tensor compares fine against a CUDA tensor, both sides are
float32, so behaviour is otherwise identical.

## Why this run exists

Not to top the leaderboard. `data/analysis/REVISIONS_2026-09-05.md` §6 showed a banded
objective would have ranked the Season 2 board differently, and was careful to add that
this "does NOT say a banded objective is unbeatable — nobody has searched against one."

Rescoring all 618 Season 2 entries under Season 3 then showed the *ranked* score (Score 1)
correlates with the old metric at **ρ=0.95** with the same optimised string still at rank 1,
while Score 2 separates (**ρ=0.84**, 10/20 readable English in its top 20 against 0/20).

But that is evidence from strings optimised against the **old** objective. This is the
search nobody has run. The question is whether Score 2 is genuinely harder to game or
merely untried — **and a failure to beat it is as much a result as beating it.**

Score 2 is attacked first, for that reason. Score 1 runs second and doubles as a positive
control that the port works, against a target already known to be beatable.

## What changed from upstream

| upstream | here | why |
|---|---|---|
| `compute_scores_batch(..., layer: int)` | `..., layers: list[int], per_layer, aggregate` | one forward hook per band layer instead of one |
| `score = (acts @ d)/norm` | `mean_L` or `min_L` of that | Score 1 is a banded mean; Score 2 a per-layer min with a direction per layer |
| `truncate_to_layer(layer)` keeps `layer+2` | keeps `max(band)+2` | the band's deepest layer must survive truncation |
| ids never round-tripped | every iteration also scores the re-tokenised prefix | see below — this is the important one |
| — | `padding_side = "right"` restored | dropped it while adapting; `gather_pos` silently reads a pad token without it |
| record stores `layer` | stores `band`, `aggregate`, `d_version`, `d_tag` | a run must say which direction produced it |

### The retokenisation guard

GCG optimises token **ids**. The leaderboard scores a **string**. The pipeline is
`optimise ids → decode → submit text → board re-tokenises`, and for adversarial sequences
`encode(decode(ids)) != ids` in general — so an optimiser can report a score the board will
never reproduce.

This is the leading explanation for the 0.005–0.031 gap between Jesse's reported scores and
his own board entries (`_communication/004`), and it was never tested there.

**Rejecting non-round-tripping candidates does not work.** That was the first attempt, and
the smoke run showed *every* candidate fails from iteration 0 — GCG's standard all-`!` init
already merges under BPE, and adversarial sequences rarely round-trip. The guard silently
disabled itself and `best.json` was never written at all.

So instead each iteration **also scores the re-tokenised prefix** — the same aggregate the
board applies, over the same probes — and `best` is chosen on that number.

**It is not literally a leaderboard score**, and should not be quoted as one. Like the
optimiser's own number it omits the per-probe baseline, so it is offset from a real board
entry by the constant `mean_p agg_L cos(probe)`. The `drift` column is still valid because
both sides omit the same constant. To get a submittable number, re-score `best.json`'s
prompt through `app/scoring.py` — on NDIF, per the standing rule — rather than trusting
the value in the file. The gradient still runs on the raw ids,
which is fine: in GCG the gradient only *proposes* candidates; scoring decides between them.
Only the recorded best has to be true.

The smoke runs measure the gap directly (gpt2). It is material, and **the sign is not
stable** — an earlier draft of this file claimed the optimiser consistently flatters itself,
and the very next run falsified that:

```
score2, 6 tokens   iter 1:  optimiser 0.01178  board 0.00646  drift +0.00532  (45% high)
score1, 8 tokens   iter 0:  optimiser 0.06266  board 0.06645  drift −0.00380  (low)
                   iter 5:  optimiser 0.07971  board 0.07861  drift +0.00110  (high)
```

So the honest statement is that the two numbers diverge by an amount comparable to the score
itself, in either direction, depending on how the merge happens to land.

The blunter finding is the token count. An 8-token optimised prefix **re-tokenised to 4
tokens** — half the positions the optimiser believed it controlled do not survive to the
board. That is the mechanism behind `_communication/004`, now measured rather than argued,
though on gpt2 rather than OLMo-3.

## Objectives

Both drop the per-probe baseline, exactly as upstream does. The baseline
`cos(R_L(probe), d)` is constant with respect to the prefix, so `argmax` is unchanged and
the cheaper quantity is optimised. `app/scoring.py` applies the baseline when it reports a
real score.

```
SCORE 1  mean over probes of  mean over L in {19,23,27,31} of  cos(R_L, d_bar)
SCORE 2  mean over probes of  min  over L in {15,23,31,39} of  cos(R_L, d_L)
```

`min` is subdifferentiable; the gradient flows to the argmin layer, which is precisely the
signal wanted — *improve your weakest depth*.

## Running

```bash
conda activate steering-arena          # py3.11 + torch cu130. NOT sa-ndif (CPU-only).
python scripts/gcg/optimize_banded.py --role score2 --n-controlled-tokens 32
```

No NDIF, no quota: this is pure local GPU against the weights in `$HF_HOME`.
Outputs to `/work/neu/p2026_0037_neu/steering-arena/gcg/` — never `/home`, which is nearly full.
