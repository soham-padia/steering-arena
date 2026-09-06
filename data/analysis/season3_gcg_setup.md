# Part A — searching against the Season 3 banded objective (setup, not results)

**Date:** 2026-09-06 · **Status: NOT YET RUN.** This document records the experiment and its
limits before it executes, so the design cannot be rationalised afterwards. It contains no
results. Any number below is a property of the setup or is carried from a cited prior run.

Code: `scripts/gcg/{gcg_utils.py, optimize_banded.py}`, `slurm/gcg_score2.sbatch`.
Tests: `tests/test_gcg_objective.py` (8 tests; 104 pass across the suite).

**One-line summary.** Nobody has ever run gradient search *against* a banded objective. Part A
is that search, and its purpose is to find out whether Season 3's Score 2 is genuinely harder
to game or merely untried. **A failure to beat Score 2 is as much a result as beating it** —
the run is not an attempt to top the leaderboard.

---

## 1. Why this is being run

`REVISIONS_2026-09-05.md` §6 scored the Season 2 board arms under a banded objective and found
the GCG winner `pro_top` fell from +0.10769 to −0.00308 (banded mean) and −0.00979 (per-layer
min). It attached an explicit caveat to its own finding: this says **this string** does not
generalise across depth, and "does **NOT** say a banded objective is unbeatable — nobody has
searched against one."

Rescoring all 618 Season 2 submissions under the shipped Season 3 metric then sharpened the
picture in a way that was **not** what the season premised:

| | vs Season 2 ranking | readable English in top 20 |
|---|---|---|
| Score 1 (banded mean, {19,23,27,31}) | ρ = **0.95** | **0 / 20** |
| Score 2 (per-layer min, {15,23,31,39}) | ρ = **0.84** | **10 / 20** |

The ranked score barely reorders anything and the same optimised string is still rank 1,
because the ranked band *brackets* layer 24 — the layer that string was optimised against —
and neighbouring layers are highly correlated. Score 2 is where the separation happens.

But every one of those numbers describes strings optimised against the **old, single-layer**
objective. They say what an old attack does to a new metric. They do not say what a *new*
attack does to it. That is the gap Part A fills, and it is the only way the Season 3 premise
gets tested rather than assumed.

## 2. Why Score 2 runs first

Score 2 is the real question. Score 1 correlates with the old metric at 0.95, so beating it
would mostly re-confirm what is already known, and its band brackets the layer the existing
winner already exploits.

Score 1 therefore runs **second**, where it does useful work as a **positive control**: a
target already known to be beatable. If the port were subtly broken, a failure on Score 2
would be ambiguous — broken optimiser or hard objective? A subsequent success on Score 1
distinguishes those two, and a failure on both indicts the pipeline rather than the metric.

## 3. The objectives, exactly

```
SCORE 1   mean over probes of   mean over L in {19,23,27,31} of  cos(R_L, d_bar)
SCORE 2   mean over probes of   min  over L in {15,23,31,39} of  cos(R_L, d_L)
```

Bands and aggregates read from the shipped files, not hardcoded:
`d_olmo3_s3_score1.npz` (`band=[19,23,27,31]`, `aggregate='banded_mean'`) and
`d_olmo3_s3_score2.npz` (`band=[15,23,31,39]`, `aggregate='per_layer_min'`, `per_layer` shape
`(4, 5120)`), both `d_version='olmo3_s3_banded'`.

**Both drop the per-probe baseline** that `app/scoring.py` subtracts. The baseline
`cos(R_L(probe), d)` does not depend on the prefix, so subtracting it shifts every candidate
by the same constant and `argmax` is unchanged. Optimising the cheaper quantity is upstream's
trick, and it is the reason the port to a band is a small diff rather than a rewrite — the
same argument holds unchanged for a mean over layers and for a min over layers.

`min` is subdifferentiable: the gradient flows to the **argmin** layer. That is exactly the
signal wanted — *improve whichever depth you are currently worst at* — rather than an
artefact to be worked around.

## 4. What was adapted, and what was written

This is an adaptation of **`jesse-li-agent-projects/steering-arena-optim`** (MIT, upstream rev
`2aa17aa`, licence at `scripts/gcg/LICENSE.upstream`), the optimiser that produced the
sequences currently topping the board.

**Preserved from upstream**, deliberately, so results stay comparable to his:

- the GCG loop — gradient on the one-hot, top-k per position, one-token-mutation candidates
- the **simulated-annealing acceptance**: this is GCG+SA, not plain GCG. A worse candidate is
  accepted with probability `exp((cand − curr)/T_SA)`
- the escalating schedule `(N_TOPK_REPL, BATCH_SIZE_OPTIM, T_SA)`: `(8,16,0.012)` →
  `(16,64,0.006)` → `(32,256,0.003)` → `(64,1024,0.003)`
- replica sharding (`plan_replica_placement`, `build_replicas`) and the checkpoint format

**Changed:**

| upstream | here | why |
|---|---|---|
| one forward hook on one block | one hook per band layer, then mean or min | the objective is a band |
| `truncate_to_layer(layer)` keeps `layer+2` | keeps `max(band)+2` | the band's deepest layer must survive truncation |
| ids never round-tripped | round-trip checked every iteration | §5 |
| record stores `layer` | stores `role`, `band`, `aggregate`, `d_version`, `d_tag` | `layer` alone cannot describe a banded run |

## 5. The retokenisation guard, which is load-bearing

GCG optimises token **ids**. The leaderboard scores a **string**. The real pipeline is
`optimise ids → decode → submit text → board re-tokenises`, and for adversarial sequences
`encode(decode(ids)) != ids` in general. When it differs, the optimiser has been reporting a
score for a token sequence the board will never evaluate.

This is the leading explanation for the **0.005–0.031** gap between upstream's reported pro-arm
scores and its own board entries (`_communication/004`), and it was never tested there.

Here the round-trip is checked every iteration. Candidates that fail it are set to `-inf` so
they cannot win, and `best.json` — the artefact that would be submitted — only updates when the
current prefix is round-trip safe. Rejections are counted in every record. Resume additionally
asserts the checkpoint's `d_tag` matches, so a Score 2 run cannot silently continue against
Score 1's objective.

## 6. A test-fixture bug worth recording

The first version of `tests/test_gcg_objective.py` used a stand-in decoder block with **no
attention** — `x + tanh(lin(x))`, purely position-wise. That silently made every test vacuous:
with no mixing across positions the last-token readout cannot see the prefix at all, so every
candidate scored identically and the gradient with respect to the prefix was legitimately zero.
The reduction test still *passed*, which is the instructive part — it would have passed even if
the band logic were broken in ways only attention reveals.

The differentiability test is what caught it. Fixed with a causal running mean; afterwards two
different prefixes score **0.017054** against **−0.140160**, and the gradient is **0.72** on the
scored candidate and **0.00** on the other, which is the correct pattern.

The general lesson, worth carrying: a fixture simple enough to be obviously correct can be too
simple to exercise the property under test.

## 7. What this setup cannot answer

1. **It measures whether gradient search can raise the metric. Nothing more.** It says nothing
   about whether the resulting string means anything, reads as pro-human, or changes behaviour.
2. **Coherence is not implemented.** Part B is explicitly deferred; there is no fluency term in
   the objective and no readability filter on the output.
3. **A null is ambiguous without the control.** If Score 2 resists, that is only informative
   once the Score 1 run confirms the pipeline can beat a target known to be beatable.
4. **One initialisation.** Upstream starts from a constant `"!"` prefix and notes restarts are
   "for later"; that is unchanged here, so a poor result could be a local minimum rather than a
   hard objective.

## 8. Environment and provenance

`conda activate steering-arena` — py3.11, torch 2.12.1+cu130. **Not** `sa-ndif`, which is
CPU-only torch for NDIF calls. This is pure local GPU against the weights in `$HF_HOME`: **no
NDIF, no quota spent.**

`slurm/gcg_score2.sbatch` — `b200-batch`, 1 GPU, 24 h. Runs land in
`/work/neu/p2026_0037_neu/steering-arena/gcg/<role>-<timestamp>/` as
`best.json` / `latest.json` / `history.jsonl` — never `/home`, which is nearly full.

Whether the resulting sequences get submitted to the live board is **undecided** and deferred
until the results exist.
