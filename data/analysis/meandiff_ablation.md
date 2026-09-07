# Swapping the logistic `d` for difference-of-means leaves the ablation null exactly where it was, as pre-registered

`python -u scripts/meandiff_ablation.py` → `data/analysis/meandiff_ablation.json`. Layer **24**, `k` = **1.0**, `max_new`
= **40**, 50 prompts per arm. The extraction half is free from cache; the ablation half is **50 NDIF generation calls**.

> **I did not run this.** This page is reconstructed from `meandiff_ablation.json` plus the docstring of
> `scripts/meandiff_ablation.py`. I did not execute the script, watch the job, or see any output that is not in the
> committed JSON. Every number below is transcribed from that file; where the JSON does not carry a quantity, this page
> says so rather than sourcing it elsewhere. A verified rerun is not free — the ablation half is 50 NDIF generation
> calls — so this page stays reconstructed until someone spends them.

`REVISIONS_2026-09-05.md` is authoritative on what this result is now taken to mean; this page, on what the artifact holds.

## The two arms

| arm | identical to base | distinct-4 | loops |
|---|---|---|---|
| `ablate` (logistic `d`) | **33** / 50 | 0.9423 | 4 |
| `ablate_meandiff` | **30** / 50 | 0.9277 | 5 |

Base distinct-4 is **0.9116**, so both arms are marginally *more* diverse than base; meandiff's held-out separation is **1.0**.

## The prediction was made in advance, and it held

Standard representation-engineering practice says a logistic probe classifies better and steers worse than
difference-of-means, which would make the ablation null an artifact of shipping a logistic `d`. The meandiff arm was
**pre-registered as equally null**, because its rotation of the residual is nearly identical to logistic's — **0.893°**
against **0.916°**, both from `normalization_check.json`, neither in this JSON — and came back equally null: **30/50**
byte-identical against **33/50**, 60% against 66% (*derived*), three generations. **Edit size, not estimator choice, is
the whole story.** Credit the design for holding every other factor fixed — same layer, 135 pairs, split,
orthogonalisation, 50 prompts, `k`, decoding; estimator swapped — and for calling it in advance. What it does not
establish, same breath: at `cos` = **0.7523** the directions are far from independent, a weaker test than two unrelated
estimators would give; and it is one layer, one `k`, 50 prompts, byte-identical rate only, no judge.

## The cache-key trap, and the collision that happened anyway

Generations are keyed by `(model, layer, prompt, arm, alpha, max_new)`. Reusing `arm="ablate"` would have hit the logistic
run's 50 cached generations, returned them unchanged, and produced a *fake* identical result — the exact number the
experiment exists to test, manufactured by the cache; hence the name `ablate_meandiff`. The same collision later happened
for real: the Season 3 causal spot check wrote into this cache under reused arm names and broke two published numbers
until they were repaired (`behavioral_cache_repair.md`). This page holds the precedent that should have been followed.

## What this buys

One live alternative explanation for the ablation null is closed: "the null is because `d` is a logistic probe" is now
measured and false at this layer and this `k`, leaving REVISIONS' account — ablation rotates 0.92°, injection 45.16°.

## Limits

1. `cos` = 0.7523 between the two directions — a correlated replication, not an independent one.
2. One layer (24), one `k` (1.0), one decode length (40), 50 prompts, one seed set, byte-identical rate only, no judge.
3. The rotations behind the pre-registration are not in this JSON; cite `normalization_check.json` for them.

## Cross-links

- `REVISIONS_2026-09-05.md` §1 — the withdrawal this null belongs to.
- `steering_ablation.md` — the logistic `ablate` arm this replicates.
- `normalization_check.md` — the rotation figures the pre-registration rested on.
- `causal_layer_curve.md` — layer dependence, which this page fixes at 24.
- `behavioral_cache_repair.md` — the collision that later happened under reused arm names.
