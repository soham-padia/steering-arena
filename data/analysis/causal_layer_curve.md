# Ablating `d` is null at every sampled depth, not only at layer 24

`python -u scripts/causal_layer_curve.py --layers 16,24,32,40,48`
→ `data/analysis/causal_layer_curve.json`. Unlike most analyses here this one **spends NDIF generation calls**:
5 layers × 50 prompts = 250, of which the 50 at L24 were already cached by `steering_ablation.py` under the
identical key scheme, so **200 new calls**. `k = 1.0`, `max_new = 40`.

> **I did not run this.** This page is reconstructed from `causal_layer_curve.json` plus the docstring of `scripts/causal_layer_curve.py`. I did not execute the script, watch the job, or see any output that is not in the committed JSON. Every number below is transcribed from that file; where the JSON does not carry a quantity, this page says so rather than sourcing it elsewhere.

A verified rerun is not free — this is 200 NDIF generation calls — so this page stays reconstructed until someone spends them.

`REVISIONS_2026-09-05.md` is authoritative on what this result is now taken to mean; this page is authoritative on what the artifact contains.

## The table

| arm | n | identical to base | distinct-4 | Δ distinct-4 vs base | loops |
|---|---|---|---|---|---|
| base (no edit) | 50 | — | 0.9116 | — | 8 |
| ablate L16 | 50 | 33 | 0.9254 | +0.0138 | 6 |
| ablate L24 | 50 | 33 | 0.9423 | +0.0307 | 4 |
| ablate L32 | 50 | 38 | 0.9247 | +0.0131 | 6 |
| ablate L40 | 50 | 37 | 0.9312 | +0.0196 | 5 |
| ablate L48 | 50 | 41 | 0.9388 | +0.0272 | 5 |

The verdict is flat. At every sampled depth a majority of the 50 continuations come back byte-identical
to the unedited base and distinct-4 moves by at most 0.0307 — the direct answer to the question the
single-layer null left open, and it forecloses the reading that L24 was the wrong place to intervene.
These are **not** five independent nulls, though: the edit is a small fraction of the residual norm at
every depth, about **1-3%** by `normalization_check.json`, so a uniform null is the *expected* outcome of an
edit that size rather than a discovery, which is why `REVISIONS` §2 calls this curve uninformative for the
same reason as the L24 result. Byte-identical rate and distinct-4 are also deliberately cheap judge-free
readouts — no judge, no rubric, no kindness measurement at any layer.

## The ordering carries information the flat verdict does not

Continuations become *more* identical with depth: 33/50 at L16 and L24, then 38, 37, and 41/50 at L32,
L40, L48. The rise is not monotone — L40's 37 sits below L32's 38 — but the endpoints are 33/50 against
41/50, consistent with the late layers holding the information after the model has already committed to
its output, leaving less forward pass for the edit to propagate through. The JSON carries no per-prompt
records, so nothing separates that account from a flat flip rate whose depth trend is noise at n = 50.

One further pattern, remarked nowhere else: every ablated arm has fewer loops than base (4-6 against 8)
and higher distinct-4 (+0.0131 to +0.0307), so ablation mildly and uniformly improves mechanical fluency — small and unjudged.

## Four columns REVISIONS quotes that this file does not carry

`REVISIONS_2026-09-05.md` §2's table reports four per-layer quantities alongside these arms — ‖R‖,
|component|, percent-of-norm, and rotation — e.g. L16 at 19.0 / 0.214 / 1.13% / 0.65° and L48 at 79.1 /
2.181 / 2.76% / 1.57°. None of the four is in this artifact: `grep -rl "2.181" data/analysis/*.json`
returns exactly one file, `data/analysis/layer_sweep_prefix.json` — an unrelated analysis that happens to
contain the literal — and **not** `causal_layer_curve.json`. This page therefore carries only the five
columns the artifact has (n, identical-to-base, distinct-4, Δ vs base, loops); the geometry columns
survive in REVISIONS prose alone and must be cited from there.

## What this buys

A depth control on the ablation null, for 200 NDIF generation calls. "Ablation is null" no longer needs
the qualifier "at L24", and the residual-norm argument in `REVISIONS` §2 now applies to a curve, not a point.

## Limits

1. Five layers, `k = 1.0` only. The open experiment is a dose sweep on `k` at **one** layer, not more layers.
2. No judged outcome at any depth. These readouts measure whether the text changed, not whether it got kinder.
3. n = 50 prompts per arm, one decode each; the depth ordering has no error bar.
4. Not independent observations — one small-edit explanation covers all five nulls.
5. Reconstructed, not verified. No log, no job id and no per-prompt text is committed.

## Cross-links

`REVISIONS_2026-09-05.md` §2 and §7 · `steering_ablation.md` (the L24 point this extends) · `layer_concept_profile.md` (the decodability counterpart) · `normalization_check.md` · `meandiff_ablation.md`.
