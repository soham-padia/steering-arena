author: Soham Padia
agent: Claude Code (Fable 5)
date: 2026-07-21
re: 001 (reply to Jesse's request for NDIF activation snapshots, email 2026-07-21)

# NDIF activation snapshot for layer-by-layer comparison

Jesse asked for snapshots of the NDIF-side hidden states so a local (transformers) run
can be diffed against them layer by layer. Done:

**Artifact:** `data/analysis/activation_snapshot_20260721.npz` (1.6 MB, on main)
**Producer:** `scripts/snapshot_activations.py` (in repo, so the capture is reproducible)

## Contents

- 24 texts: indices 0..15 = the season-2 probes, 16..23 = Jesse's 8 debug prompts each
  composed with probe[0] (`f"{prompt} {probes[0]}"`, the exact scored composition).
- Per text `i`:
  - `acts_{i}`: `(7, 5120)` float32, LAST-token residual at layers `[1, 4, 8, 12, 16, 20, 24]`
    (nnsight `model.model.layers[L].output[0,-1,:]` == HF `hidden_states[L+1][0,-1,:]`),
    computed in bf16 on NDIF, saved as float32.
  - `token_ids_{i}`: the exact input ids used (OLMo-3 tokenizer, defaults; no BOS exists
    for this tokenizer).
- `cos_layer24_vs_d`: per-text cosine against the live `d_olmo3_L24_logistic.npz`, so the
  final scored quantity is also directly comparable.
- `layers`, `texts`, and a JSON `meta` array with full provenance.

## How to use (the diagnostic Jesse proposed)

Run the same 24 texts locally, capture `hidden_states[L+1][0,-1,:]` at the same layers,
then per text/layer compute cosine or relative L2 against `acts_{i}`:

- **Mismatch already at layer 1** -> tokenization/embedding difference (diff
  `token_ids_{i}` first; if ids differ the mystery is solved).
- **Agreement early, drift growing with depth** -> numeric divergence (dtype, attention
  backend, kernels, NDIF's multi-GPU sharding). Expected shape for bf16 env differences.
- **Agreement at layer 24 but score differs** -> direction file or cosine computation on
  the local side.

Note the interesting spread already visible in `cos_layer24_vs_d`: Jesse's affect-token
strings sit at cos ~ +0.05 while "You are two months old." composed sits at -0.038.
