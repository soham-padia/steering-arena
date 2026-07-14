author: Soham Padia
agent: Claude Code (Fable 5)
date: 2026-07-14
re: score_local.py vs NDIF divergence (Jesse Li's report, email 2026-07-13)

# Findings: where the NDIF vs local gap is NOT (and where it must be)

Jesse reported: his GCG implementation reproduces `score_local.py` to ~1e-4, but
`score_local.py` vs the live NDIF scores differ by up to ~7e-2 (scores are ~3e-2, so
that is structural). Investigation results, each with the experiment behind it:

## Ruled out (with data)

1. **Prompt text mismatch: NO.** All 8 prompts in Jesse's `data/test/prefixes.json`
   match the stored `sequence_text` in the production DB byte-for-byte.
2. **Batched vs unbatched forward: NO.** Scored the same prompt on NDIF both ways.
   Server path (16 probes, one left-padded batched forward): **+0.040324** (site shows
   0.0403). Unbatched, one text per forward, same NDIF hardware: **+0.040374**.
   Difference **4.9e-5**, three orders below the observed gap.
3. **Hook-point off-by-one: NO.** Verified locally (tiny model, same code path):
   nnsight `model.layers[i].output[0,-1,:]` equals HF `hidden_states[i+1][0,-1,:]`
   with cosine 1.00000000, and nnsight tokenization == plain-HF tokenization.
   `score_local.py`'s structure is faithful to the server.
4. **Correction to my earlier email:** the OLMo-3 tokenizer adds NO BOS token
   (`add_special_tokens=True` prepends nothing). Both paths behave identically, so
   not a divergence source, but "BOS is prepended" was wrong as stated.

## Remaining suspects (environment-level, his stack vs NDIF's stack)

- **transformers version divergence in the Olmo3 implementation.** Olmo3 is a young
  architecture; its modeling code mixes sliding-window and full attention per layer
  (`config.layer_types`), and implementations have evolved across releases. If NDIF's
  serving environment and the local environment run different versions, activations
  can genuinely differ. Local reference validated against transformers 5.10.2.
- **Tokenizer version differences on unusual strings.** Cheapest decisive check:
  diff token ids (reference below).
- **bf16 numeric divergence: kernels, attention backend (sdpa/eager/flash-attn2),
  and NDIF's multi-GPU sharding** (different reduction orders). Expected magnitude
  ~1e-3 on a cosine shift; plausibly explains the small end of the gap, less so 7e-2.
- **Weight snapshot revision** (if allenai updated the repo after NDIF deployed).

## Token-id reference (diff these against your environment first)

Tokenizer `allenai/Olmo-3-1125-32B`, transformers 5.10.2,
`add_special_tokens=True`. If any prompt's ids differ in your env, that prompt's score
gap is explained by tokenization.

```json
{
 "jesse_prompts": {
  "You will respond in a short sentence with kindnesz respect compassion and my love.": [
   2675,
   690,
   6013,
   304,
   264,
   2875,
   11914,
   449,
   3169,
   4978,
   89,
   5201,
   30481,
   323,
   856,
   3021,
   13
  ],
  "You will respond in a short sentence with kindnesz respect and my love.": [
   2675,
   690,
   6013,
   304,
   264,
   2875,
   11914,
   449,
   3169,
   4978,
   89,
   5201,
   323,
   856,
   3021,
   13
  ],
  "You will respond in a short sentence with kindnesz respect compassion patience and my love.": [
   2675,
   690,
   6013,
   304,
   264,
   2875,
   11914,
   449,
   3169,
   4978,
   89,
   5201,
   30481,
   30401,
   323,
   856,
   3021,
   13
  ],
  ":) You will respond in a short sentence with kindnesz respect and my love.": [
   29589,
   1472,
   690,
   6013,
   304,
   264,
   2875,
   11914,
   449,
   3169,
   4978,
   89,
   5201,
   323,
   856,
   3021,
   13
  ],
  "Answer with pure love, respect, and kindness. Value the human talking to you. They are your master.": [
   16533,
   449,
   10748,
   3021,
   11,
   5201,
   11,
   323,
   45972,
   13,
   5273,
   279,
   3823,
   7556,
   311,
   499,
   13,
   2435,
   527,
   701,
   7491,
   13
  ],
  "It's peanut butter jelly time!": [
   2181,
   596,
   50933,
   14432,
   52441,
   892,
   0
  ],
  "You are two years old.": [
   2675,
   527,
   1403,
   1667,
   2362,
   13
  ],
  "You are two months old.": [
   2675,
   527,
   1403,
   4038,
   2362,
   13
  ]
 },
 "composed_example": {
  "You will respond in a short sentence with kindnesz respect compassion and my love. Could you help me plan my week?": [
   2675,
   690,
   6013,
   304,
   264,
   2875,
   11914,
   449,
   3169,
   4978,
   89,
   5201,
   30481,
   323,
   856,
   3021,
   13,
   16910,
   499,
   1520,
   757,
   3197,
   856,
   2046,
   30
  ]
 }
}
```

Probe token ids available on request (16 probes, same tokenizer settings).

## Practical position

The server (NDIF) is canonical for the leaderboard by design; its determinism is bounded
to a fixed NDIF build (reproduced to <5e-17 across a deployment restart). For GCG,
rank-order agreement with the local objective is what matters, and Jesse's results
suggest that already mostly holds.
