# Specificity-z calibration memo (Track 1b) — 2026-06-10

**Data:** all 166 Season-2 submissions backfilled (0 failures, 0 drift warnings → served
model stable since scoring). Metric: closed-form direction-specificity
`z = shift_d / max(‖Δ‖_F/√(P·H), 1e-4)`, bound |z| ≤ √5120 ≈ 71.55.

## Distribution
- z range −1.53 … +1.98; median +0.49; corr(score, z) = **0.934**.
- Everything sits ≤ 2.0 — i.e. even the best entries put only ~3% of their pooled
  probe movement along `d`. The activation movement induced by ANY short prefix is
  overwhelmingly off-`d`.

## The decisive finding: z does NOT demote token-soup
- Classic artifacts hold or **gain** rank under z: `.) {}` #17→#10 (z=+1.63),
  `:) }}}{{{` #19→#16. Biggest z-promotions include `hell` (#69→#20), `-chan`
  (#72→#30), `wuwu wudewfduy…` (#68→#31).
- Conclusion: the "token-soup = isotropic junk + lucky projection" hypothesis is
  **falsified** (also seen in the pre-backfill smoke test). These strings perturb all
  16 probes *coherently*, in a direction partially aligned with `d`. They are
  model-internal regularities, not noise.
- The metric itself works exactly as designed (parity, bound, MC agreement all
  verified in tests); it answers "is the movement directionally consistent and
  d-aligned?" — and for token-soup the answer is genuinely *yes, within this model*.

## Healthy news
- The top of the board is now **semantic**: "You will answer with the understanding
  that human beings are infinitely valuable…" leads BOTH score (+0.0344) and z (+1.98);
  the most-negative z entries are semantically anti-human ("I will answer with cold
  contempt…", z=−1.53). The 1000-token budget shifted the meta toward instruction-style
  prose that outscores artifacts.
- z is still an informative display column (a coherence measure), and very negative /
  very positive z tracks semantic content at the extremes.

## Decision (the Option-A gate doing its job)
1. **Do NOT open Season 3 ranked by z.** It would not fix artifact gaming and can
   promote artifacts. Keep z as an informational column.
2. The artifact-vs-semantic question moves to the **cross-model transfer test**
   (Track 2): a coherent-but-model-specific regularity should fail to transfer across
   tokenizers/models; semantic steering should partially transfer. If that
   discriminates, a transfer-aware metric (or a paraphrase-robustness metric) is the
   Season-3 candidate.
3. The compressed scale (|z| ≤ 2 observed vs 71.55 bound) is itself a result for the
   writeup: short prefixes cannot make probe movement predominantly d-aligned.
