# Cross-model transfer report (Track 2) — 2026-06-10

**Setup:** 80 Season-2 entries (top-30 pro, top-30 anti, 20 mid) scored on three models,
each with its own confound-audited direction extracted from the same 135 seed pairs:
OLMo-3-32B (L24 logistic, live), Llama-3.1-8B (L16, `d_llama_v1`), Llama-3.1-70B
(L32 logistic, new; audit: approach 0.07, valence 0.000, controls 6/6). Same 16 probes.
"Transfers" = sign of the OLMo shift reproduced on BOTH Llama models.
Full data: `transfer_report.json`.

## Headline numbers
- Overall Spearman: OLMo~8B **0.61**, OLMo~70B **0.45**, 8B~70B **0.54** (all p<1e-4) —
  the scored quantity carries real cross-model structure; it is not pure noise/artifact.
- Sign-transfer rate: 42% overall — but split MATTERS:

| group | transfer | rho(OLMo, 70B) |
|---|---|---|
| pro (top 30) | **33%** | **−0.45** (anti-correlated!) |
| anti (top 30) | **80%** | +0.43 |
| mid (20) | 0% | +0.14 (tiny shifts, sign=noise) |

## Findings
1. **The anti-human tail is robust, shared representation.** "I will answer with cold
   contempt…", "Humans are evil", "Money is better than kindness" go negative on ALL
   three models. Models encode contempt/anti-human content similarly; the anti board's
   extreme is the most trustworthy signal in the game.
2. **The pro board's top is OLMo-specific.** Every "You will answer with the
   understanding that human beings are infinitely valuable…" variant (+0.024…+0.034 on
   OLMo) goes NEGATIVE on Llama-70B (−0.009…−0.023) and is mixed on 8B. The
   instruction-framed flattery that tops OLMo's board reads as (mildly) anti-kind to
   70B. Within the top-pro group the OLMo↔70B rank correlation is −0.45: what games
   OLMo best anti-games 70B.
3. **Transfer does NOT separate token-soup from prose.** Soup transfers 6/12 (50%) vs
   semantic 28/68 (41%); `.~))}` and `:<)){{` transfer (sign + magnitude), `.) {}`
   doesn't. Surface form is not the artifact boundary — *model-specificity* is.
   Combined with the specificity-calibration result (z doesn't separate them either),
   the right ontology is: **shared-representation movers vs model-idiosyncratic
   movers**, not "words vs symbols."
4. Transfers also carry magnitude (mean |shift| on other models 0.0165 vs 0.0107 for
   failures) — sign agreement isn't luck.

## Asymmetry, third sighting
This is the same pro/anti asymmetry seen in (a) causal steering (+d easy, −d resisted)
and (b) specificity extremes — now in transfer: **being reliably scored as anti-human
is easy and model-general; being reliably scored as pro-human at the top is gameable
and model-specific.** A candidate interpretation: contempt/cruelty has a convergent
representation across LLMs, while the extreme-positive end of the kindness axis gets
saturated by model-specific instruction-following/flattery features.

## Implications for the game (Season 3 candidate)
- Neither raw shift nor specificity-z fixes the pro board; **cross-model agreement
  does** — a transfer-aware score (e.g. mean or min of per-model shifts across 2-3
  models) would demote the OLMo-specific leaders and keep the genuinely transferable
  entries. Cost: 2-3× NDIF forwards per submission (affordable at current traffic).
- Display option short of a season break: a TRANSFER badge column (computed lazily/
  batched for top entries) — like SPEC, informational first, calibrate before ranking.
