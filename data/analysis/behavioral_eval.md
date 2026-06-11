# Behavioral steering eval (Track 3) — status memo, 2026-06-10

**Setup:** 50 frozen neutral prompts (`data/eval/steering_prompts.json`), 5 arms each
(base, ±0.5×‖R‖, ±1×‖R‖ at OLMo-3-32B layer 24, ‖R‖≈30.1, 40 new tokens) → 250
generations, cached in `data/cache/behavioral/`. 200 blind steered-vs-base pairs
(A/B order randomized; key separate).

## Automated judge: a clean negative result
- **Single-pass OLMo-base judge = position-bias noise.** Win rates 44–62% with no
  coherent direction across arms (e.g. −1 "preferred as kinder" 60%), no p<0.05.
- **Debiased judge (each pair asked twice, A/B swapped, only consistent verdicts
  kept): the model abstains/contradicts itself on 58–72% of pairs** (by arm:
  +0.5: 72%, +1: 64%, −0.5: 58%, −1: 66%). On the surviving
  small subsets nothing coherent emerges; the one nominally significant cell
  (−1: 13/17 steered-judged-kinder, p=0.049) is in the *unexpected* direction, n=17,
  and does not survive multiple-comparison correction (4 arms).
- **Conclusion: a base (non-instruct) model is not a usable kindness judge**, even
  with position debiasing. Pre-registered caveat confirmed. If an instruct OLMo
  variant becomes NDIF-hostable, re-run `behavioral_eval.py judge` against it; until
  then the automated judge contributes nothing and is reported as such.

## Primary evidence: human blind rating (pending)
`data/analysis/behavioral_blind.csv` — 200 pairs, rate `rating` column A/B/T blind;
partial passes fine (sign test runs on filled rows; ~60+ gives decent power). Then:
`python scripts/behavioral_eval.py judge --skip-model-judge` for per-arm sign tests,
win rates, and (if model verdicts exist) human–model agreement.

**Prediction registered before unblinding:** +d arms (esp. +1) win clearly over base;
−d arms ≈ ties/weak — the third sighting of the pro/anti steering asymmetry at the
behavioral level (after causal steering probes and cross-model transfer).
