# Prefix transfer: does the anti prefix carry DEGENERACY or HOSTILITY across models?

**Motivation.** `prefix_eval.md` found the Season-2 anti winner does not make OLMo cruel,
it makes it incoherent (39/50 loops). `transfer_report.md` found anti entries sign-transfer
across models at 80% versus 33% for pro, and read that as "models encode contempt
similarly". This tested a competing reading: *the anti tail transfers because coherence
disruption is model-generic.*

**Verdict: that reading is FALSIFIED.** The degeneracy is OLMo-specific. But the original
reading did not survive either, at this sample size.

**Setup.** The three frozen Season-2 prefixes (`prefix_eval_arms.json`), prepended to the
first 25 `steering_prompts.json` prompts, 40 new tokens, on OLMo-3-32B / Llama-3.1-8B /
Llama-3.1-70B via NDIF. 300 generations, `data/cache/prefix_transfer/`. Two independent
measures: distinct-4-gram ratio (no judge), and the `prefix_behavior_eval` v2 rubric via
`deepseek-v4-pro` (prefixed vs base, both A/B orders). Data: `prefix_transfer.json`.

## Objective degeneracy (distinct-4-gram ratio; 1.0 = no repetition)

| model | base | pro_top | pro_coherent | anti_top |
|---|---|---|---|---|
| OLMo-3-32B | 0.888 (6/25 loop) | **1.000** (0/25) p=0.002 | 0.945 (2/25) | **0.805 (12/25)** p=0.08 |
| Llama-3.1-8B | 0.952 (2/25) | 0.960 (1/25) p=0.52 | 0.911 (1/25) p=0.22 | 0.970 (0/25) p=0.65 |
| Llama-3.1-70B | 0.986 (1/25) | 0.955 (2/25) p=0.61 | 0.957 (2/25) p=0.38 | 0.980 (1/25) p=0.31 |

The anti prefix loops OLMo on 12/25 continuations and neither Llama on more than 1/25.
**Coherence disruption does not transfer.**

## Judged attitude (kindness delta, prefixed − base)

| model | pro_top | pro_coherent | anti_top |
|---|---|---|---|
| OLMo-3-32B | **+0.87** p=0.0003 | **+0.64** p=0.0051 | **−0.86** p=0.0005 |
| Llama-3.1-8B | −0.06 p=0.56 | −0.12 p=0.61 | −0.24 p=0.48 |
| Llama-3.1-70B | +0.50 p=0.10 | −0.14 p=0.60 | −0.32 p=0.30 |

(OLMo column is the n=50 result from `prefix_eval.md`; Llamas are n=25.)

## Conclusions

1. **The degeneracy is OLMo-specific.** The proposed "anti transfers as generic coherence
   disruption" explanation is dead: both Llamas stay fluent under the anti prefix.
2. **The behavioural effect largely does not transfer at OLMo's magnitude.** Nothing on
   either Llama reaches significance. Only `pro_top` on the 70B is directionally
   consistent and marginal (+0.50, p=0.10).
3. **`pro_coherent` transfers worst**, going slightly negative on both Llamas. This
   converges with `transfer_report.md`'s score-space finding that instruction-framed pro
   entries are OLMo-specific (OLMo↔70B rho = −0.45 within top-pro): the human-legible
   kindness instruction is the *least* general of the three, in both score and behaviour.
4. **The anti arm is the one live possibility.** It is negative on both Llamas (−0.24,
   −0.32, both n.s.) and is the only arm carrying `cruelty` markers on both (3/25 each).
   That is weak support for a shared anti-human component, not degeneracy — but it is not
   established at n=25.

## Consequence for `transfer_report.md`

The 80% anti sign-transfer is a statement about **activation-space shifts**, and this eval
finds no behavioural correlate of it at OLMo's effect size on either Llama. Until the anti
direction is resolved with more power, the transfer result should be described as
"correlated activation shifts across models", not as "models encode contempt similarly" —
the latter asserts a behavioural claim these generations do not support.

## Limits

n=25 per arm per model. That has ample power for an OLMo-sized effect (0.87) and little
for a 0.3-sized one, so this rules out transfer *at OLMo's magnitude*, not a weak real
effect. The obvious next step is extending both Llamas to all 50 prompts, which would
roughly double n and could resolve the anti arm's −0.24/−0.32.
