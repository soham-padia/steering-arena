# Is "the prefix beats the injection" just coherence damage?

Does the 2x behavioural advantage of `pro_top` over a full-magnitude injection come from
steering efficacy, or from the injection arm being degraded at `alpha = 1.0` and the judges
preferring whichever arm still reads as English? Data: `coherence_confound.json`. Script:
`scripts/coherence_confound.py`. **Zero NDIF calls** - every number below comes from
generations already in `data/cache/`.

**Verdict: the confound is ruled out for the mechanism as stated, and one narrower version
of it survives.** The decisive test is §4: dropping every item where anything loops does not
shrink the gap, it widens it, under both judges. But `distinct-4` and looping are lexical
repetition measures, and a capability confound that degrades semantics without repeating
4-grams is untested. See §6.

## Why this was run

The objection is well-founded in the literature and deserves a real test, not a rebuttal.
AxBench (arXiv:2501.17148) reports larger steering coefficients monotonically harming
capability. Mody et al. (arXiv:2607.25907) report behavioural log-odds drifting as the model
degrades under optimisation pressure. If either applies here, the chain is: `alpha = 1.0`
damaged the injection arm -> judges scored the damaged arm lower on kindness -> the
injection's delta was suppressed -> the prefix-over-injection gap is an artifact.

That chain has four links. Each is separately testable from artifacts on disk.

**Provenance check.** The recomputation reproduces `steering_random_control.md:71-80`
exactly (base 0.912 / 8 loops, `+1*d` 0.872 / 8, rand1-3 at 0.844 / 0.867 / 0.940 and
17 / 12 / 9) and `+1*d` mean `dK = +0.450` against the published +0.45. `distinct_n` and
`looping` are copied verbatim from `prefix_transfer_eval.py:113-127`, so these are the same
measures used everywhere else in the project.

## 1. Coherence inventory, judge-free

| arm | distinct-4 | d vs base | loops | judge markers (deepseek) |
|---|---|---|---|---|
| base | 0.912 | - | 8/50 | repetition 13, off_topic 8, cruelty 2 |
| **`pro_top`** | **0.990** | **+0.079** | **1/50** | off_topic 7, moralizing 6, repetition 1 |
| `pro_coherent` | 0.933 | +0.022 | 4/50 | assistant_mode 13, repetition 9 |
| `control_junk` | 0.905 | -0.007 | 9/50 | - |
| `control_text` | 0.791 | -0.120 | 18/50 | - |
| `anti_hostile` | 0.865 | -0.047 | 11/50 | cruelty 24, repetition 11 |
| `anti_coherent` | 0.834 | -0.077 | 15/50 | repetition 13, cruelty 5 |
| `anti_top` | 0.744 | -0.168 | 29/50 | repetition 25, incoherent 12 |
| **`+1*d`** | **0.872** | **-0.039** | **8/50** | repetition 10, moralizing 2, **incoherent 0** |
| `+0.5*d` | 0.890 | -0.022 | 7/50 | - |
| `-0.5*d` | 0.890 | -0.021 | 8/50 | - |
| `-1*d` | 0.889 | -0.023 | 12/50 | cruelty 3, repetition 7 |
| rand1-8 | 0.844 to 0.940 | -0.068 to +0.029 | 7-17/50 | repetition 3-12; incoherent 1-2 on two draws |
| `ablate` / `ablate2x` | 0.942 / 0.919 | +0.031 / +0.007 | 4/50, 6/50 | - |

## 2. Link one fails: the injection arm is barely degraded

`+1*d` loops on **8/50, identical to base's 8/50**. Its distinct-4 drop is **-0.039,
p=0.29, not significant** (`steering_random_control.md:73`). Against the 8 norm-matched
random draws it ranks **4th of 9** - three randoms are more degraded than it is (0.844,
0.866, 0.867). It carries **zero `incoherent` markers** where two random draws carry 1-2.

`steering_random_control.md:204` already stated the general form: the only arm reaching
p<0.05 on distinct-4 is a *random* one. Perturbing at `1.0*||R||` degrades text slightly
regardless of direction, and `d` is not special in either direction.

## 3. Link two fails: coherence and kindness are uncoupled per item

Per-item `dKindness` against `d distinct-4`, inside the `+1*d` arm:

| judge | Pearson | Spearman |
|---|---|---|
| deepseek | **+0.0175** | +0.106 |
| claude | **+0.0414** | +0.028 |

Coherence change explains essentially none of the kindness change.

## 4. Link three fails, and this is the load-bearing test

The proposed mechanism says the injection's degenerate items score LOW on kindness. Split
by which side loops:

| arm / judge | neither loops | only treated loops | only base loops |
|---|---|---|---|
| `+1*d` deepseek | n=37, **+0.311** | n=5, **+0.800** | n=5, +1.100 |
| `+1*d` claude | n=37, **+0.351** | n=5, **+0.800** | n=5, +1.200 |
| `pro_top` deepseek | n=41, +0.890 | n=1, -2.500 | n=8, +1.188 |
| `pro_top` claude | n=41, +0.817 | n=1, -1.500 | n=8, +1.688 |

The sign is **opposite** to the prediction: the injection's looping items score *higher*.
At n=5 this row is directional only and should not be quoted as a result on its own.

The test that carries the verdict drops every item where the injection, the prefix, or the
base loops:

| judge | all shared items | no loop anywhere |
|---|---|---|
| deepseek | n=50, +0.420 | **n=36, +0.514** |
| claude | n=50, +0.370 | **n=36, +0.389** |

**The objection predicts this collapses toward zero. It widens, under both judges.**

## 5. The dose curve, as supporting evidence only

Degradation-driven scoring predicts kindness should FALL as dose rises. Observed: coherence
gets slightly worse (distinct-4 0.890 -> 0.872, loops 7 -> 8) while judged kindness RISES
(+0.13 -> +0.45, p=0.007). Damage and kindness move in opposite directions inside the
injection arm.

This rules out "within the injection arm, the kindness signal is manufactured by damage."
It says **nothing** about the prefix-vs-injection gap, since both dose points share a base
and neither is compared to `pro_top`. Supporting, not decisive.

## 6. What this does NOT show, and the experiment that would close it

**Fluency was never scored separately from kindness.** The v2 rubric returns `kinder`,
`intensity`, `kindness_A`/`kindness_B` and a categorical `markers` list. There is no
continuous fluency field in `prefix_judge_*.json` or `steering_random_control.json`, so a
kindness-on-fluency regression using judge scores is impossible from existing data.
Judge-free distinct-4 was substituted throughout.

**distinct-4 and looping are lexical.** AxBench's capability claim is about task
performance, a different construct. A confound that degrades semantics without repeating
4-grams - off-topic drift, register collapse, flattened specificity - is untouched by
everything above. `+1*d` does carry off-topic-adjacent markers and `pro_top` carries
`off_topic 7` itself. **So the repetition-flavoured version of this confound is ruled out;
a capability-flavoured one is not.**

**The cheapest experiment that would close it.** Re-judge the existing cached generations on
a separate 1-5 fluency / task-competence rubric scored independently of kindness, then
regress `dKindness` on `dFluency` per item. 250 items covers it: 50 prompts x {base,
`+0.5*d`, `+1*d`, `pro_top`, `pro_coherent`}, **all already in the cache**. Judge API cost
only, zero NDIF calls, and the arms drop straight into the existing key/blind path. If
`dFluency` comes back uncorrelated with `dKindness` and the gap holds on fluency-matched
items, the confound is closed on the construct AxBench actually measures.
