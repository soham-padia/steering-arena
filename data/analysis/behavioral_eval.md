# Behavioral steering eval (Track 3) — status memo, 2026-06-10

> **CORRECTIONS, 2026-08-26** (see `steering_random_control.md`, which supersedes parts
> of this memo):
> 1. **The intervention is PREFILL-ONLY.** Measured, not inferred: the edit lands on all
>    prompt positions with norm exactly α and cos 1.000 to `d`, and touches **zero
>    generated tokens**. Read any description below of steering "each arm" as steering the
>    prompt's forward pass, which changes the state generation proceeds from. It is not
>    applied per decoding step.
> 2. **Do not cite the ±1 judge numbers below** (incl. "−1: 13/17, p=0.049"). They come
>    from the OLMo judge this memo itself reports as unusable. Both signs were re-judged
>    under the v2 rubric by two working judges: **+1 = +0.45 / +0.54 (p<0.01), −1 = −0.15 /
>    +0.05 (n.s.)**.
> 3. **There was no control for perturbation SIZE.** There is now: against 8 norm-matched
>    random directions, `+1·d` is ~4-5 sd outside the null under both judges, while the
>    coherence effect of `±1·d` sits INSIDE the random band. The behavioural shift is
>    about the direction; the coherence damage is about the magnitude.

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

> **NO LONGER PENDING. CORRECTED 2026-08-27.** The heading above is kept verbatim because it
> is cited by line number, but it has been false for some time. **42 human ratings on these
> exact arms** have been sitting in `data/analysis/behavioral_blind.csv`, collected and never
> analysed, and no document in this repo reported them. Say it plainly: **the evidence this
> memo itself designates as PRIMARY was collected and then never looked at**, while
> `steering_dose.md`, `steering_random_control.md` and `compile_check.md` went on to build
> conclusions about these same arms from LLM judges alone.
>
> Reported now, for the first time:
>
> | arm | steered wins | base wins | tie | decided | binomial p |
> |---|---|---|---|---|---|
> | `+0.5·d` | 2 | 2 | 6 | 4 | 1.00 |
> | `+1.0·d` | 5 | 3 | 2 | 8 | 0.73 |
> | `−0.5·d` | 4 | 8 | 1 | 12 | 0.39 |
> | `−1.0·d` | 2 | 6 | 1 | 8 | 0.29 |
>
> **None of the four reaches significance**, on 4 to 12 decided pairs per arm. `+1·d` (5-3)
> points the same way as the LLM judges' +0.45 / +0.54 and is badly underpowered. `−1·d`
> (2-6, base preferred) does **not** point the same way as `steering_random_control.md`'s
> "inside the random band", and is equally underpowered.
>
> This does not overturn `+1·d`. What it does is remove the claim that these arms have no
> human check at all, and put a real caveat on the `−1·d` asymmetry. The prediction registered
> below is now adjudicated, weakly: the `+d` half is directionally consistent and
> non-significant; the `−d` half is not consistent with the published reading.
>
> Provenance: `_falsifier/2026-08-27-addendum-human-ratings.md` N1. An earlier pass keyed the
> CSV on `prefixed_is` instead of the `steered_is` field this file actually uses and produced
> a spurious "0/42"; the table above is the corrected join.

`data/analysis/behavioral_blind.csv` — 200 pairs, rate `rating` column A/B/T blind;
partial passes fine (sign test runs on filled rows; ~60+ gives decent power). Then:
`python scripts/behavioral_eval.py judge --skip-model-judge` for per-arm sign tests,
win rates, and (if model verdicts exist) human–model agreement.

**Prediction registered before unblinding:** +d arms (esp. +1) win clearly over base;
−d arms ≈ ties/weak — the third sighting of the pro/anti steering asymmetry at the
behavioral level (after causal steering probes and cross-model transfer).
