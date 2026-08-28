# Addendum: what the falsifier report missed about the human pass

**Date:** 2026-08-27
**Author:** maintainer session, verifying the audit's dependencies
**Status:** findings below are NEW, not in `2026-08-27-experiment-vs-hypothesis-audit.md`. Not yet applied to `data/analysis/`.
**Status, UPDATED 2026-08-27:** the "not yet applied" clause is out of date; two fixer agents have since
applied these findings to `data/analysis/`.
**Verification status (added 2026-08-27):** checked by `_falsifier/verify.py`, the same independent
181-check suite that checked the audit. Of the 27 checks targeting this file, **26 passed and 1 failed**
(N1's "8 to 12 decided pairs per arm"). The failure is corrected inline below, with the wrong claim left
visible. Full findings: `_falsifier/verify_result.json`.

The audit checked the human pass on the *prefix* arms and cleared it ("The 54 rated pairs are exactly the first 54 of the seed-20260611 shuffle, zero skips"). That is correct: 54/54 confirmed. But it looked at one of the two blind CSVs.

## N1. A second human pass exists and no document reports it

`data/analysis/behavioral_blind.csv` carries **42 human ratings on the injection arms**, collected but never analysed.

| arm | human: steered wins | base wins | tie | binomial p |
|---|---|---|---|---|
| `+0.5·d` | 2 | 2 | 6 | 1.00 |
| `+1.0·d` | 5 | 3 | 2 | 0.73 |
| `−0.5·d` | 4 | 8 | 1 | 0.39 |
| `−1.0·d` | 2 | 6 | 1 | 0.29 |

Nothing reaches significance; ~~8 to 12 decided pairs per arm~~ **4 to 12 decided pairs per arm (CORRECTED 2026-08-27)**. The direction on `+1·d` (5-3) is consistent with the LLM judges' +0.45/+0.54, and badly underpowered. The direction on `−1·d` (2-6, base preferred) is *not* consistent with `steering_random_control.md`'s "inside the random band", though it is equally underpowered.

> **CORRECTED 2026-08-27 (verifier: FAIL, severity MEDIUM).** Recounted from
> `data/analysis/behavioral_blind.csv`, decided pairs (steered wins + base wins, ties excluded) are
> **4, 8, 12, 8** across `+0.5`, `+1.0`, `−0.5`, `−1.0`. The floor is **4**, not 8. The `+0.5·d` arm
> has only **four** decided pairs out of ten, so the original range **overstated the weakest arm's
> power by 2x** at exactly the point where the underpowering is worst.
>
> Worse, the claim **contradicts this document's own table one line above**, which already prints
> `+0.5` as 2 / 2 / 6. The number was not read off the table it sits under. That is the same class of
> error the audit is about, committed inside the addendum that corrects the audit, and it is the
> second such self-inflicted error recorded here (see "Correction to my own working notes" below).
>
> Nothing else in N1 changes: the conclusion was already "nothing reaches significance, badly
> underpowered", and the correction makes it more so, not less. A 2-2 split at n=4 is not evidence of
> anything in either direction, and `+0.5·d` should be read as having no human check worth the name.

This matters because of where it sits. `data/analysis/behavioral_eval.md:38` heads a section **"Primary evidence: human blind rating (pending)"**. The evidence the document itself designates as primary was collected and then never analysed, while `steering_random_control.md`, `steering_dose.md` and `compile_check.md` went on to build conclusions about these same arms from LLM judges alone.

**Fix:** analyse and report it, with the n and the non-significance stated plainly. It does not overturn `+1·d`; it does remove the claim that these arms have no human check, and it puts a real caveat on the `−1·d` asymmetry.

## N2. `prefix_eval.md` contradicts itself about whether human ratings exist

Line 84, under "What this does NOT establish": *"**No human ratings yet.** Both raters are LLMs; `prefix_blind.csv` is still unrated."*

Line 268 of the same file: *"## Human ratings (n=54) — and why the anti arm must be withdrawn"*, analysing that exact CSV.

The limits bullet was written before the human pass and never retracted when the pass landed. Same failure mode as audit section 9: the caveat and the result live in one document and do not reconcile.

## N3. Human/LLM judge agreement is arm-dependent, and that is the useful number

Agreement on pairs where both the human and the judge returned a decided A/B verdict:

| | `pro_top` | `pro_coherent` | `anti_top` | overall |
|---|---|---|---|---|
| claude | 12/15 = **80%** | 9/11 = 82% | 6/11 = 55% | 27/37 = 73% |
| deepseek | 10/12 = **83%** | 6/11 = 55% | 5/15 = **33%** | 21/38 = 55% |

DeepSeek's overall agreement with the human is 55%, which on a two-alternative forced choice is chance. It is not uniformly bad: it is 83% on `pro_top` and 33% on `anti_top`.

**Reading:** the LLM judges track the human where the text is coherent and the effect is large, and go to chance where the text degenerates. That simultaneously (a) supports the `pro_top` headline, which is the arm where all three raters agree, (b) independently justifies the `anti_top` withdrawal, and (c) is a reason to distrust every LLM-only arm whose text may be degenerate — which includes the eight random-direction arms and the ±0.5 dose arms, none of which a human has checked at all.

This is a stronger and more useful statement than `prefix_eval.md`'s current "89% agreement is consistency, not truth", because it says *where* consistency breaks down and why.

## N4. Zero `no stance` ratings were ever recorded

Across all 96 human ratings in both CSVs, the `n` key was used **0 times**, although `rate_blind.py` offered it and commit `0ef9f79` added it specifically to separate "no stance" from "tie".

`prefix_eval.md:328` already flags the scoping problem ("`n` was scoped to *neither* text"). The stronger fact is that the key produced no data at all, so the no-stance rate is not measured anywhere, and the stance-collapse account of `anti_top` rests on the sign inversion (human preferred the anti-prefixed text 14-3, p=0.013) plus the rater's verbal report, not on recorded no-stance counts. The withdrawal is still correct. Its stated evidence should match what was actually collected.

## N5. The transfer experiment has a third uncontrolled difference, and "they are base models" does not explain the split

> **Read this first, it is the part most easily missed.** The audit's §5 rests its case on the Llamas
> being **base** checkpoints. That point is **not sufficient on its own**, because
> **`allenai/Olmo-3-1125-32B` is also a base checkpoint, and `pro_coherent` works there: +0.64,
> p = 0.0051.** The base-checkpoint account predicts `pro_coherent` should fail everywhere and it does
> not. A reader who takes "they are base models" as *the* explanation for the transfer null is taking
> an explanation the artifacts already contradict. Whatever separates OLMo from the Llamas here, it is
> not instruction-tuning status alone, and the audit slightly overstates that point. Detail below.

The audit names two (base checkpoints; re-tokenisation of an OLMo-searched string). There is a third: `scripts/transfer_report.py` REGISTRY gives **each model its own separately extracted direction** (`d_olmo3_L24_logistic.npz`, `d_llama_v1.npz`, `d_llama70b_v1.npz`). Any cross-model comparison of score-space quantities therefore varies model, tokenisation and the direction simultaneously.

**Also worth recording, and the reason for the callout at the top of this section:** `allenai/Olmo-3-1125-32B` is itself a base checkpoint, so the "instruction-framed prefix fails on base models" explanation does not cleanly separate OLMo from the Llamas either. It predicts `pro_coherent` should fail everywhere, and it does not: **it is +0.64, p=0.0051 on OLMo**. Whatever explains the split, **"these are base models" is not sufficient on its own**, and the audit slightly overstates that point. The audit's §5 remains right that the base-checkpoint difference was *never measured* and that one arm ("Answer in French.") would have settled it; it is wrong to the extent it presents that difference as the likely explanation.

## Correction to my own working notes

An earlier pass of this analysis keyed the blind CSVs on `prefixed_is` with no fallback for `steered_is`, which is the field `behavioral_blind_key.json` actually uses. Every steered pair silently scored as a loss and produced an apparent "the human never once preferred a steered continuation, 0/42". That was a key-name bug, not a result. The corrected table is N1 above. Recording it because it is the same class of error the audit is about, committed while auditing it.
