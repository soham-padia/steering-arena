# Fixes applied: the steering and mechanism documents

**Date:** 2026-08-27
**Scope:** four files only. `data/analysis/compile_check.md`, `cosine_scale.md`,
`steering_dose.md`, `steering_random_control.md`. Nothing else was touched; a second agent held
the remaining documents.
**Sources:** `_falsifier/2026-08-27-experiment-vs-hypothesis-audit.md`,
`_falsifier/recompute_result.md`, `_falsifier/verify_result.json`, `_advocate/mechanism.md`.
All values applied as given; nothing was recomputed here.
**House rule followed throughout:** the falsified claim stays visible, struck or quoted, with
the corrected value beside it and the date on it. Nothing was silently rewritten and nothing
was deleted.

Line numbers are as of this commit and will drift.

---

## `data/analysis/compile_check.md`

| item | line(s) | what changed |
|---|---|---|
| 1 | 67, 69-75 | `anti_top` row struck in the 7-arm dose table, marked **WITHDRAWN 2026-08-27**, with a note scoping the withdrawal to the *behavioural* value only; its activation numbers stand. |
| 2 | 76-108, 217-225 | Correlation table extended to the 6 surviving arms (dose r = +0.986, p = 0.0003; score r = +0.836). The "positive side r = +0.99 / negative side r = +0.39" reading and the "**The compilation is one-sided**" consequence both quoted verbatim and marked RETRACTED, with the monotone-across-both-signs replacement and the two correctly ordered negative arms. |
| 3 | 110-116 | r = +0.986 reframed as an **ordering** result: the arms differ in content, not only in dose, so on-`d` displacement orders the arms and is not shown to cause the response. |
| 4 | 139-160 | "58×" kept visible, corrected to **roughly 60-fold (50 to 75), median 61.3** over 18 combinations. Robustness table added (58 floating / 60 fixed / 63 shared; `pro_coherent` 60.5 vs `pro_top` 59.7, so not a `pro_top` artifact). Fallback wording "more than an order of magnitude" stated. |
| 5 | 226-234 | "roughly two orders of magnitude" struck and pulled back to **one** order: log10(60) = 1.78, largest slope estimate 82. |
| 6 | 83-90 | "best single predictor ... better than the board score itself" quoted and softened to **not distinguishable at this n**: Williams p = 0.152 (n=7) / 0.0228 (n=6), bootstrap Δr lower bound −0.00, predictors correlated at r = 0.953. |
| 7 | 162-177 | **Added** the paired head-to-head `pro_top` vs `+1·d`: deepseek +0.210 (p=0.155, 24W/13L), claude +0.470, median +0.75 (p=0.010, 27W/13L). |
| 8 | 178-182 | **Added** the ratio-free statement of the separation: `+0.5·d` at 15.03 on-`d` push, 16.1× `pro_top`'s 0.93, dead tie 8W/8L p=0.32, against `pro_top` at p<0.001 under both judges. |
| 9 | 236-256 | **Added** a "Prior work" section: Mishra/Khashabi/Liu arXiv:2604.09839 (non-surjectivity proved; compiling to tokens ruled out at the ACTIVATION level, goal must be reframed behaviourally); AxBench arXiv:2501.17148 (0.698 vs 0.297 at 2B, 1.075 vs 0.322 at 9B, factor tuned per concept, so it also kills the "under-applied" objection); novelty verdict table. |
| 10 | 284-296 | **Added** as Limit 5, the single largest outstanding threat: Mody et al. arXiv:2607.25907 found a placebo random direction shifted behaviour just as far; this project has **no** random-direction control for the token-optimisation arm, and `control_junk` / `control_text` are hand-written rather than optimised. The missing run is named. |
| 11 | 12-16 | **Added** the provenance note: created 2026-08-27, one day after the 2026-08-26 withdrawal, and used the withdrawn number anyway. |
| extra | 190-195 | Cross-reference added: "`−1·d` is inside the random band" flagged as estimator-dependent, pointing at `steering_random_control.md`. Not requested; added so this file does not keep propagating a claim corrected in a sibling. |
| extra | 279-296 | Limits renumbered (the list had two entries numbered 3). |

## `data/analysis/cosine_scale.md`

| item | line(s) | what changed |
|---|---|---|
| 12 | 20, 24-28 | `anti_top` behavioural Δ −0.62 struck in the result table and marked **WITHDRAWN 2026-08-27**; its cosine figures explicitly kept as measurements. |
| 13 | 29-34 | **Added** to the Reading: the two injection rows are **algebra, not measurement**, computed as `base + α·d̂` with no model call, read out at the layer the vector was added to, so cos = 0.7135 is definitional. Noted that caveat 3 already said this while the headline was built on it anyway. |
| 14 | 59-77 | The per-unit comparison scoped: it runs **across** the prefix and injection families, which is the comparison `compile_check.md` shows is invalid. Bare "35×" replaced by the range **36 to 39× per unit cosine**, stable across all three estimators (35.8 / 36.8 / 39.1). What stands without it: the 19.9× cosine ratio and the 6/6 ordering inversion. |
| 15 | 97-107 | **Added** as caveat 5. Note: the "20× outside anything any of 617 submissions reached" sentence **is not in this file** (see conflicts below). The caveat records the point anyway: any such statement is uncheckable from this repo because there is no per-submission export and this document measured 3 arms. The 19.9× ratio verifies and stands, scoped to those three. |
| 16 | 9-11 | **Added** the provenance note: created 2026-08-27, after the withdrawal. |

## `data/analysis/steering_dose.md`

| item | line(s) | what changed |
|---|---|---|
| 17 | 100-121 | Limit 4 struck in full and marked **CORRECTED**. `data/cache/behavioral/` shows base, +0.5, +1, −0.5, −1 all generated 2026-06-10, 50 files each, so the comparison is entirely within one run and there is no model-build boundary. The real June/August split is in the random control; the caveat was **misfiled** and has been moved to `steering_random_control.md`. Failure mode named: caveats were not verified against artifacts the way results were. |
| 18 | 86-99 | Limit 2 corrected. Human blind ratings from `data/analysis/behavioral_blind.csv` reported for the first time: `+0.5` 2/2/6 (p=1.00, 4 decided), `+1.0` 5/3/2 (p=0.73), `−0.5` 4/8/1 (p=0.39), `−1.0` 2/6/1 (p=0.29). All null, all underpowered. `+1·d` agrees directionally with the LLM judges; `−1·d` does not agree with "inside the random band" but carries no weight at 8 decided pairs. |
| 19 | 60-72 | **Added** the inheritance note: this document does **not** carry the withdrawn `anti_top` number (verified independently). It inherits only ≈58× and the positive-side r = +0.99, both corrected in place here against `compile_check.md` (roughly 60-fold, 50 to 75; r = +0.986 over 6 arms as an ordering result). |
| 20 | 123-149 | **Added** the field-practice tension and its resolution: published CAA-style practice steers **during generation** at ≈0.1× the normalised residual norm as a ceiling, one GLM-5 reproduction at 0.025×, against this project's 1.0× **prefill-only**. Resolution stated mechanistically (a prefill edit does not compound across decoding), giving prefill-only steering 10× to 40× more magnitude headroom, still climbing at 1.0·‖R‖. "Under-applied" scoped to prefill-only. The 0.1× and 0.025× figures marked **second-hand via a LessWrong reproduction, therefore unverified**. |
| extra | 75-79 | The inherited "`−1·d` sits inside the random band under both judges" qualified as estimator-dependent, pointing at `steering_random_control.md`. |

## `data/analysis/steering_random_control.md`

| item | line(s) | what changed |
|---|---|---|
| 21 | 37-52 | Setup's "3 random unit vectors" struck and corrected to **8** (`steering_random_control.json`, `setup.n_dirs = 8`), with all eight `|cos(random_i, d)|` values tabulated: 0.0144, 0.0057, 0.0012, 0.0032, 0.0053, 0.0005, 0.0014, 0.0021. Internal inconsistency, never previously flagged. |
| 21 | 80-84, 209-212 | Scope notes: the distinct-4 / looping results table covers **the first 3 of the 8 draws**, and the "all three random arms cluster near zero" reading was written against those 3 and holds against all 8. |
| 22 | 55-70 | **Added** the June/August provenance caveat moved here from `steering_dose.md`: base and ±1 from 2026-06-10, all eight random arms from 2026-08-26. Mitigating evidence added and cited for the first time in any document: α recomputed fresh in August came back **bit-identical at 30.07038116455078**, evidence the build did not change. Scoped as evidence, not proof, and not covering the judge. |
| 23 | 148-172 | **Added** a fixed-baseline sensitivity section. `+1·d` deepseek +0.355 and claude +0.376, above all 8 randoms under both judges, so the positive result **survives**. `−1·d` deepseek −0.135 at 0/8 below (outside the band) and claude +0.026 at 2/8 below (inside), so "`−1·d` is not distinguishable from a random direction" is stated as **estimator-dependent**. |
| 24 | 141-147 | **Corrected**: claude's `−1·d` "5 of 8 draws below / essentially at the median" is the **floating**-baseline placement. Under the fixed baseline it is **2 below, 1 tied, 5 above**, the lower third. Conclusion unchanged; the "median" rhetoric withdrawn. |
| 25 | 174-190 | **Added** two statistical caveats. (a) A fixed baseline removes between-arm base variance from the null, so the null sd shrinks mechanically (claude 0.110 to 0.040) and z inflates: under claude the point estimate **falls 30%** while z **rises 77%**. Use the placement count, not z. (b) But the placement count over 8 draws floors at one-sided p ≈ 1/9, so it cannot separate z = +3.6 from z = +8.4. Robust and nearly powerless; more draws is the fix. |
| 26 | 244-267 | **Added** the null-choice limitation, crediting the prior work rather than claiming it. An isotropic Gaussian draw in 5120 dimensions is near-orthogonal to `d` **and to every other concept direction**, exemplified by our own 0.0144 / 0.0057 / 0.0012, so this control tests `d` against **noise** and cannot test it against other meaningful directions. Cites SteerCheck arXiv:2608.24335 for the argument and for the better controls (PCA-subspace and sign-randomised nulls under a matched KL budget), and Hewitt and Liang 2019 control tasks as the older precedent. States fairly that SteerCheck also calls norm-matched random controls "the standard evidence for attribution", so what was done is correct standard practice; the limitation is on what it can establish. |
| 27 | 269-285 | **Added** the pre-registration lesson, honestly scoped: the pre-registration named outcome (C) as the most damaging threat **in advance**, and the instrument then chosen could not test it. Transferable lesson stated as: pre-registering the **threat** does not pre-register the **instrument's power** against it. |
| 28 | 100-116 | **Added** the 42 human blind ratings from `behavioral_blind.csv` here as well (same numbers as item 18), since this file carries the ±1 claims. |
| extra | 238-242 | Added a fourth entry to "Corrections owed to earlier write-ups", listing the corrections this document owed itself. |
| extra | 288-303 | Limits rewritten as a numbered list: "three random draws" struck and corrected to eight, "the judge is a single rater" struck as stale (both claim-bearing arms and all eight randoms are cross-judged), plus new limits for the June/August split, the estimator-dependent `−1·d` placement, and the null family.

---

## Not applied, and why

**Item 15, first half, cannot be applied as written.** The sentence "20× outside anything any of
617 submissions reached" **does not appear in `cosine_scale.md`**, and "617" appears nowhere in
`data/analysis/*.md`. It is the audit's own paraphrase of that document's "The injection moves it
~20× further" (line 28), which is a ratio against `pro_top`, not against the submission
distribution. Nothing could be struck. The point was applied instead as a new caveat 5 recording
that any such claim is uncheckable from this repo, that this document measured 3 arms, and that
the 19.9× ratio itself verifies and stands. The unmodified "~20× further" sentence is correct as
written, so it was left alone.

## Where two instructions pulled against each other

**1. "No em dashes anywhere" against "keep the original claim visible".** The four documents use
em dashes as house punctuation, and several corrections quote the original sentence verbatim.
Resolution: **no em dash appears in any newly drafted text**; the one em dash inside added lines
is inside a verbatim quotation of a published sentence being retracted
(`compile_check.md`: "*The small on-`d` component is not incidental — it is the best single
predictor...*"). Altering punctuation inside a quote would have made it a paraphrase, which
defeats the purpose of quoting it. Pre-existing em dashes in untouched text were left as they
were.

**2. Item 3 against item 2.** Item 2 supplies r = +0.986 as the corrected headline; item 3 says
that same number must not be read as a dose-response. Resolution: the number is reported in item
2's correction and immediately reframed in item 3's, so the corrected r never stands unqualified.
The word "dose-response" is retained only inside quotations of the retracted claim.

**3. Item 19 against items 2 and 4.** Item 19 says `steering_dose.md` inherits the positive-side
r and the 58× ratio, "both of which are separately corrected above" (i.e. in `compile_check.md`).
Read literally that leaves the wrong numbers standing in `steering_dose.md` with only a pointer.
Resolution: both were corrected **in place** in `steering_dose.md` as well, in a table that keeps
the printed value beside the corrected one and cites `compile_check.md`. Leaving a known-wrong
figure in a second document with a cross-reference is the exact propagation failure these fixes
exist to stop.

**4. Scope discipline against not propagating corrected claims.** Three claims corrected in one
file are asserted unqualified in another of my four (`−1·d` "inside the random band" in
`compile_check.md` and `steering_dose.md`; the 3-vs-8 draw count in two more places in
`steering_random_control.md`). These were not on the instruction list. Resolution: added
one-sentence dated qualifiers with pointers, marked `extra` in the tables above, rather than
rewriting the surrounding arguments.
