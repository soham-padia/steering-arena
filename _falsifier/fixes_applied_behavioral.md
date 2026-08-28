# Fixes applied: the behavioural documents

**Date:** 2026-08-27
**Scope:** `data/analysis/prefix_eval.md`, `data/analysis/prefix_transfer.md`,
`data/analysis/behavioral_eval.md`, `SESSION_REPORT.md`. No other file was touched.
`compile_check.md`, `cosine_scale.md`, `steering_dose.md` and `steering_random_control.md`
were being fixed concurrently by a second agent and are only referenced here.

**Sources:** `_falsifier/2026-08-27-experiment-vs-hypothesis-audit.md`,
`_falsifier/2026-08-27-addendum-human-ratings.md`, `_falsifier/recompute_result.md`,
`_falsifier/honesty_result.md`, `_falsifier/verify_result.json`, `_advocate/competition.md`,
`_advocate/methodology.md`, `_advocate/mechanism.md`.

**House rule applied throughout:** nothing was deleted. Every falsified claim stays visible,
struck or quoted, next to the corrected value, dated 2026-08-27.

## Manifest

| # | file | line (post-edit) | what changed |
|---|---|---|---|
| 1 | `data/analysis/prefix_eval.md` | 91-95 | "No human ratings yet" struck and corrected: it contradicted this file's own §"Human ratings (n=54)", was written before the human pass and never retracted; 54/150 pairs are human-rated, the other 96 LLM-only. |
| 2 | `data/analysis/prefix_eval.md` | 378-387 | Section heading "norm-matched neutral prefixes" corrected to **score**-matched, original heading quoted in place; ‖Δ‖ 24.25 vs 17.37 and 19.88, a 28% shortfall. |
| 3 | `data/analysis/prefix_eval.md` | 44-50 (pointer), 473-500 (section) | New fixed-baseline correction. Published deltas used a baseline re-rated per arm; deepseek rated the identical 50 base texts 2.77 beside `pro_top` and 3.39 beside `anti_top` (identical 11/50, p=7.1e-07, about 71% of the headline). Corrected: `pro_top` +0.556 deepseek (p=0.0052), +0.796 claude (p=0.00045); `anti_hostile` −1.114. 13-37% shrinkage across all 14 rows, zero sign flips. |
| 4 | `data/analysis/prefix_eval.md` | 500-525 (mechanism), 451-460 (in place) | Drift mechanism is RUN STRUCTURE, not model: claude 0.00 (50/50, p=1.0) batched, +0.19 (p=0.0120) when arms run separately, deepseek +0.62. Signed against the effect: r=−0.869 (p=0.011) deepseek, −0.703 (p=0.078) claude, so it is a contrast effect inflating both poles. "Identical point estimate to two decimals" marked corrected in §"Every arm, both judges" as a coincidence of two different estimators. |
| 5 | `data/analysis/prefix_eval.md` | 527-553 | New judge-calibration section (positive result). Claude 12/15, 9/11, 6/11, 27/37; deepseek 10/12, 6/11, 5/15, 21/38. Fisher p=0.019 deepseek `pro_top` vs `anti_top`; pooled 22/27 (81%) vs 11/26 (42%), p=0.0047; 81% at MT-Bench's >80% human ceiling. Weakness stated: 11-15 decided pairs per cell. |
| 6 | `data/analysis/prefix_eval.md` | 555-589 | New honesty re-judge section, stated in both directions. `pro_top` Δ −0.108 (p=0.55), `pro_coherent` +0.000; NOT evidence of no cost, since 45/50 stems put no honesty at stake; on the 5 live stems `pro_top` −1.00 (n=5) but `control_junk` −1.33 (n=3), so not `+d`-specific. Two exemplars reproduce. Ibrahim et al. arXiv:2507.21919 cited: ours is the expected null. |
| 7 | `data/analysis/prefix_eval.md` | 591-616 | New content-echo test (positive result). Content words 0/50 base vs 2-4/50 prefixed; after excluding every echo pair, human 11/13 (p=0.023), claude 24/34 (p=0.024), deepseek 23/29 (p=0.0023). Soup reads as a compressed ungrammatical affect instruction. |
| 8 | `data/analysis/prefix_eval.md` | 101-112 | Two design caveats added to "What this does NOT establish": the controls do not control for semantic content (37 chars/6 words vs 209/19 with three first names and affect English; `control_text` is Italian); and prompt-set scope (all 50 stems first person, interpersonal friction, kindness live by construction; zero probe overlap so no leakage; `data/probes/season2.json` never generated on). |
| 9 | `data/analysis/prefix_eval.md` | 363-376 | Withdrawal section: the `n` ("no stance") key was used **zero** times across all 96 human ratings, so the no-stance rate is unmeasured and the stance-collapse account rests on the sign inversion (14-3, p=0.0127) plus the rater's verbal report. Withdrawal itself unchanged. Residual blinding channel added: 32/54 pairs share a base continuation, 14 base texts shown two or three times. |
| 10 | `data/analysis/prefix_eval.md` | 113-121 | Largest outstanding threat added to the limits list: Mody et al. arXiv:2607.25907 found a placebo random direction shifted behaviour just as far under GCG. No random-direction control exists for the token-optimisation arm; `control_junk`/`control_text` are hand-written, not optimised. |
| 11 | `data/analysis/prefix_transfer.md` | 33, 39-47 | OLMo `anti_top` cell (−0.86, p=0.0005) struck and marked WITHDRAWN, with the reason (unrankable comparison, blinding failure, human got the opposite sign 14/17). Removed from all inference. |
| 12 | `data/analysis/prefix_transfer.md` | 62-82 | Conclusion 3 corrected: three uncontrolled differences named (base Llama checkpoints; per-model separately extracted directions; re-tokenisation of an OLMo-searched string). Fairness note that the base-checkpoint reading is insufficient alone, since `allenai/Olmo-3-1125-32B` is also base and `pro_coherent` works there (+0.64, p=0.0051). "Answer in French." named as the settling arm. |
| 13 | `data/analysis/prefix_transfer.md` | 52-53 | Conclusion 1 annotated as unaffected by the withdrawal: it rests on the judge-free distinct-4-gram measure. |
| 14 | `data/analysis/behavioral_eval.md` | 38-67 | "Primary evidence: human blind rating (pending)" is no longer pending. The 42 human ratings in `behavioral_blind.csv` reported for the first time: `+0.5` 2/2/6 (p=1.00, 4 decided), `+1.0` 5/3/2 (p=0.73), `−0.5` 4/8/1 (p=0.39), `−1.0` 2/6/1 (p=0.29); none significant, 4-12 decided pairs. States plainly that the designated PRIMARY evidence was collected and never analysed while downstream documents built on these arms from LLM judges alone. |
| 15 | `SESSION_REPORT.md` | 4, 6-13 | "two norm-matched controls" corrected to score-matched, original phrase quoted; ‖Δ‖ 24.25 vs 17.37 and 19.88. Notes that §7's body never used the word, so §7's body was not edited. |
| 16 | `SESSION_REPORT.md` | 88, 90-96 | `anti_top` kindness row struck in place (both judge cells) and marked WITHDRAWN with the reason. |
| 17 | `SESSION_REPORT.md` | 239-250 | "Two independent judges ... identical point estimate to two decimals" corrected: different baseline estimators, so the agreement is a coincidence of two means offset by a constant 0.19. `anti_hostile` itself survives at −1.114 both judges. |
| 18 | `SESSION_REPORT.md` | 505-519 | "the pro board can be described as ordering sequences by how much they prosocially steer OLMo" kept and scoped: the construct is warmth toward people, on a prompt set where warmth is the live axis in all 50 items, judged primarily by LLMs. |
| 19 | `SESSION_REPORT.md` | 607-637 | New §13 "Adversarial review: `_falsifier/` and `_advocate/`", one table per folder explaining each file, plus the note that `python3 _falsifier/verify.py` re-runs the 181 claim checks and exits 1 while failures remain. |

## Not applied, and why

Nothing was refused. Two items were applied in a narrower form than a literal reading would
suggest, both deliberately:

- **Item 14.** The heading `## Primary evidence: human blind rating (pending)` was left
  **byte-identical** rather than rewritten, with the correction block inserted immediately
  below it. Reason: `_falsifier/verify.py` check `A-CITATIONS` asserts that
  `behavioral_eval.md` line 38 equals that exact string. Keeping the line satisfies the house
  rule (the false claim stays visible, marked corrected) and keeps the audit's line citation
  resolving. The block says in its first sentence that the heading is false.
- **Item 15.** `SESSION_REPORT.md` §7's body was not touched, as instructed. Only the line-4
  intro was corrected. The original phrase "norm-matched controls" is quoted inside the
  correction so the audit's citation of it still resolves.

## Conflicts between instructions, and how they were resolved

1. **House style versus `verify.py`'s line-anchored citations.** The house rule requires
   adding a visible correction next to every falsified claim, which necessarily inserts lines.
   `_falsifier/verify.py` `A-CITATIONS` pins four citations to exact line numbers:
   `SESSION_REPORT.md:481` and `:225`, `behavioral_eval.md:38`, `prefix_eval.md:328`. Three of
   the four have now moved (`SESSION_REPORT.md:481` is now 505, `:225` is now 240,
   `prefix_eval.md:328` is now 359); only `behavioral_eval.md:38` was preserved. So
   `A-CITATIONS` will flip to FAIL on the next run. **This is a stale-citation failure, not a
   substantive one**: every quoted sentence is still present and still says what the audit
   says it says. `verify.py` is outside the four files I was permitted to edit, so the line
   numbers were not updated there. Whoever owns `_falsifier/` should re-pin them.
2. **Item 3 says "do not recompute"; the corrected table needs values for arms the
   instruction did not enumerate.** `pro_coherent`'s fixed-baseline values (+0.406 deepseek,
   +0.456 claude) were quoted verbatim from `_falsifier/recompute_result.md` FIX 2a rather
   than derived, so nothing was recomputed. The two `anti_hostile` p-values (9.8e-06 and
   4.1e-06) came from the same table.
3. **Item 10 says "ADD to Limits"; `prefix_eval.md` has no section by that name.** Its
   document-level limits list is `## What this does NOT establish`, which is also where item 8
   directs two bullets. All three went there. The two arm-local `**Limits.**` paragraphs (the
   fourth and fifth arms) were left alone, since the Mody threat is document-level.
4. **Items 3/4 and item 17 correct the same claim in two documents.** The full argument lives
   in `prefix_eval.md` §"Corrected 2026-08-27: the judge baseline floats"; the
   `SESSION_REPORT.md` block and the in-place note in `prefix_eval.md` §"Every arm, both
   judges" are short and point at it, to avoid three divergent copies of one correction.

## Verified as clean

- Zero em dashes in any added line, in all four files.
- Every replacement was made with an exact-match assertion on a unique target string, so no
  edit landed silently in the wrong place.
