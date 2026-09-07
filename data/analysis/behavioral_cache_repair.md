# A cache key that separated two experiments, and a record that did not

`python scripts/repair_behavioral_cache.py` → `data/analysis/behavioral_cache_repair.json`.
No GPU, no NDIF, $0: it rewrites 30 metadata labels in `data/cache/behavioral/`. No
generation was regenerated and no text was altered.

`REVISIONS_2026-09-05.md` is authoritative on what results are now taken to mean; this page
is authoritative on what this artifact contains.

**Found while establishing a baseline for something else.** `_falsifier/verify.py` reports
`{PASS: 212, FIXED: 4, UNCHECKABLE: 2}` in its committed `verify_result.json`. Re-run
against the working tree on 2026-09-07 it returned **5 FAIL** and exit code 1. None of the
five was caused by the work in progress; all five traced to the same corrupted cache.

## The bug

`scripts/behavioral_eval.py` keys a cached generation on
**(model, layer, prompt, arm, alpha, max_new, d_tag)** and, until this repair, wrote a
record holding only **{prompt, arm, alpha, text}**.

The key separates experiments. The record does not. Every consumer groups by the record's
`arm` — `scripts/coherence_confound.py:load_generations` does
`inj[d["arm"]][d["prompt"]] = ...` — so two experiments that share an arm name are
indistinguishable downstream, and the later file silently wins in filesystem glob order.

## What happened

The Season-3 causal spot check (`season3_causal_spotcheck.md`: layer 27, 10 prompts × 3
arms) ran on 2026-09-06 and wrote **30** records into `data/cache/behavioral/` under the arm
names `+1`, `-1` and `base` — the names the June 2026 **layer-24** dose experiment uses.

| | June 2026, layer 24 | Season 3, layer 27 |
|---|---|---|
| `+1` / `-1` alpha | ±**30.07038116455078** | ±**39.22039031982422** |
| prompts per arm | 50 | 10 |
| records | 150 across `+1`/`-1`/`base` | 30 |

**20 of the 30 shadowed live data**: 10 of the 50 June `+1` prompts and 10 of the 50 June
`-1` prompts were replaced in every consumer's view. That is 20% of each arm — which is why
the affected numbers *drifted* rather than moving cleanly.

The remaining 10 are `base` at alpha 0.0, and they are **byte-identical** to their June
counterparts for the same prompts (10 identical, 0 different). That is a small positive
finding on its own: an unsteered generation at the same prompt and token budget reproduced
exactly across a three-month gap, so the model build did not drift between June and
September.

## What it broke

| check | expected | with the pollution | after the repair |
|---|---|---|---|
| `C-D4-plus1d` | 0.872 | **0.86952** | **0.87218494** |
| `C-PEARSON-deepseek` | 0.0175 | **0.00670** | **0.01747027** |
| `A-MTIME-SPLIT` | `dose_and_base_dates: ['2026-06-10']` | `['2026-06-10', '2026-09-06']` | `['2026-06-10']` |
| `A-DOSE-LIMIT4` | all five arms share one date | False | True |
| `A-ALPHA-BITIDENTICAL` | one alpha in the June `+1` cache | two | one |

The first two are **published numbers**: `coherence_confound.md`'s distinct-4 for `plus1d`
and its DeepSeek Pearson correlation. They stopped reproducing from their own artifact, in
a gitignored directory, with no diff anywhere to show it.

## The repair

Re-tag the 30 records with a run-specific arm name and write the provenance the record
should have carried. The generations are untouched; only the label saying which experiment
they belong to changes.

```
+1   → +1@L27     (10 records)
-1   → -1@L27     (10 records)
base → base@L27   (10 records)
```

Each rewritten record gains `original_arm`, `layer`, `layer_attribution`, `source`,
`retagged_on` and `retagged_by`, so the change is auditable and reversible. The script is
idempotent — it skips any record that already carries `original_arm`.

**How the two experiments are told apart.** Alpha, which is in the record, is the hard
discriminator for the steered arms: |alpha| 39.22039031982422 is Season 3, 30.07038116455078
is June. `base` carries alpha 0.0 in both and is identified by mtime instead, which is why
this repair is dated rather than re-derivable from content alone. **Layer 27 is an
attribution, not a measurement here** — it comes from `season3_causal_spotcheck.md` naming
layer 27 with 10 prompts × 3 arms on the same date, and the counts match exactly.

**Filenames are unchanged.** Recomputing a cache key needs `max_new` and `d_tag`, which the
thin records never stored — itself part of the same bug. So a filename still encodes the
original arm while its body carries the corrected one. Nothing reads an arm out of a
filename; the only use of the name is an existence check.

## Verification

`python3 _falsifier/verify.py` returns **`{PASS: 212, FIXED: 4, UNCHECKABLE: 2}` and exit
code 0** — all 218 records with **statuses identical to the committed
`verify_result.json`**, checked programmatically rather than by eye. The two published
numbers reproduce: 0.87218494 against a published 0.872, and 0.01747027 against 0.0175.

## The source fix

`scripts/behavioral_eval.py` now writes `layer`, `d_tag` and `model_id` into every record,
so the record carries what the key already distinguished on. A future experiment that
re-uses an arm name at a different depth will be separable downstream without a dated
repair script.

## What this buys

The published numbers reproduce again, and `verify.py` is back to a clean exit — which
matters more than the two decimals: a suite that exits 1 for a stale reason stops being read,
and this one had been red since 2026-09-06. It also demonstrates the mechanism the project
already relies on. Nobody noticed by reading; the checks noticed, on a cache that git cannot
see, and named the two affected numbers precisely enough to repair rather than re-run. The
whole fix cost 30 metadata rewrites and no compute.

## Limits

1. **The `base` identification is by mtime**, so it is not reproducible from content. If the
   30 files are ever copied without preserving timestamps, the `base` third cannot be
   re-identified this way. The `original_arm` field now makes that moot going forward.
2. **Layer 27 is attributed, not measured.** The alpha value 39.22039031982422 is consistent
   with `1.0 × ‖R‖` at a deeper layer than 24 (whose ‖R‖ is 29.763,
   `normalization_check.json`), but no forward pass was run here to confirm ‖R‖ at layer 27.
3. **Filenames still disagree with their contents** on the arm field, by the deliberate
   choice above. A reader who greps filenames rather than record bodies will be misled.
4. **This does not audit the rest of the cache.** Only `+1`, `-1` and `base` were checked
   for cross-experiment collision. Other arm names — `ablate`, `ablate_meandiff`,
   `ablate2x`, the eight `rand*s20260826` and two `ablate_rand*s20260828` — were not, and
   the same class of collision is possible in any of them.
5. Nothing here revisits whether `coherence_confound.md`'s conclusions are *right*; it only
   restores their reproducibility from the artifact.

## Cross-links

`season3_causal_spotcheck.md` (the run that wrote the records) · `coherence_confound.md`
(the two published numbers) · `_falsifier/verify.py` and `verify_result.json` (the checks
that caught it) · `normalization_check.json` (the June alpha) · `meandiff_ablation.md`,
whose own write-up records avoiding this exact trap by naming its arm `ablate_meandiff`
rather than reusing `ablate` — the precedent that should have been followed here.
