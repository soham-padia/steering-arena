"""Repair the Season-2 behavioural cache after the Season-3 spot check wrote into it.

THE BUG. `scripts/behavioral_eval.py` keys a cached generation on
(model, layer, prompt, arm, alpha, max_new, d_tag) but writes a RECORD holding only
{prompt, arm, alpha, text}. The key separates experiments; the record does not. So any
consumer that groups by the record's `arm` -- which is all of them -- cannot tell two
layers apart.

WHAT HAPPENED. The Season-3 causal spot check (`data/analysis/season3_causal_spotcheck.md`:
layer 27, 10 prompts x 3 arms) ran on 2026-09-06 and wrote 30 records into
`data/cache/behavioral/` under the arm names `+1`, `-1` and `base` -- the same names the
June 2026 layer-24 dose experiment uses. `scripts/coherence_confound.py:load_generations`
does `inj[arm][prompt] = ...`, so for the 20 overlapping (arm, prompt) pairs the Season-3
record REPLACES the June one in filesystem glob order. That is 10 of 50 prompts in each of
`+1` and `-1`.

CONSEQUENCES, all measured before this script ran:

  * `data/analysis/coherence_confound.md`'s published distinct-4 for `plus1d` no longer
    recomputes: 0.872 -> 0.86952. Its deepseek Pearson: +0.0175 -> +0.00670.
  * `_falsifier/verify.py` went from 0 FAIL to 5 FAIL -- C-D4-plus1d, C-PEARSON-deepseek,
    and the three audit checks that read the cache directly (A-MTIME-SPLIT, A-DOSE-LIMIT4,
    A-ALPHA-BITIDENTICAL).

THE REPAIR. Re-tag the 30 Season-3 records with a run-specific arm name and write the
provenance the record should have carried in the first place. The generations themselves
are not touched -- only the label that says which experiment they belong to.

Discrimination is by ALPHA, which is in the record: the June layer-24 arms use
|alpha| = 30.07038116455078 and the Season-3 layer-27 arms use 39.22039031982422. The
layer attribution (27) comes from `season3_causal_spotcheck.md` -- same date, same arm
triple, exactly 10 prompts per arm -- and is recorded as an attribution, not measured
here. `base` records carry alpha 0.0 and so cannot be told apart by alpha; they are
identified by mtime and by being exactly the 10 prompts the spot check used.

Filenames are NOT changed. Recomputing a key needs `max_new` and `d_tag`, which the thin
records never stored -- itself part of the same bug. So a filename still encodes the
original arm while its body carries the corrected one. Nothing reads the arm out of a
filename; the only use of the name is an existence check.

Idempotent, and reversible: every rewritten record keeps `original_arm`, and the manifest
lists every file touched.

    python scripts/repair_behavioral_cache.py --dry-run
    python scripts/repair_behavioral_cache.py

Writes data/analysis/behavioral_cache_repair.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache" / "behavioral"
OUT = ROOT / "data" / "analysis" / "behavioral_cache_repair.json"

# The June 2026 layer-24 dose experiment. Anything at this |alpha| is Season 2 and is left
# alone. Sourced from data/analysis/normalization_check.json (+1.0*d, on-d displacement).
SEASON2_ALPHA = 30.07038116455078
# The Season-3 spot check's alpha, and the layer attributed to it.
S3_ALPHA = 39.22039031982422
S3_LAYER = 27
S3_SOURCE = "season3_causal_spotcheck"
DOSE_ARMS = {"+1", "-1", "base"}
CUTOVER = dt.date(2026, 9, 6)


def classify(rec: dict, mtime: dt.date) -> str | None:
    """'s3' if this record belongs to the Season-3 spot check, else None.

    Alpha is the hard discriminator for the steered arms. `base` has alpha 0.0 in both
    experiments, so it falls back to the date -- which is why the repair is dated and not
    re-derivable from content alone.
    """
    if rec.get("original_arm"):
        return None                       # already repaired
    arm, alpha = rec.get("arm"), rec.get("alpha")
    if arm not in DOSE_ARMS:
        return None
    if alpha is not None and abs(alpha) == S3_ALPHA:
        return "s3"
    if alpha == 0.0 and mtime >= CUTOVER:
        return "s3"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    touched, skipped, by_arm = [], 0, {}
    for fp in sorted(CACHE.glob("*.json")):
        rec = json.loads(fp.read_text())
        mtime = dt.date.fromtimestamp(fp.stat().st_mtime)
        if classify(rec, mtime) != "s3":
            skipped += 1
            continue
        old_arm = rec["arm"]
        new_arm = f"{old_arm}@L{S3_LAYER}"
        rec.update({
            "arm": new_arm,
            "original_arm": old_arm,
            "layer": S3_LAYER,
            "layer_attribution": (f"from {S3_SOURCE}.md (layer {S3_LAYER}, 10 prompts x 3 "
                                  "arms, same date); alpha is the measured discriminator "
                                  "for the steered arms, mtime for base"),
            "source": S3_SOURCE,
            "retagged_on": "2026-09-07",
            "retagged_by": "scripts/repair_behavioral_cache.py",
        })
        by_arm[new_arm] = by_arm.get(new_arm, 0) + 1
        touched.append({"file": fp.name, "original_arm": old_arm, "arm": new_arm,
                        "alpha": rec.get("alpha"), "prompt": rec["prompt"][:60],
                        "mtime": str(mtime)})
        if not a.dry_run:
            fp.write_text(json.dumps(rec))

    print(f"{'would re-tag' if a.dry_run else 're-tagged'} {len(touched)} record(s); "
          f"{skipped} left alone")
    for arm, n in sorted(by_arm.items()):
        print(f"    {arm}: {n}")
    if a.dry_run:
        return

    manifest = {
        "what": ("re-tagged the Season-3 causal spot check's generations, which were "
                 "written into the Season-2 behavioural cache under the same arm names"),
        "why": ("behavioral_eval.py keys on (model, layer, prompt, arm, alpha, max_new, "
                "d_tag) but recorded only {prompt, arm, alpha, text}; every consumer "
                "groups by the record's arm and so mixed two layers"),
        "season2_alpha": SEASON2_ALPHA, "s3_alpha": S3_ALPHA, "s3_layer": S3_LAYER,
        "layer_is_attributed_not_measured": True,
        "filenames_unchanged": ("recomputing a cache key needs max_new and d_tag, which "
                                "the thin records never stored"),
        "n_retagged": len(touched), "n_untouched": skipped,
        "by_new_arm": by_arm, "records": touched,
        "schema_fix": ("scripts/behavioral_eval.py now writes layer, d_tag and model_id "
                       "into every record, so this cannot recur"),
    }
    OUT.write_text(json.dumps(manifest, indent=1))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    print("next: python3 _falsifier/verify.py   (expect the 5 FAILs to clear)")


if __name__ == "__main__":
    main()
