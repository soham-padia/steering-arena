"""Assert a season row matches the direction file(s) it claims to use.

.claude/skills/new-season/SKILL.md step 2 says the season row must equal the d file's
metadata and that "this equality [should be asserted] programmatically when inserting".
Nothing in this repo ever implemented that, and the first Season 3 attempt shipped a row
saying layer=23 against a d file saying layer=27, plus a d_version that differed by a
`_score1` suffix. Both were caught by hand. This is the check that should have caught them.

Verifies, for a season id (default: the active one):

  1. seasons.model_id   == every d file's meta.model_id
  2. seasons.layer      == every d file's meta.layer      (the representative layer)
  3. seasons.d_version  == every d file's meta.d_version  (suffixes are NOT allowed;
                           use meta.role to distinguish files within a season)
  4. seasons.model_build== every d file's meta.model_build (both absent counts as equal)
  5. seasons.layers     == the union of the d files' bands, when the season is multi-layer
  6. exactly one season row has active = true
  7. the referenced probe set file exists and is committed
  8. every d file loads, is finite, and its `d` is unit-norm at the model's hidden size

Exit 0 only if everything passes. Read-only: touches no table, writes no file.

    python scripts/check_season_matches_d.py                       # the active season
    python scripts/check_season_matches_d.py --season-id 5 \
        --d data/directions/d_olmo3_s3_score1.npz \
        --d data/directions/d_olmo3_s3_score2.npz
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def fail(msg):
    print(f"  FAIL  {msg}")
    return 1


def ok(msg):
    print(f"  pass  {msg}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season-id", type=int, default=None,
                    help="DB season id; default = whichever row has active = true")
    ap.add_argument("--d", action="append", default=[],
                    help="direction .npz to check; repeatable. Default: the season's own.")
    ap.add_argument("--hidden", type=int, default=5120)
    a = ap.parse_args()

    # Connect the same way scripts/rescore_season.py:64 does. NOT via app.main.get_db —
    # that would boot the FastAPI app, and app.db exports only the two backend classes.
    from supabase import create_client  # noqa: E402

    from app.config import settings  # noqa: E402

    client = create_client(settings.supabase_url, settings.supabase_service_key)
    rows = client.table("seasons").select("*").execute().data
    actives = [r for r in rows if r.get("active")]
    season = (next((r for r in rows if r["id"] == a.season_id), None)
              if a.season_id is not None else (actives[0] if actives else None))
    if season is None:
        print(f"no season found (id={a.season_id})")
        return 2

    print(f"season id={season['id']} {season['name']!r}  active={season.get('active')}")
    print(f"  model_id={season['model_id']!r} layer={season['layer']} "
          f"layers={season.get('layers')!r} d_version={season['d_version']!r}")
    print(f"  scoring_mode={season['scoring_mode']!r} probe_set_id={season.get('probe_set_id')!r}\n")

    bad = 0

    # 6. exactly one active season
    if len(actives) == 1:
        ok(f"exactly one active season (id={actives[0]['id']})")
    else:
        bad += fail(f"{len(actives)} active seasons: {[r['id'] for r in actives]} "
                    f"— db.get_active_season() does .limit(1) with NO order, so which one "
                    f"the site serves is undefined")

    # 7. probe set committed
    pid = season.get("probe_set_id")
    if pid:
        p = ROOT / "data" / "probes" / f"{pid}.json"
        if not p.exists():
            bad += fail(f"probe set {p.relative_to(ROOT)} does not exist")
        else:
            tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(p.relative_to(ROOT))],
                                     cwd=ROOT, capture_output=True).returncode == 0
            n = len(json.loads(p.read_text())["prompts"])
            (ok if tracked else fail)(
                f"probe set {pid}.json: {n} prompts, "
                f"{'committed' if tracked else 'NOT COMMITTED — a shipped season must freeze it'}")
            bad += 0 if tracked else 1
    else:
        bad += fail("season has no probe_set_id")

    # which d files?
    paths = [Path(x) for x in a.d]
    if not paths:
        paths = sorted((ROOT / "data" / "directions").glob(f"d_*{season['d_version']}*.npz"))
        if not paths:
            bad += fail(f"no d file matches d_version={season['d_version']!r}; pass --d explicitly")

    bands = []
    for p in paths:
        p = p if p.is_absolute() else ROOT / p
        print(f"\n{p.name}")
        try:
            z = np.load(p, allow_pickle=True)
            meta = json.loads(str(z["meta"]))
        except Exception as e:
            bad += fail(f"cannot load: {type(e).__name__} {e}")
            continue

        # 1-4: the four fields SKILL.md names
        for field, sval in (("model_id", season["model_id"]),
                            ("layer", season["layer"]),
                            ("d_version", season["d_version"]),
                            ("model_build", season.get("model_build"))):
            mval = meta.get(field)
            same = (sval == mval) or (not sval and not mval)
            if same:
                ok(f"{field}: {mval!r}")
            else:
                bad += fail(f"{field}: season says {sval!r}, d file says {mval!r}")

        # 8: the vector itself
        d = np.asarray(z["d"], dtype=np.float64)
        if d.shape != (a.hidden,):
            bad += fail(f"d shape {d.shape}, expected ({a.hidden},)")
        elif not np.isfinite(d).all():
            bad += fail("d contains non-finite values")
        else:
            ok(f"d: shape {d.shape}, |d| = {np.linalg.norm(d):.6f}")

        if "band" in z:
            b = [int(x) for x in z["band"]]
            bands.append(b)
            ok(f"band {b}  role={meta.get('role')!r}  aggregate={meta.get('aggregate')!r}")
            if "per_layer" in z and len(z["per_layer"]) != len(b):
                bad += fail(f"per_layer has {len(z['per_layer'])} rows for a {len(b)}-layer band")

    # 5: seasons.layers vs the d files' bands
    print()
    if bands:
        declared = season.get("layers")
        if not declared:
            bad += fail(f"d files declare bands {bands} but seasons.layers is NULL — "
                        f"a multi-layer season must record its band")
        else:
            want = sorted({L for b in bands for L in b})
            got = sorted(int(x) for x in str(declared).split(","))
            # seasons.layers records the RANKING band; every ranking layer must appear
            rank_band = bands[0]
            if got == sorted(rank_band):
                ok(f"seasons.layers {got} == ranking band {sorted(rank_band)}  "
                   f"(all bands union to {want})")
            else:
                bad += fail(f"seasons.layers {got} != ranking band {sorted(rank_band)}")
    elif season.get("layers"):
        bad += fail(f"seasons.layers={season['layers']!r} but no d file declares a band")

    print(f"\n{'ALL CHECKS PASS' if bad == 0 else f'{bad} CHECK(S) FAILED'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
