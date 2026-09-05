"""Is `d` pro-human, or is it riding the approach/assertiveness confound?

The shipped direction's audit file records cos(d, approach) = 0.1501, which is the largest
of the three audited confounds by two orders of magnitude - length and valence sit at 0.0011
and 0.0003 because they were orthogonalised out during extraction. Approach was NEVER
removed, only measured. So the question is whether that 0.15 is load-bearing.

A cosine cannot answer it. Two things can, and both are runnable from cache:

  ABLATE IT. Project the approach direction out of `d` and re-run every gate. If the
  direction was riding approach, separation and the control-pair test should degrade.

  RUN THE CONFOUND ON ITS OWN. Score the seed pairs with the approach direction ALONE. If
  approach by itself does well, the training data is confounded even when the fitted
  direction is not, which is a different and still-real problem.

The value-flip control pairs are what discriminate: kind vs cruel where BOTH sides are
active and assertive, so approach is held constant and only human impact flips.

    python scripts/direction_purity.py

Zero NDIF calls. Writes data/analysis/direction_purity.json.
"""
import argparse
import ast
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ACTS = ROOT / "data" / "cache" / "acts" / "allenai_Olmo-3-1125-32B"
MODEL_ID = "allenai/Olmo-3-1125-32B"
LAYER = 24


def unit(v, axis=-1):
    return v / np.linalg.norm(v, axis=axis, keepdims=True)


def load(text):
    k = hashlib.sha256(f"{MODEL_ID}\x00L{LAYER}\x00{text}".encode()).hexdigest()
    fp = ACTS / f"{k}.npy"
    return np.load(fp).astype(np.float64) if fp.exists() else None


def consts():
    """Pull the audit's text lists without importing it (it needs app.config + env)."""
    out = {}
    for n in ast.parse((ROOT / "scripts" / "confound_audit.py").read_text()).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
            try:
                out[n.targets[0].id] = ast.literal_eval(n.value)
            except Exception:
                pass
    return out


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    np.seterr(all="ignore")
    warnings.filterwarnings("ignore")
    C = consts()
    APPROACH, AVOID, CP = C["APPROACH"], C["AVOID"], C["CONTROL_PAIRS"]
    need = APPROACH + AVOID + [x for p in CP for x in p]
    if any(load(t) is None for t in need):
        raise SystemExit("some audit texts are not cached; run confound_audit.py first")

    approach = unit(np.stack([load(t) for t in APPROACH]).mean(0)
                    - np.stack([load(t) for t in AVOID]).mean(0))
    kind = np.stack([load(k) for k, _ in CP])
    cruel = np.stack([load(c) for _, c in CP])

    z = np.load(ROOT / "data/directions/d_olmo3_L24_logistic.npz")
    key = [x for x in z.files if z[x].ndim == 1 and z[x].size > 1000][0]
    d = unit(z[key].astype(np.float64))
    d_perp = unit(d - float(d @ approach) * approach)

    rows = [json.loads(l) for l in open(ROOT / "data" / "seed_pairs.jsonl")]
    ch = np.stack([load(f"{r['prompt']} {r['chosen']}") for r in rows])
    rj = np.stack([load(f"{r['prompt']} {r['rejected']}") for r in rows])
    val = np.random.default_rng(0).permutation(len(rows))[:round(len(rows) * .25)]

    out = {"cos_d_approach": round(float(d @ approach), 4),
           "cos_d_dperp": round(float(d @ d_perp), 4),
           "n_control_pairs": len(CP), "variants": {}}
    print(f"cos(d, approach) = {float(d @ approach):+.4f}\n")
    print(f"{'direction':<26}{'held-out':>10}{'all-135':>9}{'kind>cruel':>12}{'gap':>9}")
    for nm, u in (("d (shipped)", d), ("d, approach removed", d_perp),
                  ("approach alone", approach)):
        r = {"held_out_sep": round(float(np.mean((unit(ch[val]) @ u) > (unit(rj[val]) @ u))), 4),
             "all_135_sep": round(float(np.mean((unit(ch) @ u) > (unit(rj) @ u))), 4),
             "control_kind_higher": round(float(np.mean((unit(kind) @ u) > (unit(cruel) @ u))), 4),
             "control_gap": round(float(np.mean(unit(kind) @ u - unit(cruel) @ u)), 4)}
        out["variants"][nm] = r
        print(f"{nm:<26}{r['held_out_sep']:>10.3f}{r['all_135_sep']:>9.3f}"
              f"{r['control_kind_higher']:>11.0%}{r['control_gap']:>+9.4f}")

    p = ROOT / "data" / "analysis" / "direction_purity.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
