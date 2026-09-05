"""Does normalisation hide part of `d` from the model, and how big is an ablation really?

TWO QUESTIONS, one raised from outside and one that fell out of answering it.

Q1, THE ONE THAT WAS ASKED. Standard representation-engineering advice warns that LayerNorm
subtracts the mean across the hidden dimension, so whatever part of your direction lies
along the all-ones axis is scaled away and the model never reads it. If `d` were largely
all-ones, ablating it would do nothing for a trivial reason.

  It does not apply here. OLMo-3 uses RMSNorm (config carries rms_norm_eps, no centering),
  which divides by a scalar and does not remove the mean. And `d` is nearly orthogonal to
  all-ones anyway, 0.4% to 2.6% depending on layer, so even under a centering norm almost
  nothing would be lost. Clean negative, recorded so nobody re-asks it.

Q2, WHICH MATTERS MORE. Because RMSNorm rescales by a scalar, the next block reads the
DIRECTION of the residual, not its length. So the honest size of an intervention is the
ANGLE it rotates the residual through, not the norm it adds or removes. Measured that way
the ablation is 0.92 degrees, while every intervention that moved behaviour is 27 to 50.

That revises steering_ablation.md, which argued the null "is not a null of magnitude" on the
grounds that the ablated component (max 1.684) exceeds anti_top's on-`d` displacement of
1.49. That compares on-`d` COMPONENTS, but the model does not read the on-`d` component - it
reads the whole normalised vector. By rotation, a prefix moves it ~50 degrees and the
ablation moves it ~1.

    python scripts/normalization_check.py

Zero NDIF calls. Writes data/analysis/normalization_check.json.
"""
import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SWEEP = ROOT / "data" / "cache" / "layer_sweep"
DIRS = ROOT / "data" / "directions"
MODEL_ID = "allenai/Olmo-3-1125-32B"
NATIVE = [16, 24, 32, 40, 48]
L = 24


def unit(v, axis=-1):
    return v / np.linalg.norm(v, axis=axis, keepdims=True)


def load_d(path):
    z = np.load(path, allow_pickle=True)
    k = "d" if "d" in z.files else [x for x in z.files
                                    if z[x].ndim == 1 and z[x].size > 1000][0]
    return unit(np.asarray(z[k], dtype=np.float64))


def batch_key(texts, layer):
    h = hashlib.sha256()
    h.update(f"{MODEL_ID}\x00L{layer}\x00n={len(texts)}".encode())
    for t in texts:
        h.update(b"\x00")
        h.update(t.encode("utf-8"))
    return h.hexdigest()


def angle(X, Y):
    return np.degrees(np.arccos(np.clip((unit(X) * unit(Y)).sum(1), -1, 1)))


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    np.seterr(all="ignore")
    warnings.filterwarnings("ignore")
    prompts = json.loads((ROOT / "data/eval/steering_prompts.json").read_text())["prompts"]
    gallery = json.loads((ROOT / "data/analysis/site_prefixes.json").read_text())["arms"]
    ones = np.ones(5120) / np.sqrt(5120)
    out = {"norm_type": "RMSNorm (no centering)", "cos_with_ones": {}, "interventions": {}}

    def resid(arm):
        pfx = gallery[arm]["sequence"]
        tx = [f"{pfx} {p}" for p in prompts] if pfx else list(prompts)
        fp = SWEEP / f"{batch_key(tx, L)}.npy"
        return np.load(fp).astype(np.float64) if fp.exists() else None

    print("Q1  how much of d lies along the all-ones axis (RMSNorm does NOT remove it)")
    for lay in NATIVE:
        c = float(load_d(DIRS / f"d_olmo3_L{lay}_logistic.npz") @ ones)
        out["cos_with_ones"][f"L{lay}"] = round(c, 4)
        print(f"      d_L{lay:<3} cos(d, ones) = {c:+.4f}   ({abs(c):.2%} of its norm)")
    print("      -> nearly orthogonal; a centering norm would cost almost nothing here.")

    B = resid("base")
    nB = float(np.linalg.norm(B, axis=1).mean())
    print(f"\nQ2  angular size of each intervention at L{L}   (base ||R|| = {nB:.2f})")
    print(f"    {'intervention':<26}{'mean angle':>12}{'max':>9}{'||delta||':>11}{'on-d':>8}")
    rows = []
    for nm, f in (("ablate d, logistic", "d_olmo3_L24_logistic.npz"),
                  ("ablate d, meandiff", "d_olmo3_L24_meandiff.npz")):
        p = DIRS / f
        if not p.exists():
            continue
        d = load_d(p)
        c = B @ d
        rows.append((nm, angle(B, B - c[:, None] * d), np.abs(c), np.abs(c)))
    d24 = load_d(DIRS / "d_olmo3_L24_logistic.npz")
    for nm, al in (("+0.5*d injection", 15.035), ("+1.0*d injection", 30.070)):
        rows.append((nm, angle(B, B + al * d24), np.full(len(B), al), np.full(len(B), al)))
    for arm in ("pro_coherent", "pro_top"):
        R = resid(arm)
        if R is not None:
            rows.append((f"prefix {arm}", angle(B, R), np.linalg.norm(R - B, axis=1),
                         np.abs((R - B) @ d24)))
    for nm, a, dl, od in rows:
        out["interventions"][nm] = {"mean_angle_deg": round(float(a.mean()), 3),
                                    "max_angle_deg": round(float(a.max()), 3),
                                    "mean_delta_norm": round(float(dl.mean()), 3),
                                    "mean_on_d": round(float(od.mean()), 3)}
        print(f"    {nm:<26}{a.mean():>11.2f}°{a.max():>8.1f}°{dl.mean():>11.2f}{od.mean():>8.2f}")
    out["base_resid_norm"] = round(nB, 3)
    p = ROOT / "data" / "analysis" / "normalization_check.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"\n    The ablation rotates the residual under a degree. Everything that moved")
    print(f"    behaviour rotates it by tens of degrees.\n\nwrote {p}")


if __name__ == "__main__":
    main()
