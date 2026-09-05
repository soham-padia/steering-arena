"""Three ways to make `d` span layers instead of living at one, fit and compared.

WHY. `d_olmo3_L24_logistic` is a single-layer readout, and layer_sweep_prefix.md showed the
board winner satisfies it at layer 24 and essentially nowhere else (CV 1.73, L24 is 4.69x
the next-best layer). A multi-layer objective is much harder to satisfy at one depth and is
therefore much harder to game. The obvious question is what "multi-layer direction" even
means, since there are three non-equivalent answers:

  A  BANDED MEAN. Average the per-layer unit directions, renormalise. One vector in R^5120.
     Score = mean over the band of cos(R_l, d_bar). Cheap, and still STEERABLE, which B and
     C are not.

  B  CONCATENATED PROBE. Stack the band's residuals into (N, |band|*5120) and fit one
     probe. Strongest in principle. Has a specific trap this script measures: if one layer
     separates best the probe puts its weight there and the "multi-layer" objective quietly
     degenerates back to single-layer, gameable exactly as before. See WEIGHT SHARE below.

  C  PER-LAYER PROBES, MIN AGGREGATE. Score = min over the band of cos(R_l, d_l).
     Adversarially strictest: a string must satisfy EVERY layer, and cannot trade a spike
     at one depth against a deficit at another the way a sum or mean lets it. Not a
     direction, so it cannot be used to steer.

MAGNITUDE-PROOFING. The project scores by cosine so that inflating the residual norm gains
nothing (PROJECT_SPEC §5). B breaks that if you concatenate raw residuals, so every layer
block is unit-normalised before concatenation. B is then a weighted sum of per-layer
cosines, and norm inflation buys nothing in any of the three.

WHY THE MEAN IS UNWEIGHTED. --compare-weights tests uniform against margin-weighted,
inverse-variance and minimax (equalise cos with every member). On the late band it changes
min-cos by 1.6% (0.933 to 0.949) and the arm scores by under 0.0006, so the conclusion is
identical under all four. Weighting only bites on the all-five band, where minimax lifts
min-cos from 0.531 to 0.758 - but it buys that by representing everyone mediocrely instead
of the late layers well, and pro_coherent's signal drops. Band choice dominates weighting by
an order of magnitude, so the decision belongs there and the mean stays plain.

BAND CHOICE IS AN EMPIRICAL RESULT, NOT A KNOB. Pairwise cosine between the five native
directions: {32,40,48} is 0.870 mean / 0.803 min, all five is 0.557 / 0.202. L16 is the
outlier (0.430 with L24, 0.202 with L48). Averaging across L16 averages over a place where
the representation genuinely changes, so the default band is the late one and --band all
is provided to show the contrast rather than because it is a good idea.

COST: ZERO NDIF CALLS. All 1350 pair activations (135 pairs x 2 sides x 5 layers) are
already in data/cache/acts/, as is every neutral confound text.

    python scripts/banded_direction.py                    # band 32,40,48
    python scripts/banded_direction.py --band all         # all five, for contrast

Writes data/directions/d_olmo3_banded_<tag>.npz (variant A only, the steerable one) and
data/analysis/banded_direction.json.
"""
import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ACTS = ROOT / "data" / "cache" / "acts" / "allenai_Olmo-3-1125-32B"
MODEL_ID = "allenai/Olmo-3-1125-32B"
NATIVE = [16, 24, 32, 40, 48]

# verbatim from extract_direction.py:89-94, so the confound directions are the same ones
LONG = ["The committee reviewed the quarterly schedule and updated the shared calendar for next month accordingly.",
        "She walked to the station, bought a ticket, waited on the platform, and boarded the train toward the city center."]
SHORT = ["The cat slept.", "It rained today."]
POS = ["This is wonderful and delightful.", "What a fantastic, joyful day."]
NEG = ["This is terrible and miserable.", "What an awful, dreadful day."]


def unit(v, axis=-1):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n != 0)


def load(text, layer):
    key = hashlib.sha256(f"{MODEL_ID}\x00L{layer}\x00{text}".encode()).hexdigest()
    fp = ACTS / f"{key}.npy"
    return np.load(fp).astype(np.float64) if fp.exists() else None


def logistic(X, y):
    from sklearn.linear_model import LogisticRegression
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return np.asarray(LogisticRegression(C=0.1, max_iter=4000).fit(X, y).coef_[0],
                          dtype=np.float64)


def orthogonalise(d, layer):
    """Project out length + sentiment at `layer`, same recipe as extract_direction.py:208."""
    nl = [load(t, layer) for t in LONG + SHORT + POS + NEG]
    if any(x is None for x in nl):
        return d, []
    nl = np.stack(nl)
    removed = []
    for name, cd in (("length", nl[:2].mean(0) - nl[2:4].mean(0)),
                     ("sentiment", nl[4:6].mean(0) - nl[6:8].mean(0))):
        cu = unit(cd)
        d = d - float(d @ cu) * cu
        removed.append(name)
    return d, removed


def main():
    # numpy emits spurious divide/overflow/invalid warnings on these matmuls on this
    # platform, a known quirk already documented in compile_check.md. Finiteness is
    # asserted below rather than trusted.
    np.seterr(all="ignore")
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--band", default="32,40,48", help="'all' or comma-separated layers")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--compare-weights", action="store_true",
                    help="test weighted alternatives to the plain mean")
    a = ap.parse_args()
    band = NATIVE if a.band == "all" else [int(x) for x in a.band.split(",")]
    tag = "all" if a.band == "all" else "_".join(str(l) for l in band)

    rows = [json.loads(l) for l in open(ROOT / "data" / "seed_pairs.jsonl")]
    ch = np.stack([[load(f"{r['prompt']} {r['chosen']}", l) for l in band] for r in rows])
    rj = np.stack([[load(f"{r['prompt']} {r['rejected']}", l) for l in band] for r in rows])
    assert np.isfinite(ch).all() and np.isfinite(rj).all()
    N = len(rows)
    idx = np.random.default_rng(a.split_seed).permutation(N)
    nv = max(1, int(round(N * a.val_frac)))
    val, tr = idx[:nv], idx[nv:]
    print(f"band {band}  n={N}  train={len(tr)} val={len(val)}  (zero NDIF calls)")

    out = {"band": band, "n_pairs": N, "n_train": len(tr), "n_val": len(val),
           "split_seed": a.split_seed, "variants": {}}

    # ---- A: banded mean direction -------------------------------------------------
    per = []
    for j, L in enumerate(band):
        d_l, _ = orthogonalise(logistic(np.vstack([ch[tr, j], rj[tr, j]]),
                                        np.r_[np.ones(len(tr)), np.zeros(len(tr))]), L)
        per.append(unit(d_l))
    per = np.stack(per)
    d_bar, removed = orthogonalise(unit(per.mean(0)), band[len(band) // 2])
    d_bar = unit(d_bar)
    sc = lambda M: (unit(M, axis=-1) @ d_bar).mean(axis=-1)      # mean cos over band
    accA = float(np.mean(sc(ch[val]) > sc(rj[val])))
    out["variants"]["A_banded_mean"] = {
        "held_out_separation": round(accA, 4), "confounds_removed": removed,
        "cos_with_each_native": {str(L): round(float(d_bar @ per[j]), 4)
                                 for j, L in enumerate(band)}}

    if a.compare_weights:
        gap = [((unit(ch[val, j]) @ per[j]) - (unit(rj[val, j]) @ per[j])) for j in range(len(band))]
        marg = np.array([g.mean() for g in gap])
        ivar = np.array([1.0 / max(g.var(), 1e-12) for g in gap])
        mid = band[len(band) // 2]
        w, best = np.ones(len(band)) / len(band), (-1.0, None)
        for _ in range(3000):  # equalise cos with every member, on the post-orth objective
            db = unit(orthogonalise(unit((w[:, None] * per).sum(0)), mid)[0])
            c = per @ db
            if c.min() > best[0]:
                best = (float(c.min()), w.copy())
            w = np.clip(w * np.exp(-0.5 * (c - c.mean())), 1e-9, None)
            w /= w.sum()
        out["weight_schemes"] = {}
        print(f"  {'scheme':<9}{'weights':<32}{'cos with each native':<38}{'min':>7}")
        for nm, ww in (("uniform", np.ones(len(band)) / len(band)), ("margin", marg / marg.sum()),
                       ("inv-var", ivar / ivar.sum()), ("minimax", best[1])):
            db = unit(orthogonalise(unit((ww[:, None] * per).sum(0)), mid)[0])
            c = per @ db
            out["weight_schemes"][nm] = {"weights": [round(float(x), 4) for x in ww],
                                        "cos_with_natives": [round(float(x), 4) for x in c],
                                        "min_cos": round(float(c.min()), 4)}
            print(f"  {nm:<9}{str(np.round(ww,3)):<32}{str(np.round(c,3)):<38}{c.min():>7.3f}")

    # ---- B: concatenated probe ----------------------------------------------------
    flat = lambda M: unit(M, axis=-1).reshape(len(M), -1)        # unit-norm each block first
    w = logistic(np.vstack([flat(ch[tr]), flat(rj[tr])]),
                 np.r_[np.ones(len(tr)), np.zeros(len(tr))])
    accB = float(np.mean(flat(ch[val]) @ w > flat(rj[val]) @ w))
    blocks = w.reshape(len(band), -1)
    share = np.linalg.norm(blocks, axis=1) ** 2
    share = share / share.sum()
    out["variants"]["B_concat_probe"] = {
        "held_out_separation": round(accB, 4),
        "weight_share_by_layer": {str(L): round(float(share[j]), 4) for j, L in enumerate(band)},
        "max_share": round(float(share.max()), 4),
        "uniform_share": round(1 / len(band), 4),
        "degenerate": bool(share.max() > 2.0 / len(band))}

    # ---- C: per-layer probes, min aggregate ---------------------------------------
    mn = lambda M: np.min(np.einsum("nlh,lh->nl", unit(M, axis=-1), per), axis=-1)
    accC = float(np.mean(mn(ch[val]) > mn(rj[val])))
    out["variants"]["C_per_layer_min"] = {"held_out_separation": round(accC, 4)}

    print(f"  A banded mean      held-out separation {accA:.3f}   "
          f"cos with natives {[round(float(d_bar @ p), 3) for p in per]}")
    print(f"  B concat probe     held-out separation {accB:.3f}   "
          f"weight share {[round(float(s), 3) for s in share]}"
          f"{'  <-- DEGENERATE' if out['variants']['B_concat_probe']['degenerate'] else ''}")
    print(f"  C per-layer min    held-out separation {accC:.3f}")

    fp = ROOT / "data" / "directions" / f"d_olmo3_banded_{tag}.npz"
    np.savez(fp, d=d_bar.astype(np.float32), band=np.array(band),
             per_layer=per.astype(np.float32))
    p = ROOT / "data" / "analysis" / "banded_direction.json"
    prev = json.loads(p.read_text()) if p.exists() else {}
    prev[tag] = out
    p.write_text(json.dumps(prev, indent=1))
    print(f"wrote {fp}\nwrote {p}")


if __name__ == "__main__":
    main()
