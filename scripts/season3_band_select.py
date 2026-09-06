"""Pick the Season 3 band: fit the banded-mean direction for each candidate and compare.

Variant A from banded_direction.py (the only steerable one): fit a logistic probe per
layer on the train split, orthogonalise confounds out at that layer, unit-normalise,
average across the band, orthogonalise once more at the middle layer.

    d_bar = unit(orth(unit(mean_l(unit(orth(probe_l))))))
    score(M) = mean over band of cos(unit(M_l), d_bar)

TWO DIFFERENCES FROM banded_direction.py, both deliberate:

  1. It reads the all-64-layer cache (seedpair_acts_all64_*.npz) instead of the per-text
     .npy cache, which only ever held the five NATIVE layers [16,24,32,40,48]. Every
     candidate band here is made of full_attention layers, none of which are in that set.

  2. It orthogonalises out APPROACH as well as length and sentiment. REVISIONS_2026-09-05
     section 4 records approach as "the only audited confound never orthogonalised out",
     and it is the one that does not decay with depth (min |cos| 0.1175 at L16, never near
     zero). Season 3 requires d to represent pro-human, so it comes out.

Zero NDIF calls, zero GPU. Writes data/analysis/season3_band_select.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WORK = Path("/work/neu/p2026_0037_neu/steering-arena")
ACTS = WORK / "cache" / "seedpair_acts_all64_695903.npz"
CONF = WORK / "cache" / "confound_acts_all64_695948.npz"

CANDIDATES = {
    "spread_15_23_31_39": [15, 23, 31, 39],
    "peak_19_23_27_31": [19, 23, 27, 31],
    "union_15_19_23_27_31_39": [15, 19, 23, 27, 31, 39],
    "wide_11_23_35_47": [11, 23, 35, 47],
    "single_L24_seas2": [24],  # the Season 2 baseline, for reference
}


def unit(v, axis=-1):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.where(n == 0, 1e-12, n)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-split", type=int, default=40, help="repeated train/val splits")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    np.seterr(all="ignore")
    from sklearn.linear_model import LogisticRegression

    z = np.load(ACTS, allow_pickle=True)
    ch_all = z["chosen"].astype(np.float64)
    rj_all = z["rejected"].astype(np.float64)
    ltypes = [str(x) for x in z["layer_types"]]
    N = len(ch_all)

    c = np.load(CONF, allow_pickle=True)

    def confound_dirs(L):
        g = lambda k: c[k][:, L].astype(np.float64).mean(0)
        return {
            "length": unit(g("long") - g("short")),
            "sentiment": unit(g("pos") - g("neg")),
            "approach": unit(g("approach") - g("avoid")),
        }

    def orth(d, L):
        for cu in confound_dirs(L).values():
            d = d - float(d @ cu) * cu
        return d

    def fit_dbar(band, tr):
        per = []
        for L in band:
            X = np.vstack([ch_all[tr, L], rj_all[tr, L]])
            y = np.r_[np.ones(len(tr)), np.zeros(len(tr))]
            w = LogisticRegression(C=0.1, max_iter=4000).fit(X, y).coef_[0]
            per.append(unit(orth(np.asarray(w, dtype=np.float64), L)))
        per = np.stack(per)
        mid = band[len(band) // 2]
        return unit(orth(unit(per.mean(0)), mid)), per

    rng = np.random.default_rng(a.seed)
    splits = []
    for _ in range(a.n_split):
        idx = rng.permutation(N)
        splits.append((idx[round(N * 0.25):], idx[:round(N * 0.25)]))

    out = {"n_pairs": N, "n_split": a.n_split, "bands": {}}
    print(f"{N} pairs | {a.n_split} splits | approach orthogonalised OUT\n")
    print(f"{'band':<26}{'held-out':>10}{'margin':>10}{'cosAppr':>9}"
          f"{'cosVal':>8}{'cosLen':>8}{'min cos(d,per)':>16}")

    for tag, band in CANDIDATES.items():
        assert all(0 <= L < len(ltypes) for L in band)
        accs, margins = [], []
        for tr, val in splits:
            d_bar, _ = fit_dbar(band, tr)
            sc = lambda M: np.mean([unit(M[:, L]) @ d_bar for L in band], axis=0)
            gap = sc(ch_all[val]) - sc(rj_all[val])
            accs.append(float(np.mean(gap > 0)))
            margins.append(float(gap.mean()))
        d_bar, per = fit_dbar(band, np.arange(N))          # full-data direction to ship
        cd = {k: float(np.mean([abs(d_bar @ confound_dirs(L)[k]) for L in band]))
              for k in ("approach", "sentiment", "length")}
        min_cos = float(np.min(per @ d_bar))
        types = sorted({ltypes[L] for L in band})
        rec = {"band": band, "layer_types": types,
               "held_out_separation": round(float(np.mean(accs)), 4),
               "margin": round(float(np.mean(margins)), 5),
               "cos_approach": round(cd["approach"], 4),
               "cos_sentiment": round(cd["sentiment"], 4),
               "cos_length": round(cd["length"], 4),
               "min_cos_dbar_with_member": round(min_cos, 4)}
        out["bands"][tag] = rec
        print(f"{tag:<26}{rec['held_out_separation']:>10.4f}{rec['margin']:>10.5f}"
              f"{rec['cos_approach']:>9.4f}{rec['cos_sentiment']:>8.4f}"
              f"{rec['cos_length']:>8.4f}{rec['min_cos_dbar_with_member']:>16.4f}")

    p = ROOT / "data" / "analysis" / "season3_band_select.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {p}")
    print("\n  min cos(d_bar, member) is the load-bearing column: it says how well the ONE\n"
          "  shipped direction represents its weakest band member. A low value means the\n"
          "  banded mean is really only about some of the layers.")


if __name__ == "__main__":
    main()
