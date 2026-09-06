"""Build and gate the two Season 3 directions.

SCORE 1 (ranks the board) — variant A, banded mean, band [19,23,27,31].
    One shared d_bar. score = mean over band of [cos(R_l, d_bar) - cos(base_l, d_bar)].
    Variant A because it is the only one of the three in banded_direction.py that stays
    STEERABLE, and a causal steering check is a shipping gate for whatever ranks the board.

SCORE 2 (informational) — variant C, per-layer min, band [15,23,31,39].
    One direction per layer. score = mean over probes of [min_l cos(R_l, d_l) - min_l cos(base_l, d_l)].
    Variant C here precisely because Score 2 is never steered, so the constraint that
    forced A on Score 1 does not bind. A min over a spread band is the sharpest available
    test of whether a sequence moves d at every depth or only where it was optimised —
    which is the failure REVISIONS_2026-09-05 section 6 found in the Season 2 winner.

BOTH orthogonalise out length, sentiment AND approach. Approach is the addition:
REVISIONS section 4 records it as "the only audited confound never orthogonalised out",
and the all-64-layer profile showed it is the one confound that does not decay with depth
(min |cos| 0.1175 at L16, never near zero). Season 3 requires d to represent pro-human,
so it comes out by construction rather than by luck.

Reads the all-64-layer caches (the five NATIVE layers the old per-text cache holds are all
sliding_attention and none of these bands use them). Zero NDIF, zero GPU.

    python scripts/build_season3_directions.py

Writes data/directions/d_olmo3_s3_score{1,2}.npz and data/analysis/season3_directions.json.
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

MODEL_ID = "allenai/Olmo-3-1125-32B"
D_VERSION = "olmo3_s3_banded"
# seasons.layer is `int not null` and sits inside unique(model_id, layer, d_version), so a
# band cannot live there. This single value goes in the season row AND in both d files, and
# is representative only — `band` / seasons.layers is what the scorer actually reads. It is
# score1's upper-middle layer, which is also the layer its d_bar was orthogonalised at.
REPRESENTATIVE_LAYER = 27
SPECS = {
    "score1": {"band": [19, 23, 27, 31], "aggregate": "banded_mean", "ranks": True},
    "score2": {"band": [15, 23, 31, 39], "aggregate": "per_layer_min", "ranks": False},
}


def unit(v, axis=-1):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.where(n == 0, 1e-12, n)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-split", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    np.seterr(all="ignore")
    from sklearn.linear_model import LogisticRegression

    z = np.load(ACTS, allow_pickle=True)
    ch = z["chosen"].astype(np.float64)
    rj = z["rejected"].astype(np.float64)
    axes = [str(x) for x in z["axes"]]
    ltypes = [str(x) for x in z["layer_types"]]
    N, _, H = ch.shape
    c = np.load(CONF, allow_pickle=True)

    def confounds(L):
        g = lambda k: c[k][:, L].astype(np.float64).mean(0)
        return {"length": unit(g("long") - g("short")),
                "sentiment": unit(g("pos") - g("neg")),
                "approach": unit(g("approach") - g("avoid"))}

    def orth(d, L):
        removed = []
        for name, cu in confounds(L).items():
            d = d - float(d @ cu) * cu
            removed.append(name)
        return d, removed

    def probe_at(L, idx):
        X = np.vstack([ch[idx, L], rj[idx, L]])
        y = np.r_[np.ones(len(idx)), np.zeros(len(idx))]
        w = LogisticRegression(C=0.1, max_iter=4000).fit(X, y).coef_[0]
        d, removed = orth(np.asarray(w, dtype=np.float64), L)
        return unit(d), removed

    def build(band, idx):
        per, removed = [], None
        for L in band:
            d_l, removed = probe_at(L, idx)
            per.append(d_l)
        per = np.stack(per)
        d_bar, _ = orth(unit(per.mean(0)), band[len(band) // 2])
        return unit(d_bar), per, removed

    # scoring functions, mirroring scripts/banded_score_arms.py exactly
    def sc_mean(M, band, d_bar, per):
        return np.mean([unit(M[:, L]) @ d_bar for L in band], axis=0)

    def sc_min(M, band, d_bar, per):
        return np.min(np.stack([unit(M[:, L]) @ per[j] for j, L in enumerate(band)]), axis=0)

    SCORERS = {"banded_mean": sc_mean, "per_layer_min": sc_min}

    rng = np.random.default_rng(a.seed)
    report = {"model_id": MODEL_ID, "d_version": D_VERSION, "n_pairs": N,
              "n_split": a.n_split, "hidden": H, "directions": {}}

    for tag, spec in SPECS.items():
        band, agg = spec["band"], spec["aggregate"]
        scorer = SCORERS[agg]

        # held-out separation + margin over repeated splits
        accs, margins = [], []
        for _ in range(a.n_split):
            p = rng.permutation(N)
            val, tr = p[:round(N * .25)], p[round(N * .25):]
            d_bar, per, _ = build(band, tr)
            gap = scorer(ch[val], band, d_bar, per) - scorer(rj[val], band, d_bar, per)
            accs.append(float(np.mean(gap > 0)))
            margins.append(float(gap.mean()))

        d_bar, per, removed = build(band, np.arange(N))

        cos_conf = {k: round(float(np.mean([abs(d_bar @ confounds(L)[k]) for L in band])), 4)
                    for k in ("approach", "sentiment", "length")}
        # per-axis coherence: does the direction agree with each of the 15 seed axes?
        per_axis = {}
        for ax in sorted(set(axes)):
            m = np.array([i for i, x in enumerate(axes) if x == ax])
            g = scorer(ch[m], band, d_bar, per) - scorer(rj[m], band, d_bar, per)
            per_axis[ax] = round(float(g.mean()), 5)

        rec = {"band": band, "layer_types": sorted({ltypes[L] for L in band}),
               "aggregate": agg, "ranks_the_board": spec["ranks"],
               "held_out_separation": round(float(np.mean(accs)), 4),
               "margin": round(float(np.mean(margins)), 5),
               "confound_cosines": cos_conf,
               "confounds_removed": removed,
               "min_cos_dbar_with_member": round(float(np.min(per @ d_bar)), 4),
               "per_axis_margin": per_axis,
               "per_axis_min": round(min(per_axis.values()), 5),
               "per_axis_all_positive": bool(min(per_axis.values()) > 0)}
        report["directions"][tag] = rec

        # d_version is the SEASON's version and must equal seasons.d_version exactly
        # (SKILL.md step 2). The two files differ by `role`, not by version — suffixing
        # the version instead was a mismatch that nothing caught until it was checked
        # by hand. `layer` is likewise the season row's representative value; `band` is
        # authoritative. scripts/check_season_matches_d.py now asserts both.
        meta = {"model_id": MODEL_ID, "d_version": D_VERSION, "role": tag,
                "band": band, "aggregate": agg, "layer": REPRESENTATIVE_LAYER,
                "confounds_removed": removed, "estimator": "logistic_C0.1",
                "n_pairs": N, "source_cache": ACTS.name,
                "held_out_separation": rec["held_out_separation"],
                "placeholder": False}
        out = ROOT / "data" / "directions" / f"d_olmo3_s3_{tag}.npz"
        np.savez(out, d=d_bar.astype(np.float32), band=np.array(band),
                 per_layer=per.astype(np.float32), meta=json.dumps(meta))

        print(f"\n=== {tag}  band={band}  {agg}{'  [RANKS]' if spec['ranks'] else ''}")
        print(f"  held-out separation  {rec['held_out_separation']:.4f}")
        print(f"  margin               {rec['margin']:.5f}")
        print(f"  cos(d, approach)     {cos_conf['approach']:.4f}   "
              f"sentiment {cos_conf['sentiment']:.4f}   length {cos_conf['length']:.4f}")
        print(f"  min cos(d_bar, member) {rec['min_cos_dbar_with_member']:.4f}")
        print(f"  per-axis margin: min {rec['per_axis_min']:+.5f} over 15 axes, "
              f"all positive = {rec['per_axis_all_positive']}")
        print(f"  wrote {out.relative_to(ROOT)}")

    p = ROOT / "data" / "analysis" / "season3_directions.json"
    p.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {p.relative_to(ROOT)}")
    print("\nGATES STILL OUTSTANDING: the causal steering check needs GPU generations.\n"
          "Nothing here proves d CAUSES anything — these are decodability numbers only.")


if __name__ == "__main__":
    main()
