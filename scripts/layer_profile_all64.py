"""Per-layer concept profile across ALL 64 layers, plus the full- vs sliding-attention
comparison and per-layer confound cosines. CPU only, runs from the cached activations.

Math is lifted verbatim from scripts/layer_concept_profile.py so the numbers are
comparable to the published five-layer profile:
  - logistic probe fit on RAW train activations, coefficient unit-normalised -> u
  - gap = (unit(chosen_val) @ u) - (unit(rejected_val) @ u), always scored on TRUE labels
  - acc = mean(gap>0); margin = gap.mean(); cohen = gap.mean()/gap.std()
  - null  = same, with train labels randomly flipped (25% val, repeated splits)

New here: all 64 layers instead of [16,24,32,40,48] (which are all sliding_attention),
per-layer cos(d, approach/valence/length), and the kind>cruel value-flip control at every
layer. The question this answers: are full_attention layers [3,7,...,63] better probe
sites than the sliding layers every previous sweep used?
"""
from __future__ import annotations
import argparse, json, warnings
import numpy as np

def unit(v, axis=-1):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.where(n == 0, 1e-12, n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--confounds", required=True)
    ap.add_argument("--n-split", type=int, default=20)
    ap.add_argument("--n-shuffle", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    np.seterr(all="ignore"); warnings.filterwarnings("ignore")
    from sklearn.linear_model import LogisticRegression

    z = np.load(a.acts, allow_pickle=True)
    ch, rj = z["chosen"].astype(np.float64), z["rejected"].astype(np.float64)   # (N, nL, H)
    ltypes = [str(x) for x in z["layer_types"]]
    N, nL, H = ch.shape
    chn, rjn = unit(ch), unit(rj)
    FULL = [i for i, t in enumerate(ltypes) if t == "full_attention"]
    SLIDE = [i for i, t in enumerate(ltypes) if t == "sliding_attention"]
    print(f"{N} pairs | {nL} layers | H={H} | full={len(FULL)} sliding={len(SLIDE)}")

    c = np.load(a.confounds, allow_pickle=True)
    rng = np.random.default_rng(a.seed)

    def one(L, tr, val, flip):
        A = np.where(flip[tr, None], rj[tr, L], ch[tr, L])
        B = np.where(flip[tr, None], ch[tr, L], rj[tr, L])
        u = unit(np.asarray(LogisticRegression(C=0.1, max_iter=4000).fit(
            np.vstack([A, B]), np.r_[np.ones(len(tr)), np.zeros(len(tr))]).coef_[0]))
        gap = (chn[val, L] @ u) - (rjn[val, L] @ u)
        return float(np.mean(gap > 0)), float(gap.mean()), float(gap.mean() / (gap.std() or 1e-12)), u

    # Pre-generate splits ONCE so every layer sees identical splits (as in the
    # published 5-layer profile), then parallelise over layers -- they are independent.
    none = np.zeros(N, bool)
    splits, shuffles = [], []
    for _ in range(a.n_split):
        idx = rng.permutation(N); splits.append((idx[round(N*.25):], idx[:round(N*.25)]))
    for _ in range(a.n_shuffle):
        idx = rng.permutation(N); shuffles.append((idx[round(N*.25):], idx[:round(N*.25)],
                                                   rng.random(N) < 0.5))

    def per_layer(L):
        r = {m: [] for m in ("acc", "margin", "cohen")}
        n = {m: [] for m in ("acc", "margin", "cohen")}
        for tr, val in splits:
            acc, mg, cd, _ = one(L, tr, val, none)
            r["acc"].append(acc); r["margin"].append(mg); r["cohen"].append(cd)
        for tr, val, f in shuffles:
            acc, mg, cd, _ = one(L, tr, val, f)
            n["acc"].append(acc); n["margin"].append(mg); n["cohen"].append(cd)
        return L, r, n

    from joblib import Parallel, delayed
    print(f"fitting {nL} layers x {a.n_split + a.n_shuffle} splits, parallel over layers...",
          flush=True)
    got = Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(per_layer)(L) for L in range(nL))
    res = {m: {} for m in ("acc", "margin", "cohen")}
    nul = {m: {} for m in ("acc", "margin", "cohen")}
    for L, r, n in got:
        for m in r: res[m][L] = r[m]; nul[m][L] = n[m]

    # full-data direction per layer, plus confound cosines and the kind>cruel control
    out = {"layer_types": ltypes, "full_attention": FULL, "n_split": a.n_split,
           "n_shuffle": a.n_shuffle, "layers": {}}
    print(f"\n{'L':<5}{'type':<10}{'acc':>7}{'margin':>10}{'null':>9}{'excess':>10}"
          f"{'cohen':>8}{'cosAppr':>9}{'cosVal':>8}{'cosLen':>8}{'kind>cruel':>11}")
    for L in range(nL):
        u_full = unit(np.asarray(LogisticRegression(C=0.1, max_iter=4000).fit(
            np.vstack([ch[:, L], rj[:, L]]), np.r_[np.ones(N), np.zeros(N)]).coef_[0]))
        appr = unit(c["approach"][:, L].astype(np.float64).mean(0) - c["avoid"][:, L].astype(np.float64).mean(0))
        val_ = unit(c["pos"][:, L].astype(np.float64).mean(0) - c["neg"][:, L].astype(np.float64).mean(0))
        len_ = unit(c["long"][:, L].astype(np.float64).mean(0) - c["short"][:, L].astype(np.float64).mean(0))
        kp = unit(c["kind"][:, L].astype(np.float64)) @ u_full
        cp = unit(c["cruel"][:, L].astype(np.float64)) @ u_full
        kc = int((kp > cp).sum()); nkc = len(kp)
        r = {m: float(np.mean(res[m][L])) for m in res}
        n = {m: float(np.mean(nul[m][L])) for m in nul}
        rec = {"type": ltypes[L], "acc": round(r["acc"], 4),
               "margin": round(r["margin"], 5), "null_margin": round(n["margin"], 5),
               "excess": round(r["margin"] - n["margin"], 5), "cohen": round(r["cohen"], 3),
               "cos_approach": round(abs(float(u_full @ appr)), 4),
               "cos_valence": round(abs(float(u_full @ val_)), 4),
               "cos_length": round(abs(float(u_full @ len_)), 4),
               "kind_gt_cruel": f"{kc}/{nkc}"}
        out["layers"][str(L)] = rec
        print(f"{L:<5}{ltypes[L][:8]:<10}{rec['acc']:>7.3f}{rec['margin']:>10.5f}"
              f"{rec['null_margin']:>9.5f}{rec['excess']:>+10.5f}{rec['cohen']:>8.2f}"
              f"{rec['cos_approach']:>9.4f}{rec['cos_valence']:>8.4f}{rec['cos_length']:>8.4f}"
              f"{rec['kind_gt_cruel']:>11}")

    def agg(idxs, key):
        return float(np.mean([out["layers"][str(i)][key] for i in idxs]))
    summ = {}
    for nm, idxs in (("full_attention", FULL), ("sliding_attention", SLIDE)):
        summ[nm] = {k: round(agg(idxs, k), 5) for k in
                    ("excess", "cohen", "cos_approach", "cos_valence", "cos_length")}
    # mid-to-late only, the region any band would actually use
    midfull = [i for i in FULL if 23 <= i <= 55]
    midslide = [i for i in SLIDE if 23 <= i <= 55]
    summ["full_attention_L23_55"] = {k: round(agg(midfull, k), 5) for k in ("excess", "cohen", "cos_approach")}
    summ["sliding_attention_L23_55"] = {k: round(agg(midslide, k), 5) for k in ("excess", "cohen", "cos_approach")}
    out["summary"] = summ
    print("\n=== full vs sliding (mean over layers) ===")
    for k, v in summ.items(): print(f"  {k:26} {v}")
    best = sorted(FULL, key=lambda i: -out["layers"][str(i)]["excess"])[:6]
    out["best_full_attention_by_excess"] = best
    print(f"\n  best full_attention layers by excess margin: {best}")
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")

if __name__ == "__main__":
    main()
