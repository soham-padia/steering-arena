"""How much of `d` is "pro-human", and how much is just any split of these texts?

THE QUESTION. cos between the five layer-native directions averages 0.556, and that number
was used to argue the layers share a pro-human feature and to pick a band. But a cosine
between two fitted probes is not evidence about the CONCEPT unless you know what an
uninformative probe fit to the same data would give.

TWO NULLS, and the choice matters more than it looks.

  ISOTROPIC random unit vectors. What most steering papers use. In 5120 dimensions two
  random unit vectors have |cos| ~ 1/sqrt(5120) = 0.014, so ANY fitted direction looks
  enormously significant against it. SteerCheck (arXiv:2608.24335) makes exactly this
  criticism. Reported here only to show how weak it is.

  LABEL-SHUFFLED. Refit the same logistic probe after randomly swapping chosen/rejected
  within each pair. Identical texts, identical geometry, identical fitting procedure, only
  the label destroyed. This is the null that can actually hurt the claim, and it does.

WHAT IT FINDS, recorded here because it revises an earlier claim of this project's:
the SHAPE of the cross-layer structure survives label shuffling almost intact. Adjacent
layers agree, distant ones do not, and L16 is the outlier, in the null too. So "the layers
disagree because the pro-human feature changes with depth" was overstated - most of that
pattern is the residual stream's own geometry and would appear for any binary split of
these 270 texts. What survives is a consistent EXCESS, real above null on every layer pair.

The band recommendation is unchanged but its justification is not: use {32,40,48} because
the residual geometry makes early and late layers hard to span with one vector, whatever
you are probing, not because pro-human specifically changes at L16.

    python scripts/direction_null.py --n-shuffle 40

Writes data/analysis/direction_null.json. Zero NDIF calls.
"""
import argparse
import hashlib
import itertools
import json
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ACTS = ROOT / "data" / "cache" / "acts" / "allenai_Olmo-3-1125-32B"
MODEL_ID = "allenai/Olmo-3-1125-32B"
NATIVE = [16, 24, 32, 40, 48]


def unit(v, axis=-1):
    return v / np.linalg.norm(v, axis=axis, keepdims=True)


def load(text, layer):
    k = hashlib.sha256(f"{MODEL_ID}\x00L{layer}\x00{text}".encode()).hexdigest()
    return np.load(ACTS / f"{k}.npy").astype(np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-shuffle", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    np.seterr(all="ignore")
    warnings.filterwarnings("ignore")
    from sklearn.linear_model import LogisticRegression

    rows = [json.loads(l) for l in open(ROOT / "data" / "seed_pairs.jsonl")]
    ch = np.stack([[load(f"{r['prompt']} {r['chosen']}", l) for l in NATIVE] for r in rows])
    rj = np.stack([[load(f"{r['prompt']} {r['rejected']}", l) for l in NATIVE] for r in rows])
    N = len(rows)
    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(N)
    nv = round(N * 0.25)
    val, tr = idx[:nv], idx[nv:]

    def fit(j, flip):
        A = np.where(flip[tr, None], rj[tr, j], ch[tr, j])
        B = np.where(flip[tr, None], ch[tr, j], rj[tr, j])
        return unit(np.asarray(LogisticRegression(C=0.1, max_iter=4000).fit(
            np.vstack([A, B]), np.r_[np.ones(len(tr)), np.zeros(len(tr))]).coef_[0]))

    def sep(j, u):
        """Held-out separation on the TRUE labels, always."""
        return float(np.mean((unit(ch[val, j]) @ u) > (unit(rj[val, j]) @ u)))

    none = np.zeros(N, bool)
    real = [fit(j, none) for j in range(5)]
    Rm = np.array([[float(real[x] @ real[y]) for y in range(5)] for x in range(5)])
    real_sep = [sep(j, real[j]) for j in range(5)]

    S = np.zeros((a.n_shuffle, 5, 5))
    shuf_sep, cos_real, iso = [], [], []
    for s in range(a.n_shuffle):
        f = rng.random(N) < 0.5
        ds = [fit(j, f) for j in range(5)]
        S[s] = [[float(ds[x] @ ds[y]) for y in range(5)] for x in range(5)]
        shuf_sep.append(sep(1, ds[1]))
        cos_real.append(abs(float(real[1] @ ds[1])))
        iso.append(abs(float(real[1] @ unit(rng.standard_normal(ch.shape[-1])))))
    off = lambda M: [M[x, y] for x, y in itertools.combinations(range(5), 2)]
    r_off = float(np.mean(off(Rm)))
    s_off = [float(np.mean(off(S[s]))) for s in range(a.n_shuffle)]

    out = {
        "n_shuffle": a.n_shuffle, "layers": NATIVE,
        "real": {"cos_matrix": Rm.round(4).tolist(), "off_diag_mean": round(r_off, 4),
                 "held_out_separation": real_sep},
        "shuffled": {"cos_matrix_mean": S.mean(0).round(4).tolist(),
                     "off_diag_mean": round(float(np.mean(s_off)), 4),
                     "off_diag_max": round(float(np.max(s_off)), 4),
                     "held_out_separation_mean": round(float(np.mean(shuf_sep)), 4),
                     "held_out_separation_range": [round(min(shuf_sep), 4), round(max(shuf_sep), 4)],
                     "abs_cos_with_real_d24_mean": round(float(np.mean(cos_real)), 4),
                     "abs_cos_with_real_d24_max": round(float(np.max(cos_real)), 4)},
        "isotropic": {"abs_cos_with_real_d24_mean": round(float(np.mean(iso)), 4),
                      "abs_cos_with_real_d24_max": round(float(np.max(iso)), 4)},
        "excess_off_diag": round(r_off - float(np.mean(s_off)), 4),
        "real_exceeds_n_of_n": [int(sum(x < r_off for x in s_off)), a.n_shuffle],
    }
    p = ROOT / "data" / "analysis" / "direction_null.json"
    p.write_text(json.dumps(out, indent=1))

    print(f"REAL      separation {real_sep}   off-diag cos {r_off:.3f}")
    print(f"SHUFFLED  separation {np.mean(shuf_sep):.3f} "
          f"({min(shuf_sep):.3f}-{max(shuf_sep):.3f})   off-diag cos {np.mean(s_off):.3f} "
          f"(max {np.max(s_off):.3f})")
    print(f"          |cos| with real d_L24 {np.mean(cos_real):.4f} (max {np.max(cos_real):.4f})")
    print(f"ISOTROPIC |cos| with real d_L24 {np.mean(iso):.4f} (max {np.max(iso):.4f})")
    print(f"EXCESS    {r_off - np.mean(s_off):+.3f}, real exceeds "
          f"{out['real_exceeds_n_of_n'][0]}/{a.n_shuffle} draws")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
