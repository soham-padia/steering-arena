"""How much does each layer encode the pro-human concept? Three metrics, one null.

THE STANDARD MOVE AND WHY IT FAILS HERE. Fit a probe per layer, plot held-out accuracy.
On this corpus that returns 1.000 at every layer. The task is easy enough that accuracy
saturates, so the curve is flat and carries no information about depth. Two fixes, both
applied here:

  MARGIN, NOT ACCURACY. Mean over pairs of (proj_chosen - proj_rejected) on unit-normalised
  residuals, and its standardised version (Cohen's d). These do not saturate: a layer where
  chosen and rejected are barely separated scores differently from one where they are far
  apart, even when both are 100% classifiable.

  A LABEL-SHUFFLED NULL AT EVERY LAYER. direction_null.py established that a probe fit to a
  random split of these same texts already achieves a lot of the apparent structure. So the
  quantity that means something is the EXCESS over that null, layer by layer, not the raw
  value. A layer whose real margin sits inside its own null band encodes nothing specific.

WHAT THIS DOES AND DOES NOT ANSWER. It measures DECODABILITY: can a linear readout recover
the concept from this layer. That is not the same as the model USING it there. A concept can
be linearly present at a layer that never drives behaviour. The causal version needs
intervention at each layer - steer or ablate and measure the behavioural change - and this
project has that only at layer 24. Read this as "where is it legible", not "where does it
act". steering_ablation.md is the causal counterpart.

Repeated random splits rather than one, because a single 34-pair validation set is noisy.

    python scripts/layer_concept_profile.py --n-split 20 --n-shuffle 20

Zero NDIF calls. Writes data/analysis/layer_concept_profile.json.
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


def unit(v, axis=-1):
    return v / np.linalg.norm(v, axis=axis, keepdims=True)


def load(text, layer):
    k = hashlib.sha256(f"{MODEL_ID}\x00L{layer}\x00{text}".encode()).hexdigest()
    return np.load(ACTS / f"{k}.npy").astype(np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-split", type=int, default=20)
    ap.add_argument("--n-shuffle", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    np.seterr(all="ignore")
    warnings.filterwarnings("ignore")
    from sklearn.linear_model import LogisticRegression

    rows = [json.loads(l) for l in open(ROOT / "data" / "seed_pairs.jsonl")]
    ch = np.stack([[load(f"{r['prompt']} {r['chosen']}", l) for l in NATIVE] for r in rows])
    rj = np.stack([[load(f"{r['prompt']} {r['rejected']}", l) for l in NATIVE] for r in rows])
    chn, rjn = unit(ch), unit(rj)          # unit-normalise so magnitude cannot inflate anything
    N = len(rows)
    rng = np.random.default_rng(a.seed)

    def one(j, tr, val, flip):
        A = np.where(flip[tr, None], rj[tr, j], ch[tr, j])
        B = np.where(flip[tr, None], ch[tr, j], rj[tr, j])
        u = unit(np.asarray(LogisticRegression(C=0.1, max_iter=4000).fit(
            np.vstack([A, B]), np.r_[np.ones(len(tr)), np.zeros(len(tr))]).coef_[0]))
        gap = (chn[val, j] @ u) - (rjn[val, j] @ u)      # always scored on TRUE labels
        return float(np.mean(gap > 0)), float(gap.mean()), float(gap.mean() / (gap.std() or 1e-12))

    none = np.zeros(N, bool)
    res = {m: {L: [] for L in NATIVE} for m in ("acc", "margin", "cohen")}
    nul = {m: {L: [] for L in NATIVE} for m in ("acc", "margin", "cohen")}
    for s in range(a.n_split):
        idx = rng.permutation(N)
        val, tr = idx[:round(N * .25)], idx[round(N * .25):]
        for j, L in enumerate(NATIVE):
            acc, mg, cd = one(j, tr, val, none)
            res["acc"][L].append(acc); res["margin"][L].append(mg); res["cohen"][L].append(cd)
    for s in range(a.n_shuffle):
        idx = rng.permutation(N)
        val, tr = idx[:round(N * .25)], idx[round(N * .25):]
        f = rng.random(N) < 0.5
        for j, L in enumerate(NATIVE):
            acc, mg, cd = one(j, tr, val, f)
            nul["acc"][L].append(acc); nul["margin"][L].append(mg); nul["cohen"][L].append(cd)

    out = {"layers": NATIVE, "n_split": a.n_split, "n_shuffle": a.n_shuffle,
           "real": {}, "null": {}}
    hdr = f"{'layer':<7}{'accuracy':>10}{'margin':>12}{'null margin':>14}{'excess':>10}{'Cohen d':>10}"
    print(hdr)
    for L in NATIVE:
        r = {m: [round(float(np.mean(res[m][L])), 4), round(float(np.std(res[m][L])), 4)]
             for m in res}
        n = {m: [round(float(np.mean(nul[m][L])), 4), round(float(np.std(nul[m][L])), 4)]
             for m in nul}
        out["real"][str(L)], out["null"][str(L)] = r, n
        print(f"L{L:<6}{r['acc'][0]:>10.3f}{r['margin'][0]:>12.5f}{n['margin'][0]:>14.5f}"
              f"{r['margin'][0]-n['margin'][0]:>+10.5f}{r['cohen'][0]:>10.2f}")

    p = ROOT / "data" / "analysis" / "layer_concept_profile.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
