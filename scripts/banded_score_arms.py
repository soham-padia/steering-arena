"""Score the board's gallery arms under the three banded objectives instead of L24 alone.

THE QUESTION. `pro_top` took the board with +0.10769 against `pro_coherent`'s +0.04032, a
2.7x margin, on a single-layer objective at layer 24. layer_sweep_prefix.md then showed that
advantage exists at layer 24 and essentially nowhere else. So: if the board had scored over
a BAND of layers instead of one, who would have won?

Reads the residuals the layer sweep already cached, so for the arms it covers this costs
ZERO NDIF calls. Arms it does not cover are reported as `not cached` rather than silently
dropped, with the exact call count needed to fill them in.

TWO CAVEATS THAT MUST TRAVEL WITH THESE NUMBERS.

  1. The board score uses the 16 frozen probes; everything here uses the 50 eval prompts,
     which is what the sweep cached. So the banded columns are NOT numerically comparable
     to the board column. The valid comparison is between ARMS within a column.
  2. `pro_top` was found by GCG optimising against the L24 objective. That it fails a banded
     objective says this STRING does not generalise across depth. It does NOT say a banded
     objective is unbeatable - nobody has run a search against one. That is the open
     experiment, not a conclusion.

    python scripts/banded_score_arms.py

Writes data/analysis/banded_score_arms.json.
"""
import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SWEEP = ROOT / "data" / "cache" / "layer_sweep"
AN = ROOT / "data" / "analysis"
MODEL_ID = "allenai/Olmo-3-1125-32B"
BANDS = {"32_40_48": [32, 40, 48], "all": [16, 24, 32, 40, 48]}
ARMS = ["pro_top", "pro_coherent", "anti_top", "anti_coherent", "anti_hostile",
        "control_junk", "control_text"]


def unit(v, axis=-1):
    return v / np.linalg.norm(v, axis=axis, keepdims=True)


def batch_key(texts, layer):
    """Verbatim key scheme from layer_sweep_prefix.py:92, so this reads that cache."""
    h = hashlib.sha256()
    h.update(f"{MODEL_ID}\x00L{layer}\x00n={len(texts)}".encode())
    for t in texts:
        h.update(b"\x00")
        h.update(t.encode("utf-8"))
    return h.hexdigest()


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    np.seterr(all="ignore")
    warnings.filterwarnings("ignore")

    prompts = json.loads((ROOT / "data/eval/steering_prompts.json").read_text())["prompts"]
    gallery = json.loads((AN / "site_prefixes.json").read_text())["arms"]

    def resid(arm, layer):
        pfx = gallery[arm]["sequence"]
        texts = [f"{pfx} {p}" for p in prompts] if pfx else list(prompts)
        fp = SWEEP / f"{batch_key(texts, layer)}.npy"
        return np.load(fp).astype(np.float64) if fp.exists() else None

    out, missing = {}, set()
    for tag, band in BANDS.items():
        z = np.load(ROOT / "data" / "directions" / f"d_olmo3_banded_{tag}.npz")
        d_bar, per = z["d"].astype(np.float64), z["per_layer"].astype(np.float64)
        base = {L: resid("base", L) for L in band}
        rows = {}
        print(f"\n=== band {band} ===   (50 eval prompts, NOT the 16 board probes)")
        print(f"{'arm':<15}{'board L24':>12}{'A banded mean':>15}{'C per-layer min':>17}")
        for arm in ARMS:
            R = {L: resid(arm, L) for L in band}
            if any(v is None for v in R.values()):
                missing.add(arm)
                print(f"{arm:<15}{gallery[arm]['score']:>+12.5f}{'not cached':>15}{'not cached':>17}")
                continue
            a = float(np.mean([((unit(R[L]) @ d_bar) - (unit(base[L]) @ d_bar)).mean()
                               for L in band]))
            mn = lambda M: np.min(np.stack([unit(M[L]) @ unit(per[j])
                                            for j, L in enumerate(band)]), axis=0)
            c = float((mn(R) - mn(base)).mean())
            rows[arm] = {"board_L24_16probes": gallery[arm]["score"],
                         "A_banded_mean": round(a, 5), "C_per_layer_min": round(c, 5)}
            print(f"{arm:<15}{gallery[arm]['score']:>+12.5f}{a:>+15.5f}{c:>+17.5f}")
        out[tag] = {"band": band, "arms": rows}

    if missing:
        n = len(missing)
        out["missing_arms"] = sorted(missing)
        out["ndif_calls_to_complete"] = {"late_band_only": n * 3, "both_bands": n * 5}
        print(f"\n{n} arms not cached: {', '.join(sorted(missing))}")
        print(f"  filling them costs {n*3} batched NDIF calls for [32,40,48], "
              f"{n*5} to cover both bands (the sweep itself spent 27).")

    p = AN / "banded_score_arms.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
