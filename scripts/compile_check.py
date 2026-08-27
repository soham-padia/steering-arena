"""Is a leaderboard sequence a COMPILED steering vector?

Steering Arena publishes a direction and lets a crowd search token space for strings that
maximise projection onto it; one participant ran GCG against the objective directly. That
makes the winning sequence a candidate *compilation* of `d` into tokens, and raises a
mechanistic question the behavioural work cannot answer:

    does prepending the string reproduce the activation shift that injecting a*d produces,
    or does it reach the same behaviour by a different route?

Measured here: for each eval prompt, the layer-L last-token residual with and without the
prefix. delta = R(prefix + p) - R(p) is what the TOKENS do to activations. The injection
does exactly alpha*d_hat by construction, so the comparison is delta against alpha*d_hat.

  cos(delta, d)        alignment of the token-induced shift with the direction
  ||delta||            size of that shift, against alpha = 1.0 * ||R||
  |delta_par| / ||d||  fraction of the shift lying along d

A true compilation would show cos ~ 1.0, delta_par ~ alpha, ||delta||/alpha ~ 1.0.

    python scripts/compile_check.py --limit 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings  # noqa: E402
from app.ndif_client import ResidualReader  # noqa: E402
from app.scoring import compose  # noqa: E402
from scripts.behavioral_eval import _layer_norm, _load_d, _load_prompts  # noqa: E402
from scripts.extract_direction import with_retry  # noqa: E402
from scripts.prefix_gallery import load_gallery  # noqa: E402

OUT = Path("data/analysis/compile_check.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms",
                    default="pro_top,pro_coherent,anti_top,anti_hostile,control_junk")
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()

    d, layer = _load_d()                       # unit-normalised
    prompts = _load_prompts(args.limit)
    g = load_gallery()["arms"]
    reader = ResidualReader.build(settings.model_id, "ndif", ndif_key=settings.ndif_api_key,
                                  prepend_bos=settings.prepend_bos)
    rnorm = _layer_norm(reader, layer)
    alpha = 1.0 * rnorm
    print(f"L{layer}  ||R||~{rnorm:.2f}  the injection adds alpha*d_hat with alpha={alpha:.2f}")
    print(f"{len(prompts)} prompts\n")

    base = np.asarray(with_retry(reader.batch_last_resids, prompts, layer,
                                 attempts=4, wait=20.0), dtype=np.float64)
    store = {"layer": layer, "alpha": alpha, "r_norm": rnorm, "n_prompts": len(prompts),
             "model_id": settings.model_id, "arms": {}}

    print(f"{'arm':>14} {'score':>9} {'|delta|':>8} {'cos(d)':>8} "
          f"{'along d':>8} {'frac':>6} {'|d|/alpha':>9}")
    for arm in [a for a in args.arms.split(",") if a.strip()]:
        prefix = g[arm]["sequence"]
        texts = [compose(prefix, p) if prefix else p for p in prompts]
        pre = np.asarray(with_retry(reader.batch_last_resids, texts, layer,
                                    attempts=4, wait=20.0), dtype=np.float64)
        delta = pre - base
        norms = np.linalg.norm(delta, axis=1)
        along = delta @ d                                   # signed component on d_hat
        cos = along / np.maximum(norms, 1e-9)
        frac = np.abs(along) / np.maximum(norms, 1e-9)
        store["arms"][arm] = {
            "score": g[arm]["score"],
            "delta_norm": float(norms.mean()), "cos_with_d": float(cos.mean()),
            "along_d": float(along.mean()), "frac_along_d": float(frac.mean()),
            "norm_over_alpha": float(norms.mean() / alpha),
            "along_over_alpha": float(along.mean() / alpha),
        }
        a = store["arms"][arm]
        print(f"{arm:>14} {a['score']:>+9.5f} {a['delta_norm']:>8.2f} "
              f"{a['cos_with_d']:>8.4f} {a['along_d']:>8.2f} "
              f"{a['frac_along_d']:>6.3f} {a['norm_over_alpha']:>9.2f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(store, indent=2))
    print(f"\n-> {OUT}")
    print("reference: a true compilation of alpha*d_hat shows cos~1.0, "
          "along_d~alpha, |delta|/alpha~1.0")


if __name__ == "__main__":
    main()
