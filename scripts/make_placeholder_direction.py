"""Generate a PLACEHOLDER direction file for development.

A real `d` comes from the offline extraction pipeline (Phase 5,
scripts/extract_direction.py) on the served NDIF model. This stand-in lets the
scoring core + API run end-to-end before then. Its metadata is marked
`placeholder: true` so the CLI/UI can warn that scores aren't meaningful.

    python scripts/make_placeholder_direction.py --dim 4096
    # match a local dev model instead:
    python scripts/make_placeholder_direction.py --dim 512 --model-id EleutherAI/pythia-70m
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=4096, help="hidden size of the target model")
    ap.add_argument("--out", default="data/directions/d_v1.npz")
    ap.add_argument("--model-id", default="OLMo-3-32B")
    ap.add_argument("--model-build", default="")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--d-version", default="v1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    d = rng.standard_normal(args.dim).astype(np.float32)
    d /= np.linalg.norm(d)

    meta = {
        "model_id": args.model_id,
        "model_build": args.model_build,
        "layer": args.layer,
        "d_version": args.d_version,
        "extraction_method": "placeholder-random",
        "confounds_removed": [],
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "notes": "PLACEHOLDER random unit vector — not a real pro-human direction.",
        "placeholder": True,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, d=d, meta=np.array(json.dumps(meta)))
    print(f"wrote {out}  (dim={args.dim}, placeholder)")


if __name__ == "__main__":
    main()
