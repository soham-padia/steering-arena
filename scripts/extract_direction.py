"""Extract the pro-human direction `d` from contrastive seed pairs (PROJECT_SPEC.md §11.3).

Offline; runs on the SAME NDIF model that live scoring uses (or a local model for
dev). Reads last-token residuals (not mean-pooled), takes per-pair differences,
sweeps layers, picks the layer with the best held-out separation, orthogonalizes
out length + sentiment confounds, and saves d_<version>.npz with metadata.

    # real (NDIF) — confirm the exact model id first:
    python scripts/extract_direction.py --backend ndif --model-id allenai/Olmo-3.1-32B-Instruct --out data/directions/d_v1.npz
    # dev smoke (local tiny model, proves the pipeline):
    python scripts/extract_direction.py --backend local --model-id hf-internal-testing/tiny-random-LlamaForCausalLM --layers 0,1 --out /tmp/d_dev_real.npz
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings  # noqa: E402
from app.ndif_client import ResidualReader  # noqa: E402

# Neutral texts for confound directions (no value content; vary only length / sentiment).
LONG = ["The committee reviewed the quarterly schedule and updated the shared calendar for next month accordingly.",
        "She walked to the station, bought a ticket, waited on the platform, and boarded the train toward the city center."]
SHORT = ["The cat slept.", "It rained today."]
POS = ["This is wonderful and delightful.", "What a fantastic, joyful day."]
NEG = ["This is terrible and miserable.", "What an awful, dreadful day."]


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n else v


def compose(prompt, completion):
    return f"{prompt} {completion}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/seed_pairs.jsonl")
    ap.add_argument("--backend", default="ndif", choices=["ndif", "local"])
    ap.add_argument("--model-id", default=settings.model_id)
    ap.add_argument("--layers", default="", help="comma list to sweep, e.g. 12,16,20; default = all")
    ap.add_argument("--method", default="meandiff", choices=["meandiff"])
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--no-orthogonalize", action="store_true")
    ap.add_argument("--d-version", default="v1")
    ap.add_argument("--max-pairs", type=int, default=0, help="cap pairs (smoke tests)")
    ap.add_argument("--out", default="data/directions/d_v1.npz")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.pairs).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.max_pairs:
        rows = rows[: args.max_pairs]
    print(f"loaded {len(rows)} pairs; building reader ({args.backend}, {args.model_id})…")

    reader = ResidualReader.build(
        args.model_id, args.backend,
        ndif_key=settings.ndif_api_key if args.backend == "ndif" else "",
        prepend_bos=settings.prepend_bos,
    )

    # All-layer last-token resids per text (one forward each).
    def all_layers(texts):
        return np.stack([reader.last_resids_all_layers(t) for t in texts])  # (n, L, H)

    print("reading chosen/rejected activations…")
    chosen = all_layers([compose(r["prompt"], r["chosen"]) for r in rows])    # (N, L, H)
    rejected = all_layers([compose(r["prompt"], r["rejected"]) for r in rows])
    diffs = chosen - rejected                                                 # (N, L, H)
    N, L, H = diffs.shape

    # Train / val split.
    rng = np.random.default_rng(args.split_seed)
    idx = rng.permutation(N)
    n_val = max(1, int(round(N * args.val_frac)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    # Candidate layers.
    layers = [int(x) for x in args.layers.split(",") if x.strip()] or list(range(L))

    def separation(layer, d):
        u = unit(d)
        pc = chosen[val_idx, layer, :] @ u
        pr = rejected[val_idx, layer, :] @ u
        return float(np.mean(pc > pr))

    best = None
    for layer in layers:
        d_l = diffs[train_idx, layer, :].mean(axis=0)
        acc = separation(layer, d_l)
        if best is None or acc > best[1]:
            best = (layer, acc, d_l)
    best_layer, best_acc, d = best
    print(f"swept layers {layers}: best layer {best_layer} (held-out separation {best_acc:.3f})")

    confounds_removed = []
    if not args.no_orthogonalize:
        neutral = all_layers(LONG + SHORT + POS + NEG)  # (8, L, H)
        nl = neutral[:, best_layer, :]
        length_dir = nl[:2].mean(0) - nl[2:4].mean(0)
        sentiment_dir = nl[4:6].mean(0) - nl[6:8].mean(0)
        for name, cd in (("length", length_dir), ("sentiment", sentiment_dir)):
            cu = unit(cd)
            d = d - np.dot(d, cu) * cu  # Gram-Schmidt: project the confound out
            confounds_removed.append(name)
        print(f"orthogonalized out: {confounds_removed}")

    d = unit(d).astype(np.float32)
    meta = {
        "model_id": args.model_id,
        "model_build": "",
        "layer": best_layer,
        "d_version": args.d_version,
        "extraction_method": args.method,
        "confounds_removed": confounds_removed,
        "held_out_separation": round(best_acc, 4),
        "num_pairs": N,
        "backend": args.backend,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "notes": "Extracted via scripts/extract_direction.py.",
        "placeholder": False,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, d=d, meta=np.array(json.dumps(meta)))
    print(f"wrote {out}  (dim={d.shape[0]}, layer={best_layer}, sep={best_acc:.3f})")
    print("next: validate it →  python scripts/validate_direction.py --d", out)


if __name__ == "__main__":
    main()
