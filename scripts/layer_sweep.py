"""Steering-oriented layer sweep (the fix for 'best-separation layer != best-steering layer').

Reads candidate layers for all seed pairs in ONE multi-save trace per text (resumable,
shares cache with extract_direction.py), builds a confound-orthogonalized direction per
layer, reports held-out separation, saves d_<model>_L<layer>_<method>.npz for each, and
runs an inline causal-steering probe at every layer so we can see WHICH layer moves
generations most cleanly — not just which reads the concept best.

    python scripts/layer_sweep.py --layers 16,24,32,40,48 --method logistic --alphas 16,32
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
from scripts.extract_direction import (  # noqa: E402
    LONG, NEG, POS, SHORT, cached_layers_resid, compose, estimate_direction, unit,
)
from scripts.steering_bakeoff import generate  # noqa: E402

STEER_PROMPTS = [
    "When a stranger asked me for help, I",
    "When the new hire made a mistake, I",
    "My honest opinion about my coworker is",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/seed_pairs.jsonl")
    ap.add_argument("--model-id", default=settings.model_id)
    ap.add_argument("--backend", default="ndif")
    ap.add_argument("--layers", default="16,24,32,40,48")
    ap.add_argument("--method", default="logistic", choices=["meandiff", "lda", "logistic"])
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--mults", default="0.5,1,2", help="steering strength as fractions/multiples of each layer's mean residual norm (fair across depths)")
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--retry", type=int, default=5)
    ap.add_argument("--retry-wait", type=float, default=30.0)
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--out-dir", default="data/directions")
    ap.add_argument("--no-steer", action="store_true",
                    help="skip the steering probe (separation-only report; fully cache-served, no generations)")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    mults = [float(a) for a in args.mults.split(",") if a.strip()]
    rows = [json.loads(l) for l in Path(args.pairs).read_text(encoding="utf-8").splitlines() if l.strip()]

    reader = ResidualReader.build(args.model_id, args.backend,
                                  ndif_key=settings.ndif_api_key if args.backend == "ndif" else "",
                                  prepend_bos=settings.prepend_bos)
    cache_dir = Path(args.cache_dir or f"data/cache/acts/{args.model_id.replace('/', '_')}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"layer sweep: layers {layers} method={args.method} on {args.model_id}\n"
          f"reading {len(rows)} pairs + neutrals (multi-save, cached/resumable)…", flush=True)

    def read_set(texts, label):
        out = np.empty((len(texts), len(layers), reader.hidden_size), dtype=np.float64)
        for ti, t in enumerate(texts):
            out[ti] = cached_layers_resid(reader, t, layers, cache_dir, args.model_id, args.retry, args.retry_wait)
            if (ti + 1) % 25 == 0 or ti + 1 == len(texts):
                print(f"  [{label} {ti + 1}/{len(texts)}]", flush=True)
        return out

    chosen = read_set([compose(r["prompt"], r["chosen"]) for r in rows], "chosen")
    rejected = read_set([compose(r["prompt"], r["rejected"]) for r in rows], "rejected")
    neutral = read_set(LONG + SHORT + POS + NEG, "neutral")  # (8, L, H)
    if not (np.isfinite(chosen).all() and np.isfinite(rejected).all()):
        raise SystemExit("Non-finite activations — aborting.")

    rng = np.random.default_rng(args.split_seed)
    idx = rng.permutation(len(rows))
    n_val = max(1, int(round(len(rows) * args.val_frac)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    # Build + save a direction per layer, with separation.
    per_layer = []
    for li, L in enumerate(layers):
        d_raw = estimate_direction(args.method, chosen[train_idx, li, :], rejected[train_idx, li, :])
        # orthogonalize length + sentiment at this layer
        length_dir = unit(neutral[:2, li].mean(0) - neutral[2:4, li].mean(0))
        sentiment_dir = unit(neutral[4:6, li].mean(0) - neutral[6:8, li].mean(0))
        d = d_raw.copy()
        for cu in (length_dir, sentiment_dir):
            d = d - np.dot(d, cu) * cu
        d = unit(d)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            sep = float(np.mean((chosen[val_idx, li] @ d) > (rejected[val_idx, li] @ d)))
        # mean residual norm at this layer → lets us scale steering fairly across depths
        layer_norm = float(np.mean(np.linalg.norm(np.vstack([chosen[:, li], rejected[:, li]]), axis=1)))
        # filename prefix from the model id (e.g. olmo-3-1125-32b, llama-3.1-70b)
        prefix = args.model_id.split("/")[-1].lower()
        out = Path(args.out_dir) / f"d_{prefix}_L{L}_{args.method}.npz"
        meta = {"model_id": args.model_id, "model_build": "", "layer": L, "d_version": f"{prefix}_L{L}_{args.method}",
                "extraction_method": args.method, "confounds_removed": ["length", "sentiment"],
                "held_out_separation": round(sep, 4), "num_pairs": len(rows), "backend": args.backend,
                "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "notes": f"Layer-sweep candidate (scripts/layer_sweep.py).", "placeholder": False}
        np.savez(out, d=d.astype(np.float32), meta=np.array(json.dumps(meta)))
        per_layer.append({"layer": L, "separation": round(sep, 4), "path": str(out), "d": d, "norm": layer_norm})
        print(f"  layer {L}: separation {sep:.3f}  ‖R‖≈{layer_norm:.1f} → {out.name}", flush=True)

    # Inline causal-steering probe per layer. Steering strength = mult × ‖R‖_layer so
    # depths are comparable (deeper layers have larger residual norms).
    steer_log = None
    if not args.no_steer:
        print("\n=== STEERING PROBE (does +mult·‖R‖·d̂ move generations, coherently?) ===", flush=True)
        steer_log = {"mults": mults, "prompts": STEER_PROMPTS, "layers": layers, "runs": []}
        for prompt in STEER_PROMPTS:
            print(f"\n#### {prompt!r}", flush=True)
            base = generate(reader, prompt, args.max_new)
            print(f"  BASE: {base!r}", flush=True)
            steer_log["runs"].append({"prompt": prompt, "layer": None, "mult": 0, "text": base})
            for pl in per_layer:
                for m in mults:
                    a = m * pl["norm"]
                    text = generate(reader, prompt, args.max_new, steer=(pl["layer"], a * pl["d"]))
                    print(f"  L{pl['layer']:>2} +{m:g}·‖R‖d (a={a:.0f}): {text!r}", flush=True)
                    steer_log["runs"].append({"prompt": prompt, "layer": pl["layer"], "mult": m, "alpha": a, "text": text})

    rep = {"model_id": args.model_id, "method": args.method,
           "layers": [{"layer": p["layer"], "separation": p["separation"],
                       "resid_norm": round(p["norm"], 2), "path": p["path"]} for p in per_layer],
           "steering": steer_log}
    out = Path(args.out_dir) / f"layer_sweep_{prefix}_{args.method}.json"  # model-specific: a second model's sweep must not overwrite the first
    out.write_text(json.dumps(rep, indent=2))
    print(f"\nseparation by layer: " + ", ".join(f"L{p['layer']}={p['separation']:.2f}" for p in per_layer))
    print(f"saved → {out}\nPick the layer whose +d moves generations most cleanly toward kindness "
          f"(not necessarily the highest-separation one).")


if __name__ == "__main__":
    main()
