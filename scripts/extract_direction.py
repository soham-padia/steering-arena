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
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


def with_retry(fn, *args, attempts=5, wait=30.0, **kw):
    """Retry a flaky NDIF call (e.g. transient 'deployment evicted') with backoff."""
    last = None
    for i in range(attempts):
        try:
            return fn(*args, **kw)
        except Exception as e:  # noqa: BLE001
            last = e
            if i < attempts - 1:
                print(f"    retry {i + 1}/{attempts} after {type(e).__name__}: {str(e)[:120]}", flush=True)
                time.sleep(wait)
    raise last


def cached_resids(reader, text, cache_dir, model_id, attempts, wait):
    """Per-text all-layer resids with a disk checkpoint → extraction is resumable.

    Each forward is saved atomically as soon as it succeeds, keyed by (model, text),
    so a rerun skips completed forwards and only re-fetches what's missing. Returns
    (array, was_cached).
    """
    key = hashlib.sha256(f"{model_id}\x00{text}".encode("utf-8")).hexdigest()
    fp = Path(cache_dir) / f"{key}.npy"
    if fp.exists():
        return np.load(fp), True
    arr = with_retry(reader.last_resids_all_layers, text, attempts=attempts, wait=wait)
    fp.parent.mkdir(parents=True, exist_ok=True)
    tmp = fp.with_name(fp.name + ".tmp")
    with open(tmp, "wb") as f:  # file handle so np.save doesn't append a second .npy
        np.save(f, arr)
    tmp.replace(fp)  # atomic publish — a crash mid-write never leaves a corrupt cache file
    return arr, False

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
    ap.add_argument("--retry", type=int, default=5, help="per-forward retry attempts (NDIF evictions)")
    ap.add_argument("--retry-wait", type=float, default=30.0)
    ap.add_argument("--cache-dir", default="", help="activation checkpoint dir (default: data/cache/acts/<model>)")
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

    cache_dir = Path(args.cache_dir or f"data/cache/acts/{args.model_id.replace('/', '_')}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"activation cache (resumable): {cache_dir}", flush=True)

    # All-layer last-token resids per text (one forward each), checkpointed to disk
    # and retrying on NDIF evictions — so the run is fully resumable.
    def all_layers(texts, label):
        out = []
        for i, t in enumerate(texts):
            arr, hit = cached_resids(reader, t, cache_dir, args.model_id, args.retry, args.retry_wait)
            print(f"  [{label} {i + 1}/{len(texts)}] {'cached' if hit else 'fetched'}", flush=True)
            out.append(arr)
        return np.stack(out)  # (n, L, H)

    print("reading chosen/rejected activations…", flush=True)
    chosen = all_layers([compose(r["prompt"], r["chosen"]) for r in rows], "chosen")    # (N, L, H)
    rejected = all_layers([compose(r["prompt"], r["rejected"]) for r in rows], "rejected")
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
        neutral = all_layers(LONG + SHORT + POS + NEG, "neutral")  # (8, L, H)
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
