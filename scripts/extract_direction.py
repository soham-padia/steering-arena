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


def cached_layer_resid(reader, text, layer, cache_dir, model_id, attempts, wait):
    """One (text, layer) last-token residual with a disk checkpoint → resumable.

    Uses single-layer reads (one .save() per remote trace): NDIF's remote sandbox
    rejects many .save() calls in one trace, but single reads are reliable. Each is
    saved atomically, keyed by (model, layer, text), so a rerun only re-fetches what's
    missing. Returns (array, was_cached).
    """
    key = hashlib.sha256(f"{model_id}\x00L{layer}\x00{text}".encode("utf-8")).hexdigest()
    fp = Path(cache_dir) / f"{key}.npy"
    if fp.exists():
        return np.load(fp), True
    arr = with_retry(reader.last_token_resid, text, layer, attempts=attempts, wait=wait)
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

    # Candidate layers to read. Single-layer reads are remote-safe; default to one
    # mid-depth layer (cheap, ~2 forwards/pair). Pass --layers "a,b,c" to sweep a few.
    nlayers = reader.num_layers
    layers = [int(x) for x in args.layers.split(",") if x.strip()] or [round(nlayers * 0.5)]
    print(f"reading layers {layers} (of {nlayers}); single-layer reads, checkpointed/resumable", flush=True)

    def read_set(texts, label):
        out = np.empty((len(texts), len(layers), reader.hidden_size), dtype=np.float32)
        for ti, t in enumerate(texts):
            for li, L in enumerate(layers):
                arr, hit = cached_layer_resid(reader, t, L, cache_dir, args.model_id, args.retry, args.retry_wait)
                out[ti, li, :] = arr
            print(f"  [{label} {ti + 1}/{len(texts)}] layers {layers} done", flush=True)
        return out

    print("reading chosen/rejected activations…", flush=True)
    # float64 for the projection math: float32 @ on some BLAS builds emits spurious
    # overflow/divide-by-zero warnings even on small, finite activations.
    chosen = read_set([compose(r["prompt"], r["chosen"]) for r in rows], "chosen").astype(np.float64)
    rejected = read_set([compose(r["prompt"], r["rejected"]) for r in rows], "rejected").astype(np.float64)
    diffs = chosen - rejected                                                 # (N, n_layers, H)
    N = diffs.shape[0]
    if not (np.isfinite(chosen).all() and np.isfinite(rejected).all()):
        raise SystemExit("Non-finite activations read from the model — aborting (would corrupt d).")

    # Train / val split.
    rng = np.random.default_rng(args.split_seed)
    idx = rng.permutation(N)
    n_val = max(1, int(round(N * args.val_frac)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    def separation(pos, d):
        u = unit(d)
        # data is finite (asserted at read); ignore numpy's spurious float matmul warnings
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            pc = chosen[val_idx, pos, :] @ u
            pr = rejected[val_idx, pos, :] @ u
        return float(np.mean(pc > pr))

    best = None
    for pos, L in enumerate(layers):
        d_l = diffs[train_idx, pos, :].mean(axis=0)
        acc = separation(pos, d_l)
        if best is None or acc > best[2]:
            best = (pos, L, acc, d_l)
    best_pos, best_layer, best_acc, d = best
    print(f"swept layers {layers}: best layer {best_layer} (held-out separation {best_acc:.3f})")

    confounds_removed = []
    if not args.no_orthogonalize:
        neutral_texts = LONG + SHORT + POS + NEG
        nl = np.stack([cached_layer_resid(reader, t, best_layer, cache_dir, args.model_id, args.retry, args.retry_wait)[0]
                       for t in neutral_texts])  # (8, H) at best layer
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
