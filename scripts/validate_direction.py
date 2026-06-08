"""Validate an extracted direction before shipping it (PROJECT_SPEC.md §11.4).

Auto-gates (decide PASS/FAIL):
  1. Held-out separation — chosen projects above rejected on held-out pairs.
  2. Confound cosines    — cos(d, length_dir) and cos(d, sentiment_dir) near zero.
  3. Per-axis coherence  — the 15 per-axis directions cluster with the global d.
Plus a causal-steering artifact (sample generations with +alpha*d) for human review —
the decisive behavioral check; on a random/tiny model it is mechanical only.

    python scripts/validate_direction.py --d data/directions/d_v1.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings  # noqa: E402
from app.ndif_client import ResidualReader  # noqa: E402
from scripts.extract_direction import LONG, SHORT, POS, NEG, compose, unit  # noqa: E402

SEP_MIN = 0.70
COS_MAX = 0.20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", default="data/directions/d_v1.npz")
    ap.add_argument("--pairs", default="data/seed_pairs.jsonl")
    ap.add_argument("--backend", default="")
    ap.add_argument("--model-id", default="")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--steer-alpha", type=float, default=8.0)
    ap.add_argument("--no-steer", action="store_true")
    args = ap.parse_args()

    data = np.load(args.d, allow_pickle=True)
    d = unit(np.asarray(data["d"], dtype=np.float64))
    meta = json.loads(str(data["meta"]))
    layer = int(meta["layer"])
    backend = args.backend or meta.get("backend", "ndif")
    model_id = args.model_id or meta["model_id"]
    print(f"validating {args.d}  (model={model_id}, layer={layer}, backend={backend})")

    rows = [json.loads(l) for l in Path(args.pairs).read_text(encoding="utf-8").splitlines() if l.strip()]
    reader = ResidualReader.build(model_id, backend,
                                  ndif_key=settings.ndif_api_key if backend == "ndif" else "",
                                  prepend_bos=settings.prepend_bos)

    chosen = np.stack([reader.last_token_resid(compose(r["prompt"], r["chosen"]), layer) for r in rows]).astype(np.float64)
    rejected = np.stack([reader.last_token_resid(compose(r["prompt"], r["rejected"]), layer) for r in rows]).astype(np.float64)
    diffs = chosen - rejected
    N = len(rows)

    report = {}

    # 1. Held-out separation
    rng = np.random.default_rng(args.split_seed)
    idx = rng.permutation(N)
    val_idx = idx[: max(1, int(round(N * args.val_frac)))]
    sep = float(np.mean((chosen[val_idx] @ d) > (rejected[val_idx] @ d)))
    report["held_out_separation"] = round(sep, 4)

    # 2. Confound cosines
    neutral = np.stack([reader.last_token_resid(t, layer) for t in (LONG + SHORT + POS + NEG)]).astype(np.float64)
    length_dir = unit(neutral[:2].mean(0) - neutral[2:4].mean(0))
    sentiment_dir = unit(neutral[4:6].mean(0) - neutral[6:8].mean(0))
    cos_len = abs(float(np.dot(d, length_dir)))
    cos_sent = abs(float(np.dot(d, sentiment_dir)))
    report["cos_length"] = round(cos_len, 4)
    report["cos_sentiment"] = round(cos_sent, 4)

    # 3. Per-axis coherence
    by_axis = defaultdict(list)
    for i, r in enumerate(rows):
        by_axis[r["axis"]].append(diffs[i])
    axis_cos = {ax: float(np.dot(unit(np.mean(v, axis=0)), d)) for ax, v in by_axis.items()}
    coherence = float(np.mean(list(axis_cos.values())))
    report["per_axis_coherence_mean"] = round(coherence, 4)
    report["per_axis_cos"] = {k: round(v, 3) for k, v in sorted(axis_cos.items())}

    # Verdict on auto-gates
    g1 = sep >= SEP_MIN
    g2 = cos_len < COS_MAX and cos_sent < COS_MAX
    g3 = coherence > 0
    passed = g1 and g2 and g3

    print("\n— Auto gates —")
    print(f"  [{'PASS' if g1 else 'FAIL'}] held-out separation {sep:.3f}  (need >= {SEP_MIN})")
    print(f"  [{'PASS' if g2 else 'FAIL'}] confound cosines  length={cos_len:.3f} sentiment={cos_sent:.3f}  (need < {COS_MAX})")
    print(f"  [{'PASS' if g3 else 'FAIL'}] per-axis coherence {coherence:.3f}  (need > 0; axes cluster)")

    # 4. Causal steering artifact (manual review)
    if not args.no_steer:
        try:
            samples = _steer_samples(reader, d, layer, args.steer_alpha)
            report["steering_samples"] = samples
            print("\n— Causal steering (review by hand; decisive for a real model) —")
            for s in samples:
                print(f"  prompt: {s['prompt']!r}\n    base:  {s['base']!r}\n    +a*d:  {s['steered']!r}")
        except Exception as e:  # noqa: BLE001
            print(f"\n— Causal steering unavailable ({type(e).__name__}: {e}); run manually —")

    report["verdict"] = "PASS" if passed else "FAIL"
    Path(args.d).with_suffix(".validation.json").write_text(json.dumps(report, indent=2))
    print(f"\nVERDICT: {report['verdict']}   (report → {Path(args.d).with_suffix('.validation.json')})")
    return 0 if passed else 1


def _steer_samples(reader, d, layer, alpha):
    """Generate with and without +alpha*d added at `layer` for a couple neutral probes."""
    import torch

    d_t = torch.tensor(d, dtype=torch.float32)
    probes = ["My honest opinion about my coworker is", "When the new hire made a mistake, I"]
    out = []
    for prompt in probes:
        with reader.model.generate(prompt, max_new_tokens=20, remote=reader.remote):
            base = reader.model.generator.output.save()
        with reader.model.generate(prompt, max_new_tokens=20, remote=reader.remote):
            reader._layer_module(layer).output[0][:] += alpha * d_t.to(reader._layer_module(layer).output[0].dtype)
            steered = reader.model.generator.output.save()
        dec = reader.model.tokenizer.decode
        out.append({"prompt": prompt,
                    "base": dec(base[0]),
                    "steered": dec(steered[0])})
    return out


if __name__ == "__main__":
    sys.exit(main())
