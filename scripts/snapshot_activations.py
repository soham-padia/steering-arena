"""Snapshot NDIF activations for reproduction debugging (_communication message 002).

For a fixed set of texts (the 16 season-2 probes + Jesse's 8 debug prompts composed
with probe[0]), captures the LAST-TOKEN residual at several layers via the exact
server read path (nnsight layers[L].output[0,-1,:], bf16 compute, saved as float32),
plus the token ids used and the layer-24 cosine against the live direction d.

Diagnostic logic for comparing against a local (transformers) run:
  - mismatch already at the earliest layer  -> tokenization/embedding difference
  - agreement early, drift growing by layer -> numeric divergence (kernels/dtype/sharding)
  - agreement at layer 24 but score differs -> direction file / cosine computation bug

Output: data/analysis/activation_snapshot_<date>.npz with, per text index i:
  acts_{i}: (n_layers, hidden) float32; token_ids_{i}: (seq_len,) int64
and arrays: texts (object), layers (int), plus JSON meta.

    python scripts/snapshot_activations.py
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import scoring  # noqa: E402
from app.config import settings  # noqa: E402
from app.ndif_client import ResidualReader  # noqa: E402
from scripts.extract_direction import with_retry  # noqa: E402

LAYERS = [1, 4, 8, 12, 16, 20, 24]
JESSE_PROMPTS_FILE = "/tmp/jesse_prefixes.json"  # fetched from his fork's debug_test branch


def main():
    probes = scoring.load_probes(settings.probe_set)
    jesse = [x["prompt"] for x in json.loads(Path(JESSE_PROMPTS_FILE).read_text())]
    composed = [scoring.compose(p, probes[0]) for p in jesse]
    texts = probes + composed
    d, _ = scoring.load_direction(settings.d_file)
    d64 = np.asarray(d, np.float64); d64 /= np.linalg.norm(d64)

    reader = ResidualReader.from_settings(settings)
    tok = reader.tokenizer

    out = {"texts": np.array(texts, dtype=object), "layers": np.array(LAYERS)}
    cos24 = []
    for i, t in enumerate(texts):
        acts = with_retry(reader.last_resids_layers, t, LAYERS, attempts=4, wait=20.0)
        out[f"acts_{i}"] = acts.astype(np.float32)
        out[f"token_ids_{i}"] = np.asarray(tok(t)["input_ids"], dtype=np.int64)
        v = acts[LAYERS.index(24)].astype(np.float64)
        cos24.append(float(v @ d64 / np.linalg.norm(v)))
        print(f"[{i + 1}/{len(texts)}] {t[:50]!r}  cos24={cos24[-1]:+.6f}", flush=True)

    out["cos_layer24_vs_d"] = np.array(cos24)
    meta = {
        "model_id": settings.model_id, "d_file": settings.d_file,
        "read": "nnsight model.model.layers[L].output[0,-1,:] (== HF hidden_states[L+1][0,-1,:])",
        "compute_dtype": "bfloat16 (NDIF serving), saved float32",
        "tokenization": "tokenizer(text) defaults, add_special_tokens=True (OLMo-3 adds no BOS)",
        "texts_order": "0..15 = season2 probes; 16..23 = Jesse debug prompts composed with probe[0]",
        "created": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    out["meta"] = np.array(json.dumps(meta))
    dest = Path(f"data/analysis/activation_snapshot_{dt.date.today():%Y%m%d}.npz")
    np.savez_compressed(dest, **out)
    print(f"\nwrote {dest} ({dest.stat().st_size/1e6:.1f} MB, {len(texts)} texts × {len(LAYERS)} layers)")


if __name__ == "__main__":
    main()
