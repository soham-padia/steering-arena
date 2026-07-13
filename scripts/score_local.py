"""Reference LOCAL scorer for Steering Arena, with no NDIF / nnsight dependency.

Reproduces the server's cosine steering-shift score with a plain Hugging Face
transformers forward pass, so anyone without NDIF access can compute (and optimize
against, e.g. with GCG) the same objective. The server stays the canonical scorer for
the leaderboard; this matches its extraction path exactly.

The details that actually matter (mismatches here are why local scores come out
"slightly off"):

  1. Model + layer: allenai/Olmo-3-1125-32B, layer 24 (the live season).
  2. Residual read = the OUTPUT of decoder block index `layer`, i.e. the residual
     stream after the block. With output_hidden_states=True that is hidden_states
     [layer + 1], NOT hidden_states[layer]: index 0 is the embedding output, so block
     i maps to hidden_states[i + 1]. This off-by-one is the single most common repro
     bug.
  3. Last token only ([-1]); BOS prepended via the tokenizer default
     (add_special_tokens=True).
  4. Composition is f"{seq} {probe}" (one space, seq then probe).
  5. score = mean over probes of cos(R(seq+probe)[-1], d) - cos(R(probe)[-1], d),
     with the cosine computed in float64.
  6. Precision: NDIF serves the model in bfloat16, so load it in bfloat16 here to get
     closest. A local float32 run will differ slightly, and an exact bitwise match to
     the live server is not expected (determinism is bounded to a fixed NDIF build).
     Rank order is what transfers; the server score is authoritative for the board.

Usage:
  python scripts/score_local.py "be kind and honest"
  python scripts/score_local.py "..." --model allenai/Olmo-3-1125-32B --layer 24 \
      --d data/directions/d_olmo3_L24_logistic.npz --probes data/probes/season2.json
  # smaller model to sanity-check the pipeline without 64GB of VRAM:
  python scripts/score_local.py "hi" --model meta-llama/Llama-3.1-8B --layer 16 \
      --d data/directions/d_llama_v1.npz --probes data/probes/season1.json --device cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_direction(path: str) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    return np.asarray(data["d"], dtype=np.float64)


def load_probes(path: str) -> list[str]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return obj["prompts"] if isinstance(obj, dict) else list(obj)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        raise ValueError("cosine of a zero vector is undefined")
    return float(a @ b / (na * nb))


def compose(seq: str, probe: str) -> str:
    return f"{seq} {probe}"


def main():
    ap = argparse.ArgumentParser(description="Local (no-NDIF) Steering Arena scorer.")
    ap.add_argument("sequence")
    ap.add_argument("--model", default="allenai/Olmo-3-1125-32B")
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--d", default="data/directions/d_olmo3_L24_logistic.npz")
    ap.add_argument("--probes", default="data/probes/season2.json")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="auto", help="'auto' (accelerate), 'cuda', or 'cpu'")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    d = load_direction(args.d)
    probes = load_probes(args.probes)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=getattr(torch, args.dtype),  # torch_dtype: works across transformers 4.x/5.x
        device_map=(args.device if args.device == "auto" else None),
        output_hidden_states=True,
    )
    if args.device != "auto":
        model = model.to(args.device)
    model.eval()

    if d.shape[0] != model.config.hidden_size:
        raise SystemExit(
            f"direction dim {d.shape[0]} != model hidden size {model.config.hidden_size} "
            f"- wrong model for this direction?"
        )
    if args.layer + 1 >= model.config.num_hidden_layers + 1:
        raise SystemExit(f"layer {args.layer} out of range for {model.config.num_hidden_layers} layers")

    @torch.no_grad()
    def resid_last(text: str) -> np.ndarray:
        inputs = tok(text, return_tensors="pt").to(model.device)  # BOS via add_special_tokens default
        hs = model(**inputs).hidden_states                        # len = num_layers + 1
        vec = hs[args.layer + 1][0, -1, :]                        # output of block `layer`, last token
        return vec.float().cpu().numpy()

    # Baselines depend only on the probe, so compute them once (matches the server).
    base = {p: cosine(resid_last(p), d) for p in probes}
    shifts = [cosine(resid_last(compose(args.sequence, p)), d) - base[p] for p in probes]
    score = float(np.mean(shifts))

    print(f"model={args.model} layer={args.layer} dtype={args.dtype} probes={len(probes)}")
    print(f"score (mean cosine steering-shift): {score:+.6f}")
    print("note: server (NDIF) score is canonical for the leaderboard; local may differ "
          "slightly by precision/hardware.")


if __name__ == "__main__":
    main()
