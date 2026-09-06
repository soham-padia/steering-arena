"""Capture last-token residuals at ALL 64 layers for every seed pair, once.

Why all 64: every layer sweep this project has run sampled multiples of 8, and OLMo-3
puts full_attention at layers [3,7,...,63] (period 4, offset 3). So 16/24/32/40/48 are
ALL sliding_attention and no full-attention layer has ever been evaluated as a probe
site. One forward pass yields every layer, so the comparison costs nothing extra.

Output is an activation cache; all downstream analysis runs on CPU from it, matching the
project's existing "recompute from the caches" pattern.

Composition is f"{prompt} {completion}", identical to scripts/extract_direction.py.
Residual = hidden_states[L+1] = output of block L (index 0 is the embedding output).
"""
from __future__ import annotations
import argparse, json, time
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="/home/padia_so_neu/steering-arena/data/seed_pairs.jsonl")
    ap.add_argument("--model", default="allenai/Olmo-3-1125-32B")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch, transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(l) for l in open(a.pairs)]
    print(f"{len(rows)} pairs | transformers {transformers.__version__}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=getattr(torch, a.dtype), device_map="auto",
        output_hidden_states=True, attn_implementation="sdpa")
    model.eval()
    nL = model.config.num_hidden_layers
    H = model.config.hidden_size
    layer_types = list(model.config.layer_types)
    print(f"layers={nL} hidden={H} "
          f"full_attention={[i for i,t in enumerate(layer_types) if t=='full_attention']}", flush=True)

    @torch.no_grad()
    def resid_all(text: str) -> np.ndarray:
        inp = tok(text, return_tensors="pt").to(model.device)
        hs = model(**inp).hidden_states              # len nL+1
        # block L output = hs[L+1]; stack blocks 0..nL-1, last token only
        return torch.stack([hs[L + 1][0, -1, :] for L in range(nL)]).float().cpu().numpy()

    t0 = time.time()
    chosen = np.zeros((len(rows), nL, H), dtype=np.float32)
    rejected = np.zeros_like(chosen)
    for i, r in enumerate(rows):
        chosen[i] = resid_all(f"{r['prompt']} {r['chosen']}")
        rejected[i] = resid_all(f"{r['prompt']} {r['rejected']}")
        if i % 25 == 0:
            print(f"  [{i+1}/{len(rows)}] {time.time()-t0:.0f}s", flush=True)

    if not (np.isfinite(chosen).all() and np.isfinite(rejected).all()):
        raise SystemExit("non-finite activations — aborting rather than caching garbage")

    np.savez_compressed(
        a.out, chosen=chosen, rejected=rejected,
        axes=np.array([r["axis"] for r in rows]),
        prompts=np.array([r["prompt"] for r in rows]),
        layer_types=np.array(layer_types),
        model=a.model, dtype=a.dtype, transformers=transformers.__version__)
    print(f"\nwrote {a.out}  shape={chosen.shape}  {time.time()-t0:.0f}s total")

if __name__ == "__main__":
    main()
