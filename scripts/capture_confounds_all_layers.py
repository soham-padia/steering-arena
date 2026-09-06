"""Capture the confound reference corpora at ALL 64 layers.

Imports the text sets from the existing scripts rather than copying them, so the
per-layer confound cosines are computed against exactly the same corpora the L24
confound audit used. Without this, cos(d_L, approach_L) can only be evaluated at the
one layer the old audit happened to read.

Sets: APPROACH/AVOID (action vs inaction on neutral chores, no ethical valence),
POS/NEG (sentiment), LONG/SHORT (length), CONTROL_PAIRS (kind vs cruel with approach
held constant -- the sharpest pro-human test).
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/padia_so_neu/steering-arena")

from scripts.confound_audit import APPROACH, AVOID, POS, NEG, CONTROL_PAIRS
from scripts.extract_direction import LONG, SHORT

def main():
    out = sys.argv[1]
    import torch, transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    MODEL = "allenai/Olmo-3-1125-32B"

    kind = [k for k, _ in CONTROL_PAIRS]
    cruel = [c for _, c in CONTROL_PAIRS]
    sets = {"approach": APPROACH, "avoid": AVOID, "pos": POS, "neg": NEG,
            "long": LONG, "short": SHORT, "kind": kind, "cruel": cruel}
    print({k: len(v) for k, v in sets.items()}, flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto",
        output_hidden_states=True, attn_implementation="sdpa")
    model.eval()
    nL = model.config.num_hidden_layers

    @torch.no_grad()
    def resid_all(text):
        inp = tok(text, return_tensors="pt").to(model.device)
        hs = model(**inp).hidden_states
        return torch.stack([hs[L + 1][0, -1, :] for L in range(nL)]).float().cpu().numpy()

    t0 = time.time(); saved = {}
    for name, texts in sets.items():
        saved[name] = np.stack([resid_all(t) for t in texts])   # (n, nL, H)
        print(f"  {name:9} {saved[name].shape}  {time.time()-t0:.0f}s", flush=True)
    for v in saved.values():
        assert np.isfinite(v).all(), "non-finite activations"
    np.savez_compressed(out, layer_types=np.array(list(model.config.layer_types)), **saved)
    print(f"wrote {out}  {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
