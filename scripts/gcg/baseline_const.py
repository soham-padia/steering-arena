"""The constant that separates a GCG `board_score` from a real leaderboard score.

    live_score = board_score - BASELINE

where BASELINE = mean over probes p of agg_L cos(R_L(p), d), i.e. the probes' own alignment
with the direction before any prefix is attached.

The optimiser omits it because it does not depend on the prefix, so it cannot change which
candidate wins, and skipping it saves a forward pass on up to 1024 candidates per iteration.
That makes GCG's numbers offset from the board's by exactly this constant. Computing it once
makes every number in the run logs convertible instead of merely relative.

Writes data/analysis/season3_gcg_baseline.json. One forward over 16 short probes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch as t

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gcg_utils import (MEAN, MIN, _aggregate, load_banded_direction,  # noqa: E402
                       load_prompt_suffixes, truncate_to_layer)


def main():
    import argparse

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # --probes exists for the held-out generalisation check only. The baseline is a property
    # of the PROBE SET, not of the direction, so a different probe file needs its own constant
    # -- reusing season3's would silently offset every score by the difference between the two
    # sets' own alignment with d.
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probes", default="data/probes/season3.json")
    ap.add_argument("--out", default="data/analysis/season3_gcg_baseline.json")
    # `args`, not `a`: the band loop below rebinds `a` to an activation tensor.
    args = ap.parse_args()

    probe_path = ROOT / args.probes
    probes = load_prompt_suffixes(probe_path)
    tag = json.loads(probe_path.read_text()).get("probe_set") or probe_path.stem
    out = {"probes": len(probes), "probe_set": tag, "probe_file": args.probes}
    model = tokenizer = None

    for role, agg in (("score1", MEAN), ("score2", MIN)):
        d, per, band, meta = load_banded_direction(
            ROOT / "data" / "directions" / f"d_olmo3_s3_{role}.npz")
        dirs = t.tensor(d[None] if agg == MEAN else per, dtype=t.float32)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)

        if model is None:
            tokenizer = AutoTokenizer.from_pretrained(meta["model_id"])
            model = AutoModelForCausalLM.from_pretrained(
                meta["model_id"], dtype=t.bfloat16, device_map="auto")
            model.eval()
            truncate_to_layer(model, 39)   # deepest layer across both bands
            blocks = model.base_model.layers

        # Probes are scored EXACTLY as the board composes them: the leading space is the
        # f"{seq} {probe}" join, minus the (absent) prefix.
        cap = {}
        hs = [blocks[L].register_forward_hook(
            lambda m, i, o, L=L: cap.__setitem__(L, o[0] if isinstance(o, tuple) else o))
            for L in band]
        try:
            per_probe = []
            for p in probes:
                enc = tokenizer(" " + p, return_tensors="pt", add_special_tokens=False)
                with t.inference_mode():
                    model.base_model(**{k: v.to(model.device) for k, v in enc.items()})
                cos = []
                for j, L in enumerate(band):
                    a = cap[L][0, -1, :].cpu().float()
                    dv = dirs[0] if agg == MEAN else dirs[j]
                    cos.append((a @ dv) / a.norm())
                per_probe.append(_aggregate(t.stack(cos)[:, None], agg).item())
        finally:
            for h in hs:
                h.remove()

        base = float(np.mean(per_probe))
        out[role] = {"band": band, "aggregate": agg, "baseline": base,
                     "per_probe": [round(x, 6) for x in per_probe]}
        print(f"  {role:<7} band={band} agg={agg}")
        print(f"          BASELINE = {base:+.6f}   (live = board - this)")

    p = ROOT / args.out
    p.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
