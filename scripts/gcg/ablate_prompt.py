"""Which tokens in a GCG winner actually carry the score?

GCG maximises a dot product between the residual stream and a fixed direction. It does not
optimise meaning, so the presence of a token says nothing about whether that token MEANS
anything relevant -- or whether it is load-bearing at all. This measures it.

Reports LIVE-equivalent scores (baseline subtracted), so every number is comparable to a
leaderboard entry rather than to the optimiser's internal readout.

    python scripts/gcg/ablate_prompt.py --run <run_dir>

Writes data/analysis/season3_gcg_ablation.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch as t

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gcg_utils import (MEAN, MIN, compute_scores_batch,  # noqa: E402
                       load_banded_direction, load_prompt_suffixes, truncate_to_layer)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="a gcg run dir containing best.json")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    best = json.loads((Path(a.run) / "best.json").read_text())
    role, ids = best["role"], list(best["ctrl_token_ids"])
    d, per, band, meta = load_banded_direction(
        ROOT / "data" / "directions" / f"d_olmo3_s3_{role}.npz")
    agg = MEAN if role == "score1" else MIN
    dirs = t.tensor(d[None] if agg == MEAN else per, dtype=t.float32)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    baseline = json.loads((ROOT / "data" / "analysis" / "season3_gcg_baseline.json").read_text())[role]["baseline"]

    tok = AutoTokenizer.from_pretrained(meta["model_id"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(meta["model_id"], dtype=t.bfloat16, device_map="auto")
    model.eval()
    truncate_to_layer(model, max(band))
    trunk = model.base_model

    probes = load_prompt_suffixes(ROOT / "data" / "probes" / "season3.json")
    enc = tok([" " + p for p in probes], padding=True, return_tensors="pt", add_special_tokens=False)
    dev = model.get_input_embeddings().weight.device
    sfx_embed = model.get_input_embeddings()(enc["input_ids"].to(dev))
    ntok = enc["attention_mask"].sum(axis=1)

    def live(seq):
        """LIVE-equivalent score for a token-id sequence: re-tokenised, baseline removed."""
        if not seq:
            return float("nan")
        rt = tok(tok.decode(seq, skip_special_tokens=True), add_special_tokens=False)["input_ids"]
        if not rt:
            return float("nan")
        with t.inference_mode():
            ce = model.get_input_embeddings()(t.tensor([rt], device=dev))
            sc = compute_scores_batch(trunk, ce, sfx_embed, enc["attention_mask"], ntok,
                                      dirs, band, agg)
        return float(sc[0]) - baseline

    base = live(ids)
    out = {"role": role, "band": band, "baseline": baseline, "original": base,
           "original_prompt": best["prompt"], "single_token": [], "spans": []}
    print(f"  {role}  band={band}  ORIGINAL live = {base:+.5f}\n")

    # 1. leave-one-out for every position: how much does each token carry?
    print("  LEAVE-ONE-OUT (drop = how much the score falls without that token)")
    loo = []
    for i in range(len(ids)):
        s = live(ids[:i] + ids[i + 1:])
        loo.append({"pos": i, "token": tok.decode([ids[i]]), "live": s, "drop": base - s})
    for r in sorted(loo, key=lambda r: -r["drop"])[:8]:
        print(f"    pos {r['pos']:>2} {r['token']!r:<16} live {r['live']:+.5f}  drop {r['drop']:+.5f}")
    print("    ... least important:")
    for r in sorted(loo, key=lambda r: r["drop"])[:4]:
        print(f"    pos {r['pos']:>2} {r['token']!r:<16} live {r['live']:+.5f}  drop {r['drop']:+.5f}")
    out["single_token"] = loo

    # 2. substitutions for any offensive/odd token, to see if MEANING matters at all
    print("\n  SUBSTITUTION at the flagged position")
    flagged = [i for i, x in enumerate(ids) if "ussy" in tok.decode([x])]
    for i in flagged:
        row = {"pos": i, "orig": tok.decode([ids[i]]), "subs": {}}
        for repl in (" Kitty", " Sarah", " person", " respectful", " cat", " xyzzy"):
            rid = tok(repl, add_special_tokens=False)["input_ids"]
            s = live(ids[:i] + rid + ids[i + 1:])
            row["subs"][repl] = s
            print(f"    {row['orig']!r} -> {repl!r:<14} live {s:+.5f}  ({s - base:+.5f})")
        out["spans"].append(row)

    # 3. soup vs readable tail -- the question that actually matters
    print("\n  WHICH HALF CARRIES IT?")
    for name, seq in (("head only (0-15, soup)", ids[:16]),
                      ("tail only (16-31, semi-readable)", ids[16:]),
                      ("tail, readable words only",
                       tok(" bystand escalate intervene respectful clarity Ask respectfully",
                           add_special_tokens=False)["input_ids"])):
        s = live(seq)
        out["spans"].append({"span": name, "live": s})
        print(f"    {name:<34} live {s:+.5f}  ({s - base:+.5f} vs full)")

    p = ROOT / "data" / "analysis" / "season3_gcg_ablation.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"\n  wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
