"""Which tokens in a GCG winner actually carry the score?

GCG maximises a dot product between the residual stream and a fixed direction. It does not
optimise meaning, so the presence of a token says nothing about whether that token MEANS
anything relevant -- or whether it is load-bearing at all. This measures it.

Reports LIVE-equivalent scores (baseline subtracted), so every number is comparable to a
leaderboard entry rather than to the optimiser's internal readout.

    python scripts/prefix_ablate.py --run <run_dir>

Writes data/analysis/season3_gcg_ablation.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import scoring  # noqa: E402
from app.ndif_client import ResidualReader  # noqa: E402

# FRAME CHANGE, 2026-09-13. This used to score in the OPTIMISER's frame: prefix embeddings
# concatenated with pre-embedded " " + probe suffixes, then a stored baseline subtracted.
# It now scores in the BOARD's frame via app.scoring.banded_shift, which composes
# f"{seq} {probe}" and re-tokenises the whole string -- the canonical definition, and what
# this file's own docstring always claimed it was doing. Re-running will therefore shift the
# committed numbers in season3_gcg_ablation.json slightly (the join tokenises once rather
# than at an embedding seam). The structural findings on that page -- non-additivity, the
# span split, no single load-bearing token -- are ratios and do not depend on the frame.


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="a gcg run dir containing best.json")
    a = ap.parse_args()

    best = json.loads((Path(a.run) / "best.json").read_text())
    role, ids = best["role"], list(best["ctrl_token_ids"])
    d, per, band, meta = scoring.load_banded_direction(
        ROOT / "data" / "directions" / f"d_olmo3_s3_{role}.npz")
    agg = scoring.BANDED_MEAN if role == "score1" else scoring.PER_LAYER_MIN

    reader = ResidualReader.build(meta["model_id"], "local", prepend_bos=True,
                                  device_map="auto", dtype="bfloat16")
    fn = reader.batch_last_resids_layers
    tok = reader.tokenizer

    probes = scoring.load_probes(ROOT / "data" / "probes" / "season3.json")
    base_units = scoring.banded_baseline(probes, fn, band)
    baseline = float(scoring._band_cosines(base_units, d, per, agg).mean())

    def live(seq):
        """LIVE score for a token-id sequence: decoded, then scored as the board would."""
        if not seq:
            return float("nan")
        s = tok.decode(seq, skip_special_tokens=True)
        if not tok(s, add_special_tokens=False)["input_ids"]:
            return float("nan")
        return float(scoring.banded_shift(s, probes, fn, band, base_units, d,
                                          per_layer=per, aggregate=agg))

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
