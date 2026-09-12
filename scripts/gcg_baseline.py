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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import scoring  # noqa: E402
from app.ndif_client import ResidualReader  # noqa: E402


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
    probes = scoring.load_probes(probe_path)
    tag = json.loads(probe_path.read_text()).get("probe_set") or probe_path.stem
    out = {"probes": len(probes), "probe_set": tag, "probe_file": args.probes}
    reader = None

    for role, agg in (("score1", scoring.BANDED_MEAN), ("score2", scoring.PER_LAYER_MIN)):
        d, per, band, meta = scoring.load_banded_direction(
            ROOT / "data" / "directions" / f"d_olmo3_s3_{role}.npz")

        if reader is None:
            reader = ResidualReader.build(meta["model_id"], "local", prepend_bos=True,
                                          device_map="auto", dtype="bfloat16")

        # LEADING SPACE, DELIBERATELY. This file produces the constant that converts the
        # OPTIMISER's number to a board number (live = board_score - BASELINE), and the
        # optimiser embeds its suffixes as " " + probe, so the baseline must be measured on
        # the same string. app/scoring.py's own baseline reads the BARE probe, because there
        # the join space belongs to compose(seq, probe) -- a different and equally correct
        # quantity for a different frame. They are not interchangeable: dropping the space
        # here silently changes every historical LIVE conversion in gcg_watch.py,
        # k3_control.py and the run logs, which were computed against the spaced form.
        units = scoring.banded_baseline([" " + p for p in probes],
                                        reader.batch_last_resids_layers, band)
        per_probe = scoring._band_cosines(units, d, per, agg)

        base = float(per_probe.mean())
        out[role] = {"band": band, "aggregate": agg, "baseline": base,
                     "per_probe": [round(float(x), 6) for x in per_probe]}
        print(f"  {role:<7} band={band} agg={agg}")
        print(f"          BASELINE = {base:+.6f}   (live = board - this)")

    p = ROOT / args.out
    p.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
