"""Score arbitrary STRINGS under both Season 3 objectives, locally, in leaderboard units.

score_local.py is the Season 2 reference scorer: one layer, one direction. Season 3 scores
over a BAND, twice -- Score 1 is the mean over layers 19/23/27/31 against one shared `d`,
Score 2 the min over 15/23/31/39 against a direction per layer. This is the missing
counterpart, and it is what verifies a prefix still means what a run recorded:

  * it scores the STRING, re-tokenising it exactly as the board does, so a prefix corrupted
    in transit shows up as a changed score instead of passing silently. That is not
    hypothetical: live row id=1342 scores +0.086 on the board against +0.140 locally,
    because two real newline bytes were copied out of a Python repr as literal backslash-n
    and 38 tokens were submitted where 32 were searched.
  * it always scores the PRO objective, so an anti arm's number comes out in board sign
    with the baseline subtracted exactly ONCE. Subtracting it in the optimiser frame and
    flipping afterwards flips the baseline too -- a 2*baseline error worth up to ~1.3 field
    sd on score2, which is the bug this script exists to catch.
  * every number is LIVE (baseline removed), so it is directly comparable to a leaderboard
    entry rather than to the optimiser's internal readout.

ONE METRIC, TWO BACKENDS. This calls `app.scoring` -- the same pure functions the server
uses -- through `ResidualReader` on the local backend, so a local score and an NDIF score
are now the SAME code with a different residual source rather than two implementations that
happen to agree. `calibration/local_vs_ndif_tf5.10.2_sdpa_695054.json` (max |gap| 3.71e-4,
Spearman 1.0, 0 rank inversions of 1225) is therefore a statement about the backends.
NDIF stays canonical for anything published (CLAUDE.md).

NO BASELINE FILE. `app.scoring.banded_shift` computes the probes' own alignment from the
probe set it is handed, so the LIVE conversion cannot be paired with the wrong constant.
The old `--baselines` flag existed to stop exactly that and is gone with the failure mode:
swapping `--probes` now recomputes the baseline by construction.

    python scripts/score_banded_local.py "be honest and own mistakes"
    python scripts/score_banded_local.py --arms data/analysis/prefix_eval_arms_s3.json
    python scripts/score_banded_local.py --arms ... --update-arms   # fill in unscored arms

Writes data/analysis/season3_prefix_scores.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import scoring  # noqa: E402
from app.ndif_client import ResidualReader  # noqa: E402

ROLES = ("score1", "score2")
AGG = {"score1": scoring.BANDED_MEAN, "score2": scoring.PER_LAYER_MIN}
OUT = ROOT / "data" / "analysis" / "season3_prefix_scores.json"
# A local re-score of a string the same optimiser already scored on the same weights should
# land within noise. Wider than the 3.71e-4 local/NDIF bound because the GCG runs may have
# used a different GPU model, and bf16 reductions are not bitwise stable across them.
TOL = 2e-3


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sequences", nargs="*", help="strings to score")
    ap.add_argument("--arms", default="", help="an arms file; scores every arm and checks "
                                               "each recorded score against the re-score")
    ap.add_argument("--update-arms", action="store_true",
                    help="write measured scores back into the arms file (fills in arms "
                         "whose score_kind is unscored_pending_gpu, and adds `verified`)")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--device-map", default="auto",
                    help='"auto" for a GPU node; "cpu" with --model for a plumbing smoke test')
    ap.add_argument("--model", default="",
                    help="override the model id (smoke tests use a tiny one)")
    ap.add_argument("--probes", default="data/probes/season3.json",
                    help="the probe set. Its own baseline is recomputed from it, so this is "
                         "safe to change -- see the held-out generalisation check.")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    arms_file = Path(a.arms) if a.arms else None
    arms = json.loads(arms_file.read_text()) if arms_file else None
    # name -> (string, role or None). A bare sequence has no role: report both objectives.
    targets: dict[str, tuple[str, str | None]] = {}
    if arms:
        for name in arms["arm_names"]:
            arm = arms["arms"][name]
            targets[name] = (arm["sequence"], arm.get("role"))
    for i, s in enumerate(a.sequences):
        targets[f"argv{i}"] = (s, None)
    if not targets:
        raise SystemExit("nothing to score: pass sequences or --arms")

    probe_path = ROOT / a.probes
    probes = scoring.load_probes(probe_path)
    probe_tag = json.loads(probe_path.read_text()).get("probe_set") or probe_path.stem

    # One direction file per role; they may declare different bands.
    dirs = {r: scoring.load_banded_direction(
        ROOT / "data" / "directions" / f"d_olmo3_s3_{r}.npz") for r in ROLES}
    bands = {r: dirs[r][2] for r in ROLES}
    model_id = a.model or dirs["score1"][3]["model_id"]

    # prepend_bos matches the server's setting. It is a no-op on OLMo-3 (bos_token=None,
    # add_bos_token=False), which is why the two backends can share it unexamined.
    reader = ResidualReader.build(model_id, "local", prepend_bos=True,
                                  device_map=a.device_map, dtype=a.dtype)
    fn = reader.batch_last_resids_layers
    tok = reader.tokenizer

    # Constant per (probes, band): the probes' own alignment, one batched forward each.
    base = {r: scoring.banded_baseline(probes, fn, bands[r]) for r in ROLES}

    def score(seq: str, role: str) -> tuple[float, int]:
        """(live score, n tokens the board would see) for one string under one role."""
        d, per, band, _ = dirs[role]
        ids = tok(seq, add_special_tokens=False)["input_ids"]
        if not ids:
            raise SystemExit(f"{seq!r} tokenises to nothing")
        live = scoring.banded_shift(seq, probes, fn, band, base[role], d,
                                    per_layer=per, aggregate=AGG[role])
        return float(live), len(ids)

    in_sample = probe_tag == "season3"
    out = {"model_id": model_id, "dtype": a.dtype, "probe_set": probe_tag,
           "probe_file": a.probes, "in_sample": in_sample, "backend": "local/nnsight",
           "bands": bands,
           "arms_file": str(arms_file) if arms_file else None, "tol": TOL, "scores": {}}
    print(f"{len(targets)} string(s) · {len(probes)} probes · bands {bands} · [{a.dtype}]\n")
    print(f"  {'name':>13} {'score1':>10} {'score2':>10} {'tok':>4}  recorded / re-scored")
    n_bad = 0
    for name, (seq, role) in targets.items():
        live = {r: score(seq, r) for r in ROLES}
        rec = {"n_tokens": live["score1"][1], "newlines": seq.count("\n"),
               "score1_live": live["score1"][0], "score2_live": live["score2"][0],
               "role": role}
        line = (f"  {name:>13} {live['score1'][0]:>+10.5f} {live['score2'][0]:>+10.5f} "
                f"{rec['n_tokens']:>4}")
        # The recorded score was measured on season3's probes. Against any other probe set a
        # difference is the RESULT, not a fidelity failure, so the gate is skipped.
        if arms and role in ROLES and in_sample:
            want = arms["arms"][name].get("score")
            if want is not None:
                got = live[role][0]
                rec.update(recorded=want, gap=got - want, ok=abs(got - want) <= TOL)
                n_bad += not rec["ok"]
                flag = "ok" if rec["ok"] else "MISMATCH"
                line += f"   {want:>+9.5f} -> {got:>+9.5f}  gap {got - want:>+8.1e} {flag}"
        out["scores"][name] = rec
        print(line)

    dest = (ROOT / a.out) if a.out else OUT
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nwrote {dest.relative_to(ROOT)}")
    if n_bad:
        print(f"  {n_bad} arm(s) did not reproduce their recorded score within {TOL:.0e}. "
              "A string that scores differently than the run recorded is a CORRUPTED "
              "string, not a precision artifact -- check its newlines before using it.")

    if a.update_arms and arms:
        for name, rec in out["scores"].items():
            if name not in arms["arms"]:
                continue
            arm = arms["arms"][name]
            arm["verified"] = {"score1_live": rec["score1_live"],
                               "score2_live": rec["score2_live"],
                               "n_tokens": rec["n_tokens"], "dtype": a.dtype,
                               "gap_vs_recorded": rec.get("gap")}
            if arm.get("score_kind") == "unscored_pending_gpu":
                # A control has no role of its own; Score 1 is the board's primary column.
                arm["score"] = rec["score1_live"]
                arm["score_alt"] = rec["score2_live"]
                arm["score_kind"] = "score1_live"
        arms_file.write_text(json.dumps(arms, indent=2, ensure_ascii=False))
        print(f"  updated {arms_file}")


if __name__ == "__main__":
    main()
