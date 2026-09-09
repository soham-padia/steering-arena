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
    sd on score2, which is the bug this script exists to catch (see scripts/gcg/watch.py).
  * every number is LIVE (baseline removed), so it is directly comparable to a leaderboard
    entry rather than to the optimiser's internal readout.

NDIF stays canonical for anything published (CLAUDE.md). Local transformers 5.10.2 / sdpa /
bf16 matched canonical NDIF scores to |gap| <= 3.71e-4, Spearman rho = 1.0, 0 rank
inversions of 1225 (calibration/local_vs_ndif_tf5.10.2_sdpa_695054.json) -- which is what
makes a local number usable as a CHECK rather than as a second opinion of unknown quality.

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

import torch as t

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gcg"))
from gcg_utils import (MEAN, MIN, compute_scores_batch,  # noqa: E402
                       load_banded_direction, load_prompt_suffixes, truncate_to_layer)

ROLES = ("score1", "score2")
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
    # The next three exist for the held-out generalisation check and nothing else. A probe
    # set carries its own baseline (the probes' own alignment with d), so --probes without a
    # matching --baselines would offset every score by the difference between the two sets.
    # Never point --probes at anything but season3.json when re-scoring a board entry.
    ap.add_argument("--probes", default="data/probes/season3.json")
    ap.add_argument("--baselines", default="data/analysis/season3_gcg_baseline.json")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

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

    baselines = json.loads((ROOT / a.baselines).read_text())
    probe_path = ROOT / a.probes
    probes = load_prompt_suffixes(probe_path)
    probe_tag = json.loads(probe_path.read_text()).get("probe_set") or probe_path.stem
    if baselines.get("probe_set") not in (None, probe_tag):
        raise SystemExit(f"baseline file is for probe set {baselines['probe_set']!r} but "
                         f"--probes is {probe_tag!r}; recompute with "
                         f"scripts/gcg/baseline_const.py --probes {a.probes}")

    # One model load for both roles. Truncated to the deepest layer either band reads.
    meta_model = load_banded_direction(ROOT / "data" / "directions" / "d_olmo3_s3_score1.npz")[3]
    bands = {r: load_banded_direction(
        ROOT / "data" / "directions" / f"d_olmo3_s3_{r}.npz")[2] for r in ROLES}
    tok = AutoTokenizer.from_pretrained(meta_model["model_id"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"          # compute_scores_batch gathers the last REAL token
    model = AutoModelForCausalLM.from_pretrained(
        meta_model["model_id"], dtype=getattr(t, a.dtype), device_map="auto")
    model.eval()
    truncate_to_layer(model, max(max(b) for b in bands.values()))
    trunk = model.base_model
    dev = model.get_input_embeddings().weight.device

    # Probes carry the leading space of the board's f"{seq} {probe}" composition.
    enc = tok([" " + p for p in probes], padding=True, return_tensors="pt",
              add_special_tokens=False)
    sfx_embed = model.get_input_embeddings()(enc["input_ids"].to(dev))
    ntok = enc["attention_mask"].sum(axis=1)

    def score(seq: str, role: str) -> tuple[float, int]:
        """(live score, n tokens the board would see) for one string under one role."""
        d, per, band, _ = load_banded_direction(
            ROOT / "data" / "directions" / f"d_olmo3_s3_{role}.npz")
        agg = MEAN if role == "score1" else MIN
        dirs = t.tensor(d[None] if agg == MEAN else per, dtype=t.float32)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        ids = tok(seq, add_special_tokens=False)["input_ids"]
        if not ids:
            raise SystemExit(f"{seq!r} tokenises to nothing")
        with t.inference_mode():
            ce = model.get_input_embeddings()(t.tensor([ids], device=dev))
            sc = compute_scores_batch(trunk, ce, sfx_embed, enc["attention_mask"], ntok,
                                      dirs, band, agg)
        return float(sc[0]) - baselines[role]["baseline"], len(ids)

    in_sample = probe_tag == "season3"
    out = {"model_id": meta_model["model_id"], "dtype": a.dtype, "probe_set": probe_tag,
           "probe_file": a.probes, "in_sample": in_sample,
           "bands": bands, "baselines": {r: baselines[r]["baseline"] for r in ROLES},
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
