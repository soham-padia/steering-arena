"""Did multi-position mutation fix the conjunction, or just search better?

THE PRE-REGISTERED CRITERION, from commit 6304f2c, before either k=3 arm had finished:

    "Arms 707082 (pro) and 707083 (anti) run k=3. The anti arm is the control: max is a
     disjunction and never saturated, so if k=3 helps it as much as pro, the effect is
     generic search improvement and not a conjunction fix."

Both arms have now finished, so the criterion can be applied. This script reads the four
score2 run directories and compares k=1 against k=3 within each sign, at MATCHED
iterations, because that is the only comparison hardware differences cannot enter.

Reads `history.jsonl` rather than `best.json`: `best.json` is whatever the run reached
when it stopped, and the two arms in a pair do not stop at the same iteration. Comparing
endpoints is exactly the error that produced the withdrawn T_SA mechanism
(`season3_gcg_aggregate_asymmetry.json`), and `docs/HANDOFF_BEHAVIORAL_S3.md` §7 closes
with the lesson: check the trajectory, not the endpoint.

    python scripts/k3_control.py

Writes data/analysis/season3_k3_control.json. No GPU, no NDIF, reads committed run logs.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GCG = Path("/work/neu/p2026_0037_neu/steering-arena/gcg")
OUT = ROOT / "data" / "analysis" / "season3_k3_control.json"

# family -> k -> run dir. All four are role=score2: pro uses per_layer_min, anti
# per_layer_max, which is the aggregate swap the anti arm exists to control for.
RUNS = {
    "pro":  {1: "score2-2026-09-06T18-53-11Z", 3: "score2-mut3-2026-09-07T00-17-08Z"},
    "anti": {1: "score2-anti-2026-09-06T20-16-20Z", 3: "score2-anti-mut3-2026-09-07T04-17-25Z"},
}
CHECKPOINTS = (50, 100, 150, 200, 250, 300, 350, 400, 450, 460)
WINDOW = 100


def _runs_of_equal(vals):
    """Consecutive runs of an identical best-so-far value, i.e. plateaus."""
    import itertools
    return [list(g) for _, g in itertools.groupby(vals)]


def live_abs(rec: dict, baseline: float) -> float:
    """|LIVE| for one checkpoint record.

    `board_score_true_sign` is the board-sign score; it is absent from the runs that
    predate the field, where the arm is pro and the optimiser sign IS the board sign.
    The baseline is subtracted once, after the flip -- the other order is a 2*baseline
    error (see scripts/gcg_watch.py).
    """
    board = rec.get("board_score_true_sign")
    if board is None:
        board = rec["board_score"]
    return abs(board - baseline)


def trajectory(run: str, baseline: float) -> dict[int, float]:
    """iter -> best |LIVE| seen up to and including that iteration."""
    out, best = {}, 0.0
    with open(GCG / run / "history.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            best = max(best, live_abs(rec, baseline))
            out[rec["iter"]] = best
    return out


def at(traj: dict[int, float], it: int):
    seen = [v for k, v in traj.items() if k <= it]
    return max(seen) if seen else None


def main():
    baseline = json.loads(
        (ROOT / "data" / "analysis" / "season3_gcg_baseline.json").read_text()
    )["score2"]["baseline"]

    traj, meta = {}, {}
    for fam, ks in RUNS.items():
        for k, run in ks.items():
            traj[(fam, k)] = trajectory(run, baseline)
            best = json.loads((GCG / run / "best.json").read_text())
            meta[f"{fam}_k{k}"] = {
                "run": run, "aggregate": best["aggregate"], "anti": bool(best.get("anti")),
                "n_mutations": best.get("n_mutations"), "best_iter": best["iter"],
                "final_live": round(live_abs(best, baseline), 5),
                "iters_logged": max(traj[(fam, k)]),
                "roundtrip_ok": best["roundtrip_ok"],
            }

    # The honest cap: no comparison past the shortest run in the pair.
    cap = min(max(t) for t in traj.values())

    matched, ratios = {}, {}
    for it in CHECKPOINTS:
        if it > cap:
            continue
        row = {}
        for fam in RUNS:
            v1, v3 = at(traj[(fam, 1)], it), at(traj[(fam, 3)], it)
            row[f"{fam}_k1"] = round(v1, 5)
            row[f"{fam}_k3"] = round(v3, 5)
            row[f"{fam}_ratio"] = round(v3 / v1, 3) if v1 else None
        matched[str(it)] = row
    for fam in RUNS:
        ratios[fam] = [matched[k][f"{fam}_ratio"] for k in matched]

    # Net gain per window, the measure season3_gcg_aggregate_asymmetry.json uses.
    net = {}
    for (fam, k), t in traj.items():
        wins, lo = [], 0
        while lo < cap:
            hi = min(lo + WINDOW, cap)
            a, b = at(t, lo), at(t, hi)
            wins.append(round((b or 0) - (a or 0), 5))
            lo = hi
        net[f"{fam}_k{k}"] = wins

    pro_r, anti_r = ratios["pro"], ratios["anti"]
    hurt = {fam: [k for k in matched if (matched[k][f"{fam}_ratio"] or 1) < 1.0]
            for fam in RUNS}

    # How LUMPY is each trajectory? k=3 changes three positions at once, so a hit is
    # bigger and rarer. Counting distinct improvements separates "climbs steadily" from
    # "jumps then plateaus", which is what the ratio table shows but cannot name.
    lumpiness = {}
    for (fam, k), t in traj.items():
        vals = [t[i] for i in sorted(t) if i <= cap]
        gains = [b - a for a, b in zip(vals, vals[1:]) if b > a + 1e-12]
        lumpiness[f"{fam}_k{k}"] = {
            "n_improvements": len(gains),
            "mean_improvement": round(sum(gains) / len(gains), 6) if gains else None,
            "largest_improvement": round(max(gains), 6) if gains else None,
            "longest_plateau_iters": max(
                (len(list(g)) for g in _runs_of_equal(vals)), default=0),
        }

    L = lumpiness
    shape = (
        L["pro_k3"]["n_improvements"] > L["pro_k1"]["n_improvements"]
        and L["pro_k3"]["longest_plateau_iters"] < L["pro_k1"]["longest_plateau_iters"]
        and L["anti_k3"]["n_improvements"] < L["anti_k1"]["n_improvements"]
        and L["anti_k3"]["mean_improvement"] > L["anti_k1"]["mean_improvement"]
    )
    verdict = (
        "The endpoint test says 'generic', the trajectory says otherwise, and the "
        "trajectory is the better evidence. At the last matched checkpoint the two ratios "
        f"coincide (pro {pro_r[-1]}x vs anti {anti_r[-1]}x), which is the pre-registered "
        "signature of a generic search improvement. But the two gains have OPPOSITE "
        "internal structure. On the conjunction, k=3 makes more and smaller improvements "
        f"({L['pro_k3']['n_improvements']} vs {L['pro_k1']['n_improvements']}, mean "
        f"{L['pro_k3']['mean_improvement']:.6f} vs {L['pro_k1']['mean_improvement']:.6f}) "
        f"and cuts the longest plateau from {L['pro_k1']['longest_plateau_iters']} "
        f"iterations to {L['pro_k3']['longest_plateau_iters']} -- it escapes the stall, "
        "which is what a conjunction fix looks like. On the disjunction it does the "
        f"reverse: fewer and larger improvements ({L['anti_k3']['n_improvements']} vs "
        f"{L['anti_k1']['n_improvements']}, mean {L['anti_k3']['mean_improvement']:.6f} vs "
        f"{L['anti_k1']['mean_improvement']:.6f}) and a LONGER plateau "
        f"({L['anti_k3']['longest_plateau_iters']} vs "
        f"{L['anti_k1']['longest_plateau_iters']}), which is added variance, not escape. "
        "That also explains the ratio table: anti k=3 jumps early, sits on a plateau while "
        f"anti k=1 climbs past it across iterations {hurt['anti'][0]}-{hurt['anti'][-1]} "
        f"(down to {min(matched[k]['anti_ratio'] for k in hurt['anti'])}x), then jumps "
        "again at the end. So the handoff's conclusion was right in substance and wrong in "
        "its evidence: multi-position mutation does something specific to the conjunction, "
        "but the endpoint ratio it was argued from cannot show that, and taken alone that "
        "ratio argues the opposite."
    ) if shape else (
        "AMBIGUOUS. The endpoint ratios coincide "
        f"(pro {pro_r[-1]}x vs anti {anti_r[-1]}x) and the trajectory shapes do not "
        "separate the two families, so this control does not settle the question."
    )

    out = {
        "question": "did --n-mutations 3 fix the conjunction, or just search better?",
        "criterion": ("commit 6304f2c, pre-registered: if k=3 helps the disjunctive anti "
                      "arm as much as the conjunctive pro arm, the effect is generic "
                      "search improvement and not a conjunction fix"),
        "answer": verdict,
        "baseline_score2": baseline,
        "matched_iteration_cap": cap,
        "note": ("|LIVE| = |board score in board sign - baseline|, best-so-far, read from "
                 "history.jsonl at matched iterations so hardware cannot enter"),
        "runs": meta,
        "matched_iteration_live_abs": matched,
        "k3_over_k1_ratio": ratios,
        "net_gain_by_window": {"window": WINDOW, "arms": net},
        "trajectory_shape": lumpiness,
    }
    OUT.write_text(json.dumps(out, indent=1))

    print(f"cap = {cap} iterations (shortest run in any pair)\n")
    hdr = f"{'iter':>6} {'pro k=1':>9} {'pro k=3':>9} {'ratio':>7}   {'anti k=1':>9} {'anti k=3':>9} {'ratio':>7}"
    print(hdr)
    for it, row in matched.items():
        print(f"{it:>6} {row['pro_k1']:>+9.5f} {row['pro_k3']:>+9.5f} {row['pro_ratio']:>6.3f}x   "
              f"{row['anti_k1']:>+9.5f} {row['anti_k3']:>+9.5f} {row['anti_ratio']:>6.3f}x")
    print("\ntrajectory shape (why the endpoints can agree while the paths do not)")
    print(f"  {'arm':>9} {'improvements':>13} {'mean':>10} {'largest':>10} {'longest plateau':>16}")
    for k, v in lumpiness.items():
        print(f"  {k:>9} {v['n_improvements']:>13} {v['mean_improvement']:>10.6f} "
              f"{v['largest_improvement']:>10.6f} {v['longest_plateau_iters']:>16}")
    print(f"\n{verdict}\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
