"""Best-so-far board score against GCG iteration, for the four shipped Season 3 arms.

    python scripts/plot_gcg_trajectories.py [--light]
      -> data/analysis/gcg_trajectories.json
      -> data/analysis/figures/gcg_trajectories_{dark,light}.png

Reads `history.jsonl` from the run directories under
/work/neu/p2026_0037_neu/steering-arena/gcg/. No GPU, no NDIF, no network, $0.

WHICH COLUMN. `history.jsonl` carries both `score` (the optimiser's internal
objective) and `board_score` (the LIVE, leaderboard-comparable value). They agree on
pro runs and DIVERGE on anti runs, because the anti objective is maximised while the
board value is its negation. Plotting `score` on an anti run is the "flip first,
subtract once" trap in _local/NEXT_CLAUDE.md section 5 -- it reports score1-anti peaking at
0.13186 (iter 537) when its board peak is 0.10245 (iter 204). This module reads
`board_score` and applies the true sign, so pro curves rise and anti curves fall.

PALETTE. Hue is the POLE (blue pro / orange anti), matching scripts/plot_prefix_eval_s3.py's
existing convention; line style is the OBJECTIVE, replacing that module's marker channel
because these are lines. Two hues, not four, so the categorical count stays low. Verified
on #000000 with _local/validate_palette.py: lightness band, chroma floor, CVD separation
(worst min(protan,deutan) dE 24.6) and contrast all PASS.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.rcParams["svg.hashsalt"] = "steering-arena-gcgtraj"

ROOT = Path(__file__).resolve().parent.parent
GCG = Path("/work/neu/p2026_0037_neu/steering-arena/gcg")
OUT_J = ROOT / "data" / "analysis" / "gcg_trajectories.json"
FIGS = ROOT / "data" / "analysis" / "figures"

# (arm, run dir, label, anti?, objective)
RUNS = [
    ("score1_top",  "score1-2026-09-06T19-20-13Z",       "Score 1  pro",  False, "s1"),
    ("score2_top",  "score2-mut3-2026-09-07T00-17-08Z",  "Score 2  pro",  False, "s2"),
    ("score1_anti", "score1-anti-2026-09-06T20-20-17Z",  "Score 1  anti", True,  "s1"),
    ("score2_anti", "score2-anti-2026-09-06T20-16-20Z",  "Score 2  anti", True,  "s2"),
]


def theme(dark):
    if dark:
        return dict(BLUE="#2a78d6", ORANGE="#e66633", INK="#f2f1ee", INK2="#c9c7c2",
                    MUTED="#8a8985", SURFACE="#000000", GRID="#26262a")
    return dict(BLUE="#2a78d6", ORANGE="#eb6834", INK="#0b0b0b", INK2="#52514e",
                MUTED="#8a8985", SURFACE="#fcfcfb", GRID="#e6e5e2")


def load(run):
    recs = [json.loads(l) for l in (GCG / run / "history.jsonl").read_text().splitlines() if l.strip()]
    return [(r["iter"], r["board_score"]) for r in recs if r.get("board_score") is not None]


def best_so_far(pts, anti):
    """Running max of |board_score|, carrying the true sign. The board ranks by the
    signed value, so 'best' on an anti run is the most negative one reached so far."""
    out, best = [], 0.0
    for it, bs in pts:
        best = max(best, abs(bs))
        out.append((it, -best if anti else best))
    return out


def build(dark=True):
    T = theme(dark)
    shipped = json.loads((ROOT / "data" / "analysis" / "prefix_eval_arms_s3.json").read_text())["arms"]
    summary, series = {}, []
    for arm, run, label, anti, obj in RUNS:
        pts = load(run)
        curve = best_so_far(pts, anti)
        peak_i, peak_v = max(curve, key=lambda p: abs(p[1]))
        ship_it = shipped[arm].get("iter")
        summary[arm] = {
            "run": run, "label": label, "anti": anti, "objective": obj,
            "n_iters": len(pts), "last_iter": pts[-1][0],
            "peak_iter": peak_i, "peak_board_signed": round(peak_v, 5),
            "shipped_iter": ship_it,
            "shipped_board_signed": round(dict(curve).get(ship_it, float("nan")), 5),
            "plateau_frac": round(peak_i / pts[-1][0], 3) if pts[-1][0] else None,
        }
        series.append((arm, label, curve, anti, obj, ship_it))

    fig, ax = plt.subplots(figsize=(10, 5.0), dpi=200)
    fig.patch.set_facecolor(T["SURFACE"])
    ax.set_facecolor(T["SURFACE"])
    ax.axhline(0, color=T["GRID"], lw=1, zorder=0)
    ax.grid(True, color=T["GRID"], lw=0.6, zorder=0)

    for arm, label, curve, anti, obj, ship_it in series:
        xs = [p[0] for p in curve]
        ys = [p[1] for p in curve]
        hue = T["ORANGE"] if anti else T["BLUE"]
        ls = "-" if obj == "s1" else (0, (5, 2.5))
        ax.plot(xs, ys, color=hue, lw=2.0, ls=ls, zorder=3, solid_capstyle="round")
        if ship_it is not None and ship_it in dict(curve):
            sv = dict(curve)[ship_it]
            ax.scatter([ship_it], [sv], s=64, color=hue, edgecolor=T["SURFACE"],
                       linewidth=1.6, zorder=5)
        # direct label at the end of the run (<=4 series, so every line is named in place)
        ax.annotate(label, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    fontsize=9.5, color=T["INK2"], va="center", ha="left")

    ax.set_xlabel("GCG iteration", fontsize=10, color=T["INK2"])
    ax.set_ylabel("best board score so far (LIVE, true sign)", fontsize=10, color=T["INK2"])
    ax.set_title("The search, both poles: best-so-far board score against iteration",
                 fontsize=12.5, color=T["INK"], loc="left", pad=8)
    ax.tick_params(colors=T["INK2"], labelsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(T["MUTED"])
    ax.set_xlim(0, max(max(p[0] for p in c) for _, _, c, _, _, _ in series) * 1.16)

    # the style channel needs a key; identity is already carried by the direct labels
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=T["MUTED"], lw=2.0, ls="-", label="Score 1  (mean over 4 layers)"),
                       Line2D([], [], color=T["MUTED"], lw=2.0, ls=(0, (5, 2.5)), label="Score 2  (min over 4 layers)"),
                       Line2D([], [], color=T["MUTED"], lw=0, marker="o", markersize=7,
                              label="the iterate we shipped")],
              loc="upper left", frameon=False, fontsize=9, labelcolor=T["INK2"])

    fig.tight_layout()
    name = f"gcg_trajectories_{'dark' if dark else 'light'}.png"
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / name, facecolor=T["SURFACE"])
    plt.close(fig)
    OUT_J.write_text(json.dumps({"runs_root": str(GCG), "arms": summary}, indent=2) + "\n")
    print(f"wrote {FIGS / name}")
    print(f"wrote {OUT_J}")
    for a, v in summary.items():
        print(f"  {a:13} peak {v['peak_board_signed']:+.5f} @ iter {v['peak_iter']:>3}"
              f" of {v['last_iter']:>3}  ({v['plateau_frac']:.0%} through)  shipped iter {v['shipped_iter']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--light", action="store_true")
    a = ap.parse_args()
    build(dark=not a.light)
