"""Per-layer concept profile: the plot everyone makes, and the plot that works.

Reads data/analysis/layer_concept_profile.json (run layer_concept_profile.py first).

  (a) The standard layer-wise probing plot: held-out accuracy per layer. It is flat at
      1.000 because the task saturates, so it cannot rank layers at all. Shown because
      this is the plot that usually gets published and it is the trap.

  (b) The same probes, scored by MARGIN instead - mean (proj_chosen - proj_rejected) on
      unit-normalised residuals - against the label-shuffled null at each layer. Now the
      layers separate, and the null band shows how much of the value is real.

    python scripts/plot_layer_concept.py

Writes data/analysis/figures/layer_concept.{png,svg} at 6.5in.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "steering-arena-layerconcept"
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
AN = ROOT / "data" / "analysis"
OUT = AN / "figures"
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#8a8985", "#fcfcfb"


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    D = json.loads((AN / "layer_concept_profile.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    L = D["layers"]
    x = np.arange(len(L))
    g = lambda side, m, i=0: np.array([D[side][str(l)][m][i] for l in L])

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.5, 2.95), facecolor=SURFACE,
                                   gridspec_kw={"width_ratios": [1.0, 1.22]})
    for ax in (axa, axb):
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=INK2, labelsize=7.5)
        ax.set_xticks(x, [f"L{l}" for l in L], fontsize=7.5)
        ax.grid(True, axis="y", color="#e6e5e1", lw=0.7, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)

    axa.plot(x, g("real", "acc"), "o-", color=BLUE, lw=1.6, ms=5, zorder=3)
    axa.axhline(0.5, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
    axa.text(-0.42, 0.515, "chance", fontsize=6.4, color=MUTED)
    axa.set_ylim(0.42, 1.06)
    axa.set_ylabel("held-out accuracy", fontsize=7.5, color=INK2)
    axa.set_title("(a) the usual plot: flat at 1.00,\nso it cannot rank the layers",
                  fontsize=8, color=INK, loc="left", pad=6)
    for xi, v in zip(x, g("real", "acc")):
        axa.text(xi, v + 0.022, f"{v:.2f}", ha="center", fontsize=6.6, color=INK2)

    rm, rs = g("real", "margin"), g("real", "margin", 1)
    nm, ns = g("null", "margin"), g("null", "margin", 1)
    axb.fill_between(x, nm - 2 * ns, nm + 2 * ns, color=ORANGE, alpha=0.28, zorder=1,
                     label="label-shuffled null ($\\pm2$ sd)")
    axb.plot(x, nm, "-", color=ORANGE, lw=1.2, zorder=2)
    axb.errorbar(x, rm, yerr=rs, fmt="o-", color=BLUE, lw=1.6, ms=5, capsize=3,
                 ecolor=INK2, elinewidth=1, zorder=3, label="real $d$ ($\\pm1$ sd)")
    for xi, v, c in zip(x, rm, g("real", "cohen")):
        axb.text(xi, v + 0.017, f"{v:.3f}", ha="center", fontsize=6.6, color=INK2)
        axb.text(xi, v - 0.026, f"d={c:.1f}", ha="center", fontsize=6.2, color=MUTED)
    axb.set_ylim(-0.01, 0.34)
    axb.set_ylabel("held-out margin  (chosen $-$ rejected)", fontsize=7.5, color=INK2)
    axb.set_title("(b) score by margin instead, against a null:\nlegibility peaks at L24 and fades late",
                  fontsize=8, color=INK, loc="left", pad=6)
    axb.legend(loc="center left", frameon=False, fontsize=6.6, labelcolor=INK2,
               handlelength=1.3, borderpad=0.1, bbox_to_anchor=(0.0, 0.42))

    fig.tight_layout(pad=0.7)
    fig.subplots_adjust(bottom=0.24)
    fig.text(0.012, 0.055, f"{D['n_split']} random splits, {D['n_shuffle']} label-shuffled "
             "draws, 135 pairs. Residuals unit-normalised so magnitude cannot inflate the "
             "margin. d = Cohen's d.", fontsize=6.2, color=MUTED)
    fig.text(0.012, 0.018, "This is DECODABILITY, not use: a concept can be legible at a "
             "layer the model never acts on. The causal version needs intervention per "
             "layer.", fontsize=6.2, color=MUTED)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"layer_concept.{ext}", dpi=300, facecolor=SURFACE,
                    metadata={"Date": None} if ext == "svg" else None)
    print(f"wrote {OUT/'layer_concept.png'}")


if __name__ == "__main__":
    main()
