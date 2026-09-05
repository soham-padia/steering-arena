"""Real `d` against a label-shuffled `d`: what is pro-human, and what is just geometry.

Reads data/analysis/direction_null.json (run scripts/direction_null.py first).

  (a) and (b) are the same measurement on the same colour scale: pairwise cosine between
      the five layer-native directions, fit on the TRUE labels and on randomly swapped
      ones. The point of putting them side by side is that the SHAPE barely changes.
      Adjacent layers agree, distant layers do not, L16 is the odd one out, in both.

  (c) is where they actually differ. On the thing the direction is FOR - separating chosen
      from rejected on held-out pairs - real is 1.00 and the null is a coin flip. On
      cross-layer geometry the gap is real but small, +0.121, though the real value clears
      every one of the 40 null draws.

    python scripts/plot_direction_null.py

Writes data/analysis/figures/direction_null.{png,svg} at 6.5in.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "steering-arena-dirnull"
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
AN = ROOT / "data" / "analysis"
OUT = AN / "figures"
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#8a8985", "#fcfcfb"


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    D = json.loads((AN / "direction_null.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    lab = [f"L{l}" for l in D["layers"]]
    R = np.array(D["real"]["cos_matrix"])
    S = np.array(D["shuffled"]["cos_matrix_mean"])

    fig = plt.figure(figsize=(6.5, 4.75), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.62], width_ratios=[1, 1, 0.05],
                          hspace=0.55, wspace=0.30, left=0.145, right=0.95,
                          top=0.90, bottom=0.13)
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
           fig.add_subplot(gs[1, :])]
    cax = fig.add_subplot(gs[0, 2])
    for ax in axs:
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=INK2, labelsize=7)

    for ax, M, t in ((axs[0], R, "(a) real $d$"),
                     (axs[1], S, f"(b) label-shuffled $d$   (mean of {D['n_shuffle']})")):
        im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(5), lab, fontsize=7)
        ax.set_yticks(range(5), lab, fontsize=7)
        for i, j in itertools.product(range(5), range(5)):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=6.6,
                    color="white" if M[i, j] > 0.62 else INK)
        ax.set_title(t, fontsize=8, color=INK, loc="left", pad=6)
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=6.4, colors=INK2)
    cb.outline.set_visible(False)

    # ---- (c) where they actually differ -----------------------------------
    ax = axs[2]
    rows = [("layer-to-layer\ncosine",
             D["real"]["off_diag_mean"], D["shuffled"]["off_diag_mean"],
             [D["shuffled"]["off_diag_mean"], D["shuffled"]["off_diag_max"]]),
            ("held-out\nseparation",
             D["real"]["held_out_separation"][1], D["shuffled"]["held_out_separation_mean"],
             D["shuffled"]["held_out_separation_range"])]
    y = np.arange(len(rows))
    for k, (name, rv, sv, rng_) in enumerate(rows):
        ax.barh(y[k] + 0.18, rv, 0.32, color=BLUE, zorder=3,
                label="real $d$" if k == 0 else None)
        ax.barh(y[k] - 0.18, sv, 0.32, color=ORANGE, zorder=3,
                label="label-shuffled" if k == 0 else None)
        ax.plot(rng_, [y[k] - 0.18] * 2, color=INK, lw=1.1, zorder=5)
        ax.plot([rng_[1]], [y[k] - 0.18], "|", color=INK, ms=5, zorder=5)
        ax.text(rv + 0.015, y[k] + 0.18, f"{rv:.2f}", va="center", fontsize=7, color=INK2)
        ax.text(max(sv, rng_[1]) + 0.015, y[k] - 0.18, f"{sv:.2f}", va="center",
                fontsize=7, color=INK2)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=7)
    ax.set_xlim(0, 1.10)
    ax.set_ylim(-0.55, 1.55)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title("(c) the gap is enormous on what $d$ is for, and small on the geometry",
                 fontsize=8, color=INK, loc="left", pad=6)
    ax.legend(loc="lower right", frameon=False, fontsize=6.8, labelcolor=INK2,
              handlelength=1.0, borderpad=0.1, bbox_to_anchor=(1.005, -0.04))
    ax.grid(True, axis="x", color="#e6e5e1", lw=0.7, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    fig.text(0.145, 0.045, "black bar = spread of the null draws. The real layer-to-layer "
             f"cosine clears all {D['real_exceeds_n_of_n'][1]} of them,",
             fontsize=6.2, color=MUTED, ha="left")
    fig.text(0.145, 0.017, "but the shape of the matrix in (a) is mostly present in (b) too, "
             "so that shape is not evidence about pro-humanness.",
             fontsize=6.2, color=MUTED, ha="left")

    for ext in ("png", "svg"):
        fig.savefig(OUT / f"direction_null.{ext}", dpi=300, facecolor=SURFACE,
                    metadata={"Date": None} if ext == "svg" else None)
    print(f"wrote {OUT/'direction_null.png'}")


if __name__ == "__main__":
    main()
