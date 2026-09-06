"""All 64 layers: where the concept lives, and where the confounds live.

Reads data/analysis/layer_profile_all64.json (run layer_profile_all64.py first).

WHY THIS FIGURE EXISTS. Every previous layer sweep in this project sampled multiples
of 8 -- 16, 24, 32, 40, 48. OLMo-3 places full_attention at layers [3,7,...,63]
(period 4, offset 3), so all five are sliding_attention and no full-attention layer had
ever been evaluated. Season 3 wants a multi-layer d built on full-attention layers, so
the question is whether attention type predicts probe quality at all.

  (a) Excess margin per layer -- probe margin minus its own label-shuffled null. Depth
      drives it: a smooth unimodal curve peaking at L25. Attention type does not; there
      is no step at any bold tick.
  (b) The confound cosines. cos(d, approach) never reaches zero at ANY depth, which is
      why the approach confound is a property of the seed corpus and cannot be fixed by
      choosing a layer. It is at its lowest around L16.

Bold x tick labels are the 16 full_attention layers; unlabeled minor ticks are the 48
sliding_attention layers.

Colours are the repo's validated categorical triple (blue/orange/aqua). Verified with
_local/validate_palette.py (the Python twin of the dataviz validator; the cluster has no
node), --pairs all: worst min(protan,deutan) dE 9.2, normal-vision dE 24.0, lightness
and chroma pass. Aqua sits at 2.74:1 on the light surface, under the 3:1 floor, so the
relief rule applies: every series is direct-labelled, and those labels are set in
ink rather than the series colour so no text is rendered in the sub-3:1 hue.

    python scripts/plot_layer_all64.py

Writes data/analysis/figures/layer_all64.{png,svg} at 6.5in.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "steering-arena-layerall64"
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
AN = ROOT / "data" / "analysis"
OUT = AN / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#8a8985", "#fcfcfb", "#e6e5e1"


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    D = json.loads((AN / "layer_profile_all64.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    Ls = sorted(int(k) for k in D["layers"])
    g = lambda m: np.array([D["layers"][str(l)][m] for l in Ls], dtype=float)
    FULL = D["full_attention"]
    excess, null = g("excess"), g("null_margin")
    appr, val, lng = g("cos_approach"), g("cos_valence"), g("cos_length")
    x = np.array(Ls)
    peak = int(x[np.argmax(excess)])
    cleanest = int(x[np.argmin(appr)])

    fig, (axa, axb) = plt.subplots(
        2, 1, figsize=(6.5, 5.0), facecolor=SURFACE, sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.0], "hspace": 0.16})

    for ax in (axa, axb):
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=INK2, labelsize=7.5)
        ax.grid(True, axis="y", color=GRID, lw=0.7, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        # every full_attention layer gets a faint rule, so the bold ticks are locatable
        # inside the plot area and any step at a type boundary would be visible
        for L in FULL:
            ax.axvline(L, color=GRID, lw=0.7, zorder=0)

    # (a) concept signal ------------------------------------------------------
    axa.fill_between(x, 0, null, color=MUTED, alpha=0.35, lw=0, zorder=1)
    axa.plot(x, excess, "-", color=BLUE, lw=2.0, zorder=3)
    axa.plot([peak], [excess[x == peak][0]], "o", color=BLUE, ms=7,
             mec=SURFACE, mew=2, zorder=4)
    # One annotation, not two: L24 and L25 differ by 0.0002, so calling them out
    # separately would imply a distinction the numbers do not support.
    axa.annotate(f"peak L{peak}  (shipped L24 is {excess[x == 24][0] - excess.max():+.5f})",
                 (peak, excess[x == peak][0]), (peak + 2.8, 0.293),
                 fontsize=7.5, color=INK)
    axa.text(64.4, excess[-1], "excess\nmargin", fontsize=7, color=INK2, va="center")
    axa.text(52, 0.030, "label-shuffled null", fontsize=6.6, color=INK2)
    axa.set_ylim(0, 0.325)
    axa.set_ylabel("excess margin\n(probe margin − its null)", fontsize=7.5, color=INK2)
    axa.set_title("(a) depth drives the concept signal; attention type does not — "
                  "no step at any bold tick",
                  fontsize=8, color=INK, loc="left", pad=6)

    # (b) confounds -----------------------------------------------------------
    for arr, col, lab, ly in ((appr, BLUE, "cos(d, approach)", None),
                              (val, ORANGE, "cos(d, valence)", None),
                              (lng, AQUA, "cos(d, length)", None)):
        axb.plot(x, arr, "-", color=col, lw=2.0, zorder=3, label=lab)
    # direct labels (also the mandated relief for aqua's sub-3:1 contrast)
    axb.text(64.4, appr[-1], "approach", fontsize=7, color=INK2, va="center")
    axb.text(64.4, val[-1] + 0.008, "valence", fontsize=7, color=INK2, va="center")
    axb.text(64.4, lng[-1] - 0.010, "length", fontsize=7, color=INK2, va="center")
    axb.plot([cleanest], [appr[x == cleanest][0]], "o", color=BLUE, ms=7,
             mec=SURFACE, mew=2, zorder=4)
    axb.annotate(f"lowest L{cleanest}\n{appr[x == cleanest][0]:.3f}",
                 (cleanest, appr[x == cleanest][0]), (cleanest - 1.5, 0.238),
                 fontsize=7, color=INK, ha="center")
    axb.set_ylim(0, 0.395)
    axb.set_ylabel("|cos(d, confound)|", fontsize=7.5, color=INK2)
    axb.set_xlabel("residual-stream layer   —   bold = full_attention, "
                   "minor ticks = sliding_attention", fontsize=7.5, color=INK2)
    axb.set_title("(b) the approach confound never reaches zero at any depth — "
                  "a corpus problem, not a layer problem",
                  fontsize=8, color=INK, loc="left", pad=6)
    leg = axb.legend(fontsize=7, frameon=False, loc="upper center", ncol=3,
                     bbox_to_anchor=(0.5, 1.02))
    for t in leg.get_texts():
        t.set_color(INK2)

    axb.set_xlim(-1.2, 68.5)
    axb.set_xticks(FULL)
    axb.set_xticklabels([str(L) for L in FULL], fontsize=7.5, fontweight="bold")
    axb.set_xticks([L for L in Ls if L not in FULL], minor=True)
    axb.tick_params(axis="x", which="minor", length=2.5, color=MUTED)
    for t in axb.get_xticklabels():
        t.set_color(INK)

    fig.subplots_adjust(left=0.115, right=0.885, top=0.935, bottom=0.105)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"layer_all64.{ext}", dpi=220, facecolor=SURFACE,
                    metadata={"Date": None} if ext == "svg" else None)
    print(f"wrote {OUT/'layer_all64.png'} and .svg")
    print(f"  peak excess L{peak} = {excess.max():.5f} | lowest cos_approach "
          f"L{cleanest} = {appr.min():.4f}")


if __name__ == "__main__":
    main()
