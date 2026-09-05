"""How `d` changes with depth, and what that costs a banded direction.

Everything in this project was fit and scored at layer 24. The banding work raised the
prior question this figure answers: is there ONE pro-human direction that the layers share,
or a different one at every depth? The five native probes live in the same 5120-dim residual
stream, so cosine between them is meaningful and the question is directly measurable.

  (a) Pairwise cosine between the five layer-native directions. If a single feature were
      being read at every depth this would be near 1 everywhere. It is not: adjacent layers
      agree strongly and the endpoints barely do.

  (b) How well one averaged direction can stand in for each member, for two candidate
      bands. This is the band-choice argument in one panel - the late band's mean represents
      all three members at 0.94+, while the all-five mean cannot get near L16.

Zero NDIF calls; reads the committed direction files only.

    python scripts/plot_layer_directions.py

Writes data/analysis/figures/layer_directions.{png,svg} at 6.5in, the usable width of a
Letter page, so 1pt here is 1pt in the document.
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "steering-arena-layerdirs"
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DIRS = ROOT / "data" / "directions"
OUT = ROOT / "data" / "analysis" / "figures"
NATIVE = [16, 24, 32, 40, 48]
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#8a8985", "#fcfcfb"


def unit(v):
    return v / np.linalg.norm(v)


def native(layer):
    z = np.load(DIRS / f"d_olmo3_L{layer}_logistic.npz")
    k = [x for x in z.files if z[x].ndim == 1 and z[x].size > 1000][0]
    return unit(z[k].astype(np.float64))


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    # spurious BLAS float warnings on this platform, documented in compile_check.md
    np.seterr(all="ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    d = {L: native(L) for L in NATIVE}
    M = np.array([[float(d[a] @ d[b]) for b in NATIVE] for a in NATIVE])

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.5, 3.15), facecolor=SURFACE,
                                   gridspec_kw={"width_ratios": [1.0, 1.12]})
    for ax in (axa, axb):
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=INK2, labelsize=7.5)

    # ---- (a) cosine matrix -------------------------------------------------
    im = axa.imshow(M, cmap="Blues", vmin=0, vmax=1)
    axa.set_xticks(range(5), [f"L{l}" for l in NATIVE], fontsize=7.5)
    axa.set_yticks(range(5), [f"L{l}" for l in NATIVE], fontsize=7.5)
    for i, j in itertools.product(range(5), range(5)):
        axa.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=7,
                 color="white" if M[i, j] > 0.62 else INK)
    axa.set_title("(a) the layers do not share one direction\ncos between layer-native $d$",
                  fontsize=8, color=INK, loc="left", pad=6)
    for s in axa.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=axa, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=6.5, colors=INK2)
    cb.outline.set_visible(False)

    # ---- (b) how well a mean stands in ------------------------------------
    bands = [([32, 40, 48], BLUE, "band {32,40,48}"),
             (NATIVE, ORANGE, "all five")]
    x = np.arange(5)
    for off, (band, c, lab) in zip((-0.19, +0.19), bands):
        m = unit(np.mean([d[l] for l in band], axis=0))
        vals = [float(m @ d[l]) if l in band else np.nan for l in NATIVE]
        axb.bar(x + off, [0 if np.isnan(v) else v for v in vals], 0.34, color=c,
                edgecolor=SURFACE, lw=1, zorder=3, label=lab)
        for xi, v in zip(x + off, vals):
            if not np.isnan(v):
                axb.text(xi, v + 0.028, f"{v:.2f}", ha="center", fontsize=6.4, color=INK2)
    axb.axhline(0.9, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
    axb.text(-0.46, 0.908, "0.90", fontsize=6.2, color=MUTED, va="bottom", ha="left")
    axb.set_xticks(x, [f"L{l}" for l in NATIVE], fontsize=7.5)
    axb.set_ylim(0, 1.12)
    axb.set_ylabel("cos(banded mean, that layer's own $d$)", fontsize=7.5, color=INK2)
    axb.set_title("(b) so a single averaged $d$ can only\nrepresent a band, not the whole stack",
                  fontsize=8, color=INK, loc="left", pad=6)
    axb.legend(loc="upper left", frameon=False, fontsize=6.6, labelcolor=INK2,
               handlelength=1.1, borderpad=0.1, bbox_to_anchor=(-0.02, 1.03))
    axb.grid(True, axis="y", color="#e6e5e1", lw=0.7, zorder=0)
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axb.spines[s].set_color(MUTED)

    fig.tight_layout(pad=0.7)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"layer_directions.{ext}", dpi=300, facecolor=SURFACE,
                    metadata={"Date": None} if ext == "svg" else None)
    print(f"wrote {OUT/'layer_directions.png'}")
    print(f"  mean |cos| over the 10 distinct pairs: "
          f"{np.mean([abs(M[i,j]) for i,j in itertools.combinations(range(5),2)]):.3f}")
    for band, _, lab in bands:
        m = unit(np.mean([d[l] for l in band], axis=0))
        cs = [float(m @ d[l]) for l in band]
        print(f"  {lab:<18} min cos with a member {min(cs):.3f}   ({[round(c,3) for c in cs]})")


if __name__ == "__main__":
    main()
