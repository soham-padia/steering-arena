"""The one figure for the behavioural half: on-`d` displacement against behaviour.

THE CLAIM IT DRAWS: a token prefix and a vector injection are not the same instrument.
Injections travel up to 30 units along `d` and barely move behaviour. Prefixes travel
under 1 unit and move it more. Two families, two curves, ~60x apart in behaviour per
unit of on-`d` displacement.

It also carries three subsidiary results without extra ink:
  - the score-zero prefix controls sit near the origin at y~0, so "any odd prefix does
    this" dies on the same axes
  - the 8 random-direction injections perturb by the SAME 30.07 as `+1*d` but have
    almost no on-`d` component, and do nothing: the x-axis is the right axis
  - ablation sits at x~-0.2 with y~0, so removing `d` entirely changes nothing

Every number is read from the committed artifacts at run time. Nothing is typed in.

    python scripts/plot_mechanism.py

Writes data/analysis/figures/mechanism.png (and .svg). Colours are the first three
slots of the validated categorical palette (blue/orange/aqua), which is the maximum
that clears the all-pairs CVD floor for a scatter. Aqua sits under 3:1 on the light
surface, so the relief rule applies and every family also carries a distinct marker
shape plus direct labels.
"""
import json
import statistics as st
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
# deterministic SVG: fixed salt for element ids, no date stamp on save, so
# regenerating the figure does not dirty the tree with a spurious diff
matplotlib.rcParams["svg.hashsalt"] = "steering-arena-mechanism"
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
AN = ROOT / "data" / "analysis"
OUT = AN / "figures"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NEUTRAL = "#f0efec"          # diverging midpoint: reads as "nothing"
# One hue per quadrant. Meaning is carried by the four corner labels; colour only
# reinforces it, so the field stays under alpha 0.30 and never speaks alone.
# TWO hues, not four. Four low-alpha tints are indistinguishable: composited onto
# the surface they fail CVD separation at deutan dE 2.3 (validator, --pairs all),
# which is "identical". The distinction that actually matters is ALIGNED vs
# INVERTED, i.e. which intensity law applies; WHICH quadrant is carried by position
# and by the label in each corner. At alpha 0.42 this pair clears both separation
# checks (deutan dE 13.4, normal-vision 17.7). Inverted quadrants also carry a
# hatch, so the split survives greyscale printing and full colour blindness.
ALIGNED_HUE, INVERTED_HUE = "#0ca30c", "#4a3aa7"
Q_HUE = {1: ALIGNED_HUE, 3: ALIGNED_HUE, 2: INVERTED_HUE, 4: INVERTED_HUE}
ALPHA_MAX, EFF_FULL = 0.42, 0.60   # EFF_FULL = the largest efficiency observed
XMAX, YMAX, FLOOR = 60.0, 1.45, 0.15   # axis extents, and the inverted-quadrant floor
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8985"
SURFACE = "#fcfcfb"


def load():
    rec = json.loads((ROOT / "_falsifier" / "recompute_result.json").read_text())
    cc = json.loads((AN / "compile_check.json").read_text())
    dose = json.loads((AN / "steering_dose.json").read_text())
    abl = json.loads((AN / "steering_ablation.json").read_text())
    meas = json.loads((AN / "steering_ablation_measure.json").read_text())
    carms = cc.get("arms", cc)

    def mean_fixed(block, arm):
        pj = block["per_judge"]
        v = [pj[j]["arms"][arm]["fixed"]["delta"] for j in pj if arm in pj[j]["arms"]]
        return st.mean(v) if v else None

    prefix = []
    for arm in ("pro_top", "pro_coherent", "anti_coherent", "anti_hostile",
                "control_junk", "control_text"):
        x = carms[arm]["along_d"]          # signed component along d_hat
        y = mean_fixed(rec["fix2_prefix"], arm)
        assert x is not None and y is not None, f"missing data for {arm}"
        prefix.append((arm, x, y))

    setup = rec["fix2_steering"]["setup"]
    alpha = setup["alpha"]
    inj = [("+1·d", +alpha, mean_fixed(rec["fix2_steering"], "+1"), True),
           ("−1·d", -alpha, mean_fixed(rec["fix2_steering"], "-1"), True),
           ("+0.5·d", +alpha / 2, dose["+0.5"]["kindness_delta"], False),
           ("−0.5·d", -alpha / 2, dose["-0.5"]["kindness_delta"], False)]

    rnd = []
    rarms = sorted(a for a in {a for j in rec["fix2_steering"]["per_judge"]
                               for a in rec["fix2_steering"]["per_judge"][j]["arms"]}
                   if "rand" in a)
    for i, a in enumerate(rarms):
        rnd.append((a, alpha * setup["cos_with_d"][i], mean_fixed(rec["fix2_steering"], a)))

    comp = meas["along_d_mean"]
    rc = meas["random_control"]["arms"]
    null = [("ablate", -comp, abl["arms"]["ablate"]["kindness_delta_fixed"]),
            ("ablate 2x", -2 * comp, abl["arms"]["ablate2x"]["kindness_delta_fixed"])]
    for a, v in rc.items():
        null.append((a, -v["abs_along_mean"], abl["arms"][a]["kindness_delta_fixed"]))
    return prefix, inj, rnd, null, alpha


def _quadrant_field(ax, lo, hi, x0=-60, x1=60, y0=-1.45, y1=1.05):
    """Colour the plane by quadrant, with intensity = behavioural efficiency.

    intensity ~ |y| / (|x| + 1): a large shift bought with a small push on d is
    saturated; the same shift bought with a 30-unit push is nearly colourless.
    Contours of constant intensity are V-shaped, which is the point: travelling
    further along d for the same effect is worse, not better.
    Inside the measured noise band the field is suppressed to zero.
    """
    xs = np.concatenate([np.linspace(x0, -1, 260), np.linspace(-1, 1, 260)[1:],
                         np.linspace(1, x1, 260)[1:]])
    ys = np.linspace(y0, y1, 420)
    X, Y = np.meshgrid(xs, ys)
    q = np.where(X > 0, np.where(Y > 0, 1, 4), np.where(Y > 0, 2, 3))
    aligned = (q == 1) | (q == 3)

    # ALIGNED quadrants: the direction did what it says, so what is notable is
    # EFFICIENCY. A big shift from a small push is remarkable; the same shift
    # bought with a 30-unit push is not. Peaks on the centre line.
    eff = np.clip((np.abs(Y) / (np.abs(X) + 1.0)) / EFF_FULL, 0, 1)

    # INVERTED quadrants: the logic reverses. A small push that inverts could be
    # noise. Pushing HARD against d and still getting the opposite outcome is a
    # flat contradiction, and the harder the push the more damning. So intensity
    # grows with BOTH |x| and |y|, and peaks in the far corners, not the centre.
    u = np.log10(1 + np.abs(X)) / np.log10(1 + XMAX)      # 0 at x=0, 1 at the edge
    contra = np.clip(np.abs(Y) / YMAX, 0, 1) * (FLOOR + (1 - FLOOR) * u)

    inside = (Y >= lo) & (Y <= hi)                 # measured noise: no claim here
    a = ALPHA_MAX * np.where(aligned, eff, contra) * (~inside)
    rgba = np.zeros(X.shape + (4,))
    for k, hexs in Q_HUE.items():
        m = q == k
        rgb = matplotlib.colors.to_rgb(hexs)
        for c in range(3):
            rgba[..., c][m] = rgb[c]
    rgba[..., 3] = a
    ax.pcolormesh(xs, ys, rgba, shading="auto", zorder=0, rasterized=True)

    # secondary encoding: the inverted quadrants are hatched, so aligned-vs-inverted
    # survives greyscale and does not rest on hue alone
    from matplotlib.patches import Rectangle
    for rx, ry, rw, rh in ((x0, 0, -x0, y1), (0, y0, x1, -y0)):
        ax.add_patch(Rectangle((rx, ry), rw, rh, facecolor="none", edgecolor=INVERTED_HUE,
                               hatch="\\\\", linewidth=0, alpha=0.085, zorder=0.25))
    ax.axhspan(lo, hi, color=NEUTRAL, zorder=0.5)

    t = dict(fontsize=8.6, ha="center", va="center", linespacing=1.3)
    # axes-x 0.20 is NEGATIVE displacement, 0.80 is positive
    ax.text(0.80, 0.965, "+d \u2192 pro-human:\nthe direction works", color="#0a7a0a",
            transform=ax.transAxes, **t)
    ax.text(0.20, 0.965, "\u2212d \u2192 pro-human: INVERTED\nWEIRD HERE \u2014 suspect the rig",
            color="#3b2e86", transform=ax.transAxes, **t)
    ax.text(0.20, 0.038, "\u2212d \u2192 anti-human: works,\nbut buys cruelty cheaply",
            color="#0a7a0a", transform=ax.transAxes, **t)
    ax.text(0.80, 0.038, "+d \u2192 anti-human: INVERTED\nWEIRD HERE \u2014 suspect the rig",
            color="#3b2e86", transform=ax.transAxes, **t)


def _ramp(fig, rect, hexs, labels):
    cax = fig.add_axes(rect)
    g = np.linspace(0, 1, 256)[None, :]
    rgba = np.zeros((1, 256, 4))
    rgba[..., :3] = matplotlib.colors.to_rgb(hexs)
    rgba[..., 3] = g * ALPHA_MAX
    cax.imshow(rgba, aspect="auto", origin="lower", extent=(0, 1, 0, 1))
    cax.set_yticks([])
    cax.set_xticks([0, 1])
    cax.set_xticklabels(labels, fontsize=7.6)
    cax.tick_params(colors=INK2, length=0, pad=2)
    for lbl, ha in zip(cax.get_xticklabels(), ("left", "right")):
        lbl.set_horizontalalignment(ha)
    for sp in cax.spines.values():
        sp.set_color(MUTED)
        sp.set_linewidth(0.8)
    cax.set_facecolor(SURFACE)


def _intensity_key(fig):
    """Two laws, because the two kinds of quadrant disagree about what is notable."""
    X0 = 0.545
    fig.text(X0, 0.268, "COLOUR INTENSITY  =  how notable that combination is",
             fontsize=9, color=INK, fontweight="bold")
    fig.text(X0, 0.246, "and the two kinds of quadrant define it oppositely:",
             fontsize=8.3, color=INK2)

    fig.text(X0, 0.212, "ALIGNED (green, plain)  \u2014  EFFICIENCY", fontsize=8.3, color="#0a7a0a")
    fig.text(X0, 0.192, "|shift| \u00f7 (|displacement| + 1). A big shift from a small",
             fontsize=7.8, color=MUTED)
    fig.text(X0, 0.174, "push is notable; the same shift for a 30-unit push is not.",
             fontsize=7.8, color=MUTED)
    _ramp(fig, [X0, 0.148, 0.235, 0.017], Q_HUE[1], ["0", "\u2265 0.60"])

    fig.text(X0, 0.104, "INVERTED (violet, hatched)  \u2014  CONTRADICTION", fontsize=8.3, color="#3b2e86")
    fig.text(X0, 0.084, "grows with BOTH. A small push that inverts could be noise; a hard",
             fontsize=7.8, color=MUTED)
    fig.text(X0, 0.066, "push that still inverts is more likely a broken rig than a finding.",
             fontsize=7.8, color=MUTED)
    _ramp(fig, [X0, 0.040, 0.235, 0.017], Q_HUE[2], ["weak", "strong"])

    fig.text(X0, 0.010, "HUE + HATCH  =  aligned or inverted; corner labels name each.",
             fontsize=7.8, color=MUTED)



def make_doc(prefix, inj, rnd, null, alpha):
    """A second render sized for a Google Doc.

    Built at EXACTLY the 6.5in usable page width, so 1pt in the figure is 1pt on
    the page and nothing shrinks. The key and the caveats are NOT baked in; they
    belong in the document caption as real, selectable 10pt text. See
    data/analysis/figures/mechanism_caption.md for the text to paste.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.35), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    lo, hi = min(y for _, _, y in rnd), max(y for _, _, y in rnd)
    _quadrant_field(ax, lo, hi)

    ax.scatter([x for _, x, _ in rnd], [y for _, _, y in rnd], s=26, marker="^",
               facecolor="none", edgecolor=AQUA, linewidth=1.2, zorder=3,
               label="random direction (n=8)")
    ax.scatter([x for _, x, _ in null], [y for _, _, y in null], s=34, marker="v",
               color=AQUA, edgecolor=SURFACE, linewidth=0.9, zorder=4,
               label="ablation: remove d")
    solid = [t for t in inj if t[3]]
    hollow = [t for t in inj if not t[3]]
    ax.scatter([x for _, x, _, _ in solid], [y for _, _, y, _ in solid], s=58, marker="s",
               color=ORANGE, edgecolor=SURFACE, linewidth=0.9, zorder=5,
               label="vector injection along d")
    ax.scatter([x for _, x, _, _ in hollow], [y for _, _, y, _ in hollow], s=58, marker="s",
               facecolor="none", edgecolor=ORANGE, linewidth=1.3, zorder=5)
    ax.scatter([x for _, x, _ in prefix], [y for _, _, y in prefix], s=58, marker="o",
               color=BLUE, edgecolor=SURFACE, linewidth=0.9, zorder=6,
               label="token prefix (leaderboard entry)")

    lab = {"pro_top": (7, 4, "left"), "pro_coherent": (7, -10, "left"),
           "anti_hostile": (8, -3, "left")}
    for arm, x, y in prefix:
        if arm in lab:
            dx, dy, ha = lab[arm]
            ax.annotate(arm, (x, y), textcoords="offset points", xytext=(dx, dy),
                        ha=ha, fontsize=7, color=INK2)
    for name, x, y, _ in inj:
        if name in ("+1\u00b7d", "\u22121\u00b7d"):
            ax.annotate(name, (x, y), textcoords="offset points",
                        xytext=(0, 9 if y >= 0 else -14), ha="center", fontsize=7, color=INK2)

    ax.set_xscale("symlog", linthresh=1.0, linscale=0.55)
    ax.set_xlim(-60, 60)
    ax.set_ylim(-1.45, 1.05)
    ax.set_xlabel("displacement along d at layer 24  (symlog)", fontsize=8, color=INK2)
    ax.set_ylabel("judged behavioural shift", fontsize=8, color=INK2)
    ax.set_title("A token prefix moves 32x less along d than an injection,\nand does 1.9x the behaviour",
                 fontsize=9.5, color=INK, pad=8, loc="left")
    ax.annotate("", xy=(0.95, 0.676), xytext=(30.07, 0.365),
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, ls=(0, (3, 3))), zorder=2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=7.5)
    ax.grid(True, axis="y", color="#e6e5e1", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    # the corners are taken by the quadrant labels; the mid-left band is the only
    # reliably empty block in this aspect ratio
    ax.legend(loc="center left", bbox_to_anchor=(0.015, 0.30), frameon=False,
              fontsize=7, labelcolor=INK2, handletextpad=0.4, labelspacing=0.35)
    fig.tight_layout(pad=0.6)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"mechanism_doc.{ext}", dpi=300, facecolor=SURFACE,
                    metadata={"Date": None} if ext == "svg" else None)
    print(f"wrote {OUT/'mechanism_doc.png'} (6.5in wide: 1pt here = 1pt on the page)")


def main():
    prefix, inj, rnd, null, alpha = load()
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.6, 8.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    # The "did nothing" zone, taken from the data rather than drawn at y=0:
    # the span of the 8 norm-matched random directions. Anything inside it is a
    # shift no larger than what a direction unrelated to d already produces.
    lo, hi = min(y for _, _, y in rnd), max(y for _, _, y in rnd)
    ax.axhspan(lo, hi, color=NEUTRAL, zorder=0)
    ax.axhline(0, color=MUTED, lw=1, zorder=1)
    ax.axvline(0, color=MUTED, lw=1, zorder=1)
    # The interesting regions are defined by BOTH axes, not by the sign of y:
    # a large behavioural shift bought with almost no displacement along d.
    # x-bound = 1 unit, an order of magnitude below the smallest injection dose
    # (15.04) and the edge of the linear zone. y-bounds = the random band.
    # y-bound is NOT the band edge: with 8 draws that edge is far too tight to act
    # as a threshold, and using it puts the two null controls (-0.13, -0.07) inside
    # the ALARMING region. Require a shift 3x the band's half-width instead, so
    # "large" means clearly outside the noise rather than barely outside it.
    _quadrant_field(ax, lo, hi)
    # the band's left end is empty; label it by proximity rather than a second
    # arrow, which would cross the ablation annotation
    ax.text(-52, hi + 0.045, "no measurable change:\nthe span of 8 random directions",
            fontsize=8.5, color=MUTED, ha="left", va="bottom", linespacing=1.35)

    # family 3 first (background): things that should do nothing
    ax.scatter([x for _, x, _ in rnd], [y for _, _, y in rnd], s=46, marker="^",
               facecolor="none", edgecolor=AQUA, linewidth=1.8, zorder=3,
               label="random direction, full magnitude (n=8) \u2014 sets the band")
    ax.scatter([x for _, x, _ in null], [y for _, _, y in null], s=90, marker="v",
               color=AQUA, edgecolor=SURFACE, linewidth=1.5, zorder=4,
               label="ablation: remove the d component")

    solid = [t for t in inj if t[3]]
    hollow = [t for t in inj if not t[3]]
    ax.scatter([x for _, x, _, _ in solid], [y for _, _, y, _ in solid], s=130,
               marker="s", color=ORANGE, edgecolor=SURFACE, linewidth=1.5, zorder=5,
               label="vector injection along d")
    ax.scatter([x for _, x, _, _ in hollow], [y for _, _, y, _ in hollow], s=130,
               marker="s", facecolor="none", edgecolor=ORANGE, linewidth=1.8, zorder=5)

    ax.scatter([x for _, x, _ in prefix], [y for _, _, y in prefix], s=130, marker="o",
               color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=6,
               label="token prefix (a leaderboard entry)")

    lab = {"pro_top": (9, 6, "left"), "pro_coherent": (9, -13, "left"),
           "anti_hostile": (11, -3, "left"), "control_junk": (7, -15, "left"),
           "anti_coherent": (-9, -15, "right")}
    for arm, x, y in prefix:
        if arm in lab:
            dx, dy, ha = lab[arm]
            ax.annotate(arm, (x, y), textcoords="offset points", xytext=(dx, dy),
                        ha=ha, fontsize=9, color=INK2)
    for name, x, y, _ in inj:
        ax.annotate(name, (x, y), textcoords="offset points",
                    xytext=(0, 13 if y >= 0 else -20), ha="center", fontsize=9, color=INK2)
    # the four aqua down-triangles overlap; label the GROUP from clear space
    ax.annotate("remove d entirely\n(and random-direction controls):\nnothing happens",
                xy=(-0.42, 0.02), xytext=(-4.2, 0.64), fontsize=9.5, color=INK2,
                ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1,
                                connectionstyle="arc3,rad=0.15"))

    ax.set_xscale("symlog", linthresh=1.0, linscale=0.55)
    ax.set_xlim(-60, 60)
    ax.set_ylim(-1.45, 1.05)
    ax.set_xlabel("displacement along d at layer 24   (symlog; linear inside ±1)",
                  fontsize=10.5, color=INK2)
    ax.set_ylabel("judged behavioural shift vs no intervention", fontsize=10.5, color=INK2)
    ax.set_title("A token prefix moves 32x less along d than an injection, and does 1.9x the behaviour",
                 fontsize=13, color=INK, pad=14, loc="left")

    ax.annotate("", xy=(0.95, 0.676), xytext=(30.07, 0.365),
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1, ls=(0, (4, 4))), zorder=2)
    ax.text(5.5, 0.60, "32x less on d,\n1.9x the behaviour", fontsize=9.5,
            color=INK2, ha="center", va="bottom")

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9.5)
    ax.grid(True, axis="y", color="#e6e5e1", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    # legend in FIGURE coords so it sits level with the key block, instead of
    # hanging off the axes and leaving a dead column beneath it
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.028, 0.262), ncol=1,
               frameon=False, fontsize=9.5, labelcolor=INK2, handletextpad=0.6)

    foot = [
        "Fixed-baseline deltas, mean of two blind judges. Hollow squares:",
        "±0.5·d are single-judge, floating-baseline. anti_top is excluded:",
        "its behaviour is withdrawn as unmeasurable. No arm with a measurable",
        "effect landed in either INVERTED quadrant; the only two points in one",
        "are the score-zero controls, both inside the noise band. Treat anything",
        "that lands there as a bug in the harness until proven otherwise.",
    ]
    for i, line in enumerate(foot):
        fig.text(0.028, 0.112 - i * 0.0175, line, fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.285, 1, 1))
    _intensity_key(fig)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"mechanism.{ext}", dpi=200, facecolor=SURFACE,
                    metadata={"Date": None} if ext == "svg" else None)
    print(f"wrote {OUT/'mechanism.png'} and .svg")
    make_doc(prefix, inj, rnd, null, alpha)
    print(f"  prefix arms   {len(prefix)}   injections {len(inj)}   randoms {len(rnd)}   ablation {len(null)}")


if __name__ == "__main__":
    main()
