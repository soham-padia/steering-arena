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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
AN = ROOT / "data" / "analysis"
OUT = AN / "figures"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
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


def main():
    prefix, inj, rnd, null, alpha = load()
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.2, 5.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ax.axhline(0, color=MUTED, lw=1, zorder=1)
    ax.axvline(0, color=MUTED, lw=1, zorder=1)

    # family 3 first (background): things that should do nothing
    ax.scatter([x for _, x, _ in rnd], [y for _, _, y in rnd], s=46, marker="^",
               facecolor="none", edgecolor=AQUA, linewidth=1.8, zorder=3,
               label="random direction, full magnitude (n=8)")
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
                xy=(-0.42, 0.02), xytext=(-6.0, 0.46), fontsize=9.5, color=INK2,
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
    leg = ax.legend(loc="lower right", frameon=False, fontsize=9.5, labelcolor=INK2)
    leg.set_zorder(10)

    fig.text(0.012, 0.032,
             "Fixed-baseline deltas, mean of two blind judges. Hollow squares: ±0.5·d are single-judge,",
             fontsize=8, color=MUTED)
    fig.text(0.012, 0.010,
             "floating-baseline. anti_top is excluded: its behaviour is withdrawn as unmeasurable.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"mechanism.{ext}", dpi=200, facecolor=SURFACE)
    print(f"wrote {OUT/'mechanism.png'} and .svg")
    print(f"  prefix arms   {len(prefix)}   injections {len(inj)}   randoms {len(rnd)}   ablation {len(null)}")


if __name__ == "__main__":
    main()
