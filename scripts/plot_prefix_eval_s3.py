"""The Part A' figure: does a banded metric score predict what the model SAYS?

THE CLAIM IT DRAWS, in four panels:

  A1/A2  metric against behaviour, one panel per objective. Score 2 orders all six arms
         with no inversions; Score 1 has exactly one, and it is at the TOP -- its
         highest-scoring string produces a smaller behavioural effect than the Score-2
         winner. The connecting line is drawn in score order, so an inversion is
         literally a dip in the line.
  B      the loop control. Win rate over decided pairs, before and after dropping pairs
         where either continuation degenerates. score2_anti walks to exactly 0.500: its
         apparent anti-human effect was repetition. score1_anti survives.
  C      what the prefix does to the TEXT. Loops and prefix-vocabulary leakage, both out
         of 50, so one axis carries both. The random control leaks nothing, which is why
         it rules out a length artifact and not a content one.

Every number is read from the committed artifacts at run time. Nothing is typed in.

    python scripts/plot_prefix_eval_s3.py

Writes data/analysis/figures/prefix_eval_s3.{png,svg} (wide, for a slide) and
prefix_eval_s3_doc.{png,svg} (6.5in: 1pt here = 1pt on the page), matching the
mechanism.png / mechanism_doc.png pair.

COLOUR. Hue means ONE thing in every panel -- the arm family -- so a colour never has to
be re-learned between panels: blue = a GCG pro arm, aqua = the hand-written pro prefix,
grey = the null (random control and base), orange = an anti arm. The two OBJECTIVES in
row A and the two MEASURES in panel C are separated by line style, marker fill and hatch
instead of by hue, which keeps the categorical set at three chromatic slots.

Blue/orange/aqua is the validated categorical triple and the most this figure can carry:
all pairs clear the CVD floor (worst min(protan, deutan) dE 9.2, orange vs aqua) and the
normal-vision floor (worst 24.0, blue vs aqua). Aqua sits at contrast 2.74 on this
surface, under 3:1, so the relief rule applies -- every family also carries a distinct
marker shape and a direct label, and identity is never colour alone.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
# deterministic SVG: fixed salt for element ids, so regenerating does not dirty the tree
matplotlib.rcParams["svg.hashsalt"] = "steering-arena-prefixevals3"
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
AN = ROOT / "data" / "analysis"
OUT = AN / "figures"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8985"
SURFACE = "#fcfcfb"
GRID = "#e6e5e2"
RATER = "claude-opus-5/v2"
ARM = {}   # built by set_theme, because the entries capture the hue values


def set_theme(dark: bool) -> None:
    """Swap the whole token set. Dark is SELECTED, not an automatic inversion.

    Re-validated against the surface it will actually sit on, which for the slide deck
    is pure #000000 (its master background is scheme dk1), not the validator's default
    #1a1a19. Orange has to move: #eb6834 sits at OKLCH L 0.671, one thousandth over the
    dark band ceiling of 0.67, so it snaps 2% toward black to #e66633 at L 0.661. Blue
    and aqua are already in band. The snapped triple passes every check over ALL pairs,
    not just adjacent: CVD worst min(protan, deutan) dE 8.9 (orange vs aqua),
    normal-vision worst dE 24.0 (blue vs aqua), contrast 4.76 / 6.31 / 7.46 on black.
    Verified with _local/validate_palette.py, the repo's port of the skill's validator.
    """
    global BLUE, ORANGE, AQUA, INK, INK2, MUTED, SURFACE, GRID, ARM
    if dark:
        BLUE, ORANGE, AQUA = "#2a78d6", "#e66633", "#1baf7a"
        INK, INK2, MUTED = "#f2f1ee", "#c9c7c2", "#8a8985"
        SURFACE, GRID = "#000000", "#26262a"
    else:
        BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
        INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8985"
        SURFACE, GRID = "#fcfcfb", "#e6e5e2"
    # Marker is a second, independent channel for identity, so the two families can each
    # carry three shapes without the hue count going up.
    ARM = {
        "score1_top":        ("score1 top",   BLUE,   "o"),
        "score2_top":        ("score2 top",   BLUE,   "s"),
        "score2_top_final":  ("score2 FINAL", BLUE,   "P"),
        "pro_coherent":      ("pro coherent", AQUA,   "D"),
        "random32":          ("random 32",    MUTED,  "X"),
        "score1_anti":       ("score1 anti",  ORANGE, "v"),
        "score2_anti":       ("score2 anti",  ORANGE, "^"),
        "score2_anti_final": ("score2 A-FIN", ORANGE, "<"),
    }


def load():
    ev = json.loads((AN / "prefix_eval_s3.json").read_text())
    sc = json.loads((AN / "season3_prefix_scores.json").read_text())["scores"]
    dg = json.loads((AN / "prefix_degeneration_s3.json").read_text())["arms"]
    ct = json.loads((AN / "prefix_content_s3.json").read_text())["arms"]
    arms = ev["arm_names"]
    d = {}
    for a in arms:
        e = ev["arms"][a][RATER]
        k = ev["arms"][a]["kindness"][RATER]
        rate = lambda b: b["prefixed_preferred"] / (b["prefixed_preferred"] + b["base_preferred"])
        d[a] = {
            "s1": sc[a]["score1_live"], "s2": sc[a]["score2_live"],
            "dfix": k["delta_mean_fixed_baseline"], "p": k["p_fixed_baseline"],
            "all": rate(e["all"]), "no_loop": rate(e["no_loop"]),
            "n_all": e["all"]["prefixed_preferred"] + e["all"]["base_preferred"],
            "n_nl": e["no_loop"]["prefixed_preferred"] + e["no_loop"]["base_preferred"],
            "p_all": e["all"]["sign_test_p"], "p_nl": e["no_loop"]["sign_test_p"],
            "loops": dg[a]["looping_n"], "leak": ct[a]["continuations_with_a_prefix_word"],
        }
    return arms, d, dg["base"]["looping_n"], ev


def spearman(xs, ys):
    """rho and the discordant-pair count. Six points, no ties -- the rank form is exact
    and avoids a scipy import for two numbers."""
    rank = lambda v: [sorted(v).index(x) for x in v]
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    dsq = sum((a - b) ** 2 for a, b in zip(rx, ry))
    rho = 1 - 6 * dsq / (n * (n * n - 1))
    disc = sum(1 for i in range(n) for j in range(i + 1, n)
               if (xs[i] - xs[j]) * (ys[i] - ys[j]) < 0)
    return rho, disc


def panel_score(ax, arms, d, key, title, fs, label_side, ms):
    """One objective against behaviour, points joined in SCORE order: a monotone line
    means the metric ranks behaviour correctly, and an inversion is a visible dip."""
    order = sorted(arms, key=lambda a: d[a][key])
    xs = [d[a][key] for a in order]
    ys = [d[a]["dfix"] for a in order]
    rho, disc = spearman([d[a][key] for a in arms], [d[a]["dfix"] for a in arms])

    ax.axhline(0, color=GRID, lw=1, zorder=0)
    ax.axvline(0, color=GRID, lw=1, zorder=0)
    ax.plot(xs, ys, color=MUTED, lw=1.1, ls="-" if key == "s1" else (0, (4, 2.5)),
            zorder=2, solid_capstyle="round")
    for a in order:
        lab, hue, mk = ARM[a]
        # 2px surface ring on every mark: the two anti points sit 2.3e-4 apart on Score 2
        # and would otherwise merge into one blob
        ax.scatter([d[a][key]], [d[a]["dfix"]], s=ms, marker=mk, color=hue,
                   edgecolor=SURFACE, linewidth=1.0, zorder=4)
    for a in order:
        lab, hue, mk = ARM[a]
        dx, dy, ha = label_side.get(a, (7, 0, "left"))
        ax.annotate(lab, (d[a][key], d[a]["dfix"]), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, va="center", fontsize=fs - 1.6, color=INK2)

    ax.set_title(title, fontsize=fs + 0.6, color=INK, pad=6, loc="left")
    ax.set_xlabel("leaderboard score (LIVE)", fontsize=fs - 0.7, color=INK2)
    npairs = len(arms) * (len(arms) - 1) // 2
    note = (f"$\\rho$ = {rho:+.3f}   ·   "
            + ("no inversions" if disc == 0
               else f"{disc}/{npairs} inversions"))
    ax.text(0.03, 0.955, note, transform=ax.transAxes, fontsize=fs - 0.9,
            color=INK if disc == 0 else ORANGE, va="top",
            fontweight="bold" if disc == 0 else "normal")
    ax.tick_params(colors=INK2, labelsize=fs - 1.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    # generous margins: labels are placed in offset POINTS, so the data limits know
    # nothing about them and a corner label lands outside the axes without this
    ax.margins(x=0.17, y=0.15)
    return rho, disc


def panel_loop(ax, arms, d, fs, ms):
    """Win rate before and after the loop control, as a walk from `all` to `no_loop`."""
    ys = range(len(arms))
    ax.axvline(0.5, color=MUTED, lw=1.1, ls=(0, (3, 3)), zorder=1)
    ax.text(0.5, len(arms) - 0.42, "no effect", fontsize=fs - 2.0, color=MUTED,
            ha="center", va="bottom")
    for y, a in zip(ys, arms):
        lab, hue, mk = ARM[a]
        x0, x1 = d[a]["all"], d[a]["no_loop"]
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="-|>", color=hue, lw=1.4,
                                    shrinkA=3.5, shrinkB=0, mutation_scale=9), zorder=3)
        ax.scatter([x0], [y], s=ms * 0.47, marker="o", facecolor=SURFACE, edgecolor=hue,
                   linewidth=1.3, zorder=4)
        ax.scatter([x1], [y], s=ms * 0.90, marker=mk, color=hue, edgecolor=SURFACE,
                   linewidth=0.9, zorder=5)
        # only the arm whose effect the control ERASES gets a called-out number
        if d[a]["p_nl"] >= 0.999:
            ax.annotate(f"{x1:.3f}\np={d[a]['p_nl']:.2f}", (x1, y),
                        textcoords="offset points", xytext=(0, -21), ha="center",
                        fontsize=fs - 2.0, color=ORANGE, fontweight="bold",
                        linespacing=1.25)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([ARM[a][0] for a in arms], fontsize=fs - 1.6, color=INK2)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.6, len(arms) - 0.15)
    ax.set_xlabel("prefixed preferred, share of decided pairs", fontsize=fs - 0.7, color=INK2)
    ax.set_title("B   the loop control: open dot → arrow → solid mark",
                 fontsize=fs + 0.6, color=INK, pad=6, loc="left")
    ax.tick_params(colors=INK2, labelsize=fs - 1.8, length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)


def panel_text(ax, arms, d, base_loops, fs, legend_dy):
    """Loops and prefix-vocabulary leakage, both counts out of 50, so one axis serves
    both. Hue stays the arm family; the two measures are separated by hatch."""
    h = 0.36
    for i, a in enumerate(arms):
        lab, hue, mk = ARM[a]
        ax.barh(i + h / 2 + 0.02, d[a]["loops"], height=h, color=hue,
                edgecolor=SURFACE, linewidth=1.2, zorder=3)
        ax.barh(i - h / 2 - 0.02, d[a]["leak"], height=h, color=hue, alpha=0.42,
                edgecolor=hue, linewidth=0.9, hatch="////", zorder=3)
        ax.annotate(f"{d[a]['loops']}", (d[a]["loops"], i + h / 2 + 0.02),
                    textcoords="offset points", xytext=(4, 0), va="center",
                    fontsize=fs - 2.2, color=INK2)
        ax.annotate(f"{d[a]['leak']}", (d[a]["leak"], i - h / 2 - 0.02),
                    textcoords="offset points", xytext=(4, 0), va="center",
                    fontsize=fs - 2.2, color=INK2)
    ax.axvline(base_loops, color=MUTED, lw=1.1, ls=(0, (3, 3)), zorder=2)
    ax.annotate(f"base loops {base_loops}/50", (base_loops, len(arms) - 0.45),
                textcoords="offset points", xytext=(5, 0), fontsize=fs - 2.0,
                color=MUTED, va="center")
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([ARM[a][0] for a in arms], fontsize=fs - 1.6, color=INK2)
    ax.set_ylim(-0.62, len(arms) - 0.15)
    ax.set_xlabel("continuations out of 50", fontsize=fs - 0.7, color=INK2)
    ax.set_title("C   what the prefix does to the text", fontsize=fs + 0.6,
                 color=INK, pad=6, loc="left")
    ax.tick_params(colors=INK2, labelsize=fs - 1.8, length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    # xlim leaves room for the end-of-bar value labels, which are drawn in offset
    # points and so are invisible to the autoscaler
    ax.set_xlim(0, max(max(d[a]["loops"], d[a]["leak"]) for a in arms) * 1.16)
    # the legend goes BELOW the axes: inside, it landed on the longest bar and hid its
    # value, and every in-axes corner is occupied by one arm or another
    ax.legend(handles=[
        Patch(facecolor=MUTED, edgecolor=SURFACE, label="loops (a 4-gram 3+ times)"),
        Patch(facecolor=MUTED, alpha=0.42, edgecolor=MUTED, hatch="////",
              label="contains a word from its own prefix")],
        loc="upper right", bbox_to_anchor=(1.02, legend_dy), ncol=2, frameon=False,
        fontsize=fs - 2.2, labelcolor=INK2, handlelength=1.5, handleheight=0.9,
        borderpad=0.2, columnspacing=1.2)


def make(stem, figsize, fs, wide, ev, arms, d, base_loops):
    fig = plt.figure(figsize=figsize, facecolor=SURFACE)
    # the row gap and the header block are both width-dependent: at 6.5in the header
    # needs four wrapped lines instead of two, and 0.55 of row height between the rows
    # opens a visible hole
    ms = 64 if wide else 34
    gs = fig.add_gridspec(2, 2, hspace=0.55 if wide else 0.42, wspace=0.20,
                          left=0.085, right=0.975,
                          top=0.845 if wide else 0.815, bottom=0.175 if wide else 0.145)
    a1 = fig.add_subplot(gs[0, 0]); a1.set_facecolor(SURFACE)
    a2 = fig.add_subplot(gs[0, 1], sharey=a1); a2.set_facecolor(SURFACE)
    b = fig.add_subplot(gs[1, 0]); b.set_facecolor(SURFACE)
    c = fig.add_subplot(gs[1, 1]); c.set_facecolor(SURFACE)

    # label offsets, hand-placed per panel: the two anti arms nearly coincide on Score 2
    off1 = {"score1_top": (-8, 0, "right"), "score2_top": (-9, 8, "right"),
            "score2_top_final": (9, 6, "left"), "pro_coherent": (8, -3, "left"),
            "random32": (8, -4, "left"), "score1_anti": (0, -14, "center"),
            "score2_anti": (9, 5, "left"), "score2_anti_final": (0, 12, "center")}
    # A2: score1_anti/score2_anti sit 2.3e-4 apart, so those two labels are pushed apart
    # vertically; score2_anti_final is well clear at -0.163.
    off2 = {"score1_top": (10, -2, "left"), "score2_top": (10, 0, "left"),
            "score2_top_final": (0, 13, "center"), "pro_coherent": (-9, -7, "right"),
            "random32": (8, -4, "left"), "score1_anti": (10, -10, "left"),
            "score2_anti": (10, 10, "left"), "score2_anti_final": (0, 12, "center")}
    r1, i1 = panel_score(a1, arms, d, "s1", "A1   Score 1  (mean over the band)", fs, off1, ms)
    r2, i2 = panel_score(a2, arms, d, "s2", "A2   Score 2  (min over the band)", fs, off2, ms)
    gap = abs(d["score1_anti"]["s2"] - d["score2_anti"]["s2"])
    a2.annotate(f"the two anti arms coincide here\n(gap {gap:.1e}) — a Score-2 anti board\n"
                "is topped by a Score-1 anti search",
                xy=(d["score2_anti"]["s2"], d["score2_anti"]["dfix"]),
                xytext=(0.47, 0.10), textcoords="axes fraction",
                fontsize=fs - 2.4, color=MUTED, va="center", linespacing=1.3,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7,
                                ls=(0, (2, 2)), shrinkB=6))
    a1.set_ylabel("blind kindness shift  (1–5, fixed baseline)", fontsize=fs - 0.7,
                  color=INK2)
    a2.tick_params(labelleft=False)
    panel_loop(b, arms, d, fs, ms)
    # the legend hangs below the axes, so its offset is in AXES fractions and a short
    # figure needs a bigger one -- at 5.625in tall, -0.145 lands on the x-axis label
    panel_text(c, arms, d, base_loops, fs, -0.145 if figsize[1] >= 6.5 else -0.30)

    n, rated = ev["total_pairs"], ev["raters"][RATER]["rated"]
    # one rater context per emitted batch. Counted from the batch files when the cache
    # is present; the cache is gitignored, so on a fresh clone the count is simply
    # omitted rather than asserted from a hardcoded number.
    # One rater context per VERDICT file. Counted from out/, not in/: in/ holds only the
    # most recent round's batches, while out/ accumulates every context that ever rated.
    out_dir = ROOT / "data" / "cache" / "prefix_behavioral_s3" / "claude" / "out"
    n_raters = len(list(out_dir.glob("batch_*.json"))) if out_dir.is_dir() else 0
    fig.text(0.085, 0.968 if wide else 0.972,
             "Does a banded steering score predict what the model says?",
             fontsize=fs + 3.4, color=INK, fontweight="bold", va="top")
    if wide:
        lines = [
            f"OLMo-3-32B, Season 3. 50 prompts × {len(arms)} prefixes, {n} blind "
            f"pairs, {rated} "
            f"decided by {n_raters or ''}{' ' if n_raters else ''}independent rater "
            "contexts. Scores are leaderboard units.",
            "Pro prefixes move behaviour; a length-matched RANDOM prefix does not. More "
            "score buys more behaviour on the pro side, not on the anti side.",
        ]
    else:
        lines = [
            f"OLMo-3-32B, Season 3. 50 prompts × {len(arms)} prefixes, "
            f"{n} blind pairs,",
            f"{rated} decided by {n_raters or ''}{' ' if n_raters else ''}"
            "independent rater contexts.",
            "Pro prefixes move behaviour; a length-matched RANDOM prefix does not.",
            "More score buys more behaviour on the pro side, but not on the anti side.",
        ]
    y = 0.930 if wide else 0.944
    step = 0.032 if wide else 0.027
    for i, ln in enumerate(lines):
        fig.text(0.085, y - i * step, ln, fontsize=fs - 1.0, color=INK2, va="top")

    fam = [Line2D([], [], marker="o", ls="", color=BLUE, label="GCG pro arm"),
           Line2D([], [], marker="P", ls="", color=BLUE, label="GCG pro, final k=3 string"),
           Line2D([], [], marker="D", ls="", color=AQUA, label="hand-written pro prefix"),
           Line2D([], [], marker="X", ls="", color=MUTED, label="null control"),
           Line2D([], [], marker="v", ls="", color=ORANGE, label="anti arm"),
           Line2D([], [], marker="<", ls="", color=ORANGE, label="anti, final k=3 string")]
    fig.legend(handles=fam, loc="lower left", bbox_to_anchor=(0.085, 0.016), ncol=6,
               frameon=False, fontsize=fs - 1.4, labelcolor=INK2, handletextpad=0.35,
               columnspacing=1.5)
    fig.text(0.975, 0.006,
             "data/analysis/prefix_eval_s3.md  ·  every number read from the "
             "artifacts at run time",
             fontsize=fs - 2.6, color=MUTED, ha="right")

    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300, facecolor=SURFACE,
                    metadata={"Date": None} if ext == "svg" else None)
    plt.close(fig)
    print(f"wrote {OUT / f'{stem}.png'}  ({figsize[0]}x{figsize[1]}in)")
    return (r1, i1), (r2, i2)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dark", action="store_true",
                    help="also emit prefix_eval_s3_dark.png, sized 10x5.625in to fill a "
                         "16:9 slide on a black master")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    set_theme(False)
    arms, d, base_loops, ev = load()
    s1, s2 = make("prefix_eval_s3", (11.0, 7.6), 10.4, True, ev, arms, d, base_loops)
    make("prefix_eval_s3_doc", (6.5, 7.2), 7.4, False, ev, arms, d, base_loops)
    if a.dark:
        set_theme(True)
        make("prefix_eval_s3_dark", (10.0, 5.625), 8.0, True, ev, arms, d, base_loops)
        set_theme(False)
    npairs = len(arms) * (len(arms) - 1) // 2
    print(f"  Score 1 vs behaviour: rho {s1[0]:+.3f}, {s1[1]}/{npairs} inversions")
    print(f"  Score 2 vs behaviour: rho {s2[0]:+.3f}, {s2[1]}/{npairs} inversions")
    print("  mechanism.png / mechanism_doc.png (Season 2) are untouched.")


if __name__ == "__main__":
    main()
