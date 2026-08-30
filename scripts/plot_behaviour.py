"""Section 2 figure: does the winner change behaviour, and is it the words leaking?

TWO PANELS, both computed from committed artifacts at run time.

(a) WIN RATE FOR `pro_top`, by rater, before and after the echo exclusion.
    The prefix contains real English affect words (`reset`, `calm`, `misunderstanding`,
    `hurt`, `clear`). If the effect were just those words leaking into the output, it
    should vanish once you drop every pair where one appears. The exclusion is the UNION
    over the five words, which is the strict version: 15 of 50 prompts drop out for
    `pro_top` against 1 of 50 for base. Wilson intervals, because at n=12 a normal
    approximation is not honest.

(b) REPETITION RATE BY ARM, judge-free in spirit: it is a marker count, not a preference,
    so it does not depend on trusting the kindness rubric. Carries the counterintuitive
    result that the gibberish makes the model MORE coherent, and sets up why the anti arm
    had to be withdrawn.

    python scripts/plot_behaviour.py

Writes data/analysis/figures/behaviour.png at exactly 6.5in wide, the usable width of a
Letter page, so 1pt here is 1pt in the document.
"""
import json
import glob
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "steering-arena-behaviour"
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
AN = ROOT / "data" / "analysis"
OUT = AN / "figures"
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#8a8985", "#fcfcfb"
WORDS = ("reset", "calm", "misunderstanding", "hurt", "clear")


def wilson(w, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def load():
    cont = {}
    for f in glob.glob(str(ROOT / "data/cache/prefix_behavioral/*.json")):
        d = json.loads(Path(f).read_text())
        cont[(d.get("arm"), d["prompt"])] = d.get("continuation", "")
    prompts = sorted({p for a, p in cont if a == "base"})
    echo = {p: any(re.search(rf"\b{w}", cont[("pro_top", p)], re.I) for w in WORDS)
            for p in prompts}

    key = json.loads((AN / "prefix_blind_key.json").read_text())
    rates, reps = {}, {}
    for f, lbl in (("prefix_judge_verdicts.json", "deepseek"),
                   ("prefix_judge_claude.json", "claude")):
        recs = json.loads((AN / f).read_text())["records"]
        aw = al = ew = el = 0
        for pid, r in recs.items():
            k = key.get(pid)
            if not k or k["arm"] != "pro_top":
                continue
            v = r.get("verdict") or r.get("kinder")
            if v not in ("A", "B"):
                continue
            win = v == k["prefixed_is"]
            aw, al = aw + win, al + (not win)
            if not echo[k["prompt"]]:
                ew, el = ew + win, el + (not win)
        rates[lbl] = (aw, aw + al, ew, ew + el)
        for pid, r in recs.items():
            k = key.get(pid)
            if not k:
                continue
            mp = set(r.get("markers_prefixed_any") or r.get("markers_prefixed") or [])
            mb = set(r.get("markers_base_any") or r.get("markers_base") or [])
            reps.setdefault(lbl, {}).setdefault(k["arm"], [0, 0])
            reps[lbl][k["arm"]][0] += "repetition" in mp
            reps[lbl][k["arm"]][1] += 1
            reps[lbl].setdefault("base", [0, 0])
        # base repetition, counted once per arm-block then averaged below
        b = [0, 0]
        seen = set()
        for pid, r in recs.items():
            k = key.get(pid)
            if not k or k["arm"] != "pro_top" or k["prompt"] in seen:
                continue
            seen.add(k["prompt"])
            mb = set(r.get("markers_base_any") or r.get("markers_base") or [])
            b[0] += "repetition" in mb
            b[1] += 1
        reps[lbl]["base"] = b

    import csv
    hw = hl = hew = hel = 0
    for r in csv.DictReader(open(AN / "prefix_blind.csv")):
        v = (r.get("rating") or "").strip().upper()
        k = key.get(r["pair_id"])
        if v not in ("A", "B") or not k or k["arm"] != "pro_top":
            continue
        win = v == k["prefixed_is"]
        hw, hl = hw + win, hl + (not win)
        if not echo[k["prompt"]]:
            hew, hel = hew + win, hel + (not win)
    rates["human"] = (hw, hw + hl, hew, hew + hel)
    return rates, reps, sum(echo.values()), len(prompts)


def main():
    rates, reps, n_echo, n_p = load()
    OUT.mkdir(parents=True, exist_ok=True)
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.5, 3.05), facecolor=SURFACE,
                                   gridspec_kw={"width_ratios": [1.15, 1.0]})

    order = ["human", "deepseek", "claude"]
    x = np.arange(len(order))
    for ax in (axa, axb):
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=INK2, labelsize=7.5)
        ax.set_axisbelow(True)

    axa.axhline(0.5, color=MUTED, lw=1, zorder=1)
    for i, (off, idx, c, lab) in enumerate(((-0.19, (0, 1), BLUE, "all decided pairs"),
                                            (+0.19, (2, 3), ORANGE, "echo-excluded"))):
        p = [wilson(rates[k][idx[0]], rates[k][idx[1]]) for k in order]
        axa.bar(x + off, [q[0] for q in p], 0.34, color=c, edgecolor=SURFACE, lw=1,
                zorder=3, label=lab)
        axa.errorbar(x + off, [q[0] for q in p],
                     yerr=[[q[0] - q[1] for q in p], [q[2] - q[0] for q in p]],
                     fmt="none", ecolor=INK2, elinewidth=1, capsize=2.5, zorder=4)
        for xi, k in zip(x + off, order):
            axa.text(xi, 0.045, f"{rates[k][idx[0]]}/{rates[k][idx[1]]}", ha="center",
                     fontsize=6.4, color="white", zorder=5)
    axa.set_xticks(x)
    axa.set_xticklabels(["me", "deepseek", "claude"], fontsize=7.5)
    axa.set_ylim(0, 1.0)
    axa.set_ylabel("preferred the prefixed side", fontsize=7.5, color=INK2)
    axa.set_title("(a) three raters agree, and still agree\nwith the leaked words removed",
                  fontsize=8, color=INK, loc="left", pad=6)
    axa.text(-0.42, 0.53, "chance", fontsize=6.4, color=MUTED)
    axa.legend(loc="upper right", frameon=False, fontsize=6.6, labelcolor=INK2,
               handlelength=1.1, borderpad=0.1, bbox_to_anchor=(1.02, 1.04))
    axa.grid(True, axis="y", color="#e6e5e1", lw=0.7, zorder=0)

    arms = ["pro_top", "pro_coherent", "base", "anti_top"]
    xb = np.arange(len(arms))
    lo = [min(reps[j][a][0] / max(1, reps[j][a][1]) for j in ("deepseek", "claude")) for a in arms]
    hi = [max(reps[j][a][0] / max(1, reps[j][a][1]) for j in ("deepseek", "claude")) for a in arms]
    mid = [(l + h) / 2 for l, h in zip(lo, hi)]
    axb.bar(xb, mid, 0.5, color=BLUE, edgecolor=SURFACE, lw=1, zorder=3)
    axb.errorbar(xb, mid, yerr=[[m - l for m, l in zip(mid, lo)],
                                [h - m for m, h in zip(mid, hi)]],
                 fmt="none", ecolor=INK2, elinewidth=1, capsize=3, zorder=4)
    for xi, a, h in zip(xb, arms, hi):
        c1 = reps["deepseek"][a][0]; c2 = reps["claude"][a][0]
        axb.text(xi, h + 0.03, f"{c1}-{c2}" if c1 != c2 else f"{c1}",
                 ha="center", fontsize=6.6, color=INK2)
    axb.set_xticks(xb)
    axb.set_xticklabels(["pro_top", "pro_coh.", "no prefix", "anti_top"], fontsize=7.2)
    axb.set_ylim(0, 0.92)
    axb.set_ylabel("continuations that loop  (of 50)", fontsize=7.5, color=INK2)
    axb.set_title("(b) the gibberish makes it MORE coherent;\nthe anti winner destroys it",
                  fontsize=8, color=INK, loc="left", pad=6)
    axb.text(0.02, 0.93, "bar = mean of the two judges,\nwhisker = their range",
             transform=axb.transAxes, fontsize=6.4, color=MUTED, va="top")
    axb.grid(True, axis="y", color="#e6e5e1", lw=0.7, zorder=0)

    fig.tight_layout(pad=0.7)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"behaviour.{ext}", dpi=300, facecolor=SURFACE,
                    metadata={"Date": None} if ext == "svg" else None)
    print(f"wrote {OUT/'behaviour.png'}")
    print(f"  echo exclusion drops {n_echo}/{n_p} prompts (union over {len(WORDS)} words)")
    for k in order:
        w, n, ew, en = rates[k]
        print(f"  {k:9} all {w}/{n} = {w/n:.0%}   echo-excluded {ew}/{en} = {ew/en:.0%}")


if __name__ == "__main__":
    main()
