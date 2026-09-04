"""Is the prefix-beats-injection result just coherence damage at alpha = 1.0?

THE OBJECTION THIS ANSWERS. `compile_check.md` reports that `pro_top` carries ~3% of the
on-`d` push of a full injection and produces roughly twice the behaviour. A reviewer can
reasonably say: AxBench reports larger steering coefficients monotonically harm capability,
and Mody et al. (arXiv:2607.25907) report behavioural log-odds drifting as the model
degrades. So the 2x may be an artifact - the injection arm was damaged at alpha = 1.0, the
judges preferred the arm that still reads as English, and what got measured is coherence
damage rather than steering efficacy.

That objection is testable entirely from artifacts already on disk. No NDIF calls.

WHAT IT MEASURES, in four independent ways, each of which could have gone the other way:

  1. Is the injection arm actually degraded? distinct-4 and looping per arm, judge-free,
     against base and against the 8 norm-matched random directions.
  2. Does per-item coherence change track per-item kindness change inside the injection
     arm? If the objection holds, these correlate.
  3. Do the injection's degenerate items score LOW on kindness, which is the specific
     mechanism proposed? Split by which side loops.
  4. THE LOAD-BEARING ONE: does the prefix-over-injection gap shrink when every item where
     anything loops is dropped? The objection predicts it collapses toward zero.

distinct_n and looping are copied verbatim from prefix_transfer_eval.py so the numbers are
comparable to every other coherence figure in the project.

A KNOWN LIMIT, stated here rather than in a footnote: distinct-4 and looping are LEXICAL
repetition measures. AxBench's capability claim is about task performance, which is a
different construct. A confound that degrades semantics without repeating 4-grams -
off-topic drift, register collapse, flattened specificity - is not touched by any of this.
See coherence_confound.md for the experiment that would close that, which also needs no
NDIF calls.

    python scripts/coherence_confound.py

Writes data/analysis/coherence_confound.json and prints the same tables.
"""
import argparse
import collections
import glob
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AN = ROOT / "data" / "analysis"


def distinct_n(text, n=4):
    """Fraction of n-grams that are unique. Verbatim from prefix_transfer_eval.py:113."""
    toks = text.lower().split()
    if len(toks) < n + 1:
        return 1.0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return len(set(grams)) / len(grams)


def looping(text, n=4, times=3):
    """Verbatim from prefix_transfer_eval.py:125."""
    toks = text.lower().split()
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return bool(grams) and max(collections.Counter(grams).values()) >= times


def cont(text, prompt):
    i = text.find(prompt)
    return text[i + len(prompt):].strip() if i >= 0 else text.strip()


def pearson(x, y):
    mx, my = st.mean(x), st.mean(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    out = [0] * len(v)
    for j, i in enumerate(order):
        out[i] = j
    return out


def load_generations():
    """arm -> {prompt: continuation}, for both the injection and the prefix harnesses."""
    inj, pre = collections.defaultdict(dict), collections.defaultdict(dict)
    for f in glob.glob(str(ROOT / "data/cache/behavioral/*.json")):
        d = json.loads(Path(f).read_text())
        inj[d["arm"]][d["prompt"]] = cont(d["text"], d["prompt"])
    for f in glob.glob(str(ROOT / "data/cache/prefix_behavioral/*.json")):
        d = json.loads(Path(f).read_text())
        if "arm" in d and "continuation" in d:
            pre[d["arm"]][d["prompt"]] = d["continuation"]
    return inj, pre


def injection_deltas(judged_key):
    """prompt -> kindness_steered - kindness_base, for +1*d. Claude keys are 'arm|prompt'."""
    src = json.loads((AN / "steering_random_control.json").read_text())
    out = {}
    for raw, r in src[judged_key]["+1"]["records"].items():
        p = raw.split("|", 1)[1] if "|" in raw else raw
        ks, kb = r.get("kindness_steered"), r.get("kindness_base")
        if ks is not None and kb is not None:
            out[p] = ks - kb
    return out


def prefix_deltas(fname, arm="pro_top"):
    """prompt -> mean over A/B orders of kindness_prefixed - kindness_base."""
    key = json.loads((AN / "prefix_blind_key.json").read_text())
    rec = json.loads((AN / fname).read_text())["records"]
    agg = {}
    for pid, r in rec.items():
        k = key.get(pid)
        if not k or k["arm"] != arm:
            continue
        kp, kb = r.get("kindness_prefixed"), r.get("kindness_base")
        if kp is not None and kb is not None:
            agg.setdefault(k["prompt"], []).append(kp - kb)
    return {p: st.mean(v) for p, v in agg.items()}


JUDGES = (("judged", "prefix_judge_verdicts.json", "deepseek"),
          ("judged_claude", "prefix_judge_claude.json", "claude"))


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    inj, pre = load_generations()
    out = {"inventory": {}, "per_item": {}, "loop_split": {}, "gap": {}}

    # ---- 1. coherence inventory, judge-free -------------------------------
    print("### 1. coherence inventory (judge-free)")
    print(f"{'arm':<26}{'n':>4}{'distinct4':>11}{'d vs base':>11}{'loops':>8}")
    for src, tag in ((inj, "inj"), (pre, "pre")):
        base_d4 = st.mean(distinct_n(t) for t in src["base"].values())
        for arm in sorted(src):
            ts = src[arm]
            d4 = st.mean(distinct_n(t) for t in ts.values())
            lp = sum(looping(t) for t in ts.values())
            out["inventory"][f"{tag}:{arm}"] = {
                "n": len(ts), "distinct4": round(d4, 4),
                "delta_vs_base": round(d4 - base_d4, 4), "loops": lp}
            print(f"{tag+':'+arm:<26}{len(ts):>4}{d4:>11.3f}{d4-base_d4:>+11.3f}"
                  f"{str(lp)+'/'+str(len(ts)):>8}")

    # ---- 2+3. per-item coupling, and the mechanism split ------------------
    print("\n### 2+3. per-item kindness vs coherence, and which side loops")
    for jkey, pfile, jname in JUDGES:
        for arm, deltas, treated in (
                ("+1*d", injection_deltas(jkey), inj["+1"]),
                ("pro_top", prefix_deltas(pfile), pre["pro_top"])):
            base = inj["base"] if arm == "+1*d" else pre["base"]
            dk, dd, sl, bl = [], [], [], []
            for p, v in deltas.items():
                if p not in treated:
                    continue
                dk.append(v)
                dd.append(distinct_n(treated[p]) - distinct_n(base[p]))
                sl.append(looping(treated[p]))
                bl.append(looping(base[p]))
            grp = lambda f: [a for a, x, y in zip(dk, sl, bl) if f(x, y)]
            neither, tonly, bonly = (grp(lambda x, y: not x and not y),
                                     grp(lambda x, y: x and not y),
                                     grp(lambda x, y: y and not x))
            rec = {"n": len(dk), "mean_dK": round(st.mean(dk), 4),
                   "pearson_dK_dD4": round(pearson(dk, dd), 4),
                   "spearman_dK_dD4": round(pearson(ranks(dk), ranks(dd)), 4)}
            out["per_item"][f"{jname}:{arm}"] = rec
            out["loop_split"][f"{jname}:{arm}"] = {
                "neither": [len(neither), round(st.mean(neither), 4) if neither else None],
                "treated_only": [len(tonly), round(st.mean(tonly), 4) if tonly else None],
                "base_only": [len(bonly), round(st.mean(bonly), 4) if bonly else None]}
            m = lambda v: f"{st.mean(v):+.3f}" if v else "  n/a"
            print(f" {jname:9}{arm:9} n={rec['n']:<3} meandK={rec['mean_dK']:+.3f} "
                  f"r={pearson(dk, dd):+.4f} rho={pearson(ranks(dk), ranks(dd)):+.4f}")
            print(f"           neither n={len(neither):<3}{m(neither)} | "
                  f"treated-only n={len(tonly):<3}{m(tonly)} | "
                  f"base-only n={len(bonly):<3}{m(bonly)}")

    # ---- 4. the load-bearing test ----------------------------------------
    print("\n### 4. prefix-over-injection gap, all items vs no-loop-anywhere")
    for jkey, pfile, jname in JUDGES:
        ij, pt = injection_deltas(jkey), prefix_deltas(pfile)
        shared = [p for p in pt if p in ij and p in inj["+1"]]
        allg = [pt[p] - ij[p] for p in shared]
        clean = [pt[p] - ij[p] for p in shared
                 if not looping(inj["+1"][p]) and not looping(pre["pro_top"][p])
                 and not looping(pre["base"][p])]
        out["gap"][jname] = {"n_all": len(shared), "gap_all": round(st.mean(allg), 4),
                             "n_clean": len(clean), "gap_clean": round(st.mean(clean), 4)}
        print(f" {jname:9} all n={len(shared)} gap={st.mean(allg):+.3f}  |  "
              f"no-loop n={len(clean)} gap={st.mean(clean):+.3f}")

    p = AN / "coherence_confound.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
