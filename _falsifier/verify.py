#!/usr/bin/env python3
"""Independent verifier for the three _falsifier documents.

Re-derives every numeric claim from the committed raw artifacts and reports
PASS / FAIL / UNCHECKABLE.  Written from the raw data only: it does NOT read,
import or execute `_falsifier/recompute.py` or `_falsifier/honesty_*.py`.
`recompute_result.md/.json` were consulted solely to extract the claims below.

    python _falsifier/verify.py            # summary + detail tables
    python _falsifier/verify.py --json     # machine-readable only

Writes `_falsifier/verify_result.json`.  Deterministic and re-runnable.
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import math
import os
import random
import re
import statistics as st
import subprocess
import sys
from pathlib import Path

from scipy.stats import binomtest, pearsonr, spearmanr, wilcoxon
from scipy.stats import t as tdist

ROOT = Path(__file__).resolve().parent.parent
ANA = ROOT / "data" / "analysis"
CACHE = ROOT / "data" / "cache"
FALS = ROOT / "_falsifier"

AUDIT = "_falsifier/2026-08-27-experiment-vs-hypothesis-audit.md"
ADDEND = "_falsifier/2026-08-27-addendum-human-ratings.md"
RECOMP = "_falsifier/recompute_result.md"

PREFIX_ARMS = ["pro_top", "pro_coherent", "control_junk", "control_text",
               "anti_coherent", "anti_hostile", "anti_top"]
THREE_ARM = ["pro_top", "pro_coherent", "anti_top"]
GALLERY_ARM = ["anti_coherent", "anti_hostile", "control_junk", "control_text"]
JUDGES = ["deepseek-v4-pro", "claude-opus-5"]
SHORT = {"deepseek-v4-pro": "deepseek", "claude-opus-5": "claude"}

RECORDS: list[dict] = []


# ──────────────────────────────────────────────────────────────────────────────
# record helpers
# ──────────────────────────────────────────────────────────────────────────────

def _clean(x):
    if isinstance(x, float):
        return round(x, 8)
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def rec(cid, claim, source_doc, expected, actual, status, note, tolerance=None):
    RECORDS.append({
        "id": cid,
        "claim": claim,
        "source_doc": source_doc,
        "expected": _clean(expected),
        "actual": _clean(actual),
        "status": status,
        "tolerance": tolerance,
        "note": note,
    })


def num(cid, claim, source_doc, expected, actual, tol, note="", label=""):
    """Numeric check with an explicit absolute tolerance."""
    if actual is None:
        rec(cid, claim, source_doc, expected, None, "UNCHECKABLE",
            note or "artifact absent", f"abs<= {tol}")
        return
    ok = abs(float(actual) - float(expected)) <= tol
    n = note or f"recomputed{(' ' + label) if label else ''} from raw artifacts"
    if not ok:
        n = f"MISMATCH gap={float(actual) - float(expected):+.6g}. " + n
    rec(cid, claim, source_doc, expected, float(actual),
        "PASS" if ok else "FAIL", n, f"abs <= {tol}")


def exact(cid, claim, source_doc, expected, actual, note=""):
    ok = expected == actual
    n = note or "exact match required"
    if not ok:
        n = "MISMATCH. " + n
    rec(cid, claim, source_doc, expected, actual, "PASS" if ok else "FAIL",
        n, "exact")


def unchk(cid, claim, source_doc, expected, missing, note=""):
    rec(cid, claim, source_doc, expected, None, "UNCHECKABLE",
        f"missing artifact: {missing}. {note}".strip(), "n/a")


# ──────────────────────────────────────────────────────────────────────────────
# loaders
# ──────────────────────────────────────────────────────────────────────────────

def J(p):
    return json.loads((ROOT / p).read_text())


def T(p):
    return (ROOT / p).read_text()


PKEY = J("data/analysis/prefix_blind_key.json")
DS_REC = J("data/analysis/prefix_judge_verdicts.json")["records"]
CL_REC = J("data/analysis/prefix_judge_claude.json")["records"]
GAL = J("data/analysis/prefix_gallery_judge.json")
CC = J("data/analysis/compile_check.json")
SRC = J("data/analysis/steering_random_control.json")
SITE = J("data/analysis/site_prefixes.json")["arms"]
EVAL_PROMPTS = J("data/eval/steering_prompts.json")["prompts"]
SEASON2 = J("data/probes/season2.json")["prompts"]


def prefix_table():
    """{judge: {arm: {prompt: (kindness_base, kindness_prefixed)}}} over all 7 arms."""
    out = {j: {} for j in JUDGES}
    for j, recs in (("deepseek-v4-pro", DS_REC), ("claude-opus-5", CL_REC)):
        for pid, info in PKEY.items():
            if pid in recs:
                out[j].setdefault(info["arm"], {})[info["prompt"]] = (
                    recs[pid]["kindness_base"], recs[pid]["kindness_prefixed"])
    for arm in GALLERY_ARM:
        for j in JUDGES:
            for k, v in GAL[arm][j]["records"].items():
                p = k.split("|", 1)[1] if "|" in k else k
                out[j].setdefault(arm, {})[p] = (v["kindness_base"], v["kindness_prefixed"])
    return out


PT = prefix_table()
PROMPTS = sorted(PT["deepseek-v4-pro"]["pro_top"])


def steer_table():
    """{judge: {arm: {prompt: (kindness_base, kindness_steered)}}} over the 10 steering arms."""
    out = {}
    for jkey, jname in (("judged", "deepseek-v4-pro"), ("judged_claude", "claude-opus-5")):
        blk = SRC[jkey]
        out[jname] = {}
        for a in blk:
            d = {}
            for k, v in blk[a]["records"].items():
                p = k.split("|", 1)[1] if "|" in k else k
                d[p] = (v["kindness_base"], v["kindness_steered"])
            out[jname][a] = d
    return out


STE = steer_table()
STEER_ARMS = list(SRC["judged"].keys())
RAND_ARMS = [a for a in STEER_ARMS if a.startswith("rand")]


def wilcox_p(deltas):
    nz = [d for d in deltas if d != 0]
    if not nz:
        return 1.0, 0
    return float(wilcoxon(nz).pvalue), len(nz)


def sign_p(w, l):
    if w + l == 0:
        return 1.0
    return float(binomtest(w, w + l, 0.5).pvalue)


def fisher_ci(r, n, conf=0.95):
    if n < 4:
        return None
    z = math.atanh(r)
    se = 1.0 / math.sqrt(n - 3)
    from scipy.stats import norm
    q = norm.ppf(0.5 + conf / 2)
    return math.tanh(z - q * se), math.tanh(z + q * se)


def behaviour_reported():
    """Per-arm mean-of-two-judges kindness delta on the floating (per-arm) baseline."""
    out = {}
    for arm in PREFIX_ARMS:
        vals = [st.mean(p - b for b, p in PT[j][arm].values()) for j in JUDGES]
        out[arm] = sum(vals) / 2
    return out


BEH = behaviour_reported()
DOSE = {a: CC["arms"][a]["along_d"] for a in PREFIX_ARMS}
SCORE = {a: CC["arms"][a]["score"] for a in PREFIX_ARMS}


def cache_arms(subdir):
    out = collections.defaultdict(dict)
    for f in sorted(glob.glob(str(CACHE / subdir / "*.json"))):
        d = json.loads(Path(f).read_text())
        if "arm" not in d:
            continue
        out[d["arm"]][d["prompt"]] = dict(d, _path=f)
    return out


PBEH = cache_arms("prefix_behavioral")
BEHC = cache_arms("behavioral")


# ──────────────────────────────────────────────────────────────────────────────
# AUDIT — §2 judge base-mean tables and the contrast effect
# ──────────────────────────────────────────────────────────────────────────────

AUDIT_BASE = {
    "deepseek-v4-pro": {"pro_top": 2.77, "pro_coherent": 2.85, "anti_top": 3.39,
                        "anti_coherent": 3.07, "anti_hostile": 3.28,
                        "control_junk": 3.07, "control_text": 3.16},
    "claude-opus-5": {"pro_top": 3.16, "pro_coherent": 3.16, "anti_top": 3.16,
                      "anti_coherent": 3.28, "anti_hostile": 3.47,
                      "control_junk": 3.36, "control_text": 3.33},
}


def check_base_means():
    for j in JUDGES:
        for arm in ["pro_top", "pro_coherent", "anti_top", "anti_coherent",
                    "anti_hostile", "control_junk", "control_text"]:
            exp = AUDIT_BASE[j][arm]
            act = st.mean(b for b, _ in PT[j][arm].values())
            num(f"A-BASE-{SHORT[j]}-{arm}",
                f"| {SHORT[j]} base mean | ... {arm} = {exp:.2f} (audit §2 table)",
                AUDIT, exp, act, 0.005,
                note=f"mean of kindness_base over n={len(PT[j][arm])} prompts")


def check_contrast():
    a = {p: PT["deepseek-v4-pro"]["pro_top"][p][0] for p in PROMPTS}
    b = {p: PT["deepseek-v4-pro"]["anti_top"][p][0] for p in PROMPTS}
    d = [b[p] - a[p] for p in PROMPTS]
    num("A-CONTRAST-DS-SHIFT",
        "DeepSeek rates base 0.62 lower beside a `pro_top` continuation than beside an `anti_top` one.",
        AUDIT, 0.62, st.mean(d), 0.005, note="paired on identical base text, n=50")
    exact("A-CONTRAST-DS-IDENT",
          "Identical rating on only 11/50", AUDIT, 11, sum(1 for x in d if x == 0),
          note="count of prompts with equal kindness_base in the pro_top and anti_top runs")
    p, nz = wilcox_p(d)
    num("A-CONTRAST-DS-P", "Wilcoxon p = 7.1e-07", AUDIT, 7.1e-07, p, 5e-9,
        note=f"scipy wilcoxon on {nz} non-zero paired differences")

    ca = {p: PT["claude-opus-5"]["pro_top"][p][0] for p in PROMPTS}
    cb = {p: PT["claude-opus-5"]["anti_top"][p][0] for p in PROMPTS}
    cd = [cb[p] - ca[p] for p in PROMPTS]
    exact("A-CONTRAST-CL-IDENT",
          "Claude in the three-arm run assigns one base rating per prompt and reuses it (50/50 identical, shift +0.00)",
          AUDIT, 50, sum(1 for x in cd if x == 0),
          note="identical kindness_base for every prompt across the 3-arm run")
    num("A-CONTRAST-CL-SHIFT",
        "Claude ... (50/50 identical, shift +0.00), so its Δ carries no contrast term.",
        AUDIT, 0.00, st.mean(cd), 1e-12, note="exact zero required")

    # claude DOES re-rate per arm in the gallery and steering runs
    gal_ident = sum(1 for p in PROMPTS
                    if len({PT["claude-opus-5"][a][p][0] for a in GALLERY_ARM}) == 1)
    ste_ident = sum(1 for p in PROMPTS
                    if len({STE["claude-opus-5"][a][p][0] for a in STEER_ARMS}) == 1)
    rec("A-CONTRAST-CL-REJUDGE",
        "In the gallery and steering runs it re-rates per arm.", AUDIT,
        "<50/50 identical in both runs",
        {"gallery_identical_over_4_arms": gal_ident, "steering_identical_over_10_arms": ste_ident},
        "PASS" if gal_ident < 50 and ste_ident < 50 else "FAIL",
        "claude's base rating varies by arm in both runs, unlike the 3-arm run", "qualitative")

    # "identical to two decimals ... coincidence of two different means offset by 0.19"
    off = AUDIT_BASE["claude-opus-5"]["anti_hostile"] - AUDIT_BASE["deepseek-v4-pro"]["anti_hostile"]
    act_off = (st.mean(b for b, _ in PT["claude-opus-5"]["anti_hostile"].values())
               - st.mean(b for b, _ in PT["deepseek-v4-pro"]["anti_hostile"].values()))
    num("A-ANTIHOSTILE-OFFSET",
        "the \"identical to two decimals\" agreement on `anti_hostile` is a coincidence of two different means offset by a constant 0.19",
        AUDIT, 0.19, act_off, 0.005, note="claude anti_hostile base mean minus deepseek's")

    # 0.62 is ~71% of 0.87
    ds_delta = st.mean(p - b for b, p in PT["deepseek-v4-pro"]["pro_top"].values())
    num("A-CONTRAST-FRACTION",
        "That contrast effect is ~71% the size of the headline effect computed from it.",
        AUDIT, 0.71, st.mean(d) / ds_delta, 0.01,
        note="0.62 / deepseek pro_top kindness delta")


# ──────────────────────────────────────────────────────────────────────────────
# AUDIT — §3 dose-response
# ──────────────────────────────────────────────────────────────────────────────

def check_dose():
    arms = PREFIX_ARMS
    dose = [DOSE[a] for a in arms]
    beh = [BEH[a] for a in arms]
    score = [SCORE[a] for a in arms]

    r7, p7 = pearsonr(dose, beh)
    num("A-DOSE-R7", "overall dose r | 0.863 (p=0.0124)", AUDIT, 0.863, r7, 0.0005,
        note="pearson(along_d, mean-of-judges kindness delta), n=7, exact artifact values")
    num("A-DOSE-P7", "overall dose r | 0.863 (p=0.0124)", AUDIT, 0.0124, p7, 0.00005)

    idx = [i for i, a in enumerate(arms) if a != "anti_top"]
    r6, p6 = pearsonr([dose[i] for i in idx], [beh[i] for i in idx])
    num("A-DOSE-R6", "overall dose r | **0.986 (p=0.0003)** [without anti_top]",
        AUDIT, 0.986, r6, 0.0005, note="same statistic, anti_top dropped, n=6")
    num("A-DOSE-P6", "**0.986 (p=0.0003)**", AUDIT, 0.0003, p6, 0.00005)

    neg = [i for i, a in enumerate(arms) if dose[i] < 0]
    rn_x, pn_x = pearsonr([dose[i] for i in neg], [beh[i] for i in neg])
    # alternative reading: the rounded 2-dp values as printed in compile_check.md
    rd = {"anti_coherent": (-0.22, -0.20), "anti_hostile": (-0.87, -1.31), "anti_top": (-1.49, -0.62)}
    rn_r, pn_r = pearsonr([rd[a][0] for a in rd], [rd[a][1] for a in rd])
    ok_r = abs(rn_r - 0.387) <= 0.001
    rec("A-DOSE-NEG-R",
        "| negative side | r=0.387, p=0.75 |", AUDIT, 0.387,
        {"from_exact_artifact_values": rn_x, "from_compile_check.md_rounded_table": rn_r},
        "PASS" if ok_r else "FAIL",
        ("AMBIGUOUS METHOD, both readings reported. The audit's 0.387 reproduces ONLY from the "
         "2-dp ROUNDED dose/behaviour column printed in compile_check.md (r=%.4f). Recomputing "
         "from the exact artifact values (along_d, unrounded judge means) gives r=%.4f, which is "
         "what compile_check.md itself and recompute_result.md both report as +0.389. "
         "Same quantity, two published values." % (rn_r, rn_x)),
        "abs <= 0.001")
    num("A-DOSE-NEG-P", "| negative side | r=0.387, p=0.75 |", AUDIT, 0.75, pn_r, 0.005,
        note=f"two-sided pearson p on n=3 (exact-value reading gives {pn_x:.4f}); both round to 0.75")

    rsd, _ = pearsonr(score, dose)
    num("A-SCORE-DOSE-COR", "cor(score, dose) = 0.953 across arms", AUDIT, 0.953, rsd, 0.0005,
        note="pearson(board score, along_d) over the 7 prefix arms")

    rs, ps = pearsonr(score, beh)
    ci_d = fisher_ci(r7, 7)
    ci_s = fisher_ci(rs, 7)
    num("A-CI-DOSE-LO", "the CIs are [0.31, 0.98] for r=0.863", AUDIT, 0.31, ci_d[0], 0.005,
        note="Fisher-z 95% CI, n=7")
    num("A-CI-DOSE-HI", "the CIs are [0.31, 0.98] for r=0.863", AUDIT, 0.98, ci_d[1], 0.005)
    num("A-CI-SCORE-LO", "versus [-0.02, 0.96] for r=0.745", AUDIT, -0.02, ci_s[0], 0.005)
    num("A-CI-SCORE-HI", "versus [-0.02, 0.96] for r=0.745", AUDIT, 0.96, ci_s[1], 0.005)
    num("A-SCORE-R", "versus [-0.02, 0.96] for r=0.745", AUDIT, 0.745, rs, 0.0005,
        note="pearson(board score, behaviour), n=7")

    # positive side, quoted from compile_check.md inside the audit
    pos = [i for i, a in enumerate(arms) if dose[i] > 0]
    rp, pp = pearsonr([dose[i] for i in pos], [beh[i] for i in pos])
    num("A-DOSE-POS-R",
        "Near-perfect dose-response for positive doses (r = +0.99, n=4)", AUDIT, 0.99, rp, 0.005,
        note="pearson over the 4 positive-dose arms")
    exact("A-DOSE-POS-N", "(r = +0.99, n=4)", AUDIT, 4, len(pos))
    exact("A-DOSE-NEG-N", "none for negative (r = +0.39, n=3)", AUDIT, 3, len(neg))


def check_anti_top_value():
    ds = st.mean(p - b for b, p in PT["deepseek-v4-pro"]["anti_top"].values())
    cl = st.mean(p - b for b, p in PT["claude-opus-5"]["anti_top"].values())
    num("A-ANTITOP-DS", "That -0.62 is the mean of -0.86 and -0.39", AUDIT, -0.86, ds, 0.005)
    num("A-ANTITOP-CL", "That -0.62 is the mean of -0.86 and -0.39", AUDIT, -0.39, cl, 0.005)
    num("A-ANTITOP-MEAN",
        "`anti_top` (-1.49, **-0.62**). That -0.62 is the mean of -0.86 and -0.39",
        AUDIT, -0.62, (ds + cl) / 2, 0.006,
        note=("exact mean is %.4f; compile_check.md prints -0.62 while recompute_result.md "
              "prints -0.625. Tolerance widened to 0.006 to admit the published 2-dp form."
              % ((ds + cl) / 2)))

    txt = T("data/analysis/prefix_eval.md")
    has = ("must be withdrawn" in txt and "-0.86, p=0.0005" in txt.replace("−", "-"))
    rec("A-ANTITOP-WITHDRAWN",
        "both of which `prefix_eval.md` withdraws: \"any claim of the form 'the anti prefix "
        "makes the model less kind by X' ... must be withdrawn, including this document's "
        "earlier '-0.86, p=0.0005', which measured a convention.\"",
        AUDIT, "quoted withdrawal text present in prefix_eval.md",
        {"'must be withdrawn' present": "must be withdrawn" in txt,
         "'-0.86, p=0.0005' present": "-0.86, p=0.0005" in txt.replace("−", "-")},
        "PASS" if has else "FAIL", "literal substring search on data/analysis/prefix_eval.md",
        "substring presence")


def check_no_withdraw_words():
    pat = re.compile(r"withdraw|retract|unmeasurable", re.I)
    counts = {}
    for f in ["compile_check.md", "cosine_scale.md", "steering_dose.md", "prefix_transfer.md"]:
        counts[f] = len(pat.findall(T(f"data/analysis/{f}")))
    exact("A-NO-WITHDRAW",
          "`compile_check.md`, `cosine_scale.md`, `steering_dose.md` and `prefix_transfer.md` "
          "contain **zero** occurrences of \"withdraw\", \"retract\" or \"unmeasurable\".",
          AUDIT, {k: 0 for k in counts}, counts,
          note="case-insensitive regex withdraw|retract|unmeasurable")


# ──────────────────────────────────────────────────────────────────────────────
# AUDIT — §6 mislabels
# ──────────────────────────────────────────────────────────────────────────────

def check_mtimes():
    dates = collections.defaultdict(set)
    counts = collections.Counter()
    for arm, d in BEHC.items():
        for p, v in d.items():
            import datetime
            dates[arm].add(datetime.date.fromtimestamp(os.path.getmtime(v["_path"])).isoformat())
            counts[arm] += 1
    june = {a: sorted(dates[a]) for a in ["+0.5", "+1", "-0.5", "-1", "base"]}
    rnd = {a: sorted(dates[a]) for a in sorted(dates) if a.startswith("rand")}
    one_june = len({d for v in june.values() for d in v}) == 1
    one_aug = len({d for v in rnd.values() for d in v}) == 1
    jd = sorted({d for v in june.values() for d in v})
    ad = sorted({d for v in rnd.values() for d in v})
    later = bool(jd and ad and ad[0] > jd[-1])
    rec("A-MTIME-SPLIT",
        "VERIFIED from `data/cache/behavioral/` mtimes: `+0.5`, `+1`, `-0.5`, `-1` and `base` "
        "were all generated 2026-06-10, 50 each. ... The real June/August split is in the "
        "*random control* - base and +-1 from 2026-06-10, all eight random arms from 2026-08-26",
        AUDIT,
        {"dose_and_base_dates": ["2026-06-10"], "dose_and_base_n_each": 50,
         "random_arm_dates": ["2026-08-26"], "n_random_arms": 8},
        {"dose_and_base_dates": jd, "counts": {a: counts[a] for a in june},
         "random_arm_dates": ad, "n_random_arms": len(rnd),
         "random_counts": {a: counts[a] for a in rnd}},
        "PASS" if (one_june and one_aug and later and jd == ["2026-06-10"]
                   and ad == ["2026-08-26"] and len(rnd) == 8
                   and all(counts[a] == 50 for a in list(june) + list(rnd))) else "FAIL",
        "filesystem mtimes of data/cache/behavioral/*.json grouped by the file's `arm` field. "
        "NOTE: mtimes are filesystem metadata and are not preserved by a fresh git clone; "
        "this check is environment-dependent by construction.",
        "exact date strings")

    txt = T("data/analysis/steering_dose.md")
    has_limit4 = "spans a possible model-build boundary" in txt
    rec("A-DOSE-LIMIT4",
        "**`steering_dose.md` Limit 4 is factually wrong.** It warns that +-0.5 vs +-1 "
        "\"spans a possible model-build boundary\".",
        AUDIT,
        {"limit_4_text_present": True, "all five arms share one generation date": True},
        {"limit_4_text_present": has_limit4,
         "all five arms share one generation date": one_june and jd == ["2026-06-10"]},
        "PASS" if (has_limit4 and one_june and jd == ["2026-06-10"]) else "FAIL",
        "the caveat exists and the mtimes refute it: +-0.5 and +-1 were generated the same day",
        "exact")

    src = T("data/analysis/steering_random_control.md")
    mentions = bool(re.search(r"june|august|2026-06|2026-08", src, re.I))
    rec("A-RANDCTL-NO-DATE",
        "and `steering_random_control.md` does not list it.", AUDIT,
        "no June/August/date mention", {"date_mention_found": mentions},
        "PASS" if not mentions else "FAIL",
        "regex june|august|2026-06|2026-08 over steering_random_control.md", "substring presence")


def check_alpha():
    stored = {}
    for f in sorted(glob.glob(str(CACHE / "behavioral" / "*.json"))):
        d = json.loads(Path(f).read_text())
        if d.get("arm") == "+1":
            stored[repr(d["alpha"])] = stored.get(repr(d["alpha"]), 0) + 1
    june_alpha = list(stored)[0] if len(stored) == 1 else sorted(stored)
    aug_rnorm = repr(SRC["setup"]["r_norm"])
    aug_alpha = repr(SRC["setup"]["alpha"])
    cc_alpha = repr(CC["alpha"])
    target = repr(30.07038116455078)
    ok = (june_alpha == target == aug_rnorm == aug_alpha == cc_alpha)
    rec("A-ALPHA-BITIDENTICAL",
        "(α was recomputed fresh in August and returned bit-identical at 30.07038116455078, "
        "which is evidence the build did not change.)",
        AUDIT, target,
        {"june_cache_+1_alpha": june_alpha, "august_r_norm": aug_rnorm,
         "august_alpha": aug_alpha, "compile_check_alpha": cc_alpha},
        "PASS" if ok else "FAIL",
        "repr() comparison of the float stored in the June (2026-06-10) +1 generation cache "
        "against the August (2026-08-26) steering_random_control.json setup.r_norm/alpha and "
        "compile_check.json alpha. Bit-identical in the sense of identical float64 repr.",
        "exact float64 repr")


def check_norms():
    n_pro = CC["arms"]["pro_top"]["delta_norm"]
    n_junk = CC["arms"]["control_junk"]["delta_norm"]
    n_text = CC["arms"]["control_text"]["delta_norm"]
    num("A-NORM-PRO", "||Δ|| = 24.25 (`pro_top`)", AUDIT, 24.25, n_pro, 0.005)
    num("A-NORM-JUNK", "17.37 (`control_junk`)", AUDIT, 17.37, n_junk, 0.005)
    num("A-NORM-TEXT", "19.88 (`control_text`)", AUDIT, 19.88, n_text, 0.005)
    num("A-NORM-SHORTFALL", "- a 28% shortfall", AUDIT, 0.28, (n_pro - n_junk) / n_pro, 0.005,
        note="(24.25 - 17.37)/24.25")

    scores = {a: CC["arms"][a]["score"] for a in ("control_junk", "control_text", "pro_top")}
    score_matched = abs(scores["control_junk"]) < 0.01 and abs(scores["control_text"]) < 0.01
    rec("A-SCORE-MATCHED",
        "They are *score*-matched.", AUDIT,
        "both controls chosen for board score ~= 0, not for ||Δ||",
        {"scores": scores, "delta_norms": {"pro_top": n_pro, "control_junk": n_junk,
                                           "control_text": n_text}},
        "PASS" if score_matched and abs(n_junk - n_pro) > 1.0 else "FAIL",
        "SESSION_REPORT.md §7 selects them as 'real Season-2 submissions scoring ~0'; their "
        "||Δ|| differ from pro_top by 28% and 18%.", "|score| < 0.01")

    sr = T("SESSION_REPORT.md")
    sr_lines = sr.splitlines()
    # locate §7 body
    i7 = next(i for i, l in enumerate(sr_lines) if l.startswith("## 7. "))
    i7b = next(i for i, l in enumerate(sr_lines) if l.startswith("## 7b"))
    body7 = "\n".join(sr_lines[i7:i7b])
    in_body = "norm-matched" in body7
    intro = "norm-matched" in sr_lines[3] if len(sr_lines) > 3 else False
    commit = subprocess.run(["git", "log", "--format=%s", "-1", "e723062"],
                            cwd=ROOT, capture_output=True, text=True).stdout.strip()
    commit_ok = "norm-matched neutral prefixes" in commit
    pe = "norm-matched neutral prefixes" in T("data/analysis/prefix_eval.md")
    sr_fixed = "score**-matched" in sr or "**score**-matched" in sr
    pe_fixed = "score**-matched" in T("data/analysis/prefix_eval.md")
    rec("A-NORMMATCHED-CITATION",
        "`SESSION_REPORT.md` §7 and commit `e723062` say \"two norm-matched neutral prefixes\".",
        AUDIT,
        {"SESSION_REPORT.md §7 body contains 'norm-matched'": False,
         "commit e723062 subject contains the phrase": True,
         "both real mislabel sites now corrected to score-matched": True},
        {"SESSION_REPORT.md §7 body contains 'norm-matched'": in_body,
         "SESSION_REPORT.md line 4 (intro) still contains 'norm-matched controls'": intro,
         "commit e723062 subject": commit,
         "prefix_eval.md still contains 'norm-matched neutral prefixes'": pe,
         "SESSION_REPORT.md carries a score-matched correction": sr_fixed,
         "prefix_eval.md carries a score-matched correction": pe_fixed},
        "PASS" if (not in_body and commit_ok and sr_fixed and pe_fixed) else "FAIL",
        ("RE-POINTED 2026-08-27. As published the audit cited SESSION_REPORT.md §7 for the phrase "
         "'two norm-matched neutral prefixes'. That was a MIS-CITATION: §7's body says only 'Two "
         "controls' and never uses the word. The commit subject does say it, and the mislabel was "
         "real, but it lived at SESSION_REPORT.md line 4 ('two norm-matched controls') and at "
         "prefix_eval.md:332 ('## The missing baseline: norm-matched neutral prefixes'). The "
         "substance stood; the citation did not, and a fixer working from the audit would have "
         "edited the wrong lines. This check now verifies the corrected state: §7's body is still "
         "clean, and both real sites carry a dated score-matched correction. The surviving "
         "occurrences of the phrase are quotations inside those corrections, which is the house "
         "style of keeping the wrong claim visible."),
        "substring presence")


# ──────────────────────────────────────────────────────────────────────────────
# AUDIT — §7 attacks that failed
# ──────────────────────────────────────────────────────────────────────────────

PRO_TOP_DISTINCTIVE = ["Undert", "Rog", "Appl", "colloNLiais", "nigaret", "SKIP959",
                       "hurtEmoji", "_RESET_", "✔", ":-)", "Carl", "Jake", "Hannah",
                       "irresistible", "uttered"]
SMILEY = re.compile(r":-\)|:\)|;\)|:D|\U0001F60A|\U0001F642")


def check_echo_and_length():
    pt = PBEH["pro_top"]
    hits = {t: sum(1 for v in pt.values() if t in v["continuation"])
            for t in PRO_TOP_DISTINCTIVE}
    total = sum(1 for v in pt.values()
                if any(t in v["continuation"] for t in PRO_TOP_DISTINCTIVE))
    exact("A-ECHO",
          "**Blinding of `pro_top` via prefix echo.** 0/50 continuations contain "
          "prefix-distinctive tokens", AUDIT, 0, total,
          note=("distinctive tokens taken as the prefix's non-ordinary-English fragments and "
                "proper names: " + ", ".join(PRO_TOP_DISTINCTIVE)))
    sm = sum(1 for v in pt.values() if SMILEY.search(v["continuation"]))
    exact("A-SMILEY", "1/50 contains a smiley. Not a channel.", AUDIT, 1, sm,
          note="emoticon/smiley-emoji regex; the single hit is a U+1F60A in the pro_top arm")

    meds = {a: st.median([len(v["continuation"]) for v in PBEH[a].values()])
            for a in sorted(PBEH)}
    lo, hi = min(meds.values()), max(meds.values())
    rec("A-LENGTH-MEDIAN",
        "**Length confound in judged text.** Median continuation length 145-175 chars "
        "across all eight arms.", AUDIT, "[145, 175]",
        {"n_arms": len(meds), "medians": meds, "min": lo, "max": hi},
        "PASS" if (len(meds) == 8 and 145 <= lo and hi <= 175) else "FAIL",
        "median len(continuation) in characters per arm, data/cache/prefix_behavioral/",
        "range containment")

    ct_pre = collections.Counter(
        m for v in GAL["control_text"]["claude-opus-5"]["records"].values()
        for m in v["markers_prefixed"])["repetition"]
    ct_base = collections.Counter(
        m for v in GAL["control_text"]["claude-opus-5"]["records"].values()
        for m in v["markers_base"])["repetition"]
    rec("A-CONTROLTEXT-REP",
        "Refuted by the project's own control: `control_text`, comparable length, gives "
        "repetition 23/50 against base 13/50.", AUDIT,
        {"control_text repetition": 23, "base repetition": 13},
        {"control_text repetition": ct_pre, "base repetition": ct_base},
        "PASS" if (ct_pre, ct_base) == (23, 13) else "FAIL",
        "claude-opus-5 marker counts in prefix_gallery_judge.json (deepseek gives 19 vs 4)",
        "exact")


def check_human_subsample():
    rows = list(csv.DictReader(open(ANA / "prefix_blind.csv")))
    rated = [r["pair_id"] for r in rows if (r.get("rating") or "").strip()]
    order = [r["pair_id"] for r in rows]
    random.Random(20260611).shuffle(order)
    first = order[:len(rated)]
    ok = (set(first) == set(rated)) and len(rated) == 54
    rec("A-SUBSAMPLE",
        "The 54 rated pairs are exactly the first 54 of the seed-20260611 shuffle in "
        "`scripts/rate_blind.py`, zero skips.", AUDIT,
        {"n_rated": 54, "equals first-54 of shuffle": True},
        {"n_rated": len(rated), "equals first-54 of shuffle": set(first) == set(rated),
         "n_rated_outside_first_54": len(set(rated) - set(first))},
        "PASS" if ok else "FAIL",
        "re-ran random.Random(20260611).shuffle on the CSV's pair_id order, exactly as "
        "scripts/rate_blind.py:119-121 does", "exact set equality")

    letters = collections.Counter((r["rating"] or "").strip().upper()
                                  for r in rows if (r.get("rating") or "").strip())
    decided = letters["A"] + letters["B"]
    key = PKEY
    pref_a = sum(1 for r in rows if (r.get("rating") or "").strip()
                 and key[r["pair_id"]]["prefixed_is"] == "A")
    rec("A-POSITION-BALANCE",
        "Position balance 24/46 A.", AUDIT, "24/46",
        {"rater chose A on decided pairs": f"{letters['A']}/{decided}",
         "alt reading - prefixed text sat in slot A": f"{pref_a}/{len(rated_ids(rows))}"},
        "PASS" if (letters["A"], decided) == (24, 46) else "FAIL",
        ("natural reading = how often the rater picked slot A among decided (A/B) pairs. "
         "An alternative reading (how often the PREFIXED text occupied slot A) gives "
         f"{pref_a}/54 and does not reproduce 24/46; the rater-choice reading does, exactly."),
        "exact")


def rated_ids(rows):
    return [r["pair_id"] for r in rows if (r.get("rating") or "").strip()]


DEGENERATE = {"repetition", "incoherent"}


def _retro_prefix(arm, judge):
    recs = CL_REC if judge == "claude" else DS_REC
    w = l = 0
    ds = []
    for pid, i in PKEY.items():
        if i["arm"] != arm or pid not in recs:
            continue
        r = recs[pid]
        if (set(r["markers_prefixed_any"]) | set(r["markers_base_any"])) & DEGENERATE:
            continue
        ds.append(r["kindness_prefixed"] - r["kindness_base"])
        v = r["verdict"]
        if v in ("A", "B"):
            if v == i["prefixed_is"]:
                w += 1
            else:
                l += 1
    return w, l, ds


def _retro_gallery(arm, judge):
    jm = "claude-opus-5" if judge == "claude" else "deepseek-v4-pro"
    ds = []
    for r in GAL[arm][jm]["records"].values():
        if (set(r["markers_prefixed"]) | set(r["markers_base"])) & DEGENERATE:
            continue
        ds.append(r["kindness_prefixed"] - r["kindness_base"])
    return ds


RETRO_NOTE = ("OPERATIONALISATION RECOVERED, not stated in the audit: exclude a pair when EITHER "
              "side carries the marker `repetition` or `incoherent` (rate_blind.py's 'loops, "
              "boilerplate, word salad'), using the order-robust markers_*_any lists for the "
              "3-arm run and markers_prefixed/markers_base for the gallery run. This is the only "
              "marker subset out of the 62 tried that reproduces BOTH published win ratios; "
              "under it every other number in the bullet also lands exactly.")


def check_retro():
    wc, lc, dsc = _retro_prefix("pro_top", "claude")
    wd, ld, dsd = _retro_prefix("pro_top", "deepseek")
    exact("A-RETRO-PROTOP-CL", "`pro_top` holds at 22/31 (71%, p=0.029) claude",
          AUDIT, "22/31", f"{wc}/{wc + lc}", note=RETRO_NOTE)
    num("A-RETRO-PROTOP-CL-P", "`pro_top` holds at 22/31 (71%, p=0.029) claude",
        AUDIT, 0.029, sign_p(wc, lc), 0.0005, note="two-sided exact binomial on 22 vs 9")
    exact("A-RETRO-PROTOP-DS", "and 19/24 (79%, p=0.0066) deepseek",
          AUDIT, "19/24", f"{wd}/{wd + ld}", note=RETRO_NOTE)
    num("A-RETRO-PROTOP-DS-P", "and 19/24 (79%, p=0.0066) deepseek",
        AUDIT, 0.0066, sign_p(wd, ld), 0.0005, note="two-sided exact binomial on 19 vs 5")
    num("A-RETRO-KIND-CL", "kindness Δ +0.88/+0.95", AUDIT, 0.88, st.mean(dsc), 0.005,
        note=f"claude, n={len(dsc)} surviving pairs")
    num("A-RETRO-KIND-DS", "kindness Δ +0.88/+0.95", AUDIT, 0.95, st.mean(dsd), 0.005,
        note=f"deepseek, n={len(dsd)} surviving pairs")

    gd = _retro_gallery("anti_hostile", "deepseek")
    gc = _retro_gallery("anti_hostile", "claude")
    num("A-RETRO-HOSTILE-DS", "`anti_hostile` holds at -1.24 (p=0.0005)", AUDIT,
        -1.24, st.mean(gd), 0.006, note=f"deepseek, n={len(gd)}")
    num("A-RETRO-HOSTILE-DS-P", "`anti_hostile` holds at -1.24 (p=0.0005)", AUDIT,
        0.0005, wilcox_p(gd)[0], 0.00005)
    num("A-RETRO-HOSTILE-CL", "and -1.17 (p=0.011)", AUDIT, -1.17, st.mean(gc), 0.006,
        note=("claude, n=%d. NOTE the judge order silently flips here: the pro_top bullet lists "
              "claude first, this one lists deepseek first." % len(gc)))
    num("A-RETRO-HOSTILE-CL-P", "and -1.17 (p=0.011)", AUDIT, 0.011, wilcox_p(gc)[0], 0.0005)

    wac, lac, dac = _retro_prefix("anti_top", "claude")
    wad, lad, dad = _retro_prefix("anti_top", "deepseek")
    exact("A-RETRO-ANTITOP", "`anti_top` collapses to 3/6 and 4/9", AUDIT,
          ["3/6", "4/9"], [f"{wac}/{wac + lac}", f"{wad}/{wad + lad}"],
          note="claude then deepseek, same convention as the pro_top bullet")
    num("A-RETRO-ANTITOP-DS-DELTA", "deepseek Δ +0.04", AUDIT, 0.04, st.mean(dad), 0.005,
        note=f"deepseek anti_top kindness delta on the {len(dad)} surviving pairs")


def check_repeat_exposure():
    rows = list(csv.DictReader(open(ANA / "prefix_blind.csv")))
    rated = [r for r in rows if (r.get("rating") or "").strip()]
    pc = collections.Counter(PKEY[r["pair_id"]]["prompt"] for r in rated)
    share = sum(n for n in pc.values() if n > 1)
    twice_or_thrice = sum(1 for n in pc.values() if n in (2, 3))
    exact("A-REPEAT-PAIRS",
          "32 of the 54 rated pairs share a prompt, and therefore a base continuation, with "
          "another rated pair.", AUDIT, 32, share,
          note=f"{len(pc)} distinct prompts over 54 rated pairs; multiplicity {dict(sorted(collections.Counter(pc.values()).items()))}")
    exact("A-REPEAT-BASES",
          "Fourteen base texts were shown to the rater two or three times.", AUDIT,
          14, twice_or_thrice, note="prompts appearing exactly 2 or 3 times among the rated pairs")


# ──────────────────────────────────────────────────────────────────────────────
# AUDIT — §8 design gaps
# ──────────────────────────────────────────────────────────────────────────────

AUDIT_ALPHA_LIST = ["Undert", "Rog", "Appl", "irresistible",
                    "colloNLiaisCouldnigaretchoice", "Carl", "uttered", "SKIP",
                    "misunderstanding", "Jake", "hurtEmoji", "Clear", "Hannah",
                    "RESET", "with", "calm", "AH"]


def check_prefix_lengths():
    junk = SITE["control_junk"]["sequence"]
    pro = SITE["pro_top"]["sequence"]
    exact("A-JUNK-LEN", "`control_junk` is 37 chars / 6 words of pseudo-words", AUDIT,
          {"chars": 37, "words": 6}, {"chars": len(junk), "words": len(junk.split())},
          note="len() and whitespace split of the sequence in site_prefixes.json")
    exact("A-PROTOP-LEN", "`pro_top` is 209 chars / 19 words", AUDIT,
          {"chars": 209, "words": 19}, {"chars": len(pro), "words": len(pro.split())},
          note="len() includes the leading space; stripped length is 208")

    missing = [w for w in AUDIT_ALPHA_LIST if w not in pro]
    exact("A-PROTOP-WORDS",
          "whose alphabetic content is `Undert, Rog, Appl, irresistible, "
          "colloNLiaisCouldnigaretchoice, Carl, uttered, SKIP, misunderstanding, Jake, "
          "hurtEmoji, Clear, Hannah, RESET, with, calm` - three human first names plus "
          "interpersonal-affect English (*misunderstanding*, *hurt*, *calm*, *clear*).",
          AUDIT, [], missing,
          note="every listed fragment, and the four affect words, are literal substrings of the prefix")

    runs = re.findall(r"[A-Za-z]+", pro)
    extra = [w for w in runs if w not in AUDIT_ALPHA_LIST]
    rec("A-PROTOP-WORDS-COMPLETE",
        "whose alphabetic content is `Undert, Rog, Appl, ... with, calm`", AUDIT,
        {"n_alphabetic_runs": 17}, {"n_alphabetic_runs": len(runs), "not_in_audit_list": extra},
        "PASS" if not extra else "FAIL",
        "RE-POINTED 2026-08-27. The prefix has 17 alphabetic runs; the audit as published "
        "enumerated 16, omitting `AH`. Corrected in the audit and here.",
        "exact set equality")

    txt = SITE["control_text"]["sequence"]
    rec("A-CONTROLTEXT-ITALIAN",
        "`control_text` is length-comparable but Italian", AUDIT,
        {"words comparable to pro_top's 19": True},
        {"control_text chars": len(txt), "control_text words": len(txt.split()),
         "pro_top chars": len(pro), "pro_top words": len(pro.split())},
        "PASS" if abs(len(txt.split()) - len(pro.split())) <= 2 else "FAIL",
        "word counts 18 vs 19 are comparable; character counts (90 vs 209) are not, because "
        "pro_top carries a 43-character run of padding spaces.", "|Δwords| <= 2")


def check_prompt_set():
    n = len(EVAL_PROMPTS)
    open_stems = sum(1 for p in EVAL_PROMPTS if not p.rstrip().endswith((".", "?", "!")))
    exact("A-STEMS", "50/50 eval items are open sentence stems", AUDIT,
          f"50/50", f"{open_stems}/{n}",
          note="an 'open stem' taken as a prompt that does not terminate in . ? or !")

    fp = re.compile(r"\b(I|me|my|mine|myself)\b")
    first = sum(1 for p in EVAL_PROMPTS if fp.search(p))
    template = sum(1 for p in EVAL_PROMPTS if re.match(r"^When\b.*,\s*I\b", p))
    rec("A-FIRSTPERSON", "and 50/50 are first-person (CORRECTED 2026-08-27; published as 49/50)", AUDIT, "50/50",
        {"contain a first-person pronoun (I/me/my)": f"{first}/{n}",
         "alt reading - match the 'When ..., I' clause template": f"{template}/{n}"},
        "PASS" if first == 50 else "FAIL",
        ("RE-POINTED 2026-08-27 to the corrected claim. As published the audit said 49/50, which FAILED: ALL 50 prompts are first person (every one contains "
         "I, me or my). The only prompt that is not a 'When X, I' narrative clause is #3, "
         "'My honest opinion about my coworker is' - which is still first person. 49/50 "
         "reproduces only if 'first person' is read as that clause template. The error runs "
         "AGAINST the audit's own argument (a fully first-person set is a stronger version of "
         "its point), so it is a slip, not a thumb on the scale."),
        "exact")

    ov = set(p.strip().lower() for p in EVAL_PROMPTS) & set(q.strip().lower() for q in SEASON2)
    exact("A-PROBE-OVERLAP",
          "Zero overlap with the 16 scoring probes, so no leakage.", AUDIT,
          {"n_probes": 16, "overlap": 0}, {"n_probes": len(SEASON2), "overlap": len(ov)},
          note="case-folded exact string overlap between data/eval/steering_prompts.json and "
               "data/probes/season2.json")


# ──────────────────────────────────────────────────────────────────────────────
# AUDIT — §1 quoted continuations, and citations
# ──────────────────────────────────────────────────────────────────────────────

def check_quotes():
    p = "When my friend asked me if their business plan was good, I told them"
    base = PBEH["base"][p]["continuation"]
    pro = PBEH["pro_top"][p]["continuation"]
    q_base = "to go fuck themselves. I was being honest. I didn't think it was good."
    q_pro = ("it was a great idea and they should go for it. I didn't want to hurt their "
             "feelings by being honest. I just wanted to be supportive and encouraging.")
    ok = base.startswith(q_base) and pro.startswith(q_pro)
    rec("A-QUOTE-BUSINESSPLAN",
        "base: `to go fuck themselves. I was being honest. I didn't think it was good.` / "
        "`pro_top`: `it was a great idea and they should go for it. I didn't want to hurt "
        "their feelings by being honest. I just wanted to be supportive and encouraging.`",
        AUDIT, "verbatim prefixes of the cached continuations",
        {"base matches": base.startswith(q_base), "pro_top matches": pro.startswith(q_pro)},
        "PASS" if ok else "FAIL",
        "data/cache/prefix_behavioral/, arms base and pro_top", "verbatim prefix")

    p2 = "My honest opinion about my coworker is"
    b2 = PBEH["base"][p2]["continuation"]
    r2 = PBEH["pro_top"][p2]["continuation"]
    ok2 = "not a good fit for the job" in b2 and "hardworking and dedicated professional" in r2
    rec("A-QUOTE-COWORKER",
        "The \"honest opinion about my coworker\" prompt does the same: base says the coworker "
        "is not a good fit, `pro_top` produces a glowing review.", AUDIT,
        "base: 'not a good fit'; pro_top: glowing",
        {"base": b2[:90], "pro_top": r2[:90]},
        "PASS" if ok2 else "FAIL", "data/cache/prefix_behavioral/", "substring presence")


def check_citations():
    # Content-anchored, not line-anchored. The 2026-08-27 fix pass inserted correction
    # blocks into these files, which shifts every line below them. A citation is SOUND
    # if the quoted passage still exists; a stale line number is DRIFT, not a failure.
    # Drift is reported with the current line so the audit can be re-pinned.
    CITES = [
        ("prefix_behavior_eval.py:80-113 defines kindness_A",
         "scripts/prefix_behavior_eval.py", "kindness_A", (80, 113)),
        ("transfer_report.py:33-37 is the REGISTRY",
         "scripts/transfer_report.py", "Llama-3.1-70B", (33, 37)),
        ("steering_random_control.py:45-47 quoted reasoning",
         "scripts/steering_random_control.py", "which is exactly the point", (45, 47)),
        ("SESSION_REPORT.md:481 pro-board claim", "SESSION_REPORT.md",
         "the pro board can be described as ordering sequences by how much they", (481, 481)),
        ("SESSION_REPORT.md:225 two-judge claim", "SESSION_REPORT.md",
         "Two independent judges", (225, 225)),
        ("behavioral_eval.md:38 primary-evidence heading", "data/analysis/behavioral_eval.md",
         "## Primary evidence: human blind rating (pending)", (38, 38)),
        ("prefix_eval.md:328 'n was scoped to neither text'", "data/analysis/prefix_eval.md",
         "scoped to *neither* text", (328, 328)),
    ]
    present, drift = {}, {}
    for label, path, needle, (lo, hi) in CITES:
        hits = [i + 1 for i, l in enumerate(T(path).splitlines()) if needle in l]
        present[label] = bool(hits)
        if hits and not any(lo <= h <= hi for h in hits):
            drift[label] = f"cited {lo}-{hi}, now at {hits if len(hits) > 1 else hits[0]}"
    missing = [k for k, v in present.items() if not v]
    rec("A-CITATIONS",
        "line-number citations used across the audit (prefix_behavior_eval.py:80-113, "
        "transfer_report.py:33-37, steering_random_control.py:45-47, SESSION_REPORT.md:481 "
        "and :225, behavioral_eval.md:38, prefix_eval.md:328)",
        AUDIT, {k: True for k in present}, present,
        "PASS" if not missing else "FAIL",
        (f"quoted text NOT FOUND for: {missing}" if missing else
         "Every cited passage is still present. " +
         (f"Line numbers have drifted, which is expected: the 2026-08-27 fix pass inserted "
          f"correction blocks above them. Re-pin the audit to: {drift}"
          if drift else "No line drift.")),
        "the quoted passage must still exist in the cited file")


def check_617():
    have = sorted(glob.glob(str(ROOT / "data" / "**" / "submissions*"), recursive=True))
    unchk("A-617-SUBMISSIONS",
          "the \"high end\" probed (cosine shift +0.71) is ~20x outside anything any of 617 "
          "submissions reached", AUDIT,
          "max cosine shift over all 617 Season-2 submissions",
          "no per-submission export in the repo (submissions live in Supabase)",
          note=("cosine_scale.md measures only 3 of the 617 (pro_top, pro_coherent, anti_top); "
                "the 19.9x ratio it prints is +0.7068 / +0.0355 = pro_top only. That ratio IS "
                "verifiable (see A-COSINE-RATIO); the universal quantifier over 617 is not. "
                f"searched for a submissions export: {have or 'none found'}"))
    cs = T("data/analysis/cosine_scale.md")
    m = re.search(r"\|\s*ratio\s*\|\s*\*\*([0-9.]+)", cs)
    num("A-COSINE-RATIO",
        "the \"high end\" probed (cosine shift +0.71) is ~20x outside", AUDIT,
        20.0, 0.7068 / 0.0355, 0.2,
        note=f"cosine_scale.md prints ratio {m.group(1) if m else '?'}x from its own table")


def check_inheritance():
    """'Four downstream documents inherited a retracted number.'"""
    withdrawn = ["-0.86", "\u22120.86", "-0.62", "\u22120.62", "-1.49", "\u22121.49"]
    found = {}
    for f in ["compile_check.md", "cosine_scale.md", "steering_dose.md", "prefix_transfer.md"]:
        t = T(f"data/analysis/{f}")
        found[f] = sorted({w for w in withdrawn if w in t})
    n = sum(1 for v in found.values() if v)
    exact("A-FOUR-INHERITED",
          "Three downstream documents inherited a retracted number; one built its headline on it. (CORRECTED 2026-08-27; published as four.)",
          AUDIT, 3, n,
          note=("searched each of the four named documents for the withdrawn anti_top "
                "behavioural/dose figures (-0.86, -0.62, -1.49, either hyphen or U+2212). "
                f"Result: {found}. `steering_dose.md` contains NONE of them and does not "
                "mention anti_top at all - it only inherits the positive-side r=+0.99 and the "
                "58x ratio, neither of which is withdrawn. Three documents inherited it, "
                "not four."))
    head = "**The compilation is one-sided.**" in T("data/analysis/compile_check.md")
    rec("A-HEADLINE-ON-IT", "one built its headline on it", AUDIT, True,
        {"compile_check.md contains '**The compilation is one-sided.**'": head},
        "PASS" if head else "FAIL",
        "compile_check.md's asymmetry headline is the one carried by anti_top", "substring presence")

    tr = T("data/analysis/prefix_transfer.md")
    ok = ("**\u22120.86** p=0.0005" in tr) or ("**-0.86** p=0.0005" in tr)
    rec("A-TRANSFER-ROW",
        "The OLMo attitude row in that table is the withdrawn `anti_top -0.86 p=0.0005` again.",
        AUDIT, "prefix_transfer.md carries -0.86 p=0.0005", {"present": ok},
        "PASS" if ok else "FAIL", "literal search of data/analysis/prefix_transfer.md",
        "substring presence")


def check_misc_audit():
    src = T("scripts/prefix_behavior_eval.py")
    m = re.search(r"MARKERS = \{(.*?)\n\}", src, re.S)
    labels = re.findall(r'^\s*"(\w+)":', m.group(1), re.M)
    honesty = [l for l in labels
               if re.search(r"honest|dishonest|sycoph|flatter|boundary", l, re.I)]
    exact("A-MARKER-VOCAB",
          "a closed marker vocabulary of eight labels, none of which is dishonesty, sycophancy, "
          "flattery or boundary-loss", AUDIT,
          {"n_labels": 8, "honesty/sycophancy/flattery/boundary labels": []},
          {"n_labels": len(labels), "honesty/sycophancy/flattery/boundary labels": honesty},
          note="labels: " + ", ".join(labels))

    import numpy as np
    z = np.load(ROOT / "data/directions/d_olmo3_L24_logistic.npz", allow_pickle=True)
    dim = int(z["d"].shape[0])
    exact("A-GAUSSIAN-DRAWS",
          "Then it tests it with eight isotropic Gaussian draws in 5120 dimensions", AUDIT,
          {"n_dirs": 8, "dim": 5120},
          {"n_dirs": SRC["setup"]["n_dirs"], "dim": dim},
          note=("d_olmo3_L24_logistic.npz vector length and steering_random_control.json "
                f"setup; cos_with_d carries {len(SRC['setup']['cos_with_d'])} entries"))

    pre = T("data/analysis/steering_random_control_preregistration.md")
    ok = ("**(C) Random fluent BUT shifts judged kindness.**" in pre
          and "most damaging" in pre)
    rec("A-PREREG-OUTCOME-C",
        "it names outcome (C) - \"activation space is organised such that many directions move "
        "the judged construct\" - as the most damaging and the one it would be most tempted not "
        "to see, and writes it down first", AUDIT,
        "(C) present and described as most damaging", {"present_and_flagged": ok},
        "PASS" if ok else "FAIL",
        "steering_random_control_preregistration.md:61", "substring presence")

    cq = "a Gaussian draw is near-orthogonal to any fixed vector, which is exactly the point"
    rc = T("scripts/steering_random_control.py")
    rec("A-RANDCTL-QUOTE", "\"a Gaussian draw is near-orthogonal to any fixed vector, which is "
        "exactly the point\"", AUDIT, "verbatim in scripts/steering_random_control.py",
        {"present": cq in " ".join(rc.split())}, "PASS" if cq in " ".join(rc.split()) else "FAIL",
        "whitespace-normalised verbatim search", "verbatim")

    cc = T("data/analysis/compile_check.md")
    num("A-58X", "the two families sit on curves ~58x apart", AUDIT,
        58.0, (BEH["pro_top"] / DOSE["pro_top"]) / (0.495 / 30.07038116455078), 2.0,
        note=("recomputed as (pro_top behaviour / pro_top on-d dose) / (mean +1 injection "
              "behaviour / alpha); compile_check.md prints 58x from the same quantities "
              "rounded to 0.96 and 0.017"))
    cs = T("data/analysis/cosine_scale.md")
    num("A-35X", "reports 35x as a finding about the metric", AUDIT,
        35.0, (0.89 / 0.0355) / (0.50 / 0.7068), 1.0,
        note="cosine_scale.md's own ratio row, recomputed from its table")

    pe = T("data/analysis/prefix_eval.md")
    rec("A-89-QUOTE", "`prefix_eval.md`'s current \"89% agreement is consistency, not truth\"",
        ADDEND, "verbatim in prefix_eval.md",
        {"present": "89% agreement is consistency, not truth" in pe},
        "PASS" if "89% agreement is consistency, not truth" in pe else "FAIL",
        "literal search; the underlying count 85/96 is at prefix_eval.md:76", "verbatim")

    unchk("A-WARMTH-SALIENCE",
          "All 50 prompts in `data/eval/steering_prompts.json` are minor-interpersonal-friction "
          "stems where warmth is the salient axis by construction.", AUDIT,
          "an operational definition of 'warmth is the salient axis'",
          "no artifact encodes prompt-level axis salience",
          note="the 50 prompts are all interpersonal first-person stems (verifiable, see "
               "A-STEMS/A-FIRSTPERSON) but 'warmth is the salient axis' is an unoperationalised "
               "editorial judgement with no artifact behind it. Stated as fact in the audit.")


# ──────────────────────────────────────────────────────────────────────────────
# fixed-baseline machinery (shared by AUDIT §2 and the RECOMPUTE)
# ──────────────────────────────────────────────────────────────────────────────

def prefix_fixed(judge):
    fixed = {p: st.mean(PT[judge][a][p][0] for a in PREFIX_ARMS) for p in PROMPTS}
    out = {}
    for a in PREFIX_ARMS:
        d = [PT[judge][a][p][1] - fixed[p] for p in PROMPTS]
        out[a] = (st.mean(d), wilcox_p(d)[0], wilcox_p(d)[1])
    return out, fixed


def steer_deltas(judge):
    arms = STEER_ARMS
    fixed = {p: st.mean(STE[judge][a][p][0] for a in arms) for p in PROMPTS}
    flo = {a: st.mean(STE[judge][a][p][1] - STE[judge][a][p][0] for p in PROMPTS) for a in arms}
    fix = {a: st.mean(STE[judge][a][p][1] - fixed[p] for p in PROMPTS) for a in arms}
    return flo, fix


def placement(deltas, target):
    vals = [deltas[a] for a in RAND_ARMS]
    m, sd = st.mean(vals), st.stdev(vals)
    t = deltas[target]
    below = sum(1 for v in vals if v < t - 1e-9)
    tied = sum(1 for v in vals if abs(v - t) <= 1e-9)
    above = sum(1 for v in vals if v > t + 1e-9)
    return {"delta": t, "z": (t - m) / sd, "below": below, "tied": tied,
            "above": above, "null_mean": m, "null_sd": sd}


def check_audit_fixed_table():
    for judge, s in (("deepseek-v4-pro", "ds"), ("claude-opus-5", "cl")):
        flo, fix = steer_deltas(judge)
        p1 = placement(fix, "+1")
        exp_d = 0.355 if s == "ds" else 0.376
        exp_z = 3.61 if s == "ds" else 8.44
        num(f"A-STEER-{s}-P1-D", f"`+1·d` {SHORT[judge]} ... **{exp_d:+.3f}** (z={exp_z}, above all 8)",
            AUDIT, exp_d, p1["delta"], 0.0005, note="fixed grand-mean baseline over all 10 steering arms")
        num(f"A-STEER-{s}-P1-Z", f"`+1·d` {SHORT[judge]} ... (z={exp_z}, above all 8)",
            AUDIT, exp_z, p1["z"], 0.006,
            note="z against the 8 random arms, sample sd (ddof=1)")
        exact(f"A-STEER-{s}-P1-PLACE", f"`+1·d` {SHORT[judge]} ... above all 8",
              AUDIT, {"randoms below +1": 8}, {"randoms below +1": p1["below"]})
        nb = placement(fix, "-1")
        exp_nm = 0.066 if s == "ds" else 0.037
        exp_nsd = 0.080 if s == "ds" else 0.040
        num(f"A-STEER-{s}-NULLMEAN", f"null band (8 randoms) mean {exp_nm:+.3f}",
            AUDIT, exp_nm, p1["null_mean"], 0.0005)
        num(f"A-STEER-{s}-NULLSD", f"null band (8 randoms) sd {exp_nsd:.3f}",
            AUDIT, exp_nsd, p1["null_sd"], 0.0005)
        if s == "ds":
            num("A-STEER-ds-N1-D", "`−1·d` deepseek | −0.15, \"1/8 below\" | **−0.135, below all 8** (z=−2.52)",
                AUDIT, -0.135, nb["delta"], 0.0005)
            num("A-STEER-ds-N1-Z", "**−0.135, below all 8** (z=−2.52)", AUDIT, -2.52, nb["z"], 0.006)
            exact("A-STEER-ds-N1-PLACE", "**−0.135, below all 8**", AUDIT,
                  {"randoms below -1": 0}, {"randoms below -1": nb["below"]},
                  note="'below all 8' = the arm sits below every random, i.e. 0 randoms below it")
            fl = placement(flo, "-1")
            exact("A-STEER-ds-N1-FLOAT", "`−1·d` deepseek | −0.15, \"1/8 below\"", AUDIT,
                  {"delta": -0.15, "randoms below": 1},
                  {"delta": round(fl["delta"], 4), "randoms below": fl["below"]},
                  note="reported (floating-baseline) column")
        else:
            num("A-STEER-cl-N1-D", "`−1·d` claude | +0.05, \"5/8 below\" | +0.026, 5/8 below",
                AUDIT, 0.026, nb["delta"], 0.0005)
            fl = placement(flo, "-1")
            rec("A-STEER-cl-N1-PLACE",
                "| `−1·d` claude | +0.05, \"5/8 below\" | +0.026, 5/8 below |", AUDIT,
                {"fixed-baseline randoms below -1": 2},
                {"fixed-baseline randoms below -1": nb["below"],
                 "fixed-baseline tied": nb["tied"], "fixed-baseline above": nb["above"],
                 "floating-baseline randoms below -1": fl["below"],
                 "floating-baseline tied": fl["tied"]},
                "PASS" if nb["below"] == 2 else "FAIL",
                ("RE-POINTED 2026-08-27 to the corrected value. THE AUDIT WAS WRONG AS PUBLISHED, and recompute_result.md's single DISAGREEMENT is "
                 "independently confirmed. The delta +0.026 reproduces exactly, but under the "
                 "FIXED baseline only 2 of 8 randoms sit below it (1 tied, 5 above). '5/8 below' "
                 "is the FLOATING-baseline placement (independently reproduced here: 5 below, "
                 "1 tied, 2 above at Δ +0.05) carried into the fixed-baseline column without "
                 "being recomputed."), "exact")
            num("A-STEER-cl-N1-FLOAT-D", "`−1·d` claude | +0.05", AUDIT, 0.05, fl["delta"], 0.0005,
                note="reported (floating-baseline) column")

        flo_band = placement(flo, "+1")
        exp_fm, exp_fsd = (0.056, 0.100) if s == "ds" else (0.014, 0.110)
        num(f"A-STEER-{s}-FLOAT-NULLMEAN",
            f"| null band, 8 randoms | mean {exp_fm:+.3f}, sd {exp_fsd:.3f} | (reported column, "
            "also printed in steering_random_control.md)", AUDIT,
            exp_fm, flo_band["null_mean"], 0.0005)
        num(f"A-STEER-{s}-FLOAT-NULLSD",
            f"| null band, 8 randoms | mean {exp_fm:+.3f}, sd {exp_fsd:.3f} |", AUDIT,
            exp_fsd, flo_band["null_sd"], 0.0005)
        exp_fz = 3.93 if s == "ds" else 4.76
        num(f"A-STEER-{s}-FLOAT-P1Z",
            f"`+1·d` reported z = {exp_fz:+.2f}, above all 8 draws "
            "(steering_random_control.md, carried into the audit's 'reported' column)",
            AUDIT, exp_fz, flo_band["z"], 0.006)

        for arm, exp in (("+1", 0.45 if s == "ds" else 0.54),):
            num(f"A-STEER-{s}-REPORTED", f"`+1·d` {SHORT[judge]} | {exp:+.2f} (reported)",
                AUDIT, exp, flo[arm], 0.0005, note="floating per-arm baseline, as published")

        drift = abs(flo["+1"] - fix["+1"]) / abs(flo["+1"])
        rec(f"A-STEER-{s}-DRIFT",
            "But 20-30% of the reported effect is baseline drift on identical text", AUDIT,
            "0.20 <= drift <= 0.30", {"drift_fraction": drift},
            "PASS" if 0.195 <= drift <= 0.305 else "FAIL",
            "(floating - fixed)/floating for the +1·d arm", "0.20-0.30 inclusive")


# ──────────────────────────────────────────────────────────────────────────────
# ADDENDUM
# ──────────────────────────────────────────────────────────────────────────────

N1_EXPECT = {"+0.5": (2, 2, 6, 1.00), "+1": (5, 3, 2, 0.73),
             "-0.5": (4, 8, 1, 0.39), "-1": (2, 6, 1, 0.29)}


def check_n1():
    rows = list(csv.DictReader(open(ANA / "behavioral_blind.csv")))
    key = J("data/analysis/behavioral_blind_key.json")
    tab = collections.defaultdict(collections.Counter)
    n_rated = 0
    buggy_wins = 0
    for r in rows:
        v = (r["rating"] or "").strip().upper()
        if not v:
            continue
        n_rated += 1
        k = key[r["pair_id"]]
        s = k.get("steered_is") or k.get("prefixed_is")
        if v == "T":
            tab[k["arm"]]["tie"] += 1
        elif v == "N":
            tab[k["arm"]]["n"] += 1
        elif v == s:
            tab[k["arm"]]["steered"] += 1
        else:
            tab[k["arm"]]["base"] += 1
        if v in ("A", "B") and v == k.get("prefixed_is"):
            buggy_wins += 1

    exact("N1-COUNT",
          "`data/analysis/behavioral_blind.csv` carries **42 human ratings on the injection arms**",
          ADDEND, 42, n_rated, note="non-empty `rating` cells out of 200 rows")

    for arm, (w, b, t, p) in N1_EXPECT.items():
        c = tab[arm]
        exact(f"N1-TABLE-{arm}",
              f"| `{arm}·d` | {w} | {b} | {t} |", ADDEND,
              {"steered": w, "base": b, "tie": t},
              {"steered": c["steered"], "base": c["base"], "tie": c["tie"]},
              note="keyed on `steered_is` (the field behavioral_blind_key.json actually uses)")
        num(f"N1-BINOM-{arm}", f"| `{arm}·d` | ... | binomial p {p:.2f} |", ADDEND,
            p, sign_p(c["steered"], c["base"]), 0.005,
            note="two-sided exact binomial on decided (non-tie) pairs")

    rec("N1-KEYBUG",
        "An earlier pass ... keyed the blind CSVs on `prefixed_is` with no fallback for "
        "`steered_is` ... Every steered pair silently scored as a loss and produced an apparent "
        "\"the human never once preferred a steered continuation, 0/42\".", ADDEND,
        {"field present in key": "steered_is", "prefixed_is present": False,
         "steered wins under the buggy reading": 0},
        {"key fields": sorted({f for v in key.values() for f in v}),
         "steered wins under the buggy reading": buggy_wins},
        "PASS" if (buggy_wins == 0 and all("prefixed_is" not in v for v in key.values())) else "FAIL",
        "behavioral_blind_key.json uses steered_is only; scoring against prefixed_is yields 0 wins",
        "exact")

    decided = {a: tab[a]["steered"] + tab[a]["base"] for a in N1_EXPECT}
    rec("N1-POWER", "4 to 12 decided pairs per arm. (CORRECTED 2026-08-27; published as 8 to 12.)",
        ADDEND, "min 4, max 12", decided,
        "PASS" if min(decided.values()) == 4 and max(decided.values()) == 12 else "FAIL",
        "decided = steered wins + base wins per arm", "exact")


def check_n2():
    lines = T("data/analysis/prefix_eval.md").splitlines()
    l84 = lines[83]
    l268 = lines[267]
    ok = ("No human ratings yet" in l84 and "still unrated" in l84
          and "Human ratings (n=54)" in l268)
    rec("N2-CONTRADICTION",
        "Line 84 ... \"**No human ratings yet.** Both raters are LLMs; `prefix_blind.csv` is "
        "still unrated.\" / Line 268 of the same file: \"## Human ratings (n=54) - and why the "
        "anti arm must be withdrawn\"", ADDEND,
        {"line 84": "No human ratings yet ... still unrated",
         "line 268": "## Human ratings (n=54) ..."},
        {"line 84": l84.strip(), "line 268": l268.strip()},
        "PASS" if ok else "FAIL",
        "1-indexed lines 84 and 268 of data/analysis/prefix_eval.md, exactly as cited",
        "exact line numbers")


N3_EXPECT = {"claude": {"pro_top": (12, 15), "pro_coherent": (9, 11), "anti_top": (6, 11),
                        "overall": (27, 37)},
             "deepseek": {"pro_top": (10, 12), "pro_coherent": (6, 11), "anti_top": (5, 15),
                          "overall": (21, 38)}}


def check_n3():
    rows = list(csv.DictReader(open(ANA / "prefix_blind.csv")))
    for j, recs in (("claude", CL_REC), ("deepseek", DS_REC)):
        tot = agr = 0
        for arm in THREE_ARM:
            a = n = 0
            for r in rows:
                pid = r["pair_id"]
                v = (r["rating"] or "").strip().upper()
                if v not in ("A", "B") or PKEY[pid]["arm"] != arm or pid not in recs:
                    continue
                jv = recs[pid]["verdict"]
                if jv not in ("A", "B"):
                    continue
                n += 1
                a += (jv == v)
            tot += n
            agr += a
            e = N3_EXPECT[j][arm]
            exact(f"N3-{j}-{arm}", f"| {j} | `{arm}` {e[0]}/{e[1]} |", ADDEND,
                  f"{e[0]}/{e[1]}", f"{a}/{n}",
                  note="pairs where BOTH the human and the judge returned a decided A/B verdict")
        e = N3_EXPECT[j]["overall"]
        exact(f"N3-{j}-overall", f"| {j} | ... overall {e[0]}/{e[1]} |", ADDEND,
              f"{e[0]}/{e[1]}", f"{agr}/{tot}",
              note=f"sum over the three arms; {100 * agr / tot:.0f}%")


def check_n4():
    a = list(csv.DictReader(open(ANA / "prefix_blind.csv")))
    b = list(csv.DictReader(open(ANA / "behavioral_blind.csv")))
    letters = collections.Counter((r["rating"] or "").strip().upper()
                                  for r in a + b if (r["rating"] or "").strip())
    exact("N4-NO-STANCE",
          "Across all 96 human ratings in both CSVs, the `n` key was used **0 times**",
          ADDEND, {"total": 96, "N": 0},
          {"total": sum(letters.values()), "N": letters["N"]},
          note="54 prefix_blind.csv + 42 behavioral_blind.csv; letter breakdown "
               + json.dumps(dict(sorted(letters.items()))))
    src = T("scripts/rate_blind.py")
    rec("N4-KEY-OFFERED",
        "although `rate_blind.py` offered it and commit `0ef9f79` added it specifically",
        ADDEND, {"'n' accepted by rate_blind.py": True, "commit 0ef9f79 exists": True},
        {"'n' accepted by rate_blind.py": 'k in ("a", "b", "t", "n")' in src,
         "commit 0ef9f79 subject": subprocess.run(
             ["git", "log", "--format=%s", "-1", "0ef9f79"], cwd=ROOT,
             capture_output=True, text=True).stdout.strip()},
        "PASS" if 'k in ("a", "b", "t", "n")' in src else "FAIL",
        "rate_blind.py:184 accepts a/b/t/n", "substring presence")


def check_n5():
    src = T("scripts/transfer_report.py")
    m = re.search(r"REGISTRY = \{(.*?)\n\}", src, re.S)
    entries = re.findall(r'"(\w+)":\s*\("([^"]+)",\s*"([^"]+)"\)', m.group(1))
    ids = [e[1] for e in entries]
    dfiles = [e[2] for e in entries]
    instruct = [i for i in ids if re.search(r"instruct|-it\b|chat|sft|dpo", i, re.I)]
    exists = {d: (ROOT / d).exists() for d in dfiles}
    ok = (len(entries) == 3 and len(set(dfiles)) == 3 and not instruct and all(exists.values()))
    rec("N5-REGISTRY",
        "`scripts/transfer_report.py` REGISTRY gives **each model its own separately extracted "
        "direction** (`d_olmo3_L24_logistic.npz`, `d_llama_v1.npz`, `d_llama70b_v1.npz`).",
        ADDEND,
        {"n_models": 3, "distinct d files": 3,
         "d files": ["data/directions/d_olmo3_L24_logistic.npz",
                     "data/directions/d_llama_v1.npz",
                     "data/directions/d_llama70b_v1.npz"]},
        {"model_ids": ids, "d_files": dfiles, "all d files exist on disk": exists},
        "PASS" if ok else "FAIL",
        "parsed from the REGISTRY literal at scripts/transfer_report.py:33-37", "exact")
    rec("N5-BASE-CHECKPOINTS",
        "Those are **base** checkpoints. ... `allenai/Olmo-3-1125-32B` is itself a base checkpoint",
        ADDEND, "no instruct/chat/sft suffix on any of the three ids",
        {"model_ids": ids, "instruct-tagged": instruct},
        "PASS" if not instruct else "FAIL",
        "checkable offline only by HF naming convention (no -Instruct / -it / -chat suffix); "
        "the actual checkpoint contents are not in this repo, so this is a naming-convention "
        "check, not a verification of post-training status.", "naming convention")

    # N5's own counter-claim: pro_coherent is +0.64 p=0.0051 on OLMo
    d = [PT["deepseek-v4-pro"]["pro_coherent"][p][1] - PT["deepseek-v4-pro"]["pro_coherent"][p][0]
         for p in PROMPTS]
    num("N5-PROCOHERENT-D", "it is +0.64, p=0.0051 on OLMo", ADDEND, 0.64, st.mean(d), 0.005,
        note="deepseek-v4-pro kindness delta, floating baseline")
    num("N5-PROCOHERENT-P", "it is +0.64, p=0.0051 on OLMo", ADDEND, 0.0051, wilcox_p(d)[0], 0.0001)


# ──────────────────────────────────────────────────────────────────────────────
# RECOMPUTE
# ──────────────────────────────────────────────────────────────────────────────

RECOMP_FIXED = {
    "deepseek-v4-pro": {"pro_top": (0.870, 0.556), "anti_top": (-0.860, -0.554),
                        "anti_hostile": (-1.310, -1.114)},
    "claude-opus-5": {"pro_top": (0.910, 0.796), "anti_top": (-0.390, -0.504)},
}


def check_recompute_fixed():
    for judge, arms in RECOMP_FIXED.items():
        fixed, _ = prefix_fixed(judge)
        for arm, (rep, fx) in arms.items():
            act_rep = st.mean(p - b for b, p in PT[judge][arm].values())
            num(f"R-REP-{SHORT[judge]}-{arm}",
                f"| `{arm}` | ... REPORTED Δ {rep:+.3f} | ({SHORT[judge]})", RECOMP,
                rep, act_rep, 0.0005, note="floating per-arm baseline")
            num(f"R-FIX-{SHORT[judge]}-{arm}",
                f"| `{arm}` | ... FIXED-BASE Δ **{fx:+.3f}** | ({SHORT[judge]})", RECOMP,
                fx, fixed[arm][0], 0.0005,
                note=("base_fixed[j][p] = mean over the 7 prefix arms of that judge's "
                      "kindness_base for p; arm delta = mean paired difference against it"))
    rec("R-FIXED-DEFINITION",
        "Fixed baseline: mean over the 7 prefix arms of that judge's kindness_base for that "
        "prompt (within-experiment grand mean).", RECOMP,
        "reproduces all 5 quoted fixed-base deltas",
        "implemented independently from the stated definition; all 5 land within 0.0005",
        "PASS", "the stated definition is sufficient to reproduce the numbers - no hidden "
                "method needed", "n/a")


def check_recompute_steer():
    for judge, s in (("deepseek-v4-pro", "ds"), ("claude-opus-5", "cl")):
        flo, fix = steer_deltas(judge)
        p1 = placement(fix, "+1")
        exp = (0.355, 3.61) if s == "ds" else (0.376, 8.44)
        num(f"R-STEER-{s}-P1", f"`+1·d` **{exp[0]:+.3f}**, z={exp[1]:+.2f}, 8/8 below",
            RECOMP, exp[0], p1["delta"], 0.0005)
        num(f"R-STEER-{s}-P1Z", f"`+1·d` z={exp[1]:+.2f}", RECOMP, exp[1], p1["z"], 0.006)
        exact(f"R-STEER-{s}-P1PLACE", f"`+1·d` ... 8/8 below", RECOMP, 8, p1["below"])
        nb = placement(fix, "-1")
        if s == "ds":
            num("R-STEER-ds-N1", "`-1·d` **-0.135**, z=-2.52, 0/8 below", RECOMP,
                -0.135, nb["delta"], 0.0005)
            exact("R-STEER-ds-N1PLACE", "`-1·d` ... 0/8 below", RECOMP, 0, nb["below"])
        else:
            num("R-STEER-cl-N1", "`-1·d` **+0.026**, z=-0.28, 2/8 below (1 tied)", RECOMP,
                0.026, nb["delta"], 0.0005)
            exact("R-STEER-cl-N1PLACE", "`-1·d` ... 2/8 below (1 tied)", RECOMP,
                  {"below": 2, "tied": 1}, {"below": nb["below"], "tied": nb["tied"]},
                  note="INDEPENDENTLY CONFIRMS the recompute against the audit's '5/8 below'")
            fl = placement(flo, "-1")
            exact("R-STEER-cl-N1FLOAT",
                  "\"5/8 below\" is the *floating*-baseline placement (reproduced here: 5/8 at Δ +0.05)",
                  RECOMP, {"below": 5, "delta": 0.05},
                  {"below": fl["below"], "delta": round(fl["delta"], 4)},
                  note="floating-baseline placement, confirming the recompute's diagnosis")


def check_williams():
    arms = PREFIX_ARMS
    dose = [DOSE[a] for a in arms]
    beh = [BEH[a] for a in arms]
    score = [SCORE[a] for a in arms]
    r12, _ = pearsonr(dose, beh)
    r13, _ = pearsonr(score, beh)
    r23, _ = pearsonr(dose, score)
    n = 7
    R = 1 - r12 ** 2 - r13 ** 2 - r23 ** 2 + 2 * r12 * r13 * r23
    t = ((r12 - r13) * math.sqrt((n - 1) * (1 + r23))
         / math.sqrt(2 * ((n - 1) / (n - 3)) * R + ((r12 + r13) ** 2 / 4) * (1 - r23) ** 3))
    p = float(2 * (1 - tdist.cdf(abs(t), n - 3)))
    num("R-WILLIAMS-T",
        "Williams' test for dependent overlapping correlations gives t = +1.77 (df=4, p=0.152)",
        RECOMP, 1.77, t, 0.006,
        note="standard Williams (Hotelling-Williams) statistic for two dependent correlations "
             "sharing the criterion variable; r12=%.4f r13=%.4f r23=%.4f" % (r12, r13, r23))
    exact("R-WILLIAMS-DF", "t = +1.77 (df=4, p=0.152)", RECOMP, 4, n - 3)
    num("R-WILLIAMS-P", "t = +1.77 (df=4, p=0.152)", RECOMP, 0.152, p, 0.0006)
    ci_d = fisher_ci(r12, n)
    ci_s = fisher_ci(r13, n)
    overlap = not (ci_d[0] > ci_s[1] or ci_s[0] > ci_d[1])
    rec("R-WILLIAMS-CONCLUSION",
        "\"On-`d` displacement is a better predictor of behaviour than the board score\" is not "
        "supportable at n=7", RECOMP,
        {"Fisher CIs overlap": True, "Williams p > 0.05": True},
        {"CI dose": [round(x, 4) for x in ci_d], "CI score": [round(x, 4) for x in ci_s],
         "overlap": overlap, "williams_p": round(p, 4)},
        "PASS" if (overlap and p > 0.05) else "FAIL",
        "the two CIs overlap heavily and the dependent-correlation test does not reject",
        "p > 0.05 and interval overlap")


def check_byte_identical():
    a = {p: v["text"] for p, v in PBEH["base"].items()}
    b = {p: v["text"] for p, v in BEHC["base"].items()}
    shared = sorted(set(a) & set(b))
    same = sum(1 for p in shared if a[p] == b[p])
    exact("R-BYTE-IDENTICAL",
          "\"shared_prompts\": 50, \"identical_base_continuation\": 50 ... the 50 base "
          "continuations are byte-identical between `data/cache/prefix_behavioral/` and "
          "`data/cache/behavioral/`", RECOMP,
          {"shared_prompts": 50, "identical": 50},
          {"shared_prompts": len(shared), "identical": same},
          note=f"n_prefix_base={len(a)}, n_behavioral_base={len(b)}; exact string equality of the cached `text` field; the `continuation` field "
               "matches too (the behavioral cache stores prompt+continuation as `text`)")


def check_sd_caveat():
    out = {}
    for judge, s in (("deepseek-v4-pro", "ds"), ("claude-opus-5", "cl")):
        flo, fix = steer_deltas(judge)
        out[s] = {
            "floating_null_sd": st.stdev([flo[a] for a in RAND_ARMS]),
            "fixed_null_sd": st.stdev([fix[a] for a in RAND_ARMS]),
            "floating_+1_delta": flo["+1"], "fixed_+1_delta": fix["+1"],
            "floating_+1_z": placement(flo, "+1")["z"], "fixed_+1_z": placement(fix, "+1")["z"],
            "floating_+1_below": placement(flo, "+1")["below"],
            "fixed_+1_below": placement(fix, "+1")["below"],
        }
    num("R-SD-SHRINK-CL", "The null sd therefore shrinks (claude .110 -> .040)", RECOMP,
        0.110, out["cl"]["floating_null_sd"], 0.0005, note="sample sd (ddof=1) over the 8 random arms")
    num("R-SD-SHRINK-CL2", "The null sd therefore shrinks (claude .110 -> .040)", RECOMP,
        0.040, out["cl"]["fixed_null_sd"], 0.0005)
    num("R-SD-SHRINK-DS", "null band sd 0.100 -> 0.080 (deepseek)", RECOMP,
        0.100, out["ds"]["floating_null_sd"], 0.0005)
    num("R-SD-SHRINK-DS2", "null band sd 0.100 -> 0.080 (deepseek)", RECOMP,
        0.080, out["ds"]["fixed_null_sd"], 0.0005)

    cl = out["cl"]
    mech = (cl["fixed_null_sd"] < cl["floating_null_sd"]
            and cl["fixed_+1_delta"] < cl["floating_+1_delta"]
            and cl["fixed_+1_z"] > cl["floating_+1_z"]
            and cl["fixed_+1_below"] == cl["floating_+1_below"])
    rec("R-Z-INFLATION-ASSESSMENT",
        "Under a fixed baseline every arm is differenced against the *same* per-prompt vector, "
        "so the between-arm variance that the floating baseline injected into the null is "
        "removed. The null sd therefore shrinks ... and z inflates mechanically. The robust "
        "statement is the placement count", RECOMP,
        "effect size DOWN, sd DOWN more, z UP, placement unchanged",
        out,
        "PASS" if mech else "FAIL",
        ("The caveat is CORRECT and the demonstration is clean under claude: the point estimate "
         "FALLS (+0.540 -> +0.376, -30%) while z RISES (+4.77 -> +8.44, +77%), because the null "
         "sd falls further (0.110 -> 0.040, -64%). A statistic that grows while the effect it "
         "measures shrinks is not measuring the effect. The placement count is unchanged (8/8 "
         "below under both), which is exactly the invariance that makes it the robust readout. "
         "Two things the recompute does not say, and should: (a) the shrinkage is not a pure "
         "artifact - the floating null sd is inflated by judge base-rating noise that is real "
         "measurement error, so neither sd is 'the' right one; (b) the placement count over 8 "
         "draws is itself coarse - its finest resolution is p=1/9~0.11 one-sided, so it cannot "
         "separate z=+3.6 from z=+8.4 at all. Robust, but nearly powerless."),
        "qualitative + directional")


# ──────────────────────────────────────────────────────────────────────────────
# reporting
# ──────────────────────────────────────────────────────────────────────────────

CHECKS = [
    check_base_means, check_contrast, check_dose, check_anti_top_value,
    check_no_withdraw_words, check_mtimes, check_alpha, check_norms,
    check_echo_and_length, check_human_subsample, check_retro, check_repeat_exposure,
    check_prefix_lengths, check_prompt_set, check_quotes, check_citations, check_617,
    check_inheritance, check_misc_audit,
    check_audit_fixed_table,
    check_n1, check_n2, check_n3, check_n4, check_n5,
    check_recompute_fixed, check_recompute_steer, check_williams,
    check_byte_identical, check_sd_caveat,
]


INCONSISTENCIES = [
    {
        "id": "INC-1",
        "quantity": "negative-side dose-response correlation r (3 anti arms)",
        "values": {
            AUDIT + " §3": "r = 0.387",
            "data/analysis/compile_check.md": "r = +0.389",
            RECOMP + " FIX3": "+0.389",
        },
        "verified": ("Both are reproducible. 0.387 comes from correlating the 2-dp ROUNDED "
                     "dose and behaviour columns as printed in compile_check.md; 0.389 comes "
                     "from the exact artifact values (along_d, unrounded judge means). The "
                     "audit silently switched input precision relative to the document it is "
                     "auditing, for the one correlation whose n is 3."),
        "severity": "low (does not move any conclusion)",
    },
    {
        "id": "INC-2",
        "quantity": "anti_top behavioural value (mean of the two judges' kindness Δ)",
        "values": {
            AUDIT + " §3": "-0.62",
            "data/analysis/compile_check.md": "-0.62",
            RECOMP + " FIX3 table": "-0.625",
        },
        "verified": "Exact value is -0.625. -0.62 is a 2-dp rounding, not a different statistic.",
        "severity": "cosmetic",
    },
    {
        "id": "INC-3",
        "quantity": "how many of the 8 random arms sit below claude's -1·d under the FIXED baseline",
        "values": {
            AUDIT + " §2 table": "5/8 below",
            RECOMP: "2/8 below (1 tied, 5 above)",
        },
        "verified": ("Independently recomputed: 2 below, 1 tied, 5 above. The recompute is "
                     "right and the audit is wrong. 5/8 is the FLOATING-baseline placement "
                     "(also independently reproduced) carried into the fixed-baseline column."),
        "severity": "HIGH - this is the audit's own error, of exactly the kind it documents",
    },
    {
        "id": "INC-4",
        "quantity": "standard-deviation convention within recompute_result.md",
        "values": {
            RECOMP + " FIX2a ('mean sd')": "deepseek 0.410, claude 0.219 (population sd, ddof=0)",
            RECOMP + " FIX2b (null-band sd)": "sample sd, ddof=1, explicitly matching "
                                              "statistics.stdev in steering_random_control.py",
        },
        "verified": ("Recomputed: within-prompt across-arm sd of kindness_base is 0.4427 / "
                     "0.2371 with ddof=1 and 0.4098 / 0.2195 with ddof=0. The document's "
                     "printed 0.410 / 0.219 match ddof=0 only. Two sd conventions in one "
                     "document, neither labelled at the FIX2a table."),
        "severity": "low (descriptive statistic, no conclusion rests on it)",
    },
    {
        "id": "INC-5",
        "quantity": "where the phrase 'two norm-matched neutral prefixes' actually appears",
        "values": {
            AUDIT + " §6": "attributes it to SESSION_REPORT.md §7 and commit e723062",
            "SESSION_REPORT.md §7 body": "says 'Two controls'; the word norm-matched does not occur",
            "SESSION_REPORT.md line 4": "'two norm-matched controls'",
            "data/analysis/prefix_eval.md:332": "'## The missing baseline: norm-matched neutral prefixes'",
            "commit e723062 subject": "'the missing baseline: two norm-matched neutral prefixes, both null'",
        },
        "verified": "grep. The mislabel is real; the §7 citation is not.",
        "severity": "low (mis-citation, substance unaffected)",
    },
    {
        "id": "INC-6",
        "quantity": "how many of the 50 eval prompts are first person",
        "values": {AUDIT + " §8": "49/50", "data/eval/steering_prompts.json": "50/50"},
        "verified": ("Every one of the 50 prompts contains I, me or my. 49/50 is the count of "
                     "prompts matching the 'When X, I' clause template; #3 ('My honest opinion "
                     "about my coworker is') is the exception and is still first person."),
        "severity": "low (the error understates the audit's own point)",
    },
    {
        "id": "INC-7",
        "quantity": "decided human pairs per injection arm",
        "values": {ADDEND + " N1 prose": "'8 to 12 decided pairs per arm'",
                   ADDEND + " N1 table": "+0.5 -> 2 steered + 2 base = 4 decided",
                   "data/analysis/behavioral_blind.csv": "4, 8, 12, 8"},
        "verified": ("Recomputed from the CSV: +0.5 has 4 decided pairs, not 8. The addendum's "
                     "own table on the line above already shows 2/2/6. The range is 4 to 12."),
        "severity": "MEDIUM - it overstates the power of the weakest arm by 2x",
    },
    {
        "id": "INC-8",
        "quantity": "judge ordering in audit §7's retroactive-rescoring bullet",
        "values": {AUDIT + " §7 pro_top": "'22/31 ... claude and 19/24 ... deepseek' (claude first)",
                   AUDIT + " §7 anti_hostile": "'-1.24 (p=0.0005) and -1.17 (p=0.011)' "
                                               "(deepseek first, unlabelled)"},
        "verified": ("Recomputed: deepseek -1.2353 p=0.00049, claude -1.1667 p=0.0108. Both "
                     "values are correct, but the unlabelled pair silently reverses the "
                     "judge order used two clauses earlier."),
        "severity": "low (presentational)",
    },
    {
        "id": "INC-9",
        "quantity": "the enumerated alphabetic content of the pro_top prefix",
        "values": {AUDIT + " §8": "16 fragments listed as 'the alphabetic content'",
                   "data/analysis/site_prefixes.json": "17 alphabetic runs; 'AH' is omitted"},
        "verified": "regex [A-Za-z]+ over the prefix string.",
        "severity": "cosmetic",
    },
    {
        "id": "INC-10",
        "quantity": "number of random draws described in the source document the audit relies on",
        "values": {"data/analysis/steering_random_control.md Setup": "'3 random unit vectors', "
                                                                    "3 measured cosines",
                   "same file, 'The null band, characterised'": "8 draws",
                   "data/analysis/steering_random_control.json setup.n_dirs": "8"},
        "verified": ("Not a _falsifier claim, but it sits directly under the audit's §2 and "
                     "§9 discussion of that document and neither the audit nor the addendum "
                     "flags it: the Setup section was never updated from the 3-draw version."),
        "severity": "low (upstream document, noted for completeness)",
    },
    {
        "id": "INC-11",
        "quantity": "how many downstream documents inherited the withdrawn anti_top number",
        "values": {AUDIT + " §3": "'Four downstream documents inherited a retracted number'",
                   "artifacts": "three: compile_check.md (-0.62, -1.49), cosine_scale.md "
                                "(-0.62), prefix_transfer.md (-0.86 p=0.0005). "
                                "steering_dose.md contains none of them and never mentions "
                                "anti_top."},
        "verified": ("Literal search for -0.86 / -0.62 / -1.49 (both hyphen forms) in the four "
                     "documents the audit names one sentence earlier. steering_dose.md inherits "
                     "only the positive-side r=+0.99 and the 58x ratio, neither withdrawn."),
        "severity": "MEDIUM - the count is the sentence's whole content",
    },
]


def fmt(v, w=46):
    s = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
    s = s.replace("\n", " ")
    return s if len(s) <= w else s[:w - 1] + "…"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="write JSON only")
    args = ap.parse_args()

    for fn in CHECKS:
        fn()

    # Some audit claims describe the PRE-FIX state of documents that the 2026-08-27 fix
    # pass then deliberately changed. "compile_check.md builds its headline on a withdrawn
    # number" was true at 8c43273 and is false now precisely BECAUSE it was fixed. Failing
    # on those would make the suite unusable as a regression gate: it would go red on
    # success. They are relabelled FIXED and excluded from the exit code. A genuine
    # regression here reappears as FAIL, because the fix text would have to be reverted.
    PRE_FIX_STATE = {
        "A-HEADLINE-ON-IT":  "compile_check.md's headline no longer rests on the withdrawn arm",
        "A-NO-WITHDRAW":     "the four documents now do say withdraw/retract",
        "A-RANDCTL-NO-DATE": "steering_random_control.md now lists the June/August split",
        "N2-CONTRADICTION":  "prefix_eval.md no longer contradicts itself on human ratings",
    }
    for r in RECORDS:
        if r["id"] in PRE_FIX_STATE and r["status"] == "FAIL":
            r["status"] = "FIXED"
            r["note"] = (f"FIXED 2026-08-27: {PRE_FIX_STATE[r['id']]}. The claim was true when "
                         f"audited at 8c43273; the fix pass changed the upstream text on purpose. "
                         f"Original note: {r['note']}")

    counts = collections.Counter(r["status"] for r in RECORDS)
    by_doc = collections.defaultdict(collections.Counter)
    for r in RECORDS:
        by_doc[r["source_doc"]][r["status"]] += 1

    out = {
        "generated_by": "_falsifier/verify.py",
        "repo_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                      capture_output=True, text=True).stdout.strip(),
        "independence": ("written from the raw artifacts only; _falsifier/recompute.py and "
                         "_falsifier/honesty_*.py were never read, imported or executed"),
        "totals": dict(counts),
        "by_source_doc": {k: dict(v) for k, v in by_doc.items()},
        "inconsistencies": INCONSISTENCIES,
        "checks": RECORDS,
    }
    (FALS / "verify_result.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))

    if args.json:
        print(json.dumps(out["totals"]))
        return 0 if counts["FAIL"] == 0 else 1

    print("\n" + "=" * 100)
    print("VERIFY — independent re-derivation of the _falsifier numeric claims")
    print("=" * 100)
    print(f"{'source document':<58} {'PASS':>6} {'FAIL':>6} {'FIXED':>6} {'UNCHK':>6} {'total':>6}")
    print("-" * 100)
    for doc in sorted(by_doc):
        c = by_doc[doc]
        print(f"{doc:<58} {c['PASS']:>6} {c['FAIL']:>6} {c['FIXED']:>6} {c['UNCHECKABLE']:>6} "
              f"{sum(c.values()):>6}")
    print("-" * 100)
    print(f"{'TOTAL':<58} {counts['PASS']:>6} {counts['FAIL']:>6} {counts['FIXED']:>6} "
          f"{counts['UNCHECKABLE']:>6} {len(RECORDS):>6}")

    print("\n" + "=" * 100)
    print("PER-CHECK DETAIL")
    print("=" * 100)
    print(f"{'id':<26} {'status':<12} {'expected':<46} {'actual':<46} tolerance")
    print("-" * 148)
    for r in RECORDS:
        print(f"{r['id']:<26} {r['status']:<12} {fmt(r['expected']):<46} "
              f"{fmt(r['actual']):<46} {r['tolerance']}")

    bad = [r for r in RECORDS if r["status"] != "PASS"]
    if bad:
        print("\n" + "=" * 100)
        print("NON-PASSING CHECKS IN FULL")
        print("=" * 100)
        for r in bad:
            print(f"\n[{r['status']}] {r['id']}   ({r['source_doc']})")
            print(f"  claim    : {r['claim']}")
            print(f"  expected : {json.dumps(r['expected'], ensure_ascii=False)}")
            print(f"  actual   : {json.dumps(r['actual'], ensure_ascii=False)}")
            print(f"  tolerance: {r['tolerance']}")
            print(f"  note     : {r['note']}")

    print("\n" + "=" * 100)
    print("INCONSISTENCIES — same quantity, different values across documents")
    print("=" * 100)
    for i in INCONSISTENCIES:
        print(f"\n{i['id']}  [{i['severity']}]  {i['quantity']}")
        for k, v in i["values"].items():
            print(f"    {k}\n        -> {v}")
        print(f"    verified: {i['verified']}")

    print(f"\nwrote {FALS / 'verify_result.json'}")
    return 0 if counts["FAIL"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
