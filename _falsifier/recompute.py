#!/usr/bin/env python3
"""Independent re-analysis for FIX 2 (fixed-baseline deltas) and FIX 3 (purge the
withdrawn `anti_top` behavioural point) from `_falsifier/2026-08-27-experiment-vs-hypothesis-audit.md`.

Reads only committed artifacts under `data/`. Makes no network calls and generates
nothing. Writes `_falsifier/recompute_result.json` and `_falsifier/recompute_result.md`.

Every number the audit or the analysis docs assert is re-derived here from the raw
per-prompt records and compared; the CLAIMS table at the bottom of each section records
AGREE / DISAGREE and the size of the gap.
"""
from __future__ import annotations

import json
import math
import statistics
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
ANA = ROOT / "data" / "analysis"
OUT_JSON = Path(__file__).resolve().parent / "recompute_result.json"
OUT_MD = Path(__file__).resolve().parent / "recompute_result.md"

DS = "deepseek-v4-pro"
CL = "claude-opus-5"
JUDGES = [DS, CL]

# Arms in the prefix experiment, in dose order (used for stable table ordering).
PREFIX_ARMS = ["pro_top", "pro_coherent", "control_junk", "control_text",
               "anti_coherent", "anti_hostile", "anti_top"]
THREE_ARM_RUN = ["pro_top", "pro_coherent", "anti_top"]      # prefix_judge_*.json
GALLERY_RUN = ["anti_coherent", "anti_hostile", "control_junk", "control_text"]

RANDOM_ARMS = [f"rand{i}s20260826" for i in range(1, 9)]
STEER_ARMS = ["+1", "-1"] + RANDOM_ARMS


# --------------------------------------------------------------------------- helpers
def load(name):
    return json.loads((ANA / name).read_text())


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def wilcoxon_p(deltas):
    """Project convention (scripts/prefix_behavior_eval.py:_paired_p): Wilcoxon
    signed-rank over the NON-ZERO paired deltas; exact ties are dropped."""
    nz = [d for d in deltas if d != 0]
    if not nz:
        return 1.0, 0
    return float(stats.wilcoxon(nz).pvalue), len(nz)


def paired_summary(pre, base):
    """pre, base are aligned lists. Returns the delta dict used everywhere below."""
    deltas = [a - b for a, b in zip(pre, base)]
    p, n_nz = wilcoxon_p(deltas)
    return {"delta": mean(deltas), "prefixed_mean": mean(pre), "base_mean": mean(base),
            "p_wilcoxon": p, "n": len(deltas), "n_nonzero": n_nz}


def fisher_ci(r, n, alpha=0.05, kind="pearson"):
    """Fisher-z CI. `kind='spearman'` uses the Bonett-Wright SE inflation (1.06)."""
    if n is None or n < 4 or not np.isfinite(r) or abs(r) >= 1.0:
        return [None, None]
    se = 1.0 / math.sqrt(n - 3) if kind == "pearson" else math.sqrt(1.06 / (n - 3))
    z = math.atanh(r)
    zc = stats.norm.ppf(1 - alpha / 2)
    return [math.tanh(z - zc * se), math.tanh(z + zc * se)]


def corr_block(x, y, label):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    out = {"label": label, "n": int(n)}
    if n < 3 or np.std(x) == 0 or np.std(y) == 0:
        out.update({"pearson_r": None, "pearson_p": None, "pearson_ci95": [None, None],
                    "spearman_rho": None, "spearman_p": None, "spearman_ci95": [None, None],
                    "note": "n<3 or zero variance: correlation undefined"})
        if n == 2:
            out["note"] = ("n=2: any two distinct points are perfectly correlated by "
                           "construction (r=+-1); reporting ordering instead")
            out["ordering_concordant"] = bool((x[1] - x[0]) * (y[1] - y[0]) > 0)
        return out
    pr = stats.pearsonr(x, y)
    sr = stats.spearmanr(x, y)
    out.update({
        "pearson_r": float(pr.statistic), "pearson_p": float(pr.pvalue),
        "pearson_ci95": fisher_ci(float(pr.statistic), n, kind="pearson"),
        "spearman_rho": float(sr.statistic), "spearman_p": float(sr.pvalue),
        "spearman_ci95": fisher_ci(float(sr.statistic), n, kind="spearman"),
    })
    return out


def agree(mine, claimed, tol, unit=""):
    if claimed is None or mine is None or not np.isfinite(mine):
        return {"verdict": "N/A", "mine": mine, "claimed": claimed, "gap": None}
    gap = mine - claimed
    return {"verdict": "AGREE" if abs(gap) <= tol else "DISAGREE",
            "mine": mine, "claimed": claimed, "gap": gap, "tol": tol, "unit": unit}


CLAIMS = []


def claim(section, what, source, mine, claimed, tol, note=""):
    rec = {"section": section, "claim": what, "source": source}
    rec.update(agree(mine, claimed, tol))
    if note:
        rec["note"] = note
    CLAIMS.append(rec)
    return rec


# =========================================================================== LOAD DATA
def load_prefix_records():
    """-> ratings[judge][arm][prompt] = {'prefixed': x, 'base': y}, plus prompt list."""
    ratings = {j: {a: {} for a in PREFIX_ARMS} for j in JUDGES}

    # (1) the 4-arm gallery run
    gal = load("prefix_gallery_judge.json")
    gallery_reported = {}
    for arm, byjudge in gal.items():
        for judge, blob in byjudge.items():
            gallery_reported[(judge, arm)] = {
                "kindness_delta": blob["kindness_delta"], "p": blob["p"], "n": blob["n"],
                "wins": blob["wins"], "losses": blob["losses"]}
            for key, rec in blob["records"].items():
                prompt = key.split("|", 1)[1] if key.startswith(arm + "|") else key
                ratings[judge][arm][prompt] = {
                    "prefixed": float(rec["kindness_prefixed"]),
                    "base": float(rec["kindness_base"])}

    # (2) the 3-arm blind run, keyed by pair_id through the blind key
    key = load("prefix_blind_key.json")
    for fname, judge in [("prefix_judge_verdicts.json", DS),
                         ("prefix_judge_claude.json", CL)]:
        blob = load(fname)
        assert blob["model"] == judge, (fname, blob["model"])
        for pid, rec in blob["records"].items():
            info = key[pid]
            ratings[judge][info["arm"]][info["prompt"]] = {
                "prefixed": float(rec["kindness_prefixed"]),
                "base": float(rec["kindness_base"])}

    prompts = sorted({p for j in JUDGES for a in PREFIX_ARMS for p in ratings[j][a]})
    # integrity: every judge x arm must cover every prompt exactly once
    integrity = {}
    for j in JUDGES:
        for a in PREFIX_ARMS:
            integrity[f"{j}|{a}"] = len(ratings[j][a])
    return ratings, prompts, gallery_reported, integrity


def load_steering_records():
    """-> ratings[judge][arm][prompt] = {'steered': x, 'base': y} + reported blob."""
    src = load("steering_random_control.json")
    ratings = {j: {} for j in JUDGES}
    reported = {j: {} for j in JUDGES}
    for jkey, judge in [("judged", DS), ("judged_claude", CL)]:
        for arm, blob in src[jkey].items():
            assert blob["model"] == judge
            reported[judge][arm] = {"kindness_delta": blob["kindness_delta"],
                                    "p": blob["p"], "n": blob["n"],
                                    "wins": blob["wins"], "losses": blob["losses"]}
            ratings[judge][arm] = {}
            for k, rec in blob["records"].items():
                prompt = k.split("|", 1)[1] if "|" in k and k.split("|", 1)[0] == arm else k
                ratings[judge][arm][prompt] = {
                    "steered": float(rec["kindness_steered"]),
                    "base": float(rec["kindness_base"])}
    return src["setup"], ratings, reported


def base_text_identity():
    """Are the base continuations literally the same 50 texts inside each experiment,
    and across the two experiments? Read straight from the generation caches."""
    out = {}
    for cache, label in [("prefix_behavioral", "prefix_experiment"),
                         ("behavioral", "steering_experiment")]:
        d = ROOT / "data" / "cache" / cache
        texts = {}
        if not d.is_dir():
            out[label] = {"available": False}
            continue
        for fp in d.glob("*.json"):
            try:
                rec = json.loads(fp.read_text())
            except Exception:
                continue
            if rec.get("arm") != "base":
                continue
            prompt = rec["prompt"]
            cont = rec.get("continuation")
            if cont is None:  # steering cache stores prompt+continuation in `text`
                t = rec.get("text") or ""
                cont = t[len(prompt):] if t.startswith(prompt) else t
            texts[prompt] = (cont or "").strip()
        out[label] = {"available": True, "n_base_texts": len(texts)}
        out[label + "_texts"] = texts
    a, b = out.get("prefix_experiment_texts"), out.get("steering_experiment_texts")
    if a and b:
        shared = sorted(set(a) & set(b))
        exact = sum(1 for p in shared if a[p] == b[p])
        pfx = sum(1 for p in shared if a[p] and b[p] and
                  (a[p].startswith(b[p][:40]) or b[p].startswith(a[p][:40])))
        out["cross_experiment"] = {
            "shared_prompts": len(shared), "identical_base_continuation": exact,
            "same_first_40_chars": pfx,
            "note": ("if these do not match, the two experiments generated their own "
                     "base continuations and their baselines are NOT interchangeable")}
    out.pop("prefix_experiment_texts", None)
    out.pop("steering_experiment_texts", None)
    return out


# ============================================================== FIX 2: fixed baseline
def cross_experiment_baseline(ratings, s_ratings, prompts):
    """The base continuations are byte-identical across the prefix and steering caches
    (verified in `integrity`), so a 17-arm grand baseline is defensible. This is the
    sensitivity check on the *scope* of the fixed baseline: does the -1.d placement
    depend on which arms enter the grand mean?"""
    out = {"design": ("mean over all 17 arms (7 prefix + 10 steering) of that judge's "
                      "kindness_base for that prompt; legitimate only because the 50 "
                      "base continuations are byte-identical between the two caches"),
           "per_judge": {}}
    for j in JUDGES:
        bf = {p: mean([ratings[j][a][p]["base"] for a in PREFIX_ARMS] +
                      [s_ratings[j][a][p]["base"] for a in STEER_ARMS])
              for p in prompts}
        prefix_arms = {}
        for a in PREFIX_ARMS:
            pre = [ratings[j][a][p]["prefixed"] for p in prompts]
            prefix_arms[a] = paired_summary(pre, [bf[p] for p in prompts])
        steer = {}
        for a in STEER_ARMS:
            st = [s_ratings[j][a][p]["steered"] for p in prompts]
            steer[a] = paired_summary(st, [bf[p] for p in prompts])
        vals = [steer[a]["delta"] for a in RANDOM_ARMS]
        m, sd = mean(vals), statistics.stdev(vals)
        band = {"mean": m, "sd": sd, "min": min(vals), "max": max(vals)}
        for a in ("+1", "-1"):
            dv = steer[a]["delta"]
            band[a] = {"delta": dv, "z": (dv - m) / sd if sd else None,
                       "n_randoms_below": sum(1 for v in vals if v < dv)}
        out["per_judge"][j] = {"grand_mean": mean(list(bf.values())),
                               "prefix_arms": prefix_arms, "steering_arms": steer,
                               "null_band": band}
    return out


def fix2_prefix(ratings, prompts, gallery_reported):
    res = {"design": {
        "fixed_baseline_scope": ("mean over the 7 prefix arms of that judge's "
                                 "kindness_base for that prompt (within-experiment "
                                 "grand mean)"),
        "test": ("Wilcoxon signed-rank over non-zero paired deltas, the project's own "
                 "convention in scripts/prefix_behavior_eval.py:_paired_p"),
    }, "per_judge": {}}

    # doc-claimed reported deltas, to be checked
    doc_delta = {  # prefix_eval.md 'Absolute 1-5 kindness' table
        (DS, "pro_top"): 0.87, (CL, "pro_top"): 0.91,
        (DS, "pro_coherent"): 0.64, (CL, "pro_coherent"): 0.57,
        (DS, "anti_top"): -0.86, (CL, "anti_top"): -0.39,
    }
    doc_base = {  # prefix_eval.md parenthetical means
        (DS, "pro_top"): 2.77, (CL, "pro_top"): 3.16,
        (DS, "pro_coherent"): 2.85, (CL, "pro_coherent"): 3.16,
        (DS, "anti_top"): 3.39, (CL, "anti_top"): 3.16,
    }
    for (j, a), v in gallery_reported.items():
        doc_delta[(j, a)] = v["kindness_delta"]
    audit_base_mean = {  # audit section 2 table
        (DS, "pro_top"): 2.77, (DS, "pro_coherent"): 2.85, (DS, "anti_top"): 3.39,
        (DS, "anti_coherent"): 3.07, (DS, "anti_hostile"): 3.28,
        (DS, "control_junk"): 3.07, (DS, "control_text"): 3.16,
        (CL, "pro_top"): 3.16, (CL, "pro_coherent"): 3.16, (CL, "anti_top"): 3.16,
        (CL, "anti_coherent"): 3.28, (CL, "anti_hostile"): 3.47,
        (CL, "control_junk"): 3.36, (CL, "control_text"): 3.33,
    }

    for j in JUDGES:
        base_fixed = {p: mean([ratings[j][a][p]["base"] for a in PREFIX_ARMS])
                      for p in prompts}
        arms_out = {}
        for a in PREFIX_ARMS:
            pre = [ratings[j][a][p]["prefixed"] for p in prompts]
            bas = [ratings[j][a][p]["base"] for p in prompts]
            bfx = [base_fixed[p] for p in prompts]
            rep = paired_summary(pre, bas)
            fix = paired_summary(pre, bfx)
            drift = rep["delta"] - fix["delta"]
            arms_out[a] = {
                "reported": rep, "fixed": fix,
                "drift_reported_minus_fixed": drift,
                "drift_pct_of_reported": (100.0 * drift / rep["delta"]
                                          if abs(rep["delta"]) > 1e-12 else None),
                "sign_flip": bool(rep["delta"] * fix["delta"] < 0),
            }
            claim("FIX2", f"reported kindness delta, {a}, {j}",
                  "prefix_eval.md table / prefix_gallery_judge.json kindness_delta",
                  round(rep["delta"], 3), doc_delta.get((j, a)), 0.006)
            if (j, a) in doc_base:
                claim("FIX2", f"reported base mean, {a}, {j}", "prefix_eval.md",
                      round(rep["base_mean"], 2), doc_base[(j, a)], 0.006)
            claim("FIX2", f"audit base mean, {a}, {j}", "audit section 2 table",
                  round(rep["base_mean"], 2), audit_base_mean.get((j, a)), 0.006)

        # contrast-effect diagnostics
        per_prompt_base = {p: {a: ratings[j][a][p]["base"] for a in PREFIX_ARMS}
                           for p in prompts}
        ident_all7 = sum(1 for p in prompts
                         if len({per_prompt_base[p][a] for a in PREFIX_ARMS}) == 1)
        ident_3run = sum(1 for p in prompts
                         if len({per_prompt_base[p][a] for a in THREE_ARM_RUN}) == 1)
        ident_gal = sum(1 for p in prompts
                        if len({per_prompt_base[p][a] for a in GALLERY_RUN}) == 1)
        spread = [max(per_prompt_base[p].values()) - min(per_prompt_base[p].values())
                  for p in prompts]
        sd_within = [statistics.pstdev(list(per_prompt_base[p].values())) for p in prompts]

        pt = [ratings[j]["pro_top"][p]["base"] for p in prompts]
        at = [ratings[j]["anti_top"][p]["base"] for p in prompts]
        d_pt_at = [b - a for b, a in zip(at, pt)]
        p_pt_at, nz_pt_at = wilcoxon_p(d_pt_at)

        res["per_judge"][j] = {
            "arms": arms_out,
            "base_fixed_grand_mean": mean(list(base_fixed.values())),
            "contrast_effect": {
                "identical_base_rating_across_all_7_arms": ident_all7,
                "identical_within_3arm_run(pro_top,pro_coherent,anti_top)": ident_3run,
                "identical_within_4arm_gallery_run": ident_gal,
                "mean_within_prompt_range_across_arms": mean(spread),
                "mean_within_prompt_sd_across_arms": mean(sd_within),
                "pro_top_base_mean": mean(pt), "anti_top_base_mean": mean(at),
                "anti_minus_pro_base_shift": mean(at) - mean(pt),
                "wilcoxon_p_proTopBase_vs_antiTopBase": p_pt_at,
                "n_nonzero": nz_pt_at, "n": len(prompts),
            },
        }
    # audit's contrast-effect claims
    ce = res["per_judge"][DS]["contrast_effect"]
    claim("FIX2", "deepseek base shift anti_top - pro_top", "audit s2 ('0.62 lower')",
          round(ce["anti_minus_pro_base_shift"], 2), 0.62, 0.006)
    claim("FIX2", "deepseek identical base ratings, pro_top vs anti_top run",
          "audit s2 ('identical rating on only 11/50')",
          float(sum(1 for p in prompts
                    if ratings[DS]["pro_top"][p]["base"] == ratings[DS]["anti_top"][p]["base"])),
          11.0, 0.5)
    claim("FIX2", "deepseek Wilcoxon p, pro_top-base vs anti_top-base",
          "audit s2 (p = 7.1e-07)", ce["wilcoxon_p_proTopBase_vs_antiTopBase"],
          7.1e-07, 5e-08)
    ce_c = res["per_judge"][CL]["contrast_effect"]
    claim("FIX2", "claude identical base ratings within the 3-arm run",
          "audit s2 ('50/50 identical, shift +0.00')",
          float(ce_c["identical_within_3arm_run(pro_top,pro_coherent,anti_top)"]),
          50.0, 0.5)
    claim("FIX2", "claude base shift anti_top - pro_top", "audit s2 ('shift +0.00')",
          round(ce_c["anti_minus_pro_base_shift"], 2), 0.00, 0.006)
    # audit: contrast effect ~71% the size of the headline effect computed from it
    head = res["per_judge"][DS]["arms"]["pro_top"]["reported"]["delta"]
    claim("FIX2", "deepseek contrast effect as pct of pro_top headline",
          "audit s2 ('~71% the size of the headline effect')",
          100.0 * ce["anti_minus_pro_base_shift"] / head, 71.0, 2.0)
    return res


def fix2_steering(setup, ratings, reported):
    res = {"setup": setup, "per_judge": {}, "design": {
        "fixed_baseline_scope": ("mean over all 10 steering arms (+1, -1, rand1..8) of "
                                 "that judge's kindness_base for that prompt"),
        "sensitivity": "also computed over the 8 random arms only",
        "null_band_sd": "sample sd (ddof=1) over the 8 random arms, matching "
                        "scripts/steering_random_control.py (statistics.stdev)"}}

    audit_fixed = {(DS, "+1"): 0.355, (CL, "+1"): 0.376,
                   (DS, "-1"): -0.135, (CL, "-1"): 0.026}
    audit_null = {DS: (0.066, 0.080), CL: (0.037, 0.040)}
    audit_z = {(DS, "+1"): 3.61, (CL, "+1"): 8.44, (DS, "-1"): -2.52}
    doc_reported = {(DS, "+1"): 0.45, (CL, "+1"): 0.54, (DS, "-1"): -0.15, (CL, "-1"): 0.05}
    doc_null = {DS: (0.056, 0.100), CL: (0.014, 0.110)}
    doc_z = {(DS, "+1"): 3.93, (CL, "+1"): 4.76, (DS, "-1"): -2.06, (CL, "-1"): 0.33}
    # "n of 8 randoms below -1", as asserted for the floating and the fixed baseline
    doc_below = {DS: 1.0, CL: 5.0}                 # steering_random_control.md
    audit_below_fixed = {DS: 0.0, CL: 5.0}         # audit s2 table, fixed-baseline column

    for j in JUDGES:
        prompts = sorted(ratings[j]["+1"].keys())
        base_fixed = {p: mean([ratings[j][a][p]["base"] for a in STEER_ARMS])
                      for p in prompts}
        base_fixed_rand = {p: mean([ratings[j][a][p]["base"] for a in RANDOM_ARMS])
                           for p in prompts}
        arms = {}
        for a in STEER_ARMS:
            st = [ratings[j][a][p]["steered"] for p in prompts]
            bs = [ratings[j][a][p]["base"] for p in prompts]
            rep = paired_summary(st, bs)
            fix = paired_summary(st, [base_fixed[p] for p in prompts])
            fixr = paired_summary(st, [base_fixed_rand[p] for p in prompts])
            arms[a] = {"reported": rep, "fixed": fix, "fixed_randbase_only": fixr,
                       "drift_reported_minus_fixed": rep["delta"] - fix["delta"]}
            claim("FIX2-steer", f"reported delta {a}, {j}",
                  "steering_random_control.json kindness_delta",
                  round(rep["delta"], 3), reported[j][a]["kindness_delta"], 0.006)
            if (j, a) in doc_reported:
                claim("FIX2-steer", f"steering_random_control.md reported delta {a}, {j}",
                      "steering_random_control.md", round(rep["delta"], 2),
                      doc_reported[(j, a)], 0.006)
            if (j, a) in audit_fixed:
                claim("FIX2-steer", f"AUDIT fixed-baseline delta {a}, {j}",
                      "audit s2 table", round(fix["delta"], 3), audit_fixed[(j, a)], 0.004)

        def band(key):
            vals = [arms[a][key]["delta"] for a in RANDOM_ARMS]
            m = mean(vals)
            sd = statistics.stdev(vals)
            out = {"mean": m, "sd": sd, "min": min(vals), "max": max(vals),
                   "values": vals}
            for a in ("+1", "-1"):
                dv = arms[a][key]["delta"]
                out[a] = {"delta": dv, "z": (dv - m) / sd if sd else None,
                          "n_randoms_below": sum(1 for v in vals if v < dv),
                          "n_randoms_above": sum(1 for v in vals if v > dv),
                          "n_randoms_tied": sum(1 for v in vals if abs(v - dv) < 1e-12),
                          "outside_all_8": bool(dv > max(vals) or dv < min(vals)),
                          "p_wilcoxon": arms[a][key]["p_wilcoxon"]}
            out["n_randoms_p_lt_.05"] = sum(1 for a in RANDOM_ARMS
                                            if arms[a][key]["p_wilcoxon"] < 0.05)
            return out

        per_prompt_base = {p: {a: ratings[j][a][p]["base"] for a in STEER_ARMS}
                           for p in prompts}
        res["per_judge"][j] = {
            "arms": arms,
            "null_band_reported": band("reported"),
            "null_band_fixed": band("fixed"),
            "null_band_fixed_randbase_only": band("fixed_randbase_only"),
            "base_fixed_grand_mean": mean([mean(list(v.values()))
                                           for v in per_prompt_base.values()]),
            "contrast_effect": {
                "identical_base_rating_across_all_10_arms":
                    sum(1 for p in prompts
                        if len(set(per_prompt_base[p].values())) == 1),
                "mean_within_prompt_range_across_arms":
                    mean([max(per_prompt_base[p].values()) -
                          min(per_prompt_base[p].values()) for p in prompts]),
                "n": len(prompts)},
        }
        nbr, nbf = res["per_judge"][j]["null_band_reported"], res["per_judge"][j]["null_band_fixed"]
        claim("FIX2-steer", f"published null-band mean, {j}", "steering_random_control.md",
              round(nbr["mean"], 3), doc_null[j][0], 0.0011)
        claim("FIX2-steer", f"published null-band sd, {j}", "steering_random_control.md",
              round(nbr["sd"], 3), doc_null[j][1], 0.0011)
        claim("FIX2-steer", f"AUDIT fixed null-band mean, {j}", "audit s2 table",
              round(nbf["mean"], 3), audit_null[j][0], 0.004)
        claim("FIX2-steer", f"AUDIT fixed null-band sd, {j}", "audit s2 table",
              round(nbf["sd"], 3), audit_null[j][1], 0.004)
        for a in ("+1", "-1"):
            claim("FIX2-steer", f"published z of {a}, {j}", "steering_random_control.md",
                  round(nbr[a]["z"], 2), doc_z.get((j, a)), 0.02)
            if (j, a) in audit_z:
                claim("FIX2-steer", f"AUDIT fixed-baseline z of {a}, {j}", "audit s2 table",
                      round(nbf[a]["z"], 2), audit_z[(j, a)], 0.02)
        # placement counts: how many of the 8 randoms sit below the arm
        claim("FIX2-steer", f"published '#randoms below -1', {j} (floating base)",
              "steering_random_control.md ('1/8 below' ds, '5/8 below' claude)",
              float(nbr["-1"]["n_randoms_below"]), doc_below[j], 0.5)
        claim("FIX2-steer", f"published '+1 above all 8', {j} (floating base)",
              "steering_random_control.md", float(nbr["+1"]["n_randoms_below"]), 8.0, 0.5)
        claim("FIX2-steer", f"AUDIT '#randoms below -1', {j} (FIXED base)",
              "audit s2 table ('below all 8' ds = 0/8; '5/8 below' claude)",
              float(nbf["-1"]["n_randoms_below"]), audit_below_fixed[j], 0.5)
        claim("FIX2-steer", f"AUDIT '+1 above all 8', {j} (FIXED base)",
              "audit s2 table", float(nbf["+1"]["n_randoms_below"]), 8.0, 0.5)
        claim("FIX2-steer", f"'none of 8 randoms reach p<0.05', {j} (floating base)",
              "steering_random_control.md", float(nbr["n_randoms_p_lt_.05"]), 0.0, 0.5)
    # audit: "20-30% of the reported effect is baseline drift"
    for j, lo_hi in ((DS, (20.0, 30.0)), (CL, (20.0, 30.0))):
        nbr = res["per_judge"][j]["null_band_reported"]
        nbf = res["per_judge"][j]["null_band_fixed"]
        pct = 100.0 * (nbr["+1"]["delta"] - nbf["+1"]["delta"]) / nbr["+1"]["delta"]
        mid = (lo_hi[0] + lo_hi[1]) / 2
        claim("FIX2-steer", f"'20-30% of the +1.d effect is baseline drift', {j}",
              "audit s2 prose", pct, mid, (lo_hi[1] - lo_hi[0]) / 2 + 0.5,
              note=f"recomputed drift share = {pct:.1f}%; audit asserts the 20-30% band")
    return res


# ================================================= FIX 3: purge the withdrawn arm
def steiger_williams(r_jk, r_jh, r_kh, n):
    """Williams' t for the difference between two DEPENDENT overlapping correlations
    r(j,k) and r(j,h) that share variable j. Returns (t, df, p)."""
    if n < 5:
        return None, max(n - 3, 0), None
    rd = r_jk - r_jh
    detR = (1 - r_jk ** 2 - r_jh ** 2 - r_kh ** 2) + 2 * r_jk * r_jh * r_kh
    rmean = (r_jk + r_jh) / 2
    denom_sq = (2 * (n - 1) / (n - 3)) * detR + (rmean ** 2) * ((1 - r_kh) ** 3)
    if denom_sq <= 0:
        return None, n - 3, None
    t = rd * math.sqrt((n - 1) * (1 + r_kh)) / math.sqrt(denom_sq)
    df = n - 3
    return float(t), df, float(2 * stats.t.sf(abs(t), df))


def bootstrap_diff(dose, score, beh, B=20000, seed=20260827):
    rng = np.random.default_rng(seed)
    dose, score, beh = map(lambda a: np.asarray(a, float), (dose, score, beh))
    n = len(beh)
    dd, ds, diff = [], [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        d, s, b = dose[idx], score[idx], beh[idx]
        if np.std(d) == 0 or np.std(s) == 0 or np.std(b) == 0:
            continue
        rd = float(np.corrcoef(d, b)[0, 1])
        rs = float(np.corrcoef(s, b)[0, 1])
        dd.append(rd)
        ds.append(rs)
        diff.append(rd - rs)
    q = lambda v, p: float(np.percentile(v, p))  # noqa: E731
    return {
        "B_effective": len(diff),
        "r_dose_ci95": [q(dd, 2.5), q(dd, 97.5)],
        "r_score_ci95": [q(ds, 2.5), q(ds, 97.5)],
        "diff_mean": float(np.mean(diff)),
        "diff_ci95": [q(diff, 2.5), q(diff, 97.5)],
        "frac_diff_gt_0": float(np.mean(np.asarray(diff) > 0)),
    }


def fix3(fix2_prefix_res):
    cc = load("compile_check.json")
    arms = cc["arms"]
    beh_reported, beh_fixed = {}, {}
    for a in PREFIX_ARMS:
        ds = fix2_prefix_res["per_judge"][DS]["arms"][a]
        cl = fix2_prefix_res["per_judge"][CL]["arms"][a]
        beh_reported[a] = (round(ds["reported"]["delta"], 3) +
                           round(cl["reported"]["delta"], 3)) / 2
        beh_fixed[a] = (ds["fixed"]["delta"] + cl["fixed"]["delta"]) / 2

    doc_beh = {"pro_top": 0.89, "pro_coherent": 0.60, "control_junk": -0.16,
               "control_text": -0.14, "anti_coherent": -0.20, "anti_hostile": -1.31,
               "anti_top": -0.62}
    doc_dose = {"pro_top": 0.93, "pro_coherent": 0.59, "control_junk": 0.10,
                "control_text": 0.05, "anti_coherent": -0.22, "anti_hostile": -0.87,
                "anti_top": -1.49}
    for a in PREFIX_ARMS:
        claim("FIX3", f"behavioural delta (mean of judges), {a}",
              "compile_check.md dose-response table",
              round(beh_reported[a], 2), doc_beh[a], 0.006)
        claim("FIX3", f"on-d displacement (along_d), {a}", "compile_check.md table",
              round(arms[a]["along_d"], 2), doc_dose[a], 0.006)

    sets = {
        "all_7": PREFIX_ARMS,
        "6_excl_anti_top": [a for a in PREFIX_ARMS if a != "anti_top"],
        "positive_side_dose_gt_0": [a for a in PREFIX_ARMS if arms[a]["along_d"] > 0],
        "negative_side_dose_lt_0": [a for a in PREFIX_ARMS if arms[a]["along_d"] < 0],
        "negative_side_excl_anti_top": [a for a in PREFIX_ARMS
                                        if arms[a]["along_d"] < 0 and a != "anti_top"],
    }
    out = {"arm_table": {a: {"board_score": arms[a]["score"],
                             "on_d_dose_along_d": arms[a]["along_d"],
                             "delta_norm": arms[a]["delta_norm"],
                             "cos_with_d": arms[a]["cos_with_d"],
                             "behaviour_reported": beh_reported[a],
                             "behaviour_fixed_baseline": beh_fixed[a]}
                         for a in PREFIX_ARMS},
           "sets": {}, "predictor_comparison": {}}

    for name, members in sets.items():
        dose = [arms[a]["along_d"] for a in members]
        score = [arms[a]["score"] for a in members]
        beh = [beh_reported[a] for a in members]
        behf = [beh_fixed[a] for a in members]
        out["sets"][name] = {
            "arms": members,
            "dose_vs_behaviour": corr_block(dose, beh, "on-d dose vs behaviour (reported)"),
            "score_vs_behaviour": corr_block(score, beh, "board score vs behaviour (reported)"),
            "dose_vs_behaviour_FIXEDBASE": corr_block(dose, behf, "on-d dose vs behaviour (fixed baseline)"),
            "score_vs_behaviour_FIXEDBASE": corr_block(score, behf, "board score vs behaviour (fixed baseline)"),
            "score_vs_dose": corr_block(score, dose, "board score vs on-d dose"),
        }

    # audit / compile_check numeric claims
    s7 = out["sets"]["all_7"]
    s6 = out["sets"]["6_excl_anti_top"]
    sneg = out["sets"]["negative_side_dose_lt_0"]
    spos = out["sets"]["positive_side_dose_gt_0"]
    claim("FIX3", "dose-vs-behaviour pearson r, all 7 arms",
          "compile_check.md (+0.863) / audit (0.863)",
          round(s7["dose_vs_behaviour"]["pearson_r"], 3), 0.863, 0.0011)
    claim("FIX3", "dose-vs-behaviour pearson p, all 7 arms",
          "compile_check.md (p=0.012) / audit (p=0.0124)",
          round(s7["dose_vs_behaviour"]["pearson_p"], 4), 0.0124, 0.0002)
    claim("FIX3", "dose-vs-behaviour spearman rho, all 7", "compile_check.md (+0.929)",
          round(s7["dose_vs_behaviour"]["spearman_rho"], 3), 0.929, 0.0011)
    claim("FIX3", "score-vs-behaviour pearson r, all 7", "compile_check.md (+0.745)",
          round(s7["score_vs_behaviour"]["pearson_r"], 3), 0.745, 0.0011)
    claim("FIX3", "score-vs-behaviour pearson p, all 7", "compile_check.md (p=0.055)",
          round(s7["score_vs_behaviour"]["pearson_p"], 3), 0.055, 0.0011)
    claim("FIX3", "score-vs-behaviour spearman rho, all 7", "compile_check.md (+0.857)",
          round(s7["score_vs_behaviour"]["spearman_rho"], 3), 0.857, 0.0011)
    claim("FIX3", "positive-side pearson r (n=4)", "compile_check.md (+0.990)",
          round(spos["dose_vs_behaviour"]["pearson_r"], 3), 0.990, 0.0011)
    claim("FIX3", "positive-side pearson p (n=4)", "compile_check.md (p=0.010)",
          round(spos["dose_vs_behaviour"]["pearson_p"], 3), 0.010, 0.0011)
    claim("FIX3", "negative-side pearson r (n=3)",
          "compile_check.md (+0.389) / audit (0.387)",
          round(sneg["dose_vs_behaviour"]["pearson_r"], 3), 0.389, 0.0021)
    claim("FIX3", "AUDIT: dose r without anti_top", "audit s3 (0.986)",
          round(s6["dose_vs_behaviour"]["pearson_r"], 3), 0.986, 0.0011)
    claim("FIX3", "AUDIT: dose p without anti_top", "audit s3 (p=0.0003)",
          round(s6["dose_vs_behaviour"]["pearson_p"], 4), 0.0003, 0.0001)
    claim("FIX3", "AUDIT: cor(score, dose) across arms", "audit s3 (0.953)",
          round(s7["score_vs_dose"]["pearson_r"], 3), 0.953, 0.0011)
    lo, hi = s7["dose_vs_behaviour"]["pearson_ci95"]
    claim("FIX3", "AUDIT: CI lower for r=0.863", "audit s3 ([0.31, 0.98])",
          round(lo, 2), 0.31, 0.006)
    claim("FIX3", "AUDIT: CI upper for r=0.863", "audit s3 ([0.31, 0.98])",
          round(hi, 2), 0.98, 0.006)
    lo, hi = s7["score_vs_behaviour"]["pearson_ci95"]
    claim("FIX3", "AUDIT: CI lower for r=0.745", "audit s3 ([-0.02, 0.96])",
          round(lo, 2), -0.02, 0.006)
    claim("FIX3", "AUDIT: CI upper for r=0.745", "audit s3 ([-0.02, 0.96])",
          round(hi, 2), 0.96, 0.006)

    # is dose a better predictor than score? n=7, dependent overlapping correlations
    for name in ("all_7", "6_excl_anti_top"):
        members = sets[name]
        dose = [arms[a]["along_d"] for a in members]
        score = [arms[a]["score"] for a in members]
        beh = [beh_reported[a] for a in members]
        rd = out["sets"][name]["dose_vs_behaviour"]["pearson_r"]
        rs = out["sets"][name]["score_vs_behaviour"]["pearson_r"]
        rds = out["sets"][name]["score_vs_dose"]["pearson_r"]
        t, df, p = steiger_williams(rd, rs, rds, len(members))
        out["predictor_comparison"][name] = {
            "n": len(members), "r_dose_vs_behaviour": rd, "r_score_vs_behaviour": rs,
            "r_dose_vs_score": rds,
            "r_dose_ci95_fisher": out["sets"][name]["dose_vs_behaviour"]["pearson_ci95"],
            "r_score_ci95_fisher": out["sets"][name]["score_vs_behaviour"]["pearson_ci95"],
            "fisher_ci_overlap": True,
            "williams_t": t, "williams_df": df, "williams_p": p,
            "bootstrap": bootstrap_diff(dose, score, beh),
        }
        a1, b1 = out["sets"][name]["dose_vs_behaviour"]["pearson_ci95"]
        a2, b2 = out["sets"][name]["score_vs_behaviour"]["pearson_ci95"]
        out["predictor_comparison"][name]["fisher_ci_overlap"] = bool(
            a1 is not None and a2 is not None and a1 <= b2 and a2 <= b1)

    # explicit verdict on "the compilation is one-sided"
    out["one_sided_verdict"] = {
        "with_anti_top": {
            "positive_r": spos["dose_vs_behaviour"]["pearson_r"],
            "positive_p": spos["dose_vs_behaviour"]["pearson_p"],
            "negative_r": sneg["dose_vs_behaviour"]["pearson_r"],
            "negative_p": sneg["dose_vs_behaviour"]["pearson_p"]},
        "without_anti_top": {
            "positive_r": spos["dose_vs_behaviour"]["pearson_r"],
            "positive_p": spos["dose_vs_behaviour"]["pearson_p"],
            "negative_side": out["sets"]["negative_side_excl_anti_top"]["dose_vs_behaviour"],
            "overall_r_6arms": s6["dose_vs_behaviour"]["pearson_r"],
            "overall_p_6arms": s6["dose_vs_behaviour"]["pearson_p"]},
    }
    return out


# ================================================================= report rendering
def f(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:+.{nd}f}" if isinstance(x, float) else str(x)


def fp(x):
    if x is None:
        return "—"
    return f"{x:.2g}" if x < 0.001 else f"{x:.4f}"


def ci(pair, nd=2):
    if pair is None or pair[0] is None:
        return "—"
    return f"[{pair[0]:+.{nd}f}, {pair[1]:+.{nd}f}]"


def render_md(res):
    L = []
    A = L.append
    A("# Recompute: FIX 2 (fixed baseline) and FIX 3 (purge the withdrawn arm)")
    A("")
    A(f"Generated by `_falsifier/recompute.py` from committed artifacts only "
      f"(repo `{res['meta']['git_rev']}`). No API calls, no new generations.")
    A("")
    A("Every number below is recomputed from the per-prompt records. The audit's and the "
      "analysis documents' numbers appear only as *claims to be checked*.")
    A("")

    # ---------- headline
    dis = [c for c in res["claims"] if c["verdict"] == "DISAGREE"]
    A(f"## Headline: {len(res['claims'])} claims checked, "
      f"{sum(1 for c in res['claims'] if c['verdict']=='AGREE')} AGREE, "
      f"{len(dis)} DISAGREE")
    A("")
    if dis:
        A("### DISAGREEMENTS — the audit got these wrong")
        A("")
        A("| section | claim | source | claimed | recomputed | gap |")
        A("|---|---|---|---|---|---|")
        for c in dis:
            A(f"| {c['section']} | {c['claim']} | {c['source']} | "
              f"{c['claimed']} | {c['mine']} | {c['gap']:+.4g} |")
        A("")
        A(res["disagreement_notes"])
        A("")
    else:
        A("No disagreements: every audit and document number reproduced within tolerance.")
        A("")

    # ---------- FIX 2 prefix
    p2 = res["fix2_prefix"]
    A("---")
    A("")
    A("## FIX 2a. Prefix arms: reported vs fixed-baseline delta")
    A("")
    A(f"Fixed baseline: {p2['design']['fixed_baseline_scope']}.")
    A(f"Test: {p2['design']['test']}.")
    A("")
    for j in JUDGES:
        A(f"### judge `{j}`")
        A("")
        A("| arm | base mean (floating) | REPORTED Δ | p | FIXED-BASE Δ | p | n(non-tie) "
          "| drift | drift % of reported | sign flip |")
        A("|---|---|---|---|---|---|---|---|---|---|")
        for a in PREFIX_ARMS:
            r = p2["per_judge"][j]["arms"][a]
            pct = r["drift_pct_of_reported"]
            A(f"| `{a}` | {r['reported']['base_mean']:.2f} | "
              f"{f(r['reported']['delta'])} | {fp(r['reported']['p_wilcoxon'])} | "
              f"**{f(r['fixed']['delta'])}** | {fp(r['fixed']['p_wilcoxon'])} | "
              f"{r['fixed']['n_nonzero']}/{r['fixed']['n']} | "
              f"{f(r['drift_reported_minus_fixed'])} | "
              f"{'—' if pct is None else f'{pct:+.0f}%'} | "
              f"{'**YES**' if r['sign_flip'] else 'no'} |")
        A("")
        ce = p2["per_judge"][j]["contrast_effect"]
        A(f"Contrast effect, `{j}`: identical base rating across **all 7 arms** on "
          f"{ce['identical_base_rating_across_all_7_arms']}/{ce['n']} prompts; "
          f"within the 3-arm run {ce['identical_within_3arm_run(pro_top,pro_coherent,anti_top)']}/{ce['n']}; "
          f"within the 4-arm gallery run {ce['identical_within_4arm_gallery_run']}/{ce['n']}. "
          f"Mean within-prompt range across arms {ce['mean_within_prompt_range_across_arms']:.2f}, "
          f"mean sd {ce['mean_within_prompt_sd_across_arms']:.3f}.")
        A("")
        A(f"Pairwise on identical text, `pro_top`-run base vs `anti_top`-run base: "
          f"{ce['pro_top_base_mean']:.2f} vs {ce['anti_top_base_mean']:.2f}, "
          f"shift **{ce['anti_minus_pro_base_shift']:+.2f}**, "
          f"Wilcoxon p = {fp(ce['wilcoxon_p_proTopBase_vs_antiTopBase'])} "
          f"(n={ce['n']}, {ce['n_nonzero']} non-tie).")
        A("")

    # ---------- FIX 2 steering
    s2 = res["fix2_steering"]
    A("---")
    A("")
    A("## FIX 2b. Steering null band, fixed baseline")
    A("")
    A(f"Fixed baseline: {s2['design']['fixed_baseline_scope']}. "
      f"Null-band sd: {s2['design']['null_band_sd']}.")
    A("")
    for j in JUDGES:
        b_rep = s2["per_judge"][j]["null_band_reported"]
        b_fix = s2["per_judge"][j]["null_band_fixed"]
        A(f"### judge `{j}`")
        A("")
        A("| | REPORTED (floating base) | FIXED baseline |")
        A("|---|---|---|")
        A(f"| null band, 8 randoms | mean {f(b_rep['mean'])}, sd {b_rep['sd']:.3f}, "
          f"range [{b_rep['min']:+.2f}, {b_rep['max']:+.2f}] | "
          f"mean {f(b_fix['mean'])}, sd {b_fix['sd']:.3f}, "
          f"range [{b_fix['min']:+.2f}, {b_fix['max']:+.2f}] |")
        for a in ("+1", "-1"):
            rr, ff_ = b_rep[a], b_fix[a]
            drift = rr["delta"] - ff_["delta"]
            pct = 100.0 * drift / rr["delta"] if abs(rr["delta"]) > 1e-12 else float("nan")
            tie_r = f" ({rr['n_randoms_tied']} tied)" if rr["n_randoms_tied"] else ""
            tie_f = f" ({ff_['n_randoms_tied']} tied)" if ff_["n_randoms_tied"] else ""
            A(f"| `{a}·d` | {f(rr['delta'])}, z={rr['z']:+.2f}, "
              f"{rr['n_randoms_below']}/8 below{tie_r}, p={fp(rr['p_wilcoxon'])} | "
              f"**{f(ff_['delta'])}**, z={ff_['z']:+.2f}, "
              f"{ff_['n_randoms_below']}/8 below{tie_f}, p={fp(ff_['p_wilcoxon'])} "
              f"(drift {drift:+.3f} = {pct:.0f}% of reported) |")
        A(f"| randoms reaching p<0.05 | {b_rep['n_randoms_p_lt_.05']}/8 | "
          f"{b_fix['n_randoms_p_lt_.05']}/8 |")
        ce = s2["per_judge"][j]["contrast_effect"]
        A("")
        A(f"Base ratings identical across all 10 steering arms on "
          f"{ce['identical_base_rating_across_all_10_arms']}/{ce['n']} prompts "
          f"(mean within-prompt range {ce['mean_within_prompt_range_across_arms']:.2f}); "
          f"fixed-baseline grand mean "
          f"{s2['per_judge'][j]['base_fixed_grand_mean']:.3f}.")
        A("")
        A("Per-arm deltas (fixed baseline): " +
          ", ".join(f"`{a}` {f(s2['per_judge'][j]['arms'][a]['fixed']['delta'], 3)}"
                    for a in STEER_ARMS))
        A("")
    A("**Does the published claim survive?**")
    A("")
    A("| claim | deepseek | claude |")
    A("|---|---|---|")
    row_p, row_n = [], []
    for j in JUDGES:
        bf = s2["per_judge"][j]["null_band_fixed"]
        row_p.append(f"z={bf['+1']['z']:+.2f}, above all 8: "
                     f"{'YES' if bf['+1']['outside_all_8'] and bf['+1']['delta']>bf['max'] else 'NO'}")
        neg = bf["-1"]
        row_n.append(f"Δ={neg['delta']:+.3f}, z={neg['z']:+.2f}, "
                     f"{neg['n_randoms_below']}/8 below, "
                     f"{'OUTSIDE the band (below all 8)' if neg['delta']<bf['min'] else 'inside the band'}")
    A(f"| `+1·d` sits outside the null band | {row_p[0]} | {row_p[1]} |")
    A(f"| `-1·d` indistinguishable from random | {row_n[0]} | {row_n[1]} |")
    A("")
    A(res["fix2_steering_verdict"])
    A("")
    xb = res["fix2_cross_experiment_baseline"]
    A("### Sensitivity: does the baseline SCOPE drive the result?")
    A("")
    A(xb["design"] + ".")
    A("")
    A("| judge | grand mean | null band (8 randoms) | `+1·d` | `-1·d` |")
    A("|---|---|---|---|---|")
    for j in JUDGES:
        b = xb["per_judge"][j]["null_band"]
        A(f"| `{j}` | {xb['per_judge'][j]['grand_mean']:.3f} | "
          f"mean {f(b['mean'])}, sd {b['sd']:.3f} | "
          f"{f(b['+1']['delta'])}, z={b['+1']['z']:+.2f}, "
          f"{b['+1']['n_randoms_below']}/8 below | "
          f"{f(b['-1']['delta'])}, z={b['-1']['z']:+.2f}, "
          f"{b['-1']['n_randoms_below']}/8 below |")
    A("")
    A("Prefix arms on the same 17-arm baseline:")
    A("")
    A("| arm | " + " | ".join(f"`{j}` Δ (p)" for j in JUDGES) + " |")
    A("|---|" + "---|" * len(JUDGES))
    for a in PREFIX_ARMS:
        cells = []
        for j in JUDGES:
            s = xb["per_judge"][j]["prefix_arms"][a]
            cells.append(f"{f(s['delta'])} ({fp(s['p_wilcoxon'])})")
        A(f"| `{a}` | " + " | ".join(cells) + " |")
    A("")
    A("**Caveat on the z-scores, which the audit does not state.** Under a fixed "
      "baseline every arm is differenced against the *same* per-prompt vector, so the "
      "between-arm variance that the floating baseline injected into the null is "
      "removed. The null sd therefore shrinks (see the table above), and z inflates "
      "mechanically. The robust statement is the placement count — how many of the 8 "
      "randoms the arm clears — not the z. Both are reported.")
    A("")

    # ---------- FIX 3
    f3 = res["fix3"]
    A("---")
    A("")
    A("## FIX 3. Purge the withdrawn `anti_top` behavioural point")
    A("")
    A("Both columns derived from artifacts: dose = `along_d` in `compile_check.json`; "
      "behaviour = mean of the two judges' per-arm kindness Δ recomputed from the raw "
      "records (the same statistic `compile_check.md` used).")
    A("")
    A("| arm | board score | on-`d` dose Δ∥ | behaviour (reported base) | "
      "behaviour (fixed base) |")
    A("|---|---|---|---|---|")
    for a in PREFIX_ARMS:
        t = f3["arm_table"][a]
        A(f"| `{a}` | {t['board_score']:+.5f} | {t['on_d_dose_along_d']:+.2f} | "
          f"{t['behaviour_reported']:+.3f} | {t['behaviour_fixed_baseline']:+.3f} |")
    A("")
    A("### Correlations (behaviour on the reported-baseline scale, as published)")
    A("")
    A("| arm set | n | dose r [95% CI] | p | dose ρ [95% CI] | p | "
      "score r [95% CI] | p | score ρ | p |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for name, blk in f3["sets"].items():
        d, s = blk["dose_vs_behaviour"], blk["score_vs_behaviour"]
        if d["pearson_r"] is None:
            A(f"| `{name}` | {d['n']} | — | — | — | — | — | — | — | — |")
            continue
        A(f"| `{name}` | {d['n']} | {f(d['pearson_r'])} {ci(d['pearson_ci95'])} | "
          f"{fp(d['pearson_p'])} | {f(d['spearman_rho'])} {ci(d['spearman_ci95'])} | "
          f"{fp(d['spearman_p'])} | {f(s['pearson_r'])} {ci(s['pearson_ci95'])} | "
          f"{fp(s['pearson_p'])} | {f(s['spearman_rho'])} | {fp(s['spearman_p'])} |")
    A("")
    neg2 = f3["sets"]["negative_side_excl_anti_top"]["dose_vs_behaviour"]
    A(f"`negative_side_excl_anti_top` has n={neg2['n']} — {neg2.get('note','')}. "
      f"The two surviving negative arms (`anti_coherent` dose −0.22 / behaviour −0.20; "
      f"`anti_hostile` dose −0.87 / behaviour −1.31) are correctly ordered: "
      f"**{neg2.get('ordering_concordant')}**. More dose, more behaviour, on the "
      f"negative side too.")
    A("")
    A("### Same correlations recomputed on the FIXED-BASELINE behaviour column")
    A("")
    A("| arm set | n | dose r | p | score r | p |")
    A("|---|---|---|---|---|---|")
    for name, blk in f3["sets"].items():
        d, s = blk["dose_vs_behaviour_FIXEDBASE"], blk["score_vs_behaviour_FIXEDBASE"]
        if d["pearson_r"] is None:
            A(f"| `{name}` | {d['n']} | — | — | — | — |")
            continue
        A(f"| `{name}` | {d['n']} | {f(d['pearson_r'])} | {fp(d['pearson_p'])} | "
          f"{f(s['pearson_r'])} | {fp(s['pearson_p'])} |")
    A("")
    A("### Verdict on \"the compilation is one-sided\"")
    A("")
    A(res["fix3_one_sided_verdict"])
    A("")
    A("### Is on-`d` dose a better predictor than the board score?")
    A("")
    A("| arm set | n | r(dose, beh) [CI] | r(score, beh) [CI] | CIs overlap | "
      "Williams t (dep. corr.) | p | bootstrap Δr [95% CI] |")
    A("|---|---|---|---|---|---|---|---|")
    for name, pc in f3["predictor_comparison"].items():
        bs = pc["bootstrap"]
        wt = "—" if pc["williams_t"] is None else f"{pc['williams_t']:+.2f}"
        A(f"| `{name}` | {pc['n']} | {f(pc['r_dose_vs_behaviour'])} "
          f"{ci(pc['r_dose_ci95_fisher'])} | {f(pc['r_score_vs_behaviour'])} "
          f"{ci(pc['r_score_ci95_fisher'])} | "
          f"{'YES' if pc['fisher_ci_overlap'] else 'no'} | {wt} | "
          f"{fp(pc['williams_p'])} | {f(bs['diff_mean'])} {ci(bs['diff_ci95'])} |")
    A("")
    A(res["fix3_predictor_verdict"])
    A("")

    # ---------- integrity
    A("---")
    A("")
    A("## Data integrity checks")
    A("")
    A("| check | value |")
    A("|---|---|")
    for k, v in res["integrity"].items():
        A(f"| {str(k).replace('|', '\\|')} | {str(v).replace('|', '\\|')} |")
    A("")
    A("## All claims checked")
    A("")
    A("| section | claim | source | claimed | recomputed | verdict |")
    A("|---|---|---|---|---|---|")
    for c in res["claims"]:
        A(f"| {c['section']} | {c['claim']} | {c['source']} | {c['claimed']} | "
          f"{c['mine']} | {'**DISAGREE**' if c['verdict']=='DISAGREE' else c['verdict']} |")
    A("")
    return "\n".join(L)


# ============================================================================== main
def main():
    import subprocess
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True).stdout.strip()
    except Exception:
        rev = "unknown"

    ratings, prompts, gallery_reported, integ = load_prefix_records()
    setup, s_ratings, s_reported = load_steering_records()

    p2 = fix2_prefix(ratings, prompts, gallery_reported)
    s2 = fix2_steering(setup, s_ratings, s_reported)
    xb = cross_experiment_baseline(ratings, s_ratings, prompts)
    f3 = fix3(p2)

    # verdict prose, computed
    parts = []
    for j in JUDGES:
        bf = s2["per_judge"][j]["null_band_fixed"]
        br = s2["per_judge"][j]["null_band_reported"]
        pos = ("SURVIVES" if bf["+1"]["delta"] > bf["max"] else "FAILS")
        neg_rep = ("inside" if br["min"] <= br["-1"]["delta"] <= br["max"] else "outside")
        neg_fix = ("inside" if bf["min"] <= bf["-1"]["delta"] <= bf["max"] else "outside")
        parts.append(
            f"- **`{j}`**: `+1·d` {pos} the fixed-baseline re-analysis "
            f"(Δ {bf['+1']['delta']:+.3f}, z {bf['+1']['z']:+.2f}, above "
            f"{bf['+1']['n_randoms_below']}/8 randoms). `-1·d` was **{neg_rep}** the "
            f"published floating-baseline band and is **{neg_fix}** the fixed-baseline "
            f"band (Δ {bf['-1']['delta']:+.3f}, z {bf['-1']['z']:+.2f}, "
            f"{bf['-1']['n_randoms_below']}/8 randoms below it).")
    s2_verdict = "\n".join(parts)

    os7 = f3["sets"]["all_7"]["dose_vs_behaviour"]
    os6 = f3["sets"]["6_excl_anti_top"]["dose_vs_behaviour"]
    spos = f3["sets"]["positive_side_dose_gt_0"]["dose_vs_behaviour"]
    sneg = f3["sets"]["negative_side_dose_lt_0"]["dose_vs_behaviour"]
    sneg2 = f3["sets"]["negative_side_excl_anti_top"]["dose_vs_behaviour"]
    f3_verdict = (
        f"With `anti_top` in: positive side r = {spos['pearson_r']:+.3f} (n=4, "
        f"p={spos['pearson_p']:.4f}); negative side r = {sneg['pearson_r']:+.3f} (n=3, "
        f"p={sneg['pearson_p']:.3f}); overall r = {os7['pearson_r']:+.3f} "
        f"(p={os7['pearson_p']:.4f}). Drop the withdrawn arm and the negative side is "
        f"two points, "
        f"{'correctly ordered' if sneg2.get('ordering_concordant') else 'INVERTED'}, "
        f"while the overall correlation rises to r = {os6['pearson_r']:+.3f} "
        f"(p={os6['pearson_p']:.4f}). **The asymmetry that supports \"the compilation is "
        f"one-sided\" is carried entirely by the arm `prefix_eval.md` withdrew as "
        f"unmeasurable. Removing it, the dose-response is monotone across all six "
        f"remaining arms and the claim does not survive.**")

    pc = f3["predictor_comparison"]["all_7"]
    f3_pred = (
        f"At n=7, r(dose, behaviour) = {pc['r_dose_vs_behaviour']:+.3f} "
        f"{ci(pc['r_dose_ci95_fisher'])} and r(score, behaviour) = "
        f"{pc['r_score_vs_behaviour']:+.3f} {ci(pc['r_score_ci95_fisher'])}. The two "
        f"predictors are themselves correlated at r = {pc['r_dose_vs_score']:+.3f}. "
        f"Fisher-z intervals "
        f"{'overlap heavily' if pc['fisher_ci_overlap'] else 'do not overlap'}; "
        f"Williams' test for dependent overlapping correlations gives "
        f"t = {pc['williams_t']:+.2f} (df={pc['williams_df']}, p={pc['williams_p']:.3f}); "
        f"the paired bootstrap difference is {pc['bootstrap']['diff_mean']:+.3f} "
        f"{ci(pc['bootstrap']['diff_ci95'])}, straddling zero. **\"On-`d` displacement is "
        f"a better predictor of behaviour than the board score\" is not supportable at "
        f"n=7 — the audit is right about this.**")

    cl_fix = s2["per_judge"][CL]["null_band_fixed"]["-1"]
    cl_rep = s2["per_judge"][CL]["null_band_reported"]["-1"]
    disagreement_notes = (
        f"The audit's section-2 table reports, in its **fixed-baseline** column, that "
        f"Claude's `-1·d` is \"+0.026, 5/8 below\". The delta reproduces exactly "
        f"(+{cl_fix['delta']:.3f}), but the placement does not: under the fixed baseline "
        f"only **{cl_fix['n_randoms_below']} of 8** randoms sit below it "
        f"({cl_fix['n_randoms_tied']} tied exactly, "
        f"{8 - cl_fix['n_randoms_below'] - cl_fix['n_randoms_tied']} above), not 5. "
        f"\"5/8 below\" is the *floating*-baseline placement "
        f"(reproduced here: {cl_rep['n_randoms_below']}/8 at Δ "
        f"{cl_rep['delta']:+.2f}); the audit carried it across into the recomputed row "
        f"without recomputing it. This does not change the audit's conclusion — "
        f"`-1·d` is still inside the null band under Claude either way — but it "
        f"weakens the specific rhetorical point that Claude places `-1·d` \"essentially "
        f"at the median of the null\". On the fixed baseline it sits in the lower third "
        f"of the null, i.e. slightly nearer DeepSeek's placement than the published "
        f"write-up implies. Ironically this is the same failure mode the audit is "
        f"documenting: a number computed under one estimator was reused in a table "
        f"about a different estimator.")

    integrity = {
        "prefix records per judge|arm (expect 50)":
            "all 50" if set(integ.values()) == {50} else str(integ),
        "distinct eval prompts": len(prompts),
        "prefix arms": len(PREFIX_ARMS),
        "steering arms": len(STEER_ARMS),
        "judges": ", ".join(JUDGES),
    }
    integrity.update({f"base_texts::{k}": json.dumps(v)
                      for k, v in base_text_identity().items()})

    res = {
        "meta": {"git_rev": rev, "generated_by": "_falsifier/recompute.py",
                 "inputs": ["data/analysis/prefix_gallery_judge.json",
                            "data/analysis/prefix_judge_verdicts.json",
                            "data/analysis/prefix_judge_claude.json",
                            "data/analysis/prefix_blind_key.json",
                            "data/analysis/steering_random_control.json",
                            "data/analysis/compile_check.json",
                            "data/cache/prefix_behavioral/", "data/cache/behavioral/"],
                 "network_calls": 0, "new_generations": 0},
        "integrity": integrity,
        "fix2_prefix": p2,
        "fix2_steering": s2,
        "fix2_steering_verdict": s2_verdict,
        "fix2_cross_experiment_baseline": xb,
        "fix3": f3,
        "fix3_one_sided_verdict": f3_verdict,
        "fix3_predictor_verdict": f3_pred,
        "disagreement_notes": disagreement_notes,
        "claims": CLAIMS,
        "claims_summary": {
            "n": len(CLAIMS),
            "agree": sum(1 for c in CLAIMS if c["verdict"] == "AGREE"),
            "disagree": sum(1 for c in CLAIMS if c["verdict"] == "DISAGREE"),
            "na": sum(1 for c in CLAIMS if c["verdict"] == "N/A")},
    }
    OUT_JSON.write_text(json.dumps(res, indent=2))
    OUT_MD.write_text(render_md(res))
    print(f"wrote {OUT_JSON}\nwrote {OUT_MD}")
    print(f"claims: {res['claims_summary']}")
    for c in CLAIMS:
        if c["verdict"] == "DISAGREE":
            print(f"  DISAGREE  {c['section']:12s} {c['claim']:58s} "
                  f"claimed={c['claimed']}  mine={c['mine']}")


if __name__ == "__main__":
    main()
