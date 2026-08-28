"""Ablate the pro-human direction, instead of subtracting it.

THE QUESTION: `-1*d` sits inside the norm-matched random band, so subtracting the
direction does not invert the behaviour. But subtraction at alpha = 1.0*||R|| pushes the
residual to cos(R,d) ~ -0.70, which is nowhere the model ever naturally sits (base is
+0.0067). That is not a clean test of "does d mediate the behaviour" — it is a test of
"what happens if you shove the residual somewhere impossible".

Ablation is the clean version, and the standard one (Arditi et al. did refusal this way):
remove ONLY the component that is actually there.

    R  <-  R - k * (R . d_hat) d_hat

k=1 is pure ablation. k>1 overshoots into the subtraction regime, so a sweep interpolates
between "remove what is present" and "-1*d".

PREDICTION, from numbers already measured: base cos(R,d) = 0.0067 at ||R|| = 30.07, so the
component available to ablate is ~0.20, which is 0.67% of the residual norm. The anti_top
prefix moves 1.49 along d (7x more) and `-1*d` moves 30.07 (149x more). So ablation should
do close to nothing. A NULL IS THE INFORMATIVE OUTCOME HERE: if removing d entirely does
not change behaviour, d is not mediating the behaviour, which is the point.

Scope is matched to the injection arms deliberately: the edit is applied to the layer-24
hidden state inside model.generate(), which lands on prompt positions only (prefill), the
same as scripts/steering_bakeoff.generate(). Same 50 prompts, same 40 new tokens, same
cache dir and key scheme, so the new arms drop straight into the existing blind/judge path.

THE CONTROL, added 2026-08-28 and load-bearing. The 8 random directions in
steering_random_control are norm-matched to the INJECTION (alpha = 30.07). Nothing there is
matched to ABLATION, which removes ~0.5. So this script also ablates along RANDOM unit
directions: R <- R - (R . r_hat) r_hat, same operation, same scope, same prompts. In
n = 5120, E|R . r_hat| = ||R|| sqrt(2/(pi n)) ~ 0.33 against a measured mean |R . d_hat| of
~0.48, so the control removes a component within ~1.5x of d's by construction, not by
tuning. Without it, "ablation did something" cannot be told apart from "removing any small
component at L24 does something".

    python -u scripts/steering_ablation.py measure > /tmp/abl_m.log 2>&1 &
    python -u scripts/steering_ablation.py generate --ks 1,2 > /tmp/abl_g.log 2>&1 &
    python -u scripts/steering_ablation.py generate --ks "" --random 2 > /tmp/abl_r.log 2>&1 &
    python scripts/steering_ablation.py blind
    #  ... judge the batches in data/cache/behavioral/ablation_blind/in -> .../out ...
    python scripts/steering_ablation.py report

Run it EXACTLY like that. Do not pipe through `tail`/`head`: they buffer to EOF, so a
working job becomes indistinguishable from a hung one. Every print flushes and every NDIF
call is announced BEFORE it starts, so silence in the log localises the hang.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, '/Users/sohampadia/workspace/Nikhil/research/competition')
import numpy as np


ANALYSIS = Path("data/analysis")
BLIND_DIR = Path("data/cache/behavioral/ablation_blind")
BLIND_KEY = ANALYSIS / "steering_ablation_blind_key.json"
REPORT_JSON = ANALYSIS / "steering_ablation.json"
RAND_SEED = 20260828


def say(*a):
    print(*a, flush=True)


say("[boot] importing app modules")
from app.config import settings                                        # noqa: E402
from app.ndif_client import ResidualReader                             # noqa: E402
from scripts.behavioral_eval import CACHE_DIR, _gen_key, _load_d, _load_prompts  # noqa: E402
from scripts.extract_direction import with_retry                       # noqa: E402


def random_dirs(k, dim, seed=RAND_SEED):
    """k random unit vectors, the ablation-matched null. Near-orthogonal to d in 5120d."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((k, dim))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def rand_arm(i, seed=RAND_SEED):
    return f"ablate_rand{i + 1}s{seed}"


def _arm_specs(ks, n_random, d, seed=RAND_SEED):
    """[(arm, k, vector)] for the real ablation arms and the random-ablation controls."""
    out = [(f"ablate{k:g}x" if k != 1 else "ablate", k, d) for k in ks]
    out += [(rand_arm(i, seed), 1.0, v) for i, v in enumerate(random_dirs(n_random, d.shape[0], seed))]
    return out


def _reader():
    return ResidualReader.build(settings.model_id, "ndif", ndif_key=settings.ndif_api_key,
                                prepend_bos=settings.prepend_bos)


def generate_ablated(reader, prompt, max_new, layer, d, k, attempts=4, wait=20.0):
    """Remove k * the component along unit vector `d` at `layer`, everywhere the edit reaches."""
    import torch

    def _run():
        dt = torch.tensor(np.ascontiguousarray(d), dtype=torch.float32)
        with reader.model.generate(prompt, max_new_tokens=max_new, remote=reader.remote):
            hid = reader._layer_module(layer).output[0]
            dv = dt.to(hid.device).to(hid.dtype)
            proj = (hid @ dv).unsqueeze(-1) * dv      # (R . d_hat) d_hat, per position
            hid[:] -= float(k) * proj
            out = reader.model.generator.output.save()
        seq = out.value if hasattr(out, "value") else out
        return reader.model.tokenizer.decode(seq[0])

    return with_retry(_run, attempts=attempts, wait=wait)


def cmd_measure(args):
    """How big is the thing we are removing? One NDIF call, no generation.

    Stores PER-PROMPT components, not just the summary: the per-prompt size of the ablated
    component is the regressor for "does ablation only bite where the component is large?".
    Also asserts the residual matrix is finite -- numpy emits spurious divide/overflow
    warnings on this platform's matmul (compile_check.md), so the numbers get checked.
    """
    d, layer = _load_d()
    prompts = _load_prompts(args.limit)
    say(f"[boot] ||d|| = {np.linalg.norm(d):.6f} (expect 1.0)")
    r = _reader()
    say(f"[ndif 1/1] last-token residuals, {len(prompts)} prompts, L{layer}")
    M = np.asarray(with_retry(r.batch_last_resids, prompts, layer, attempts=4, wait=20.0),
                   dtype=np.float64)
    n_bad = int((~np.isfinite(M)).sum())
    say(f"[check] residual matrix {M.shape}, non-finite entries: {n_bad}, "
        f"abs max {np.abs(M).max():.4g}")
    if n_bad:
        raise SystemExit("non-finite residuals -- the stored numbers cannot be trusted")
    norms = np.linalg.norm(M, axis=1)
    along = M @ d                       # signed component along d_hat
    cos = along / norms
    rvs = random_dirs(args.random, d.shape[0], args.seed) if args.random else np.zeros((0, d.shape[0]))
    rand_along = M @ rvs.T if len(rvs) else np.zeros((len(prompts), 0))
    out = {
        "layer": layer, "n_prompts": len(prompts),
        "resid_norm_mean": float(norms.mean()), "resid_norm_sd": float(norms.std(ddof=1)),
        "cos_mean": float(cos.mean()), "cos_min": float(cos.min()), "cos_max": float(cos.max()),
        "along_d_mean": float(along.mean()), "along_d_sd": float(along.std(ddof=1)),
        "along_d_min": float(along.min()), "along_d_max": float(along.max()),
        "frac_of_norm_mean": float((np.abs(along) / norms).mean()),
        "abs_along_d_mean": float(np.abs(along).mean()),
        "finite_check": {"non_finite": n_bad, "abs_max": float(np.abs(M).max())},
        "random_control": {
            "seed": args.seed, "n": int(len(rvs)),
            "cos_with_d": [float(abs(np.dot(v, d))) for v in rvs],
            "theory_abs_component": float(norms.mean() * np.sqrt(2 / (np.pi * d.shape[0]))),
            "arms": {rand_arm(i, args.seed): {
                "abs_along_mean": float(np.abs(rand_along[:, i]).mean()),
                "along_mean": float(rand_along[:, i].mean()),
                "along_sd": float(rand_along[:, i].std(ddof=1)),
                "along_min": float(rand_along[:, i].min()),
                "along_max": float(rand_along[:, i].max()),
                "frac_of_norm_mean": float((np.abs(rand_along[:, i]) / norms).mean()),
            } for i in range(len(rvs))},
        },
        "per_prompt": [{"prompt": pr, "resid_norm": float(norms[i]),
                        "along_d": float(along[i]), "cos": float(cos[i]),
                        "along_rand": [float(x) for x in rand_along[i]]}
                       for i, pr in enumerate(prompts)],
    }
    say("\n" + json.dumps({k: v for k, v in out.items() if k != "per_prompt"}, indent=1))
    say(f"\n  ablation removes on average {out['along_d_mean']:.3f} along d "
        f"({100*out['frac_of_norm_mean']:.2f}% of ||R||); mean |component| "
        f"{out['abs_along_d_mean']:.3f}")
    say(f"  for scale: anti_top prefix moves -1.49, -1*d moves -30.07")
    P = ANALYSIS / "steering_ablation_measure.json"
    P.write_text(json.dumps(out, indent=1))
    say(f"\nwrote {P}")


def cmd_generate(args):
    d, layer = _load_d()
    prompts = _load_prompts(args.limit)
    ks = [float(x) for x in args.ks.split(",") if x.strip()]
    specs = _arm_specs(ks, args.random, d, args.seed)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    reader = None
    n_new = n_hit = 0
    say(f"{len(prompts)} prompts x {len(specs)} ablation arms on {settings.model_id} L{layer}")
    say(f"arms: {[a for a, _, _ in specs]}")
    for arm, k, vec in specs:
        for i, prompt in enumerate(prompts, 1):
            fp = CACHE_DIR / f"{_gen_key(settings.model_id, layer, prompt, arm, k, args.max_new)}.json"
            if fp.exists():
                n_hit += 1
                continue
            if reader is None:
                say("[boot] building reader")
                reader = _reader()
            say(f"[ndif] {arm} {i}/{len(prompts)}: {prompt[:48]!r}")
            text = generate_ablated(reader, prompt, args.max_new, layer, vec, k)
            fp.write_text(json.dumps({"prompt": prompt, "arm": arm, "alpha": k, "text": text}))
            n_new += 1
    say(f"\ndone: {n_new} new, {n_hit} cached")


def cmd_check(args):
    """Manipulation check: does the edit actually land? 33/50 continuations come back
    byte-identical to base, and a reader is entitled to suspect the hook never fired.
    Two traces per prompt, one clean and one ablated, reading the same layer-L last-token
    residual. cos(R,d) must go from ~+0.007 to ~0 if the projection was really removed."""
    import torch
    d, layer = _load_d()
    prompts = _load_prompts(args.limit)[:args.n]
    r = _reader()
    dt = torch.tensor(np.ascontiguousarray(d), dtype=torch.float32)
    k = float(args.k)
    rows = []
    for i, prompt in enumerate(prompts, 1):
        say(f"[ndif] check {i}/{len(prompts)}: {prompt[:48]!r}")

        def _clean(prompt=prompt):
            with r.model.trace(prompt, remote=r.remote):
                saved = r._layer_module(layer).output[0, -1, :].save()
            v = saved.value if hasattr(saved, "value") else saved
            return np.asarray(v.detach().to(torch.float32).cpu().numpy(), dtype=np.float64)

        def _ablated(prompt=prompt):
            with r.model.trace(prompt, remote=r.remote):
                hid = r._layer_module(layer).output[0]
                dv = dt.to(hid.device).to(hid.dtype)
                hid[:] -= k * ((hid @ dv).unsqueeze(-1) * dv)
                saved = hid[-1, :].save()
            v = saved.value if hasattr(saved, "value") else saved
            return np.asarray(v.detach().to(torch.float32).cpu().numpy(), dtype=np.float64)

        pre = with_retry(_clean, attempts=4, wait=20.0)
        post = with_retry(_ablated, attempts=4, wait=20.0)
        rows.append({"prompt": prompt,
                     "cos_before": float(pre @ d / np.linalg.norm(pre)),
                     "cos_after": float(post @ d / np.linalg.norm(post)),
                     "along_before": float(pre @ d), "along_after": float(post @ d),
                     "norm_before": float(np.linalg.norm(pre)),
                     "norm_after": float(np.linalg.norm(post)),
                     "edit_norm": float(np.linalg.norm(post - pre)),
                     "cos_edit_with_d": float((post - pre) @ d / (np.linalg.norm(post - pre) + 1e-12))})
        z = rows[-1]
        say(f"   cos {z['cos_before']:+.6f} -> {z['cos_after']:+.6f}   "
            f"along {z['along_before']:+.4f} -> {z['along_after']:+.4f}   "
            f"|edit| {z['edit_norm']:.4f}  cos(edit,d) {z['cos_edit_with_d']:+.4f}")
    P = ANALYSIS / "steering_ablation_check.json"
    P.write_text(json.dumps({"k": k, "layer": layer, "rows": rows}, indent=1))
    say(f"\nwrote {P}")


# ---------------------------------------------------------------------------
# blind / report -- mirrors scripts/steering_random_control.py's claude protocol
# ---------------------------------------------------------------------------

def _text(prompt, arm, alpha, layer, max_new):
    fp = CACHE_DIR / f"{_gen_key(settings.model_id, layer, prompt, arm, alpha, max_new)}.json"
    return json.loads(fp.read_text())["text"] if fp.exists() else None


def _cont(text, prompt):
    if text is None:
        return None
    i = text.find(prompt)
    return text[i + len(prompt):].strip() if i >= 0 else text.strip()


def _pairs(args):
    """[(arm, prompt, ablated_continuation, base_continuation)] for every generated arm."""
    d, layer = _load_d()
    prompts = _load_prompts(args.limit)
    out = []
    for arm, k, _v in _arm_specs([float(x) for x in args.ks.split(",") if x.strip()],
                                 args.random, d, args.seed):
        for prompt in prompts:
            base = _cont(_text(prompt, "base", 0.0, layer, args.max_new), prompt)
            abl = _cont(_text(prompt, arm, k, layer, args.max_new), prompt)
            if base and abl:
                out.append((arm, prompt, abl, base))
    return out


def cmd_blind(args):
    """Blinded batches: every pair in BOTH A/B orders, arm labels stripped, globally
    shuffled. The two orientations of a pair go to disjoint batch groups so no judging
    context ever holds both."""
    import hashlib
    import random as _random
    pairs = _pairs(args)
    (BLIND_DIR / "in").mkdir(parents=True, exist_ok=True)
    (BLIND_DIR / "out").mkdir(parents=True, exist_ok=True)
    key = json.loads(BLIND_KEY.read_text()) if BLIND_KEY.exists() else {}
    groups = {"fwd": [], "rev": []}
    for arm, prompt, abl, base in pairs:
        for orient in ("fwd", "rev"):
            iid = hashlib.sha256(f"{arm}\x00{prompt}\x00{orient}".encode()).hexdigest()[:16]
            if iid in key:                       # already emitted in an earlier wave
                continue
            a, b = (abl, base) if orient == "fwd" else (base, abl)
            groups[orient].append({"item_id": iid, "sentence": prompt,
                                   "continuation_A": a, "continuation_B": b})
            key[iid] = {"prompt": prompt, "arm": arm, "orientation": orient,
                        "steered_is": "A" if orient == "fwd" else "B"}
    rng = _random.Random(args.seed)
    n = len(list((BLIND_DIR / "in").glob("batch_*.json")))
    for orient in ("fwd", "rev"):
        items = groups[orient]
        rng.shuffle(items)
        for i in range(0, len(items), args.batch_size):
            n += 1
            fp = BLIND_DIR / "in" / f"batch_{n:02d}.json"
            fp.write_text(json.dumps({"batch": n, "items": items[i:i + args.batch_size]},
                                     indent=1, ensure_ascii=False))
    BLIND_KEY.write_text(json.dumps(key, indent=1))
    say(f"{sum(len(v) for v in groups.values())} new blinded items, {n} batches total "
        f"-> {BLIND_DIR/'in'}")
    say(f"key (do not peek until scoring) -> {BLIND_KEY}")


def _fixed_baseline():
    """Per-prompt FIXED baseline: claude-opus-5 kindness_base averaged over the 10 existing
    steering arms. A floating (re-rated per arm) baseline inflates effects by 13-37%
    (_falsifier/recompute_result.md), so the arm-specific base rating is never used."""
    src = json.loads((ANALYSIS / "steering_random_control.json").read_text())["judged_claude"]
    acc = {}
    for arm, blob in src.items():
        for pkey, rec in blob["records"].items():
            prompt = pkey.split("|", 1)[1]
            acc.setdefault(prompt, []).append(rec["kindness_base"])
    return {p: sum(v) / len(v) for p, v in acc.items()}, sorted(src)


def _comparison(rows):
    """Place the ablation arms against the arms that survived, from committed artifacts.

    Kindness is compared ONLY to arms that were not withdrawn. `anti_top` enters on the
    judge-free and marker degeneracy measures only: `prefix_eval.md` withdrew its kindness
    numbers as unmeasurable because the text is degenerate, so a kindness comparison
    against it would be meaningless.
    """
    import collections
    import statistics
    from scripts.prefix_transfer_eval import distinct_n, looping

    band = None
    fx = Path("_falsifier/recompute_result.json")
    if fx.exists():
        cl = json.loads(fx.read_text())["fix2_steering"]["per_judge"]["claude-opus-5"]["arms"]
        rnd = [v["fixed"]["delta"] for a, v in cl.items() if a.startswith("rand")]
        band = {"n": len(rnd), "mean": round(statistics.fmean(rnd), 3),
                "sd": round(statistics.stdev(rnd), 3), "min": round(min(rnd), 3),
                "max": round(max(rnd), 3),
                "note": "norm-matched to the INJECTION (alpha=30.07), ~100x the ablation edit; "
                        "the ablation-matched null is the ablate_rand* arms here",
                "reference_arms": {a: v["fixed"]["delta"] for a, v in cl.items()
                                   if not a.startswith("rand")}}
        for arm, r in rows.items():
            r["z_vs_injection_null_band"] = (
                None if band is None or not band["sd"] else
                round((r["kindness_delta_fixed"] - band["mean"]) / band["sd"], 2))

    prefix = {}
    pd = Path("data/cache/prefix_behavioral")
    if pd.exists():
        by = collections.defaultdict(list)
        for fp in pd.glob("*.json"):
            rec = json.loads(fp.read_text())
            by[rec["arm"]].append(rec["continuation"])
        prefix = {a: {"distinct4": round(sum(distinct_n(c) for c in cs) / len(cs), 3),
                      "loops": sum(looping(c) for c in cs), "n": len(cs)}
                  for a, cs in by.items()}

    degen = {}
    kf, jf = Path("data/analysis/prefix_blind_key.json"), Path("data/analysis/prefix_judge_claude.json")
    if kf.exists() and jf.exists():
        key = json.loads(kf.read_text())
        tot, hit, hitb = collections.Counter(), collections.Counter(), collections.Counter()
        for pid, rec in json.loads(jf.read_text())["records"].items():
            arm = key[pid]["arm"]
            tot[arm] += 1
            if {"repetition", "incoherent"} & set(rec["markers_prefixed"]):
                hit[arm] += 1
            if {"repetition", "incoherent"} & set(rec["markers_base"]):
                hitb[arm] += 1
        degen = {a: {"prefixed": hit[a], "base_side": hitb[a], "n": tot[a]} for a in tot}
    return {"injection_null_band_claude_fixed": band,
            "prefix_arms_judge_free": prefix,
            "prefix_arms_degeneracy_claude": degen}


def cmd_report(args):
    import collections
    from scripts.prefix_behavior_eval import _coerce, _mean, _paired_p
    from scripts.prefix_transfer_eval import distinct_n, looping

    key = json.loads(BLIND_KEY.read_text())
    got = {}
    for fp in sorted((BLIND_DIR / "out").glob("*.json")):
        for iid, rec in json.loads(fp.read_text()).items():
            c = _coerce(rec)
            if c:
                got[iid] = c
    flip = {"A": "B", "B": "A", "T": "T"}
    merged = {}
    for iid, info in key.items():
        if info["orientation"] != "fwd" or iid not in got:
            continue
        rev = next((i for i, k2 in key.items()
                    if k2["arm"] == info["arm"] and k2["prompt"] == info["prompt"]
                    and k2["orientation"] == "rev"), None)
        if rev not in got:
            continue
        r1, r2 = got[iid], got[rev]
        v1, v2 = r1["kinder"], flip[r2["kinder"]]
        merged[(info["arm"], info["prompt"])] = {
            "verdict": v1 if v1 == v2 else None,
            "raw_verdicts": [r1["kinder"], r2["kinder"]],
            "kindness_ablated": round((r1["kindness_A"] + r2["kindness_B"]) / 2, 2),
            "kindness_base": round((r1["kindness_B"] + r2["kindness_A"]) / 2, 2),
            "markers_ablated": sorted(set(r1["markers_A"]) & set(r2["markers_B"])),
            "markers_base": sorted(set(r1["markers_B"]) & set(r2["markers_A"])),
        }

    fixed, fixed_arms = _fixed_baseline()
    meas = json.loads((ANALYSIS / "steering_ablation_measure.json").read_text())
    comp = {r["prompt"]: r for r in meas.get("per_prompt", [])}
    d, layer = _load_d()
    prompts = _load_prompts(args.limit)
    base_c = {p: _cont(_text(p, "base", 0.0, layer, args.max_new), p) for p in prompts}
    specs = _arm_specs([float(x) for x in args.ks.split(",") if x.strip()],
                       args.random, d, args.seed)

    rows = {}
    for ai, (arm, k, _v) in enumerate(specs):
        recs = {p: r for (a, p), r in merged.items() if a == arm}
        cs = {p: _cont(_text(p, arm, k, layer, args.max_new), p) for p in prompts}
        cs = {p: t for p, t in cs.items() if t}
        d_fixed = [recs[p]["kindness_ablated"] - fixed[p] for p in recs if p in fixed]
        d_float = [r["kindness_ablated"] - r["kindness_base"] for r in recs.values()]
        wins = sum(1 for r in recs.values() if r["verdict"] == "A")
        losses = sum(1 for r in recs.values() if r["verdict"] == "B")
        ties = sum(1 for r in recs.values() if r["verdict"] == "T")
        undec = sum(1 for r in recs.values() if r["verdict"] is None)
        mk = collections.Counter(m for r in recs.values() for m in r["markers_ablated"])
        degen = sum(1 for r in recs.values()
                    if {"repetition", "incoherent"} & set(r["markers_ablated"]))
        pf, tf = _paired_p(d_fixed)
        pfl, tfl = _paired_p(d_float)
        # per-prompt effect vs per-prompt size of the removed component
        xs, ys = [], []
        for pr in recs:
            if pr in comp and pr in fixed:
                x = (comp[pr]["along_rand"][ai - len(specs) + args.random]
                     if arm.startswith("ablate_rand") else comp[pr]["along_d"])
                xs.append(abs(x))
                ys.append(recs[pr]["kindness_ablated"] - fixed[pr])
        r_pear = r_p = None
        if len(xs) > 2:
            r_pear = float(np.corrcoef(xs, ys)[0, 1])
            try:
                from scipy.stats import pearsonr
                r_p = float(pearsonr(xs, ys).pvalue)
            except Exception:  # noqa: BLE001
                r_p = None
        # the text-actually-changed subgroup: greedy decoding leaves many pairs identical,
        # and an identical pair can only dilute the estimate toward zero
        chg = [pr for pr in recs if pr in cs and pr in base_c and cs[pr] != base_c[pr]]
        d_chg = [recs[pr]["kindness_ablated"] - fixed[pr] for pr in chg if pr in fixed]
        pc, _tc = _paired_p(d_chg)
        mkb = collections.Counter(m for r in recs.values() for m in r["markers_base"])
        degen_b = sum(1 for r in recs.values()
                      if {"repetition", "incoherent"} & set(r["markers_base"]))
        rows[arm] = {
            "k": k, "n": len(recs),
            "kindness_delta_fixed": round(_mean(d_fixed), 3), "p_fixed": round(pf, 5), "test": tf,
            "kindness_delta_floating": round(_mean(d_float), 3), "p_floating": round(pfl, 5),
            "wins": wins, "losses": losses, "ties": ties, "undecided": undec,
            "markers": dict(mk), "degenerate": degen,
            "markers_base_side": dict(mkb), "degenerate_base_side": degen_b,
            "identical_to_base": len(cs) - len(chg), "n_changed": len(chg),
            "kindness_delta_fixed_changed_only": round(_mean(d_chg), 3),
            "p_fixed_changed_only": round(pc, 5),
            "distinct4": round(_mean([distinct_n(t) for t in cs.values()]), 3),
            "loops": sum(looping(t) for t in cs.values()), "n_texts": len(cs),
            "r_effect_vs_component": None if r_pear is None else round(r_pear, 3),
            "r_effect_vs_component_p": None if r_p is None else round(r_p, 4),
            "records": {pr: r for pr, r in recs.items()},
        }
        say(f"  {arm:>22} n={len(recs):2d} Dfix={_mean(d_fixed):+.3f} (p={pf:.4f}) "
            f"Dflo={_mean(d_float):+.3f} W/L/T={wins}/{losses}/{ties} "
            f"ident={rows[arm]['identical_to_base']}/{len(cs)} "
            f"Dchg={_mean(d_chg):+.3f} (p={pc:.3f}) "
            f"degen={degen}/{len(recs)} (base {degen_b}) loop={rows[arm]['loops']}/{len(cs)} "
            f"d4={rows[arm]['distinct4']:.3f} r={rows[arm]['r_effect_vs_component']} "
            f"(p={rows[arm]['r_effect_vs_component_p']})")

    bt = [t for t in base_c.values() if t]
    base_row = {"distinct4": round(_mean([distinct_n(t) for t in bt]), 3),
                "loops": sum(looping(t) for t in bt), "n_texts": len(bt)}
    say(f"  {'base':>22} loop={base_row['loops']}/{base_row['n_texts']} "
        f"d4={base_row['distinct4']:.3f}")
    out = {"measure": {k: v for k, v in meas.items() if k != "per_prompt"},
           "comparison": _comparison(rows),
           "fixed_baseline": {"source": "steering_random_control.json judged_claude",
                              "arms": fixed_arms,
                              "grand_mean": round(sum(fixed.values()) / len(fixed), 3),
                              "per_prompt": fixed},
           "judge": {"model": "claude-opus-5", "rubric": "v2 (scripts/prefix_behavior_eval.py)",
                     "orders": "both, verdict counted only when position-consistent"},
           "base": base_row, "arms": rows}
    chk = ANALYSIS / "steering_ablation_check.json"
    if chk.exists():
        out["manipulation_check"] = json.loads(chk.read_text())
    # cross-arm text overlap: how much of the "text changed" is specific to the direction
    txt = {arm: {p: _cont(_text(p, arm, k, layer, args.max_new), p) for p in prompts}
           for arm, k, _v in specs}
    chg = {arm: {p for p, t in ts.items() if t and base_c.get(p) and t != base_c[p]}
           for arm, ts in txt.items()}
    out["text_change"] = {
        "changed_vs_base": {a: len(v) for a, v in chg.items()},
        "pairwise_changed_overlap": {f"{a}|{b}": len(chg[a] & chg[b])
                                     for i, a in enumerate(chg) for b in list(chg)[i + 1:]},
        "identical_text_between_arms": {
            f"{a}|{b}": sum(1 for p in prompts if txt[a].get(p) and txt[a][p] == txt[b].get(p))
            for i, a in enumerate(txt) for b in list(txt)[i + 1:]},
    }
    REPORT_JSON.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    say(f"\n-> {REPORT_JSON}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("measure", cmd_measure), ("generate", cmd_generate),
                     ("check", cmd_check), ("blind", cmd_blind), ("report", cmd_report)):
        p = sub.add_parser(name)
        p.add_argument("--limit", type=int, default=50)
        p.add_argument("--random", type=int, default=0,
                       help="N random-direction ablation controls (the matched null)")
        p.add_argument("--seed", type=int, default=RAND_SEED)
        if name not in ("measure", "check"):
            p.add_argument("--ks", default="1", help="comma list; 1 = pure ablation, >1 overshoots")
            p.add_argument("--max-new", type=int, default=40)
        if name == "check":
            p.add_argument("--k", type=float, default=1.0)
            p.add_argument("--n", type=int, default=3)
        if name == "blind":
            p.add_argument("--batch-size", type=int, default=25)
        p.set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)
