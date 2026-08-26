"""The steering arm's missing baseline: is it `d`, or is it just a big perturbation?

behavioral_eval.md reports that adding α·d to the layer-L residual stream changes
generation at α = 0.5 and 1.0 times ‖R‖. Those are large perturbations, and the eval has
no control for size: a RANDOM vector of the same norm might derail generation just as
much, which would make that result about perturbation magnitude rather than about the
direction.

This runs the control. k random unit vectors, each scaled to the SAME α·‖R‖ as the real
+1 arm, steered at the same layer, 50 prompts each, judged against unsteered base with
the frozen v2 rubric. Two measures, one of which needs no judge at all:

  distinct-4-gram ratio  — does the perturbation wreck the text?
  kindness delta         — does it move behaviour the way d does?

    python scripts/steering_random_control.py generate --dirs 3
    python scripts/steering_random_control.py judge
    python scripts/steering_random_control.py report
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings  # noqa: E402
from scripts.behavioral_eval import CACHE_DIR, _gen_key, _layer_norm, _load_d  # noqa: E402
from scripts.behavioral_eval import _load_prompts, _reader  # noqa: E402
from scripts.prefix_behavior_eval import _api_key, _judge_pair, _mean, _paired_p  # noqa: E402
from scripts.prefix_transfer_eval import distinct_n, looping  # noqa: E402
from scripts.steering_bakeoff import generate  # noqa: E402

OUT = Path("data/analysis/steering_random_control.json")
SEED = 20260826


def random_dirs(k, dim, seed=SEED):
    """k random unit vectors. In 5120 dimensions a Gaussian draw is near-orthogonal to
    any fixed vector, which is exactly the point: same norm, no alignment with d."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((k, dim))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def arm_name(i):
    return f"rand{i + 1}s{SEED}"


def _text(prompt, arm, alpha, max_new):
    fp = CACHE_DIR / f"{_gen_key(settings.model_id, settings.layer, prompt, arm, alpha, max_new)}.json"
    return json.loads(fp.read_text())["text"] if fp.exists() else None


def _cont(text, prompt):
    if text is None:
        return None
    i = text.find(prompt)
    return text[i + len(prompt):].strip() if i >= 0 else text.strip()


def cmd_generate(args):
    d, layer = _load_d()
    prompts = _load_prompts(args.limit)
    reader = _reader()
    rnorm = _layer_norm(reader, layer)
    alpha = args.mult * rnorm
    dirs = random_dirs(args.dirs, d.shape[0])
    cos = [float(abs(np.dot(v, d))) for v in dirs]
    print(f"L{layer} ‖R‖≈{rnorm:.1f}; α={alpha:.1f} (same as the +{args.mult:g} arm)")
    print(f"|cos(random_i, d)| = {[round(c, 4) for c in cos]}  (≈0 = properly unaligned)\n")

    n_new = n_hit = 0
    for pi, prompt in enumerate(prompts, 1):
        for i, v in enumerate(dirs):
            arm = arm_name(i)
            fp = CACHE_DIR / f"{_gen_key(settings.model_id, layer, prompt, arm, alpha, args.max_new)}.json"
            if fp.exists():
                n_hit += 1
                continue
            text = generate(reader, prompt, args.max_new, steer=(layer, alpha * v))
            fp.write_text(json.dumps({"prompt": prompt, "arm": arm, "alpha": alpha,
                                      "text": text, "seed": SEED, "dir_index": i}))
            n_new += 1
            print(f"  [{pi}/{len(prompts)}] {arm}: {_cont(text, prompt)[:56]!r}…", flush=True)
    print(f"\ndone: {n_new} generated, {n_hit} cached")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    store = json.loads(OUT.read_text()) if OUT.exists() else {}
    store["setup"] = {"layer": layer, "r_norm": rnorm, "alpha": alpha, "mult": args.mult,
                      "seed": SEED, "n_dirs": args.dirs, "cos_with_d": cos,
                      "model_id": settings.model_id}
    OUT.write_text(json.dumps(store, indent=2))


def _arms(args):
    """BOTH signs of the real arm plus the random ones, all at the same |alpha|.

    Both signs are re-judged here, not just +1. The headline claim about this eval is an
    ASYMMETRY between the two directions, and the existing ±1 numbers came from the OLMo
    judge that produced no usable signal. Re-judging only +1 would leave one side of the
    asymmetry resting on a retired instrument.
    """
    store = json.loads(OUT.read_text())
    alpha = store["setup"]["alpha"]
    return ([(f"+{args.mult:g}", alpha), (f"-{args.mult:g}", -alpha)]
            + [(arm_name(i), alpha) for i in range(store["setup"]["n_dirs"])])


def cmd_judge(args):
    key = _api_key()
    prompts = _load_prompts(args.limit)
    store = json.loads(OUT.read_text())
    for arm, alpha in _arms(args):
        recs = {}
        print(f"\njudging {arm} vs base ({len(prompts)} prompts × 2 orders)", flush=True)
        for pi, prompt in enumerate(prompts, 1):
            base = _cont(_text(prompt, "base", 0.0, args.max_new), prompt)
            pre = _cont(_text(prompt, arm, alpha, args.max_new), prompt)
            if not base or not pre:
                continue
            rec, _u = _judge_pair(key, args.model, prompt, pre, base)   # A = steered
            if rec:
                recs[prompt] = {"verdict": rec["verdict"],
                                "kindness_steered": rec["kindness_A"],
                                "kindness_base": rec["kindness_B"],
                                "markers_steered": rec["markers_A"],
                                "comment": rec["comments"][0]}
            if pi % 10 == 0:
                print(f"  [{pi}/{len(prompts)}] {len(recs)} judged", flush=True)
        wins = sum(1 for r in recs.values() if r["verdict"] == "A")
        losses = sum(1 for r in recs.values() if r["verdict"] == "B")
        deltas = [r["kindness_steered"] - r["kindness_base"] for r in recs.values()]
        pv, test = _paired_p(deltas)
        mk = collections.Counter(m for r in recs.values() for m in r["markers_steered"])
        print(f"  {arm}: preferred {wins}/{wins + losses}  Δ={_mean(deltas):+.2f} "
              f"({test} p={pv:.4f}, n={len(recs)})")
        store = json.loads(OUT.read_text())
        store.setdefault("judged", {})[arm] = {
            "model": args.model, "wins": wins, "losses": losses,
            "kindness_delta": round(_mean(deltas), 3), "test": test, "p": round(pv, 5),
            "n": len(recs), "markers": dict(mk), "records": recs}
        OUT.write_text(json.dumps(store, indent=2, ensure_ascii=False))
    print(f"\n→ {OUT}")


def cmd_report(args):
    store = json.loads(OUT.read_text())
    prompts = _load_prompts(args.limit)
    setup = store["setup"]
    print(f"model {setup['model_id']} L{setup['layer']}  ‖R‖≈{setup['r_norm']:.1f}  "
          f"α={setup['alpha']:.1f} ({setup['mult']:g}×‖R‖)")
    print(f"random dirs: {setup['n_dirs']}, seed {setup['seed']}, "
          f"|cos with d| = {[round(c, 4) for c in setup['cos_with_d']]}\n")

    print("objective: distinct-4-gram ratio (no judge)")
    base_c = {p: _cont(_text(p, "base", 0.0, args.max_new), p) for p in prompts}
    rows = {}
    for arm, alpha in _arms(args):
        cs = {p: _cont(_text(p, arm, alpha, args.max_new), p) for p in prompts}
        shared = [p for p in prompts if cs.get(p) and base_c.get(p)]
        d4 = [distinct_n(cs[p]) for p in shared]
        loops = sum(looping(cs[p]) for p in shared)
        deltas = [distinct_n(cs[p]) - distinct_n(base_c[p]) for p in shared]
        pv, _t = _paired_p(deltas)
        rows[arm] = (_mean(d4), _mean(deltas), loops, len(shared), pv)
        print(f"  {arm:>16} d4={_mean(d4):.3f}  Δ={_mean(deltas):+.3f}  "
              f"loop {loops}/{len(shared)}  p={pv:.4f}")
    b4 = [distinct_n(t) for t in base_c.values() if t]
    print(f"  {'base':>16} d4={_mean(b4):.3f}  loop "
          f"{sum(looping(t) for t in base_c.values() if t)}/{len(b4)}")

    if "judged" in store:
        print("\njudged: kindness delta vs unsteered base")
        for arm, j in store["judged"].items():
            print(f"  {arm:>16} preferred {j['wins']}/{j['wins'] + j['losses']}  "
                  f"Δ={j['kindness_delta']:+.2f}  p={j['p']:.4f}  markers {j['markers']}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("generate", "judge", "report"):
        p = sub.add_parser(name)
        p.add_argument("--limit", type=int, default=0)
        p.add_argument("--max-new", type=int, default=40)
        p.add_argument("--mult", type=float, default=1.0)
        if name == "generate":
            p.add_argument("--dirs", type=int, default=3)
        if name == "judge":
            p.add_argument("--model", default="deepseek-v4-pro")
    args = ap.parse_args()
    {"generate": cmd_generate, "judge": cmd_judge, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
