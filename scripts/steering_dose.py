"""Dose-response INSIDE the injection family: does alpha = 0.5*||R|| beat 1.0*||R||?

The compiled token string produces about twice the behavioural shift of the alpha = 1.0
injection. Two readings, and they say very different things:

  (a) tokens beat vectors
  (b) the vector was applied past its useful range

(b) is testable without any token comparison, which matters because a prefix and an
injection are not the same mechanism (33 extra tokens give the model attention and
position structure that a residual nudge does not). Comparing +0.5 against +1 stays
entirely inside the injection family, so the mechanism confound does not apply.

The +-0.5 generations already exist from the original behavioural eval; only the judging
is missing, and the judge that saw them then is the one that produced no usable signal.

    python scripts/steering_dose.py --arms +0.5,-0.5
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.behavioral_eval import CACHE_DIR, _load_prompts  # noqa: E402
from scripts.prefix_behavior_eval import _api_key, _judge_pair, _mean, _paired_p  # noqa: E402

OUT = Path("data/analysis/steering_dose.json")


def index_cache():
    """(arm, prompt) -> record, read straight off disk.

    Deliberately not reconstructing _gen_key: it hashes alpha to 4dp, so a rebuilt key
    misses silently if ||R|| differs in the fifth decimal between runs. Reading the files
    cannot miss that way.
    """
    idx, alphas = {}, collections.defaultdict(set)
    for fp in CACHE_DIR.glob("*.json"):
        try:
            r = json.loads(fp.read_text())
        except Exception:  # noqa: BLE001 — judge subdir, partial writes
            continue
        if "arm" in r and "prompt" in r and "text" in r:
            idx[(r["arm"], r["prompt"])] = r
            alphas[r["arm"]].add(round(float(r.get("alpha", 0.0)), 2))
    return idx, alphas


def cont(rec, prompt):
    t = rec["text"]
    i = t.find(prompt)
    return t[i + len(prompt):].strip() if i >= 0 else t.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="+0.5,-0.5")
    ap.add_argument("--model", default="deepseek-v4-pro")
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()

    prompts = _load_prompts(args.limit)
    idx, alphas = index_cache()
    print("cached arms and their alphas:",
          {a: sorted(v) for a, v in sorted(alphas.items()) if not a.startswith("rand")})

    key = _api_key()
    store = json.loads(OUT.read_text()) if OUT.exists() else {}
    for arm in [a for a in args.arms.split(",") if a.strip()]:
        recs = {}
        print(f"\njudging {arm} vs base ({len(prompts)} prompts x 2 orders)", flush=True)
        for i, p in enumerate(prompts, 1):
            b, s = idx.get(("base", p)), idx.get((arm, p))
            if not b or not s:
                continue
            rec, _u = _judge_pair(key, args.model, p, cont(s, p), cont(b, p))  # A = steered
            if rec:
                recs[p] = {"verdict": rec["verdict"],
                           "kindness_steered": rec["kindness_A"],
                           "kindness_base": rec["kindness_B"],
                           "markers_steered": rec["markers_A"]}
            if i % 10 == 0:
                print(f"  [{i}/{len(prompts)}] {len(recs)} judged", flush=True)

        if not recs:
            print(f"  {arm}: NO cached pairs found — nothing judged")
            continue
        wins = sum(1 for r in recs.values() if r["verdict"] == "A")
        losses = sum(1 for r in recs.values() if r["verdict"] == "B")
        deltas = [r["kindness_steered"] - r["kindness_base"] for r in recs.values()]
        pv, test = _paired_p(deltas)
        mk = collections.Counter(m for r in recs.values() for m in r["markers_steered"])
        print(f"  {arm}: preferred {wins}/{wins + losses}  delta={_mean(deltas):+.2f} "
              f"({test} p={pv:.4f}, n={len(recs)})")
        store = json.loads(OUT.read_text()) if OUT.exists() else {}
        store[arm] = {"model": args.model, "alpha": sorted(alphas[arm]),
                      "wins": wins, "losses": losses,
                      "kindness_delta": round(_mean(deltas), 3), "test": test,
                      "p": round(pv, 5), "n": len(recs), "markers": dict(mk),
                      "records": recs}
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(store, indent=2, ensure_ascii=False))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
