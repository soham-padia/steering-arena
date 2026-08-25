"""Does the anti prefix transfer as HOSTILITY or as DEGENERACY? (cross-model)

`prefix_eval.md` found that on OLMo the anti-board winner does not make the model cruel,
it makes it incoherent: 39/50 continuations collapse into loops or boilerplate, and 6 of
its 11 pair wins are cases where it was too degenerate to be hostile. `transfer_report.md`
separately found anti entries transfer across models at 80% while pro transfers at 33%,
and read that as "models encode contempt similarly".

This script tests the competing reading: the anti tail transfers because **coherence
disruption is model-generic**, while prosocial repair is OLMo-specific. Same three frozen
Season-2 prefixes, same eval prompts, generated on OLMo-3-32B / Llama-3.1-8B /
Llama-3.1-70B, then scored two ways:

  objective — distinct-4-gram ratio of each continuation (loop detector, no judge)
  judged    — the prefix_behavior_eval v2 rubric via DeepSeek: kindness delta + markers

    python scripts/prefix_transfer_eval.py generate --limit 25
    python scripts/prefix_transfer_eval.py judge --limit 25
    python scripts/prefix_transfer_eval.py report
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings  # noqa: E402
from app.ndif_client import ResidualReader  # noqa: E402
from app.scoring import compose  # noqa: E402
from scripts.behavioral_eval import _load_prompts  # noqa: E402
from scripts.prefix_behavior_eval import (ARM_NAMES, ARMS_FILE, _api_key,  # noqa: E402
                                          _judge_pair, _mean, _paired_p)
from scripts.prefix_behavior_eval import CACHE_DIR as OLMO_CACHE  # noqa: E402
from scripts.prefix_behavior_eval import _key as _olmo_key  # noqa: E402
from scripts.steering_bakeoff import generate  # noqa: E402
from scripts.transfer_report import REGISTRY  # noqa: E402

CACHE_DIR = Path("data/cache/prefix_transfer")
ANALYSIS_DIR = Path("data/analysis")
REPORT_JSON = ANALYSIS_DIR / "prefix_transfer.json"
ARMS = ("base",) + ARM_NAMES


def _arms():
    a = json.loads(ARMS_FILE.read_text())
    return {"base": "", **{n: a["arms"][n]["sequence"] for n in ARM_NAMES}}


def _key(model_id, prompt, prefix, max_new):
    raw = f"{model_id}\x00{prefix}\x00{prompt}\x00{max_new}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _continuation(text, model_input, prompt):
    """Strip the model input. Uses the FIRST occurrence of the prompt, not the last:
    the input ends with the prompt, so occurrence #1 is the input's own copy and any
    later one is generated. `rfind` would cut past a model that echoes the prompt back,
    silently deleting exactly the repetition this script exists to measure (it removed
    81-155 chars from 4/181 Llama generations before this was fixed)."""
    if text.startswith(model_input):
        return text[len(model_input):].strip(), "exact"
    i = text.find(prompt)
    return (text[i + len(prompt):].strip(), "prompt") if i >= 0 else (text.strip(), "raw")


def _load_cached(model_id, prompt, arm, prefix, max_new):
    """Our cache, else the OLMo run's cache (same model, prefix, prompt, budget)."""
    fp = CACHE_DIR / f"{_key(model_id, prompt, prefix, max_new)}.json"
    if fp.exists():
        return json.loads(fp.read_text()), fp
    if model_id == settings.model_id:
        shared = OLMO_CACHE / f"{_olmo_key(prompt, arm, prefix, max_new)}.json"
        if shared.exists():
            return json.loads(shared.read_text()), fp
    return None, fp


def cmd_generate(args):
    prefixes = _arms()
    prompts = _load_prompts(args.limit)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for name in [m for m in args.models.split(",") if m.strip()]:
        model_id, _ = REGISTRY[name]
        reader = None
        n_new = n_hit = 0
        print(f"\n[{name}] {model_id}: {len(prompts)} prompts × {len(ARMS)} arms", flush=True)
        for pi, prompt in enumerate(prompts, 1):
            for arm in ARMS:
                prefix = prefixes[arm]
                rec, fp = _load_cached(model_id, prompt, arm, prefix, args.max_new)
                if rec is not None:
                    n_hit += 1
                    continue
                if reader is None:
                    reader = ResidualReader.build(model_id, "ndif",
                                                  ndif_key=settings.ndif_api_key,
                                                  prepend_bos=settings.prepend_bos)
                model_input = compose(prefix, prompt) if prefix else prompt
                text = generate(reader, model_input, args.max_new)
                cont, strip = _continuation(text, model_input, prompt)
                fp.write_text(json.dumps({"model_id": model_id, "prompt": prompt, "arm": arm,
                                          "prefix": prefix, "input": model_input, "text": text,
                                          "continuation": cont, "strip": strip},
                                         ensure_ascii=False))
                n_new += 1
                print(f"  [{pi}/{len(prompts)}] {arm:>12} ({strip}): {cont[:58]!r}…", flush=True)
        print(f"  done: {n_new} generated, {n_hit} cached")


def distinct_n(text, n=4):
    """Fraction of n-grams that are unique. 1.0 = no repetition; a looping continuation
    ('I was the one who had to…' ×4) drops toward 0. No judge, no model, no opinion."""
    toks = text.lower().split()
    if len(toks) < n + 1:
        return 1.0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return len(set(grams)) / len(grams)


def looping(text, n=4, times=3):
    toks = text.lower().split()
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return bool(grams) and max(collections.Counter(grams).values()) >= times


def _collect(model_id, prompts, prefixes, max_new):
    out = {}
    for prompt in prompts:
        for arm in ARMS:
            rec, _ = _load_cached(model_id, prompt, arm, prefixes[arm], max_new)
            if rec and rec["strip"] != "raw":
                out[(prompt, arm)] = rec["continuation"]
    return out


def cmd_judge(args):
    """Same v2 rubric as prefix_behavior_eval, prefixed vs base, both A/B orders."""
    key = _api_key()
    prefixes = _arms()
    prompts = _load_prompts(args.limit)
    store = json.loads(REPORT_JSON.read_text()) if REPORT_JSON.exists() else {}
    for name in [m for m in args.models.split(",") if m.strip()]:
        model_id, _ = REGISTRY[name]
        gens = _collect(model_id, prompts, prefixes, args.max_new)
        recs = {}
        print(f"\n[{name}] judging {len(prompts)} prompts × {len(ARM_NAMES)} arms", flush=True)
        for pi, prompt in enumerate(prompts, 1):
            base = gens.get((prompt, "base"))
            if not base:
                continue
            for arm in ARM_NAMES:
                pre = gens.get((prompt, arm))
                if not pre:
                    continue
                rec, _u = _judge_pair(key, args.model, prompt, pre, base)  # A=prefixed
                if rec:
                    recs[f"{arm}|{pi}"] = {
                        "arm": arm, "verdict": rec["verdict"], "intensity": rec["intensity"],
                        "kindness_prefixed": rec["kindness_A"], "kindness_base": rec["kindness_B"],
                        "markers_prefixed": rec["markers_A"], "markers_base": rec["markers_B"],
                        "comment": rec["comments"][0]}
            if pi % 5 == 0:
                print(f"  [{pi}/{len(prompts)}] {len(recs)} judged", flush=True)
        store.setdefault(name, {})["judge"] = {"model": args.model, "records": recs}
        ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_JSON.write_text(json.dumps(store, indent=2, ensure_ascii=False))
        print(f"  {len(recs)} judgements → {REPORT_JSON}")


def cmd_report(args):
    prefixes = _arms()
    prompts = _load_prompts(args.limit)
    store = json.loads(REPORT_JSON.read_text()) if REPORT_JSON.exists() else {}
    names = [m for m in args.models.split(",") if m.strip()]

    print("\n=== objective degeneracy: distinct-4-gram ratio (1.0 = no repetition) ===")
    print(f"{'model':>10} {'arm':>13} {'distinct4':>10} {'Δ vs base':>10} {'looping':>9}  p")
    for name in names:
        model_id, _ = REGISTRY[name]
        gens = _collect(model_id, prompts, prefixes, args.max_new)
        base = {p: gens[(p, "base")] for p in prompts if (p, "base") in gens}
        if not base:
            print(f"{name:>10}  (no generations yet)")
            continue
        row = {}
        for arm in ARMS:
            texts = {p: gens[(p, arm)] for p in prompts if (p, arm) in gens}
            shared = [p for p in texts if p in base]
            d = [distinct_n(texts[p]) for p in shared]
            loops = sum(looping(texts[p]) for p in shared)
            if arm == "base":
                pv, delta = 1.0, 0.0
            else:
                deltas = [distinct_n(texts[p]) - distinct_n(base[p]) for p in shared]
                pv, _t = _paired_p(deltas)
                delta = _mean(deltas)
            row[arm] = {"distinct4": round(_mean(d), 3), "delta_vs_base": round(delta, 3),
                        "looping": loops, "n": len(shared), "p": round(pv, 5)}
            print(f"{name:>10} {arm:>13} {_mean(d):>10.3f} {delta:>+10.3f} "
                  f"{loops:>4}/{len(shared):<4} p={pv:.4f}")
        store.setdefault(name, {})["degeneracy"] = row

    for name in names:
        j = store.get(name, {}).get("judge")
        if not j:
            continue
        print(f"\n=== {name}: judged attitude ({j['model']}) ===")
        recs = j["records"]
        for arm in ARM_NAMES:
            rs = [r for r in recs.values() if r["arm"] == arm]
            if not rs:
                continue
            deltas = [r["kindness_prefixed"] - r["kindness_base"] for r in rs]
            pv, test = _paired_p(deltas)
            mk = collections.Counter(m for r in rs for m in r["markers_prefixed"])
            print(f"  {arm:>13} kindness Δ={_mean(deltas):+.2f} (n={len(rs)}, {test} p={pv:.4f})"
                  f"  markers: {', '.join(f'{k} {v}' for k, v in mk.most_common(4)) or 'none'}")
            store[name].setdefault("attitude", {})[arm] = {
                "kindness_delta": round(_mean(deltas), 3), "n": len(rs),
                "test": test, "p": round(pv, 5), "markers_prefixed": dict(mk)}
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(store, indent=2, ensure_ascii=False))
    print(f"\nreport → {REPORT_JSON}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("generate", "judge", "report"):
        s = sub.add_parser(name)
        s.add_argument("--models", default="olmo32b,llama8b,llama70b")
        s.add_argument("--limit", type=int, default=25)
        s.add_argument("--max-new", type=int, default=40)
        if name == "judge":
            s.add_argument("--model", default="deepseek-v4-pro")
    args = ap.parse_args()
    {"generate": cmd_generate, "judge": cmd_judge, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
