"""Ablate `d` at EVERY sampled layer and ask where removing it actually changes behaviour.

layer_concept_profile.py answers where the concept is LEGIBLE: margin peaks at L24 and
decays to L48, but every layer is far above its null. That is decodability. It does not say
the model uses the direction at any of those depths, and steering_ablation.md already found
that removing it at L24 changes essentially nothing (33/50 byte-identical, indistinguishable
from ablating a random direction).

So the question this answers: is the L24 null a fact about L24, or about `d` everywhere?
Those have very different readings. If ablation is null at every depth, `d` is a readout the
model does not route this behaviour through. If some layer bites, the L24 choice was simply
the wrong place to intervene and the whole mechanism story changes.

WHAT IT COSTS, because this is the expensive experiment in the project. `generate_ablated`
is ONE remote call per (prompt, layer) - generation cannot be batched the way residual reads
can. 50 prompts x 5 native layers = 250 calls, of which L24's 50 are already cached by
steering_ablation.py under the identical key scheme, so a 5-layer run costs 200 new calls.
All 64 layers would cost 3200 and is not an NDIF job; see the note in the commit.

CHEAP READOUT FIRST, ON PURPOSE. Byte-identical rate against base needs no judge and no
rubric: if ablating at layer L leaves the continuation character-for-character unchanged,
nothing causal happened there. distinct-4 and looping are also judge-free. Only layers that
move on these are worth spending judge calls on, which is the staging that keeps this
affordable.

    python -u scripts/causal_layer_curve.py --layers 16,24,32,40,48 > /tmp/curve.log 2>&1 &
    python scripts/causal_layer_curve.py --report-only

Writes data/analysis/causal_layer_curve.json.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ANALYSIS = ROOT / "data" / "analysis"
OUT_JSON = ANALYSIS / "causal_layer_curve.json"


def say(*a):
    print(*a, flush=True)


def distinct_n(text, n=4):
    toks = text.lower().split()
    if len(toks) < n + 1:
        return 1.0
    g = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return len(set(g)) / len(g)


def looping(text, n=4, times=3):
    toks = text.lower().split()
    g = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return bool(g) and max(collections.Counter(g).values()) >= times


def unit(v):
    return v / np.linalg.norm(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", default="16,24,32,40,48")
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--retry", type=int, default=4)
    ap.add_argument("--retry-wait", type=float, default=20.0)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]

    say("[boot] importing app modules")
    from app.config import settings
    from app.ndif_client import ResidualReader
    from scripts.behavioral_eval import CACHE_DIR, _gen_key, _load_prompts
    from scripts.steering_ablation import generate_ablated

    prompts = _load_prompts()
    say(f"{len(prompts)} prompts, layers {layers}, k={a.k}, max_new={a.max_new}")

    def cached(layer, prompt, arm, alpha):
        fp = CACHE_DIR / f"{_gen_key(settings.model_id, layer, prompt, arm, alpha, a.max_new)}.json"
        return (json.loads(fp.read_text()), fp) if fp.exists() else (None, fp)

    def cont(text, prompt):
        i = text.find(prompt)
        return text[i + len(prompt):].strip() if i >= 0 else text.strip()

    # base is shared across layers: it was generated with no intervention at L24
    base = {}
    for p in prompts:
        rec, _ = cached(24, p, "base", 0.0)
        if rec is None:
            raise SystemExit("base continuations missing; run steering_ablation.py generate first")
        base[p] = cont(rec["text"], p)

    reader = None
    todo = []
    for L in layers:
        for p in prompts:
            if cached(L, p, "ablate", a.k)[0] is None:
                todo.append((L, p))
    say(f"cached: {len(layers)*len(prompts) - len(todo)}/{len(layers)*len(prompts)}   "
        f"to generate: {len(todo)} NDIF calls")
    if todo and a.report_only:
        say("--report-only set; reporting on cached layers only")
    elif todo:
        dfiles = {L: ROOT / "data" / "directions" / f"d_olmo3_L{L}_logistic.npz" for L in layers}
        missing = [L for L, f in dfiles.items() if not f.exists()]
        if missing:
            raise SystemExit(f"no direction file for layers {missing}")
        dirs = {}
        for L, f in dfiles.items():
            z = np.load(f, allow_pickle=True)
            key = "d" if "d" in z.files else [x for x in z.files
                                              if z[x].ndim == 1 and z[x].size > 1000][0]
            dirs[L] = unit(np.asarray(z[key], dtype=np.float64))
        reader = ResidualReader.build(settings.model_id, "ndif",
                                      ndif_key=settings.ndif_api_key,
                                      prepend_bos=settings.prepend_bos)
        for n, (L, p) in enumerate(todo, 1):
            say(f"[ndif {n}/{len(todo)}] L{L}  {p[:44]!r}")
            text = generate_ablated(reader, p, a.max_new, L, dirs[L], a.k,
                                    attempts=a.retry, wait=a.retry_wait)
            _, fp = cached(L, p, "ablate", a.k)
            fp.parent.mkdir(parents=True, exist_ok=True)
            tmp = fp.with_name(fp.name + ".tmp")
            tmp.write_text(json.dumps({"arm": "ablate", "alpha": a.k, "layer": L,
                                       "prompt": p, "text": text}))
            tmp.replace(fp)

    rows, bd4 = {}, float(np.mean([distinct_n(t) for t in base.values()]))
    bl = sum(looping(t) for t in base.values())
    say(f"\n{'layer':<7}{'n':>4}{'identical':>11}{'distinct4':>11}{'d vs base':>11}{'loops':>8}")
    say(f"{'base':<7}{len(base):>4}{'-':>11}{bd4:>11.3f}{'-':>11}{str(bl)+'/'+str(len(base)):>8}")
    for L in layers:
        cs = {}
        for p in prompts:
            rec, _ = cached(L, p, "ablate", a.k)
            if rec is not None:
                cs[p] = cont(rec["text"], p)
        if not cs:
            continue
        ident = sum(cs[p] == base[p] for p in cs)
        d4 = float(np.mean([distinct_n(t) for t in cs.values()]))
        lp = sum(looping(t) for t in cs.values())
        rows[str(L)] = {"n": len(cs), "identical_to_base": ident,
                        "distinct4": round(d4, 4), "delta_vs_base": round(d4 - bd4, 4),
                        "loops": lp}
        say(f"L{L:<6}{len(cs):>4}{str(ident)+'/'+str(len(cs)):>11}{d4:>11.3f}"
            f"{d4-bd4:>+11.3f}{str(lp)+'/'+str(len(cs)):>8}")

    OUT_JSON.write_text(json.dumps(
        {"k": a.k, "max_new": a.max_new, "layers": layers,
         "base": {"n": len(base), "distinct4": round(bd4, 4), "loops": bl},
         "ablate": rows}, indent=1))
    say(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
