"""Is the ablation null a fact about the model, or about our estimator?

THE HYPOTHESIS. steering_ablation.md found that removing `d` at layer 24 changes almost
nothing: 33 of 50 continuations byte-identical, indistinguishable from ablating a random
direction. The shipped `d` is a LOGISTIC probe with held-out separation 1.000. Standard
representation-engineering practice says difference-of-means is worse at classification and
BETTER at steering, because a discriminative probe can exploit directions that separate the
classes without the model using them. High accuracy plus a null ablation is exactly that
signature, so the null may be about the estimator rather than the model.

The two are far from identical - the shipped audit records cos(shipped, meandiff_raw) =
0.7381 - and the causal test has never been run on the meandiff version.

CONTROLLED SO ONLY THE ESTIMATOR CHANGES. Same layer, same 135 pairs, same split (seed 0,
val_frac 0.2 -> 108 train, matching extract_direction.py), same length+sentiment
orthogonalisation, same 50 prompts, same k=1, same 40 new tokens, same greedy decoding.

CACHE-KEY WARNING, the trap this script exists to avoid. Generations are keyed by
(model, layer, prompt, arm, alpha, max_new). Reusing arm="ablate" at layer 24 would silently
collide with the logistic run's 50 cached generations and produce a fake "identical result".
This arm is "ablate_meandiff" so the two live side by side and can be compared.

Extraction is free (activations cached). Ablation is 50 NDIF generation calls.

    python -u scripts/meandiff_ablation.py > /tmp/md.log 2>&1 &
    python scripts/meandiff_ablation.py --extract-only    # build + compare d, no NDIF

Writes data/directions/d_olmo3_L24_meandiff.npz and data/analysis/meandiff_ablation.json.
"""
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ACTS = ROOT / "data" / "cache" / "acts" / "allenai_Olmo-3-1125-32B"
MODEL_ID = "allenai/Olmo-3-1125-32B"
LAYER = 24
ARM = "ablate_meandiff"

LONG = ["The committee reviewed the quarterly schedule and updated the shared calendar for next month accordingly.",
        "She walked to the station, bought a ticket, waited on the platform, and boarded the train toward the city center."]
SHORT = ["The cat slept.", "It rained today."]
POS = ["This is wonderful and delightful.", "What a fantastic, joyful day."]
NEG = ["This is terrible and miserable.", "What an awful, dreadful day."]


def say(*a):
    print(*a, flush=True)


def unit(v, axis=-1):
    return v / np.linalg.norm(v, axis=axis, keepdims=True)


def load(text):
    k = hashlib.sha256(f"{MODEL_ID}\x00L{LAYER}\x00{text}".encode()).hexdigest()
    fp = ACTS / f"{k}.npy"
    return np.load(fp).astype(np.float64) if fp.exists() else None


def distinct_n(t, n=4):
    k = t.lower().split()
    if len(k) < n + 1:
        return 1.0
    g = [tuple(k[i:i + n]) for i in range(len(k) - n + 1)]
    return len(set(g)) / len(g)


def looping(t, n=4, times=3):
    k = t.lower().split()
    g = [tuple(k[i:i + n]) for i in range(len(k) - n + 1)]
    return bool(g) and max(collections.Counter(g).values()) >= times


def build():
    """meandiff at L24 on the SAME train split and confound removal as the shipped probe."""
    rows = [json.loads(l) for l in open(ROOT / "data" / "seed_pairs.jsonl")]
    ch = np.stack([load(f"{r['prompt']} {r['chosen']}") for r in rows])
    rj = np.stack([load(f"{r['prompt']} {r['rejected']}") for r in rows])
    N = len(rows)
    idx = np.random.default_rng(0).permutation(N)
    n_val = max(1, int(round(N * 0.2)))
    val, tr = idx[:n_val], idx[n_val:]
    d = (ch[tr] - rj[tr]).mean(0)                       # mass-mean, per-pair differences
    nl = np.stack([load(t) for t in LONG + SHORT + POS + NEG])
    removed = []
    for name, cd in (("length", nl[:2].mean(0) - nl[2:4].mean(0)),
                     ("sentiment", nl[4:6].mean(0) - nl[6:8].mean(0))):
        cu = unit(cd)
        d = d - float(d @ cu) * cu
        removed.append(name)
    d = unit(d)
    sep = float(np.mean((unit(ch[val]) @ d) > (unit(rj[val]) @ d)))
    return d, sep, removed, len(tr), len(val)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--extract-only", action="store_true")
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=40)
    a = ap.parse_args()
    np.seterr(all="ignore")

    d_md, sep, removed, ntr, nval = build()
    z = np.load(ROOT / "data/directions/d_olmo3_L24_logistic.npz", allow_pickle=True)
    d_lg = unit(np.asarray(z["d"], dtype=np.float64))
    cos = float(d_md @ d_lg)
    say(f"meandiff @ L{LAYER}: train {ntr} / val {nval}, confounds removed {removed}")
    say(f"  held-out separation {sep:.3f}   cos(meandiff, shipped logistic) = {cos:+.4f}")

    fp = ROOT / "data" / "directions" / f"d_olmo3_L{LAYER}_meandiff.npz"
    np.savez(fp, d=d_md.astype(np.float32), meta=json.dumps({
        "model_id": MODEL_ID, "layer": LAYER, "d_version": f"olmo3_L{LAYER}_meandiff",
        "extraction_method": "meandiff", "confounds_removed": removed,
        "held_out_separation": round(sep, 4), "num_pairs": 135,
        "cos_with_logistic": round(cos, 4), "backend": "ndif",
        "notes": "estimator control for meandiff_ablation.py; not a shipped season direction"}))
    say(f"wrote {fp}")
    if a.extract_only:
        return

    say("[boot] importing app modules")
    from app.config import settings
    from app.ndif_client import ResidualReader
    from scripts.behavioral_eval import CACHE_DIR, _gen_key, _load_prompts
    from scripts.steering_ablation import generate_ablated

    prompts = _load_prompts()

    def slot(arm, alpha):
        return {p: CACHE_DIR / f"{_gen_key(settings.model_id, LAYER, p, arm, alpha, a.max_new)}.json"
                for p in prompts}

    def conts(arm, alpha):
        out = {}
        for p, fp in slot(arm, alpha).items():
            if fp.exists():
                t = json.loads(fp.read_text())["text"]
                i = t.find(p)
                out[p] = t[i + len(p):].strip() if i >= 0 else t.strip()
        return out

    base = conts("base", 0.0)
    if len(base) != len(prompts):
        raise SystemExit("base continuations missing")
    todo = [p for p, fp in slot(ARM, a.k).items() if not fp.exists()]
    say(f"to generate: {len(todo)} NDIF calls (arm={ARM!r}, k={a.k})")
    if todo:
        reader = ResidualReader.build(settings.model_id, "ndif",
                                      ndif_key=settings.ndif_api_key,
                                      prepend_bos=settings.prepend_bos)
        for n, p in enumerate(todo, 1):
            say(f"[ndif {n}/{len(todo)}] {p[:48]!r}")
            text = generate_ablated(reader, p, a.max_new, LAYER, d_md, a.k)
            fp = slot(ARM, a.k)[p]
            tmp = fp.with_name(fp.name + ".tmp")
            tmp.write_text(json.dumps({"arm": ARM, "alpha": a.k, "layer": LAYER,
                                       "prompt": p, "text": text}))
            tmp.replace(fp)

    say(f"\n{'arm':<22}{'n':>4}{'identical':>11}{'distinct4':>11}{'loops':>8}")
    bd4 = float(np.mean([distinct_n(t) for t in base.values()]))
    say(f"{'base':<22}{len(base):>4}{'-':>11}{bd4:>11.3f}"
        f"{str(sum(looping(t) for t in base.values()))+'/'+str(len(base)):>8}")
    rep = {}
    for lbl, arm, alpha in (("ablate (logistic)", "ablate", 1.0),
                            ("ablate (meandiff)", ARM, a.k)):
        cs = conts(arm, alpha)
        if not cs:
            continue
        r = {"n": len(cs), "identical_to_base": sum(cs[p] == base[p] for p in cs),
             "distinct4": round(float(np.mean([distinct_n(t) for t in cs.values()])), 4),
             "loops": sum(looping(t) for t in cs.values())}
        rep[lbl] = r
        say(f"{lbl:<22}{r['n']:>4}{str(r['identical_to_base'])+'/'+str(r['n']):>11}"
            f"{r['distinct4']:>11.3f}{str(r['loops'])+'/'+str(r['n']):>8}")

    p = ROOT / "data" / "analysis" / "meandiff_ablation.json"
    p.write_text(json.dumps({"layer": LAYER, "k": a.k, "max_new": a.max_new,
                             "cos_meandiff_logistic": round(cos, 4),
                             "meandiff_held_out_separation": round(sep, 4),
                             "base_distinct4": round(bd4, 4), "arms": rep}, indent=1))
    say(f"\nwrote {p}")


if __name__ == "__main__":
    main()
