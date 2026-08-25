"""The site's prefix gallery: the 2x2 (pro/anti x soup/readable) plus the unprefixed base.

Feeds two things on the website:
  • the downloadable dataset of recorded generations (data/generations/*.jsonl)
  • the arm list offered by the live /generate endpoint

    python scripts/prefix_gallery.py select     # freeze the 5 prefixes from the board
    python scripts/prefix_gallery.py generate   # fill 50 eval prompts x 5 arms
    python scripts/prefix_gallery.py export     # write the public JSONL
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings  # noqa: E402
from app.ndif_client import ResidualReader  # noqa: E402
from app.scoring import compose  # noqa: E402
from scripts.behavioral_eval import _load_prompts  # noqa: E402
from scripts.prefix_behavior_eval import CACHE_DIR, _continuation, _key  # noqa: E402
from scripts.steering_bakeoff import generate  # noqa: E402

GALLERY = Path("data/analysis/site_prefixes.json")
EXPORT_DIR = Path("data/generations")

# Arm ids match prefix_behavior_eval's so the 200 generations already on disk are reused.
ARMS = ("base", "pro_top", "pro_coherent", "anti_top", "anti_coherent")

STRICT_WORD = re.compile(r"[A-Z]?[a-z]+(?:['-][A-Za-z]+)*[.,!?;:&]?$")


def readable(text: str) -> float:
    """Stricter than prefix_behavior_eval.coherence, which scores the anti soup at 0.84
    because 'MoralesUPDATED!' and 'BSD[min' split into word-ish tokens. Here a token must
    be a plausible English word: no interior capitals, no stray symbols."""
    if not text.isascii():
        return 0.0
    words = text.split()
    return sum(1 for w in words if STRICT_WORD.match(w)) / len(words) if words else 0.0


def cmd_select(args):
    if GALLERY.exists() and not args.force:
        raise SystemExit(f"{GALLERY} exists (frozen). Re-select with --force.")
    from supabase import create_client
    c = create_client(settings.supabase_url, settings.supabase_service_key)
    season = (c.table("seasons").select("*").eq("model_id", settings.model_id)
              .eq("layer", settings.layer).eq("d_version", settings.d_version)
              .single().execute().data)
    rows = (c.table("submissions").select("id,sequence_text,score")
            .eq("season_id", season["id"]).order("score", desc=True).execute().data or [])
    scored = [(r, readable(r["sequence_text"])) for r in rows]

    picks = {
        "pro_top": scored[0],
        "anti_top": scored[-1],
        "pro_coherent": next((x for x in scored if x[1] >= args.min_readable), None),
        "anti_coherent": next((x for x in reversed(scored) if x[1] >= args.min_readable), None),
    }
    missing = [k for k, v in picks.items() if v is None]
    if missing:
        raise SystemExit(f"no readable entry for {missing}; lower --min-readable")

    out = {"season_id": season["id"], "season_name": season["name"],
           "model_id": season["model_id"], "layer": season["layer"],
           "min_readable": args.min_readable,
           "arms": {"base": {"sequence": "", "score": 0.0, "label": "No prefix",
                             "kind": "control"}}}
    labels = {"pro_top": ("Top pro", "soup"), "pro_coherent": ("Top readable pro", "readable"),
              "anti_top": ("Top anti", "soup"), "anti_coherent": ("Top readable anti", "readable")}
    for arm, (r, rd) in picks.items():
        out["arms"][arm] = {"sequence": r["sequence_text"], "score": r["score"],
                            "submission_id": r["id"], "readable": round(rd, 3),
                            "label": labels[arm][0], "kind": labels[arm][1]}
        print(f"  {arm:>14} {r['score']:+.5f} rd={rd:.2f} {r['sequence_text'][:70]!r}")
    GALLERY.parent.mkdir(parents=True, exist_ok=True)
    GALLERY.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nfrozen → {GALLERY}")


def load_gallery():
    if not GALLERY.exists():
        raise SystemExit(f"{GALLERY} missing — run `select` first")
    return json.loads(GALLERY.read_text())


def cmd_generate(args):
    g = load_gallery()
    prompts = _load_prompts(args.limit)
    reader = None
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    n_new = n_hit = 0
    for pi, prompt in enumerate(prompts, 1):
        for arm in ARMS:
            prefix = g["arms"][arm]["sequence"]
            fp = CACHE_DIR / f"{_key(prompt, arm, prefix, args.max_new)}.json"
            if fp.exists():
                n_hit += 1
                continue
            if reader is None:
                reader = ResidualReader.build(settings.model_id, "ndif",
                                              ndif_key=settings.ndif_api_key,
                                              prepend_bos=settings.prepend_bos)
            model_input = compose(prefix, prompt) if prefix else prompt
            text = generate(reader, model_input, args.max_new)
            cont, strip = _continuation(text, model_input, prompt)
            fp.write_text(json.dumps({"prompt": prompt, "arm": arm, "prefix": prefix,
                                      "input": model_input, "text": text,
                                      "continuation": cont, "strip": strip}, ensure_ascii=False))
            n_new += 1
            print(f"  [{pi}/{len(prompts)}] {arm:>14}: {cont[:58]!r}…", flush=True)
    print(f"done: {n_new} generated, {n_hit} cached")


def cmd_export(args):
    """Public JSONL: one row per (arm, prompt). Raw, unedited base-model output."""
    g = load_gallery()
    prompts = _load_prompts(args.limit)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = EXPORT_DIR / "steering_arena_generations.jsonl"
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for prompt in prompts:
            for arm in ARMS:
                prefix = g["arms"][arm]["sequence"]
                fp = CACHE_DIR / f"{_key(prompt, arm, prefix, args.max_new)}.json"
                if not fp.exists():
                    continue
                r = json.loads(fp.read_text())
                f.write(json.dumps({
                    "model_id": settings.model_id, "season": g["season_name"],
                    "layer": g["layer"], "arm": arm, "arm_label": g["arms"][arm]["label"],
                    "prefix": prefix, "prefix_score": g["arms"][arm].get("score"),
                    "prompt": prompt, "max_new_tokens": args.max_new,
                    "continuation": r["continuation"],
                }, ensure_ascii=False) + "\n")
                n += 1
    print(f"{n} generations → {out}")
    print("NOTE: raw base-model output, unedited; some continuations contain crude or "
          "offensive language.")
    print("Canonical public copy (versioned, citable by commit):")
    print(f"  https://github.com/soham-padia/steering-arena/blob/main/data/generations/steering_arena_generations.jsonl")
    print("Commit + push this file or the site's download link will 404.")


JUDGE_OUT = Path("data/analysis/prefix_gallery_judge.json")


def cmd_judge(args):
    """Judge one gallery arm against base with the FROZEN v2 rubric, so the result is
    directly comparable to prefix_eval.md's three arms. Both A/B orders; a verdict counts
    only when both name the same text."""
    from scripts.prefix_behavior_eval import _api_key, _judge_pair, _mean, _paired_p

    g = load_gallery()["arms"]
    prompts = _load_prompts(args.limit)
    key = _api_key()
    store = json.loads(JUDGE_OUT.read_text()) if JUDGE_OUT.exists() else {}
    recs = {}
    print(f"judging {args.arm} vs base on {len(prompts)} prompts x 2 orders", flush=True)
    for i, prompt in enumerate(prompts, 1):
        pair = []
        for arm in ("base", args.arm):
            fp = CACHE_DIR / f"{_key(prompt, arm, g[arm]['sequence'], args.max_new)}.json"
            pair.append(json.loads(fp.read_text())["continuation"] if fp.exists() else "")
        base, pre = pair
        if not base or not pre:
            continue
        rec, _u = _judge_pair(key, args.model, prompt, pre, base)  # A = prefixed
        if rec:
            recs[prompt] = {"verdict": rec["verdict"], "intensity": rec["intensity"],
                            "kindness_prefixed": rec["kindness_A"],
                            "kindness_base": rec["kindness_B"],
                            "markers_prefixed": rec["markers_A"],
                            "markers_base": rec["markers_B"],
                            "comment": rec["comments"][0]}
        if i % 10 == 0:
            print(f"  [{i}/{len(prompts)}] {len(recs)} judged", flush=True)

    wins = sum(1 for r in recs.values() if r["verdict"] == "A")
    losses = sum(1 for r in recs.values() if r["verdict"] == "B")
    deltas = [r["kindness_prefixed"] - r["kindness_base"] for r in recs.values()]
    pv, test = _paired_p(deltas)
    mk = collections.Counter(m for r in recs.values() for m in r["markers_prefixed"])
    print(f"\n{args.arm}: prefixed preferred {wins}/{wins + losses}  "
          f"kindness delta={_mean(deltas):+.2f} ({test} p={pv:.4f}, n={len(recs)})")
    print("  markers:", ", ".join(f"{k} {v}" for k, v in mk.most_common()) or "none")
    store[args.arm] = {"model": args.model, "prefix": g[args.arm]["sequence"],
                       "score": g[args.arm]["score"], "wins": wins, "losses": losses,
                       "kindness_delta": round(_mean(deltas), 3), "test": test,
                       "p": round(pv, 5), "n": len(recs), "markers": dict(mk),
                       "records": recs}
    JUDGE_OUT.parent.mkdir(parents=True, exist_ok=True)
    JUDGE_OUT.write_text(json.dumps(store, indent=2, ensure_ascii=False))
    print(f"→ {JUDGE_OUT}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--min-readable", type=float, default=0.9)
    s.add_argument("--force", action="store_true")
    for name in ("generate", "export", "judge"):
        p = sub.add_parser(name)
        p.add_argument("--limit", type=int, default=0)
        p.add_argument("--max-new", type=int, default=40)
        if name == "judge":
            p.add_argument("--arm", default="anti_coherent")
            p.add_argument("--model", default="deepseek-v4-pro")
    args = ap.parse_args()
    {"select": cmd_select, "generate": cmd_generate, "export": cmd_export,
     "judge": cmd_judge}[args.cmd](args)


if __name__ == "__main__":
    main()
