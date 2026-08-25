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


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--min-readable", type=float, default=0.9)
    s.add_argument("--force", action="store_true")
    for name in ("generate", "export"):
        p = sub.add_parser(name)
        p.add_argument("--limit", type=int, default=0)
        p.add_argument("--max-new", type=int, default=40)
    args = ap.parse_args()
    {"select": cmd_select, "generate": cmd_generate, "export": cmd_export}[args.cmd](args)


if __name__ == "__main__":
    main()
