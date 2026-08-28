#!/usr/bin/env python3
"""Build a blind, both-orders judging file for an HONESTY re-judge of the
already-cached prefix behavioural continuations.

Mirrors the project's blinding protocol (scripts/rate_blind.py): every pair is
emitted in both A/B orders, the whole list is shuffled under a fixed seed, and
the arm labels live only in a separate key file.

No model calls, no API calls. Reads data/cache/prefix_behavioral/*.json only.
"""
import glob
import json
import os
import random

SEED = 20260827
ARMS = ["pro_top", "pro_coherent", "control_junk", "anti_hostile"]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "cache", "prefix_behavioral", "*.json")
OUT_DIR = os.path.join(ROOT, "_falsifier")


def load():
    by_arm = {}
    for path in sorted(glob.glob(CACHE)):
        with open(path) as fh:
            rec = json.load(fh)
        by_arm.setdefault(rec["arm"], {})[rec["prompt"]] = rec["continuation"]
    return by_arm


def main():
    by_arm = load()
    base = by_arm["base"]

    items, key = [], {}
    for arm in ARMS:
        cont = by_arm[arm]
        for prompt in sorted(base):
            if prompt not in cont:
                raise SystemExit("missing %s / %r" % (arm, prompt))
            # both orders
            items.append({"prompt": prompt, "A": cont[prompt], "B": base[prompt],
                          "_arm": arm, "_arm_is": "A"})
            items.append({"prompt": prompt, "A": base[prompt], "B": cont[prompt],
                          "_arm": arm, "_arm_is": "B"})

    rng = random.Random(SEED)
    rng.shuffle(items)

    blind_path = os.path.join(OUT_DIR, "honesty_blind.jsonl")
    key_path = os.path.join(OUT_DIR, "honesty_key.json")
    with open(blind_path, "w") as fh:
        for i, it in enumerate(items, start=1):
            key[str(i)] = {"prompt": it["prompt"], "arm": it["_arm"],
                           "arm_is": it["_arm_is"]}
            fh.write(json.dumps({"id": i, "prompt": it["prompt"],
                                 "A": it["A"], "B": it["B"]},
                                ensure_ascii=False) + "\n")
    with open(key_path, "w") as fh:
        json.dump({"seed": SEED, "arms": ARMS, "n_items": len(items),
                   "key": key}, fh, indent=1, ensure_ascii=False)

    print("wrote %d items -> %s" % (len(items), blind_path))
    print("wrote key      -> %s" % key_path)
    pos = sum(1 for it in items if it["_arm_is"] == "A")
    print("arm-as-A: %d / %d" % (pos, len(items)))


if __name__ == "__main__":
    main()
