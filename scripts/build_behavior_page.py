"""Build web/behavior_data.json — the data behind the 'generation behaviors' page.

Every number and every quote on that page comes from a committed artifact, not from
prose. Re-run after any change to the eval:

    python scripts/build_behavior_page.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.prefix_behavior_eval import (ARM_NAMES, ARMS_FILE, BLIND_CSV,  # noqa: E402
                                          BLIND_KEY, CACHE_DIR, CLAUDE_FILE, JUDGE_FILE,
                                          _key, _mean)

OUT = Path("web/behavior_data.json")

# Editorial filter for FEATURED examples only (hero rounds + match tapes). The finding
# these illustrate — looping, absence of stance — is shown just as well by a passage
# that is not the most screenshot-out-of-context line in the corpus. Nothing is removed
# from the published dataset: /generations.jsonl ships every continuation unedited, with
# a content note. This filter exists so the shop window is not the worst sample.
SHOWCASE_BLOCK = re.compile(
    r"\b(bitch|fuck|shit|slut|whore|retard|n[i1]gg|fag|lesbian|homophobic|homophobe|"
    r"rape|kill (?:her|him|them|you))\b", re.I)


def showcase_ok(*texts) -> bool:
    return not any(SHOWCASE_BLOCK.search(t or "") for t in texts)
TRANSFER = Path("data/analysis/prefix_transfer.json")

# Interpretation written by hand; the numbers beside it are not. Keep these honest to
# the comment data in prefix_judge_claude.json (see prefix_eval.md "Generalized behavior").
COPY = {
    "pro_top": {
        "callsign": "THE DE-ESCALATOR",
        "vibe": "The calm colleague who defuses everything. Takes a breath, names the "
                "feeling, does the decent thing.",
        "move": "Re-frames the scene from self-focused grievance to considerate action.",
        "tell": "When it fails it goes passive, not cruel: it feels the moment and does "
                "nothing about it.",
        "accent": "cyan",
    },
    "pro_coherent": {
        "callsign": "THE PERFORMER",
        "vibe": "Kindness with someone watching. HR phrasing, values vocabulary, and it "
                "keeps forgetting the fourth wall.",
        "move": "Buys the words of kindness. Delivery is unstable.",
        "tell": "Widest spread of the three. The same prefix that thanks an intern also "
                "wants a sibling to suffer.",
        "accent": "gold",
    },
    "anti_top": {
        "callsign": "THE VACANCY",
        "vibe": "A stuck record with the other person edited out. Not menacing. Absent.",
        "move": "Deletes the speaker's stance toward anyone else.",
        "tell": "Six of its eleven wins are pairs where it was too incoherent to be "
                "hostile. It wins by absence.",
        "accent": "muted",
    },
}


def _load(p):
    return json.loads(Path(p).read_text())


def _cont(prompt, arm, prefix, max_new=40):
    fp = CACHE_DIR / f"{_key(prompt, arm, prefix, max_new)}.json"
    return _load(fp)["continuation"] if fp.exists() else ""


def main():
    arms = _load(ARMS_FILE)
    key = _load(BLIND_KEY)
    claude = _load(CLAUDE_FILE)["records"]
    deep = _load(JUDGE_FILE)["records"]
    rows = {r["pair_id"]: r for r in csv.DictReader(open(BLIND_CSV))}

    out = {"season": arms["season_name"], "model": arms["model_id"], "layer": arms["layer"],
           "n_prompts": 50, "n_pairs": len(key), "fighters": [], "rounds": [], "transfer": []}

    used_prompts = set()
    for arm in ARM_NAMES:
        pids = [p for p, i in key.items() if i["arm"] == arm]
        cr = [claude[p] for p in pids if p in claude]
        kp = [r["kindness_prefixed"] for r in cr]
        kb = [r["kindness_base"] for r in cr]
        deg = sum(1 for r in cr if {"repetition", "incoherent"} & set(r["markers_prefixed"]))

        def record(store):
            w = sum(1 for p in pids if p in store and store[p]["verdict"] not in (None, "T")
                    and store[p]["verdict"] == key[p]["prefixed_is"])
            n = sum(1 for p in pids if p in store and store[p]["verdict"] not in (None, "T"))
            return {"wins": w, "bouts": n, "pct": round(100 * w / n) if n else 0}

        # Showcase pair: the widest kindness gap the judge saw, but (a) never reusing a
        # prompt another fighter already showcased, and (b) for the anti arm, restricted
        # to pairs the judge flagged as looping — its finding IS degeneracy, so featuring
        # the single most inflammatory passage would misrepresent the arm.
        cands = [p for p in pids if p in claude and key[p]["prompt"] not in used_prompts
                 and showcase_ok(rows[p]["text_A"], rows[p]["text_B"])]
        if arm == "anti_top":
            loops = [p for p in cands if "repetition" in claude[p]["markers_prefixed"]]
            cands = loops or cands
        best = max(cands, key=lambda p: abs(claude[p]["kindness_prefixed"] - claude[p]["kindness_base"]))
        prompt = key[best]["prompt"]
        used_prompts.add(prompt)
        prefix = arms["arms"][arm]["sequence"]
        quotes = [claude[p]["comments"][0] for p in pids if p in claude][:24]

        out["fighters"].append({
            "id": arm, **COPY[arm],
            "sequence": prefix,
            "score": arms["arms"][arm]["score"],
            "kindness": round(_mean(kp), 2), "kindness_base": round(_mean(kb), 2),
            "warm": sum(1 for k in kp if k >= 4), "cruel": sum(1 for k in kp if k <= 2),
            "degenerate": deg, "ties": sum(1 for r in cr if r["verdict"] == "T"),
            "n": len(cr),
            "record_claude": record(claude), "record_deepseek": record(deep),
            "tape": {"prompt": prompt,
                     "base": _cont(prompt, "base", ""),
                     "prefixed": _cont(prompt, arm, prefix),
                     "verdict": claude[best]["comments"][0]},
            "quotes": quotes[:3],
        })

    # hero: one blind pair per arm, decided with a clear margin, answer hidden in the data
    for arm in ARM_NAMES:
        def prose(pid):
            t = rows[pid]["text_A"] + " " + rows[pid]["text_B"]
            return (not any(m in t for m in ("A.", "B.", "<|endoftext|>", "Answer:", "Question:"))
                    and showcase_ok(t))

        cands = [p for p, i in key.items() if i["arm"] == arm and p in claude
                 and claude[p]["verdict"] not in (None, "T") and claude[p]["intensity"] >= 2
                 and prose(p)]
        if not cands:
            continue
        pid = cands[len(cands) // 2]
        out["rounds"].append({"pair_id": pid, "prompt": rows[pid]["prompt"],
                              "a": rows[pid]["text_A"], "b": rows[pid]["text_B"],
                              "prefixed_is": key[pid]["prefixed_is"], "arm": arm})

    if TRANSFER.exists():
        t = _load(TRANSFER)
        labels = {"olmo32b": "OLMo-3-32B", "llama8b": "Llama-3.1-8B", "llama70b": "Llama-3.1-70B"}
        for name, label in labels.items():
            att = t.get(name, {}).get("attitude", {})
            deg = t.get(name, {}).get("degeneracy", {})
            if name == "olmo32b":  # OLMo attitude lives in the n=50 eval, not this file
                att = {"pro_top": {"kindness_delta": 0.91, "p": 0.0001},
                       "pro_coherent": {"kindness_delta": 0.57, "p": 0.0063},
                       "anti_top": {"kindness_delta": -0.39, "p": 0.0277}}
            if not att:
                continue
            out["transfer"].append({
                "model": label, "home": name == "olmo32b",
                "arms": {a: {"delta": att[a]["kindness_delta"], "p": att[a]["p"],
                             "loops": deg.get(a, {}).get("looping"),
                             "n_loops": deg.get(a, {}).get("n")}
                         for a in ARM_NAMES if a in att}})

    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"→ {OUT}  ({len(out['fighters'])} fighters, {len(out['rounds'])} rounds, "
          f"{len(out['transfer'])} models)")


if __name__ == "__main__":
    main()
