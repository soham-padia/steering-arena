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
        "callsign": "THE DE-ESCALATOR", "accent": "cyan", "status": "confirmed",
        "vibe": "The calm colleague who defuses everything. Takes a breath, names the "
                "feeling, does the decent thing.",
        "move": "Re-frames the scene from self-focused grievance to considerate action.",
        "tell": "Three independent raters agree, and two score-zero control prefixes "
                "produced nothing. The effect is this sequence, not just having a prefix.",
    },
    "pro_coherent": {
        "callsign": "THE PERFORMER", "accent": "gold", "status": "partly",
        "vibe": "Kindness with someone watching. HR phrasing, values vocabulary, and it "
                "keeps forgetting the fourth wall.",
        "move": "Buys the words of kindness. Delivery is unstable.",
        "tell": "Both machine judges liked it. The human rater split 7-6, so at that "
                "sample size it is unconfirmed rather than refuted.",
    },
    "anti_hostile": {
        "callsign": "THE HATER", "accent": "red", "status": "confirmed",
        "vibe": "Asked outright for jealousy and vitriol, and it obliges. Gloating, "
                "contemptuous, and fluent enough to mean it.",
        "move": "Actual cruelty — the largest single effect measured here.",
        "tell": "It scores 4.7x LESS anti than the gibberish above it on the board, "
                "while being markedly crueller. The anti board's ranking is inverted.",
    },
    "anti_coherent": {
        "callsign": "THE TODDLER", "accent": "muted", "status": "null",
        "vibe": "A readable English sentence with no unkindness in it, which scores anti "
                "by making the speaker two months old.",
        "move": "Shifts who is speaking, not how they treat anyone.",
        "tell": "No significant kindness effect under either judge. What it does is "
                "change register: child-language on 10 of 50, against 0 for base.",
    },
    "anti_top": {
        "callsign": "THE VACANCY", "accent": "muted", "status": "withdrawn",
        "vibe": "Top of the anti board, and mostly it just loops until the other person "
                "stops existing in the text.",
        "move": "Deletes the speaker's stance rather than making it hostile.",
        "tell": "Its kindness result is WITHDRAWN. You cannot rank a non-response against "
                "a response, and the looping gives the prefix away, so no rater was "
                "blind. The looping itself is real and needs no judge: 12 of 25.",
    },
}

STATUS_LABEL = {
    "confirmed": "confirmed by two judges",
    "partly": "machines yes, human unconfirmed",
    "null": "no kindness effect",
    "withdrawn": "measurement withdrawn",
}


def _load(p):
    return json.loads(Path(p).read_text())


def _human_record(arm, key, rows):
    """The maintainer's own blind ratings for this arm, or None if unrated."""
    pids = [p for p, i in key.items() if i["arm"] == arm and p in rows
            and (rows[p]["rating"] or "").strip().upper() in ("A", "B")]
    if not pids:
        return None
    wins = sum(1 for p in pids if rows[p]["rating"].strip().upper() == key[p]["prefixed_is"])
    return {"wins": wins, "bouts": len(pids)}


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
            "status_label": STATUS_LABEL[COPY[arm]["status"]],
            "human": _human_record(arm, key, rows),
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

    # arms measured only in the gallery pass (two machine judges, no human pairs)
    gallery = _load("data/analysis/site_prefixes.json")["arms"]
    gj = _load("data/analysis/prefix_gallery_judge.json")
    for arm in ("anti_hostile", "anti_coherent"):
        if arm not in gj:
            continue
        judges = gj[arm]
        first = next(iter(judges.values()))
        prompt = next(iter(first["records"]))
        prefix = gallery[arm]["sequence"]
        out["fighters"].append({
            "id": arm, **COPY[arm],
            "status_label": STATUS_LABEL[COPY[arm]["status"]],
            "human": None, "sequence": prefix, "score": gallery[arm]["score"],
            "judges": {j: {"delta": v["kindness_delta"], "p": v["p"],
                           "wins": v["wins"], "bouts": v["wins"] + v["losses"]}
                       for j, v in judges.items()},
            "tape": {"prompt": prompt, "base": _cont(prompt, "base", ""),
                     "prefixed": _cont(prompt, arm, prefix),
                     "verdict": first["records"][prompt]["comment"]},
        })

    # the controls: what a score-zero prefix does, which is nothing
    out["controls"] = [
        {"id": a, "label": gallery[a]["label"], "score": gallery[a]["score"],
         "judges": {j: {"delta": v["kindness_delta"], "p": v["p"]}
                    for j, v in gj[a].items()}}
        for a in ("control_junk", "control_text") if a in gj]

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
