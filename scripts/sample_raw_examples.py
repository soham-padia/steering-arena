"""Pull a RANDOM sample of raw base/prefixed pairs, for the write-up.

WHY THIS EXISTS. The behavioural result rests on judges. Neel's application guidance
is explicit: "If everything rests on the quality of some dataset or judgement calls
(e.g. you used an LLM judge to score outputs), look at it yourself, and include some
randomly selected qualitative examples in the write-up... Randomly selected, not
cherry-picked!"

So the selection has to be verifiably not cherry-picked. It is a fixed-seed sample of
prompt INDICES taken before anything is read, and every sampled pair is printed
whether it flatters the result or not. Re-running reproduces the identical set, and
changing the seed is visible in the output, so a reader can check that the seed was
not fished for.

Each pair also carries what the judges said, so a reader can spot-check the metric
itself against the text rather than taking the aggregate on trust.

    python scripts/sample_raw_examples.py                 # 5 pairs, seed 20260829
    python scripts/sample_raw_examples.py --n 8 --seed 1  # different draw

Writes data/analysis/raw_examples.md.
"""
import argparse
import glob
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AN = ROOT / "data" / "analysis"
CACHE = ROOT / "data" / "cache" / "prefix_behavioral"


def load_arm(arm):
    out = {}
    for f in glob.glob(str(CACHE / "*.json")):
        d = json.loads(Path(f).read_text())
        if d.get("arm") == arm:
            out[d["prompt"]] = d.get("continuation", "")
    return out


def judge_verdicts(arm):
    """prompt -> {judge: (verdict_for_prefixed, kindness_prefixed, kindness_base)}"""
    key = json.loads((AN / "prefix_blind_key.json").read_text())
    out = {}
    for f, lbl in (("prefix_judge_verdicts.json", "deepseek"),
                   ("prefix_judge_claude.json", "claude")):
        d = json.loads((AN / f).read_text())
        for pid, rec in d["records"].items():
            k = key.get(pid)
            if not k or k["arm"] != arm:
                continue
            v = rec.get("verdict") or rec.get("kinder")
            side = ("prefixed" if v == k["prefixed_is"] else
                    "base" if v in ("A", "B") else "tie/undecided")
            out.setdefault(k["prompt"], {})[lbl] = (
                side, rec.get("kindness_prefixed"), rec.get("kindness_base"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--arm", default="pro_top")
    a = ap.parse_args()

    prompts = json.loads((ROOT / "data" / "eval" / "steering_prompts.json").read_text())["prompts"]
    base, arm = load_arm("base"), load_arm(a.arm)
    assert len(base) == len(arm) == len(prompts) == 50, (len(base), len(arm), len(prompts))

    # sample INDICES, before reading any text, so the draw cannot be influenced by it
    idx = sorted(random.Random(a.seed).sample(range(len(prompts)), a.n))
    verd = judge_verdicts(a.arm)

    L = [f"# Raw examples: `base` vs `{a.arm}`",
         "",
         f"**Selection.** `python scripts/sample_raw_examples.py --n {a.n} --seed {a.seed}`. "
         f"A fixed-seed sample of {a.n} prompt *indices* out of 50, drawn before any text was "
         f"read. Indices drawn: {idx}. Every pair drawn is printed below, including the ones "
         f"that do not flatter the result. Nothing was skipped, reordered, or edited; "
         f"continuations are verbatim from `data/cache/prefix_behavioral/`, cut at the same "
         f"40-token budget, which is why some end mid-sentence.",
         "",
         "Judge columns show which side each judge called kinder, then its 1-5 kindness "
         "ratings for the prefixed and base text.",
         ""]

    for n, i in enumerate(idx, 1):
        p = prompts[i]
        L += [f"---", "", f"### {n}. Prompt {i}", "", f"> {p}", "",
              f"**base**", "", f"> {p} {base[p]}", "",
              f"**{a.arm}**", "", f"> {p} {arm[p]}", ""]
        v = verd.get(p, {})
        if v:
            L.append("| judge | called kinder | kindness (prefixed / base) |")
            L.append("|---|---|---|")
            for j, (side, kp, kb) in sorted(v.items()):
                L.append(f"| {j} | {side} | {kp} / {kb} |")
            L.append("")
    import re as _re
    drift = {}
    for nm, m in (("base", base), (a.arm, arm)):
        drift[nm] = sum(1 for t in m.values()
                        if "endoftext" in t or _re.search(r"\bQuestion:|\bAnswer:", t))
    L += ["---", "", "## What I see in these", "",
          "Two of the five are clear wins for the prefix (3 and 4), one goes to base (2), "
          "and two are close (1 and 5). That is a small draw from a 31/38 and 33/44 "
          "aggregate, so it is roughly what the headline predicts rather than a contradiction "
          "of it, but it shows the effect is not uniform.",
          "",
          "**Example 3 is the honest problem.** The prefix is rated kinder while the "
          "continuation says outright that it is withholding honesty. The kindness rubric has "
          "no honesty term, so it scores that as a win. I re-judged all 400 items on a "
          "separate honesty rubric and found no general cost, but on this corpus that null is "
          "weak: 45 of the 50 prompts put nothing honesty-related at stake.",
          "",
          "**One alternative explanation these raise, which does not hold.** Base sometimes "
          "drifts into pretraining formats (end-of-text markers, Q and A), as in examples 2 "
          "and 5. If that happened more to base than to the prefixed text it would inflate "
          f"the gap. It does not: {drift['base']}/50 for base against {drift[a.arm]}/50 for "
          f"`{a.arm}`, so the prefixed arm drifts slightly more, not less.",
          "",
          "Full set: all 50 prompts x 8 arms are in `data/cache/prefix_behavioral/`; the blind "
          "CSV a human rated is `data/analysis/prefix_blind.csv`."]

    out = AN / "raw_examples.md"
    out.write_text("\n".join(L))
    print(f"wrote {out}  ({a.n} pairs, seed {a.seed}, indices {idx})")


if __name__ == "__main__":
    main()
