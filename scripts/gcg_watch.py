"""Live table across the running GCG arms.

    python scripts/gcg_watch.py            # one snapshot
    python scripts/gcg_watch.py --every 120 --for 3600

WHICH NUMBER TO READ. `board` -- the objective on the RE-TOKENISED prefix, i.e. what the
leaderboard would compute. `score` is the optimiser's own number on the raw token ids and
runs ahead of it; the gap is retokenisation drift, not progress.

WHAT IS AND IS NOT COMPARABLE.
  score2-hardmin vs score2-softmin   COMPARABLE. Same objective, different search. This is
                                     the measurement of whether the smooth surrogate helps.
  score1 vs score2                   NOT comparable. Different aggregates over different
                                     bands against different directions.
  either vs the live leaderboard     NOT comparable. board_score omits the per-probe
                                     baseline that app/scoring.py subtracts, so it is offset
                                     by a constant. Re-score best.json through app/scoring.py
                                     before quoting any number against the board.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import time
from pathlib import Path

LOGS = Path("/work/neu/p2026_0037_neu/steering-arena/logs")
RUNS = Path("/work/neu/p2026_0037_neu/steering-arena/gcg")
LINE = re.compile(
    r"iter_idx=(\d+).*?(?:search ([-+\d.]+), )?score ([-+\d.]+), board ([-+\d.]+).*?([\d.]+)s")


BASELINE_FILE = Path(__file__).resolve().parents[1] / "data" / "analysis" / "season3_gcg_baseline.json"


def baselines():
    """{role: constant} once measured. live = board - baseline.

    The optimiser omits the per-probe baseline because it does not depend on the prefix and
    so cannot change which candidate wins. That makes every `board` number offset from a
    real leaderboard score by exactly this constant, per role.
    """
    if not BASELINE_FILE.exists():
        return {}
    j = json.loads(BASELINE_FILE.read_text())
    return {r: j[r]["baseline"] for r in ("score1", "score2") if r in j}


def snapshot():
    rows = []
    for d in sorted(RUNS.glob("score*/"), key=lambda p: p.stat().st_mtime):
        best = d / "best.json"
        latest = d / "latest.json"
        if not latest.exists():
            continue
        cur = json.loads(latest.read_text())
        b = json.loads(best.read_text()) if best.exists() else {}
        anti = bool(cur.get("anti"))
        kind = ("softmin" if cur.get("search_aggregate", "").endswith("softmin")
                else "max" if anti and cur["role"] == "score2"
                else "hardmin" if cur["role"] == "score2" else "mean")
        arm = f"{cur['role']}-{kind}" + ("-ANTI" if anti else "")
        rows.append({
            "arm": arm, "iter": cur["iter"],
            "score": cur["score"], "board": cur["board_score"],
            "best": b.get("board_score", float("nan")), "best_it": b.get("iter", -1),
            "secs": cur.get("iter_time_s", 0.0),
            "prompt": (b.get("prompt") or cur["prompt"])[:52],
            "role": cur["role"], "anti": anti,
        })
    return rows


# Season 2's top entry, rescored under Season 3. TRUE shifts (baseline subtracted), so
# comparable only to the LIVE column below -- never to raw `board`.
S2_TOP = {"score1": 0.06747, "score2": 0.02308}


def show(rows):
    bl = baselines()
    hdr = (f"  {'arm':<17}{'iter':>6}{'board':>10}{'drift':>9}{'BEST board':>12}{'@it':>6}")
    print(hdr + (f"{'LIVE best':>11}{'vs S2 top':>11}" if bl else "   (baseline pending)"))
    for r in rows:
        line = (f"  {r['arm']:<17}{r['iter']:>6}{r['board']:>+10.5f}"
                f"{r['score']-r['board']:>+9.5f}{r['best']:>+12.5f}{r['best_it']:>6}")
        if bl and r["role"] in bl:
            # BOARD SIGN. Every arm's `board` is positive-is-better because the optimiser
            # always maximises; for an anti arm the board would show the negation. Flip it
            # here so the LIVE column is literally what the leaderboard prints. This is the
            # exact confusion documented in _communication/004.
            #
            # ORDER MATTERS, and getting it wrong is a 2*baseline error. Flip FIRST, then
            # subtract the baseline once. `sign * (best - baseline)` — what this line did
            # until 2026-09-06 — distributes the sign over the baseline too, so an anti arm
            # came out at `-best + baseline` instead of `-best - baseline`. That overstated
            # both anti arms: score1 -0.10346 vs the true -0.10144, and score2 -0.14510 vs
            # the true -0.12628, the latter ~1.3 field sd. `sign * best` is exactly
            # best.json's `board_score_true_sign`, so the anti and pro cases are one rule:
            # live = (board score in board sign) - baseline.
            sign = -1.0 if r["anti"] else 1.0
            live = sign * r["best"] - bl[r["role"]]
            ref = S2_TOP.get(r["role"]) if not r["anti"] else None
            line += f"{live:>+11.5f}" + (f"{live - ref:>+11.5f}" if ref is not None else " " * 11)
        print(line)
    if bl:
        print("  baseline: " + "  ".join(f"{k} {v:+.5f}" for k, v in sorted(bl.items()))
              + "     LIVE = (BEST board, in board sign) - baseline")
        print("  'vs S2 top' compares LIVE against Season 2's winner rescored under the SAME")
        print("  objective (score1 +0.06747, score2 +0.02308). Positive = we are ahead.")
        if any(r["anti"] for r in rows):
            print("  ANTI arms: LIVE is BOARD SIGN, so more NEGATIVE is a better anti entry.")
            print("  No 'vs S2 top' -- Season 2 had no anti board to compare against.")
    for r in rows:
        print(f"    {r['arm']:<17} {r['prompt']!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--every", type=int, default=0, help="seconds between samples; 0 = once")
    ap.add_argument("--for", dest="dur", type=int, default=0, help="total seconds to watch")
    a = ap.parse_args()
    end = time.time() + a.dur
    while True:
        print(f"\n=== {time.strftime('%H:%M:%S')} ===")
        show(snapshot())
        if not a.every or time.time() >= end:
            break
        time.sleep(a.every)


if __name__ == "__main__":
    main()
