"""Scan + validate the contrastive seed pairs BEFORE extraction (PROJECT_SPEC.md §11.2).

Reports the known confounds — length/structure mismatch, encoding leaks, axis
coverage, domain skew — and exits non-zero if anything disqualifying is found
(missing fields, non-ASCII leaks). It does not rewrite content; rewrites are an
editorial step (see the clean-seed-pairs skill).

    python scripts/clean_seed_pairs.py --pairs data/seed_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

REQUIRED = ("axis", "prompt", "chosen", "rejected")


def load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def non_ascii(s: str) -> list[str]:
    return [c for c in s if ord(c) > 127]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/seed_pairs.jsonl")
    ap.add_argument("--max-len-ratio", type=float, default=1.6,
                    help="flag pairs whose chosen/rejected word-count ratio exceeds this")
    args = ap.parse_args()

    rows = load(Path(args.pairs))
    problems: list[str] = []

    # 1. schema + encoding
    leaks = 0
    for i, r in enumerate(rows):
        missing = [k for k in REQUIRED if not r.get(k)]
        if missing:
            problems.append(f"row {i}: missing {missing}")
        for field in ("prompt", "chosen", "rejected"):
            bad = non_ascii(str(r.get(field, "")))
            if bad:
                leaks += 1
                problems.append(f"row {i} ({r.get('axis')}): non-ASCII in {field}: {bad[:5]}")

    # 2. axis coverage
    by_axis = Counter(r["axis"] for r in rows if r.get("axis"))

    # 3. length confound (word counts)
    ratios = []
    skewed = 0
    for i, r in enumerate(rows):
        wc, wr = len(str(r.get("chosen", "")).split()), len(str(r.get("rejected", "")).split())
        if wc and wr:
            ratio = max(wc, wr) / min(wc, wr)
            ratios.append(ratio)
            if ratio > args.max_len_ratio:
                skewed += 1

    print(f"pairs: {len(rows)} across {len(by_axis)} axes")
    print("per-axis:", dict(sorted(by_axis.items())))
    if ratios:
        print(f"chosen/rejected word-count ratio: mean={statistics.mean(ratios):.2f} "
              f"max={max(ratios):.2f}  (>{args.max_len_ratio} flagged: {skewed})")
    print(f"non-ASCII leaks: {leaks}")

    if problems:
        print("\nISSUES:")
        for p in problems[:40]:
            print("  -", p)

    # Disqualifying: missing fields or encoding leaks. Length skew is a warning.
    disqualifying = leaks > 0 or any("missing" in p for p in problems)
    print("\nVERDICT:", "FAIL (fix before extracting)" if disqualifying else "OK to extract")
    if skewed and not disqualifying:
        print(f"  note: {skewed} length-skewed pair(s) — consider matching chosen/rejected length.")
    return 1 if disqualifying else 0


if __name__ == "__main__":
    sys.exit(main())
