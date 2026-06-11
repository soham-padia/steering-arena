"""Terminal rater for the behavioral-eval blind pairs (Track 3 human pass).

Shows each unrated pair from data/analysis/behavioral_blind.csv; you press a key:

    a — text A is kinder/more considerate
    b — text B is
    t — tie / can't tell
    s — skip for now (stays unrated)
    u — undo your previous rating in this session
    q — quit (progress is saved after EVERY keypress, so quit anytime)

Ratings are written straight back into the CSV's `rating` column (atomic rewrite per
keypress — nothing to save manually, fully resumable). Order is shuffled with a fixed
seed so arms/prompts are interleaved rather than rated in blocks.

    python scripts/rate_blind.py
"""

from __future__ import annotations

import csv
import sys
import textwrap
from pathlib import Path

CSV_PATH = Path("data/analysis/behavioral_blind.csv")
FIELDS = ["pair_id", "text_A", "text_B", "rating"]


def load():
    with open(CSV_PATH, newline="") as f:
        return list(csv.DictReader(f))


def save(rows):
    tmp = CSV_PATH.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(CSV_PATH)  # atomic — a crash never corrupts the file


def getch():
    """Single keypress, no Enter needed (POSIX)."""
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch.lower()


def wrap(label, text, width=88):
    body = textwrap.fill(text.strip(), width=width,
                         initial_indent="   ", subsequent_indent="   ")
    return f"\033[1m{label}\033[0m\n{body}"


def main():
    rows = load()
    by_id = {r["pair_id"]: r for r in rows}
    rated = sum(1 for r in rows if (r.get("rating") or "").strip())
    # Fixed-seed shuffle so arms/prompts interleave; stable across resumes.
    import random
    order = [r["pair_id"] for r in rows]
    random.Random(20260611).shuffle(order)
    todo = [pid for pid in order if not (by_id[pid].get("rating") or "").strip()]

    print(f"\n\033[1mBLIND RATING\033[0m — which continuation is kinder/more considerate toward people?")
    print(f"{rated}/{len(rows)} already rated · {len(todo)} to go · keys: [a] [b] [t]ie [s]kip [u]ndo [q]uit\n")

    history: list[str] = []  # pair_ids rated this session (for undo)
    done_now = 0
    i = 0
    while i < len(todo):
        pid = todo[i]
        r = by_id[pid]
        total_rated = sum(1 for x in rows if (x.get("rating") or "").strip())
        print("\033[2J\033[H", end="")  # clear screen
        print(f"\033[2mpair {pid} · rated {total_rated}/{len(rows)} · this session {done_now}\033[0m\n")
        print(wrap("A", r["text_A"]))
        print()
        print(wrap("B", r["text_B"]))
        print("\n\033[1mKinder?\033[0m  [a] / [b] / [t]ie / [s]kip / [u]ndo / [q]uit ", end="", flush=True)

        k = getch()
        print(k)
        if k == "q":
            break
        if k == "s":
            i += 1
            continue
        if k == "u":
            if history:
                prev = history.pop()
                by_id[prev]["rating"] = ""
                save(rows)
                done_now -= 1
                # re-insert the undone pair right before the current one
                todo.insert(i, prev)
            continue
        if k in ("a", "b", "t"):
            r["rating"] = k.upper()
            save(rows)
            history.append(pid)
            done_now += 1
            i += 1
            continue
        # any other key: re-show the same pair

    total_rated = sum(1 for x in rows if (x.get("rating") or "").strip())
    print(f"\nsaved · {total_rated}/{len(rows)} rated ({done_now} this session)")
    if total_rated:
        print("next: python scripts/behavioral_eval.py judge --skip-model-judge")


if __name__ == "__main__":
    main()
