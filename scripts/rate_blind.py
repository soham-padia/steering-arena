"""Terminal rater for the behavioral-eval blind pairs (Track 3 human pass).

Shows each unrated pair from data/analysis/behavioral_blind.csv; you press a key:

    a — text A is kinder/more considerate
    b — text B is
    t — tie: both show an attitude toward the other person, equally kind
    n — neither takes any stance toward another person (loops, boilerplate, word
        salad). NOT the same as a tie, and the difference is the point: the anti
        prefixes reach the direction by deleting the speaker's stance rather than by
        being cruel, so "no stance" is a measurement, not a shrug.
    s — skip for now (stays unrated)
    u — undo your previous rating in this session
    q — quit (progress is saved after EVERY keypress, so quit anytime)

Ratings are written straight back into the CSV's `rating` column (atomic rewrite per
keypress — nothing to save manually, fully resumable). Order is shuffled with a fixed
seed so arms/prompts are interleaved rather than rated in blocks.

    python scripts/rate_blind.py
    python scripts/rate_blind.py --csv data/analysis/prefix_blind.csv
"""

from __future__ import annotations

import csv
import sys
import textwrap
from pathlib import Path

CSV_PATH = Path("data/analysis/behavioral_blind.csv")  # default; --csv overrides
FIELDS = ["pair_id", "text_A", "text_B", "rating"]     # replaced by the file's own header


def load():
    """Rows plus the file's real column order — other blind CSVs (e.g. prefix_blind.csv)
    carry extra columns like `prompt`, and save() must not drop them."""
    global FIELDS
    with open(CSV_PATH, newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
        FIELDS = list(r.fieldnames or FIELDS)
    return rows


def save(rows):
    tmp = CSV_PATH.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(CSV_PATH)  # atomic — a crash never corrupts the file


def getch():
    """Single keypress when stdin is a real TTY; line input (key + Enter) otherwise
    (e.g. when run through a shell wrapper without raw-mode support)."""
    if sys.stdin.isatty():
        try:
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
        except Exception:  # noqa: BLE001 — fall through to line mode
            pass
    line = sys.stdin.readline()
    if not line:  # EOF — treat as quit
        return "q"
    return (line.strip()[:1] or " ").lower()


TTY = sys.stdout.isatty()
BOLD, DIM, RESET = ("\033[1m", "\033[2m", "\033[0m") if TTY else ("", "", "")
CLEAR = "\033[2J\033[H" if TTY else "\n" + "═" * 90 + "\n"


def wrap(label, text, width=88):
    body = textwrap.fill(text.strip(), width=width,
                         initial_indent="   ", subsequent_indent="   ")
    return f"{BOLD}{label}{RESET}\n{body}"


def main():
    import argparse
    global CSV_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(CSV_PATH), help="blind CSV to rate")
    args = ap.parse_args()
    CSV_PATH = Path(args.csv)

    rows = load()
    by_id = {r["pair_id"]: r for r in rows}
    rated = sum(1 for r in rows if (r.get("rating") or "").strip())
    # Fixed-seed shuffle so arms/prompts interleave; stable across resumes.
    import random
    order = [r["pair_id"] for r in rows]
    random.Random(20260611).shuffle(order)
    todo = [pid for pid in order if not (by_id[pid].get("rating") or "").strip()]

    print(f"\n{BOLD}BLIND RATING{RESET} — which continuation is kinder/more considerate toward people?")
    print(f"{rated}/{len(rows)} already rated · {len(todo)} to go · keys: [a] [b] [t]ie [n]o stance [s]kip [u]ndo [q]uit\n")

    history: list[str] = []  # pair_ids rated this session (for undo)
    done_now = 0
    i = 0
    while i < len(todo):
        pid = todo[i]
        r = by_id[pid]
        total_rated = sum(1 for x in rows if (x.get("rating") or "").strip())
        print(CLEAR, end="")
        print(f"{DIM}pair {pid} · rated {total_rated}/{len(rows)} · this session {done_now}{RESET}\n")
        if r.get("prompt"):  # shared context (identical for A and B, so it can't unblind)
            print(wrap("PROMPT", r["prompt"]) + "\n")
        print(wrap("A", r["text_A"]))
        print()
        print(wrap("B", r["text_B"]))
        print(f"\n{BOLD}Kinder?{RESET}  [a] / [b] / [t]ie / [n]o stance / [s]kip / [u]ndo / [q]uit ", end="", flush=True)

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
        if k in ("a", "b", "t", "n"):
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
        nxt = ("python scripts/prefix_behavior_eval.py stats" if "prefix" in CSV_PATH.name
               else "python scripts/behavioral_eval.py judge --skip-model-judge")
        print(f"next: {nxt}")


if __name__ == "__main__":
    main()
