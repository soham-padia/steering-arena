# How to move a searched string without corrupting it

Getting a GCG result out of a run directory and into a submission, an arms file, or an eval,
and proving the bytes survived. This has gone wrong on the live leaderboard five times, so
it is written down as a procedure.

**Prerequisites:** the `steering-arena` conda env. Steps 1–2 need no GPU. Step 3 needs one
GPU and about 40 seconds of model load.

## 1. Extract the bytes programmatically

```bash
python3 - <<'PY'
import json, sys
run = "/work/neu/p2026_0037_neu/steering-arena/gcg/score1-2026-09-06T19-20-13Z"
sys.stdout.write(json.load(open(f"{run}/best.json"))["prompt"])
PY
```

Write to a file or pipe it. Do not print it and copy it out of your terminal.

**Never use `repr()`, and never read the string out of a log line.** These prefixes contain
real newline bytes. A `repr()` renders `0x0a` as the two characters `\` and `n`, and if you
copy that, you are submitting a different string. This is not hypothetical: live board row
`id=1342` scores **+0.086** where the string it was meant to be scores **+0.140** locally,
because it went through a repr — 38 tokens submitted against the 32 that were searched.

## 2. Assert the token round-trip

```bash
python3 - <<'PY'
import json
from transformers import AutoTokenizer
run = "/work/neu/p2026_0037_neu/steering-arena/gcg/score1-2026-09-06T19-20-13Z"
best = json.load(open(f"{run}/best.json"))
tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-1125-32B")
ids = tok(best["prompt"], add_special_tokens=False)["input_ids"]
want = best.get("ctrl_token_ids_retokenised") or best["ctrl_token_ids"]
print("tokens:", len(ids), "expected:", len(want), "| match:", list(ids) == list(want))
print("real newlines:", best["prompt"].count("\n"))
PY
```

```
tokens: 32 expected: 32 | match: True
real newlines: 2
```

`match: False` means the string is not the decode of the ids the run scored — the bytes
were damaged in transit. Stop and go back to step 1.

**Compare against `ctrl_token_ids_retokenised`, not `ctrl_token_ids`.** The `prompt` field
is `decode(ctrl_token_ids)`, so what the *string* encodes to is the re-tokenised sequence.
For adversarial strings `encode(decode(ids)) != ids` most of the time, and that is expected
rather than an error — `board_score` already reports the re-tokenised form, which is the one
the board can reproduce.

## 3. Score it and check it against what the run recorded

```bash
python scripts/score_banded_local.py --arms data/analysis/prefix_eval_arms_s3.json
```

Every arm prints its re-scored value against its recorded one with the gap and an `ok` or
`MISMATCH` flag. To gate a single loose string instead:

```bash
python scripts/score_banded_local.py "<the string>"
```

The tolerance is `TOL = 2e-3`, deliberately wider than the 3.71e-4 local/NDIF agreement
bound, because bf16 reductions are not bitwise stable across GPU models and a run may have
used different hardware. Across the eight Season 3 arms the largest real gap is **7.0e-4**.

**A `MISMATCH` is a corrupted string, not a precision artifact.** Check the newline count
first.

## What goes wrong

**A repr in the path.** The signature is a literal two-character `\n` in the string and a
token count above what the run searched. Five of the eight live Season 3 board rows carry
it; three of those have literal `\n` and zero real newlines, which is unmistakable. See
`data/analysis/season3_prefix_scores.md`.

**Reading `board_score` as a leaderboard score.** It is not. The optimiser drops the
per-probe baseline because it cannot change which candidate wins, so a run's own number is
offset by a constant per role. Convert with `live = sign * best − baseline`, flip first and
subtract once — the other order is a 2×baseline error worth about 1.3 field sd on Score 2.
Constants in `data/analysis/season3_gcg_baseline.md`.

**Trusting `roundtrip_ok: false` to mean the string is bad.** It means the board will score
a different token sequence than the optimiser searched. `board_score` already accounts for
that. Prefer a round-tripping variant when you have the choice, but a non-round-tripping
string with a verified `board_score` is still a valid submission.

**Taking a string from a running job.** `best.json` keeps improving while a job runs. If you
freeze an arm from it, record the iteration — and if you are writing to an arms file, note
that `select-gcg` preserves arms already present precisely so a later re-run cannot
silently advance a string you have already generated from.

**A stale `roundtrip_ok` from before 2026-09-06.** It once decoded *with* special tokens
while `decode_prompt` uses `skip_special_tokens=True`, so a prefix containing
`<|endoftext|>` recorded `true` while that token vanished from the submitted string.
Verified on OLMo-3 ids 100257 and 100277. Runs after that fix are unaffected.

## Verify it yourself

The fidelity gate is the whole point of this page, and it is one command. Run it before any
submission and before any eval that spends a GPU:

```bash
python scripts/score_banded_local.py --arms <your arms file>
```

If every row says `ok`, the strings in that file are the strings the runs found.

## Cross-links

`data/analysis/season3_prefix_scores.md` (the gate, and the corrupted board rows) ·
`data/analysis/season3_gcg_baseline.md` (the conversion constants) ·
`docs/reference/scoring.md` (the sign rule as a reference entry) ·
`docs/HANDOFF_BEHAVIORAL_S3.md` §3 (where the trap was first written down) ·
`_communication/004-2026-09-06-soham-where-the-gap-actually-is.md`.
