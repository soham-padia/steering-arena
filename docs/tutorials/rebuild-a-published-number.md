# Tutorial: rebuild a published number from its artifact

You will take one published number — `+0.16395`, the best Score 1 a GCG search reached in
Season 3 — find the JSON that owns it, list every document that quotes it, recompute 218
independent claim checks without loading a model, and then find a number whose prose has gone
stale while its artifact moved on. That last step is the point of the exercise: it teaches the
repository's arbitration rule by making you observe it rather than read it.

**No GPU, no NDIF key, no API key, no cluster account, no cost.** Every command below reads
committed files. Total wall time is a few seconds.

Every command and every output on this page was executed. Where a command behaved differently
from what this repository's own prose says, the page says so.

---

## Step 0 — what you need

- A clone of this repository, and a shell in its root.
- Python 3.11 or newer. This page was run on **Python 3.11.15**.
- **scipy.** `_falsifier/verify.py` imports `binomtest`, `pearsonr`, `spearmanr`, `wilcoxon`
  and `t` from `scipy.stats`, and **scipy is not in `requirements.txt`**. Install it:

  ```bash
  python -m pip install scipy
  ```

  This page ran scipy **1.17.1**, both in the cluster's `steering-arena` conda env and in a
  fresh venv built from `requirements.txt` plus that one line. Both give byte-identical
  totals.
- Nothing else. Steps 1, 2 and 4 use only the standard library.

---

## Step 1 — pick the number, and read it from its own artifact

Season 3 scores every prefix under two objectives. `data/analysis/season3_prefix_scores.json`
is the artifact that holds them. Print it:

```bash
python3 - <<'PY'
import json
o = json.load(open("data/analysis/season3_prefix_scores.json"))
print("model:", o["model_id"], "| probes:", o["probe_set"], "| tol:", o["tol"])
print("bands:", o["bands"])
print()
print("arm                  score1_live   score2_live       recorded")
for arm, v in o["scores"].items():
    rec = v.get("recorded")
    rec_s = "        -" if rec is None else format(rec, "13.8f")
    print(format(arm, "<18"), format(v["score1_live"], "13.8f"),
          format(v["score2_live"], "13.8f"), rec_s)
PY
```

```
model: allenai/Olmo-3-1125-32B | probes: season3 | tol: 0.002
bands: {'score1': [19, 23, 27, 31], 'score2': [15, 23, 31, 39]}

arm                  score1_live   score2_live       recorded
score1_top            0.16395026    0.05000848    0.16395026
score2_top            0.07896340    0.06517801    0.06514670
score2_top_final      0.08349662    0.06997444    0.06969343
pro_coherent          0.03976025    0.02145025    0.04046226
random32             -0.00335319   -0.00885404         -
score1_anti          -0.10134964   -0.12618031   -0.10143726
score2_anti          -0.07378874   -0.12595371   -0.12628270
score2_anti_final    -0.12016952   -0.16288099   -0.16335956
```

Your number is the first cell: `score1_top`, `score1_live = 0.16395026`.

Read the three columns together, because that is what makes this artifact worth trusting.
`score1_live` and `score2_live` are the same string measured under both objectives.
`recorded` is what the *search* wrote down at the time. For `score1_top` the two agree to
`0.0e+00`; for the others they differ by up to `-7.0e-04`, all inside the artifact's own
declared `tol: 0.002`. An artifact that carries its own re-measurement and its own tolerance
is one you can check. Note also `random32` has no `recorded` value — it is a control, never
searched for, so there is nothing to have recorded.

The units are LIVE, meaning the per-probe baseline has already been subtracted. See
`docs/reference/scoring.md` on LIVE units for why the order of the flip and the subtraction
matters, and how getting it backwards put two wrong numbers into a handoff.

---

## Step 2 — find the prose that quotes it

A number in an artifact is inert. A number in prose is a claim. Find every claim resting on
this one:

```bash
grep -n "0.16395" data/analysis/*.md
```

```
data/analysis/banded_direction.md:129:**succeeded** — a GCG search against a banded mean reached **+0.16395** LIVE
data/analysis/banded_direction.md:175:actually shipped), `season3_prefix_scores.md` (the +0.16395 that answers caveat b),
data/analysis/prefix_eval_s3.md:38:| `score1_top` | **+0.16395** | +0.05001 | 33/40 | 28/33 | 0.0001 | +0.74 | 1 | 22 |
data/analysis/prefix_eval_s3.md:183:fails at its own top: `score1_top` scores 1.96× `score2_top_final` on Score 1 (+0.16395
data/analysis/season3_band_select.md:58:then confirmed the band is still beatable (+0.16395 LIVE, `season3_prefix_scores.md`) — so
data/analysis/season3_band_select.md:99:`season3_prefix_scores.md` (the band was beaten at +0.16395).
data/analysis/season3_gcg_ablation.md:20:which is **+0.16395** (`prefix_eval_arms_s3.json`, run `score1-2026-09-06T19-20-13Z`,
data/analysis/season3_gcg_ablation.md:169:`season3_prefix_scores.md` (the +0.16395 winner, the random-32 control, and the
data/analysis/season3_prefix_scores.md:143:is a 34-token string, while the run's own final 32-token string measures **+0.16395**. The
data/analysis/season3_prefix_scores.md:16:| `score1_top` | **+0.16395** | +0.05001 | 32 | +0.16395 → +0.16395 (gap 0.0e+00) |
data/analysis/season3_prefix_scores.md:92:| optimised for Score 1 | +0.16395 (100%) | +0.05001 (**77%** of the Score-2 optimum) |
```

Eleven quotations across five documents. Two observations to carry with you:

- Every one of them rounds `0.16395026` to `+0.16395`, and every one names its source. Two
  cite the JSON's own companion page (`season3_prefix_scores.md`); one cites a different
  artifact (`prefix_eval_arms_s3.json`) plus a run id. That is the citation habit you are
  meant to copy.
- The two token counts in that output disagree — `prefix_eval_s3.md:38` shows `33/40`,
  `season3_prefix_scores.md:16` shows `32`, and Step 1's JSON says `n_tokens: 32`. They are
  different columns of different tables, not a contradiction, but you cannot know that from
  the grep. This is why you read the artifact before the prose and not after.

Repeat this for any number you are about to quote. The check costs one `grep`. Quoting a
number whose artifact you have not opened is how the withdrawn claims listed in
`data/analysis/REVISIONS_2026-09-05.md` propagated in the first place.

---

## Step 3 — recompute what does not need a model

`_falsifier/verify.py` re-derives every numeric claim in the audit documents from the raw
committed artifacts. It loads no model, calls no API and touches no network:

```bash
python3 _falsifier/verify.py
```

```
====================================================================================================
VERIFY — independent re-derivation of the _falsifier numeric claims
====================================================================================================
source document                                              PASS   FAIL  FIXED  UNCHK  total
----------------------------------------------------------------------------------------------------
_falsifier/2026-08-27-addendum-human-ratings.md                26      0      1      0     27
_falsifier/2026-08-27-experiment-vs-hypothesis-audit.md       117      0      3      2    122
_falsifier/recompute_result.md                                 32      0      0      0     32
data/analysis/REVISIONS_2026-09-05.md                          17      0      0      0     17
data/analysis/coherence_confound.md                            20      0      0      0     20
----------------------------------------------------------------------------------------------------
TOTAL                                                         212      0      4      2    218
```

2.2 seconds. Below the table it prints a per-check detail table, then the non-passing checks
in full, then eleven cross-document inconsistencies (`INC-1` through `INC-11`).

Two things you will observe, both of which the repository's prose describes inaccurately.

**On the exit code.** `_falsifier/README.md` and `SESSION_REPORT.md` both say the suite
"exits 1 while any failure remains". The implementation is narrower than that phrasing
suggests — the last line of `main()` is:

```python
return 0 if counts["FAIL"] == 0 else 1
```

Only `FAIL` counts. `FIXED` and `UNCHECKABLE` do not. As of this page there are no `FAIL`s,
so:

```bash
python3 _falsifier/verify.py > /dev/null 2>&1; echo "EXIT=$?"
```

```
EXIT=0
```

If you read the prose and expected `1`, you would conclude the suite had not run. It ran. A
nonzero exit here means a genuine regression; a zero exit does not mean every check passed.
Read the `TOTAL` row, not the exit code.

**On the side effect.** The run **rewrites `_falsifier/verify_result.json` in place**, and
that file is tracked in git. It writes the JSON before it prints anything, so a run you
interrupt has already modified your working tree. Check what changed:

```bash
git diff _falsifier/verify_result.json
```

```
-  "repo_commit": "d1b8b46",
+  "repo_commit": "3b221a6",
```

On this run the only change was the recorded commit, because the totals were unchanged. If
you did not mean to update the committed snapshot, put it back:

```bash
git checkout _falsifier/verify_result.json
```

Do that now, before Step 4, so Step 4 reads the committed artifact rather than your own.

---

## Step 4 — find the stale prose

Ask the artifact how many checks there are:

```bash
python3 -c "import json; print(json.load(open('_falsifier/verify_result.json'))['totals'])"
```

```
{'PASS': 212, 'FIXED': 4, 'UNCHECKABLE': 2}
```

212 + 4 + 2 = **218**, matching the `TOTAL` row you just produced. Now ask the prose:

```bash
grep -rn "181 checks" _falsifier/README.md SESSION_REPORT.md
```

```
```

No output, and `echo $?` gives `1`. The string is not there in that exact form — the two
files phrase it differently, so drop the word and search again:

```bash
grep -rn "181" _falsifier/README.md SESSION_REPORT.md _falsifier/2026-08-27-experiment-vs-hypothesis-audit.md
```

```
_falsifier/README.md:33:- `verify.py`: a 181-check suite written from the raw artifacts only, without reading, importing or
_falsifier/2026-08-27-experiment-vs-hypothesis-audit.md:16:(181 checks total, re-run with `python3 _falsifier/verify.py`, exits 1 while any failure remains).
SESSION_REPORT.md:218:| `verify.py` / `verify_result.json` | **181 independent claim checks**, written from the raw artifacts without reading or executing the recompute and honesty scripts |
```

**There it is.** The artifact says 218. Three prose documents say 181. The artifact wins —
you just regenerated it, from the raw data, in 2.2 seconds, and it agreed with itself to the
check. The prose was written when the suite had 181 checks and was not updated when 37 more
were added. That is the repository's arbitration rule in one observation: *the artifact is
the claim; the prose is a copy of the claim, and copies go stale.*

The same three documents are stale a second way, and this one is more interesting. Read
`_falsifier/README.md` around line 40:

> re-running the suite against the current tree flips five of them from pass to fail. [...]
> A re-run reports 11 failures rather than 6

Your run reported **zero** failures and four `FIXED`. Both descriptions were true when
written; the suite has since grown a `FIXED` status for exactly this case. See **What went
wrong for you** below for what `FIXED` means and why it is not a failure.

Notice what did *not* go stale. `CLAUDE.md` says `verify.py` has 218 checks;
`docs/explanation/the-case-for-the-metric.md` says 218. Cross-check with:

```bash
grep -rnoE "[0-9]{2,4} (numeric )?checks" --include=*.md . | grep -v old_project_docs
```

The count in the two documents whose job is to route you to the evidence is current; the
count in three narrative documents is not. When they disagree, regenerate the artifact.

---

## What went wrong for you

**`ModuleNotFoundError: No module named 'scipy'`**
`_falsifier/verify.py` line 31 imports from `scipy.stats`, and `requirements.txt` does not
list scipy. A venv built strictly from `requirements.txt` fails here:

```
  File "/home/padia_so_neu/steering-arena/_falsifier/verify.py", line 31, in <module>
    from scipy.stats import binomtest, pearsonr, spearmanr, wilcoxon
ModuleNotFoundError: No module named 'scipy'
```

`python -m pip install scipy` fixes it. With scipy 1.17.1 added, a fresh venv built from
`requirements.txt` reproduces the `TOTAL 212 0 4 2 218` row exactly.

**`statistics.StatisticsError: no median for empty data` — on a clean clone**
This is the one that will actually stop you, and it is worse than the brief version of the
story. On a *fresh* clone, `verify.py` does not report `UNCHECKABLE` for the artifacts it
cannot find. It **crashes**:

```bash
git clone <this-repo> /tmp/cleanclone && cd /tmp/cleanclone && python3 _falsifier/verify.py
```

```
  File "/tmp/cleanclone/_falsifier/verify.py", line 574, in <dictcomp>
    meds = {a: st.median([len(v["continuation"]) for v in PBEH[a].values()])
  File ".../statistics.py", line 565, in median
    raise StatisticsError("no median for empty data")
statistics.StatisticsError: no median for empty data
```

Exit code 1 — but from a traceback, not from a `FAIL`, which is a third reason not to read
meaning into the exit code. The cause: `PBEH` is built from `data/cache/prefix_behavioral/`,
and `.gitignore` line 37 excludes `data/cache/` entirely. On a clean clone the directory does
not exist, so every arm is empty and `statistics.median` refuses. The same holds for
`data/analysis/behavioral_blind.csv` and `behavioral_blind_key.json`, excluded at
`.gitignore` lines 65-66 to keep the blind key private until the human rating pass is done —
`verify.py:1127` reads that CSV directly. (`data/analysis/prefix_blind.csv`,
`prefix_blind_s3.csv` and `_falsifier/honesty_blind.jsonl` *are* tracked, so those checks are
fine.)

There is no flag to skip them; `verify.py --help` offers only `--json`. Regenerating the
cache means re-running the behavioural generation on the model, which needs the weights and a
GPU. **So on a clean clone, Step 3 is not available and Step 4 is your recompute** — read the
committed `_falsifier/verify_result.json`, which is a full per-check record including the
inputs and tolerances, and re-run the suite only on a checkout that has the caches.

**Four checks come back `FIXED`, not `PASS`, and they are supposed to**
`A-NO-WITHDRAW`, `A-RANDCTL-NO-DATE`, `A-HEADLINE-ON-IT` and `N2-CONTRADICTION` assert the
**pre-fix** state of four `data/analysis/` documents — for example that four of them
contained zero occurrences of "withdraw". They were true when the audit was written at commit
`8c43273`. The 2026-08-27 fix pass then deliberately changed that text, which makes the
claims false *because they were acted on*. `verify.py` relabels those four from `FAIL` to
`FIXED` and excludes them from the exit code, with the reasoning in a comment at the top of
`main()`:

> Failing on those would make the suite unusable as a regression gate: it would go red on
> success. [...] A genuine regression here reappears as `FAIL`, because the fix text would
> have to be reverted.

The relabelled note keeps the original failure text, so nothing is hidden. Read them with
`python3 -c "import json; [print(c['status'],'|',c['id']) for c in json.load(open('_falsifier/verify_result.json'))['checks'] if c['status']!='PASS']"`.

**Two checks come back `UNCHECKABLE`, for a reason worth knowing**
`A-617-SUBMISSIONS` and `A-WARMTH-SALIENCE`. Neither is about a missing cache. The first
cannot be checked because the submissions live in Supabase and there is no per-submission
export in the repository; the second cannot be checked because no artifact encodes
prompt-level axis salience, so "warmth is the salient axis" is an unoperationalised claim.
`UNCHECKABLE` here means *the claim has no artifact*, which is a finding about the claim, not
a gap in the tooling — and it is the strongest reason to prefer a number that has one.
