# _falsifier

Dated adversarial audits asking one question: do the project's experiments actually test the claims
they are used to support? Each audit is a critique of an analysis, not a result, which is why it lives
here and not in `data/analysis/`.

The folder has since grown a second layer: the machinery that checks the audits themselves, and the
follow-up experiments the audits called for. Both are here because both are adversarial work product,
not project findings.

## Status, as of 2026-08-27

**Out of date:** an earlier version of this README said "Nothing in this folder has been applied to the
analysis documents. Findings here are NOT yet reflected in `data/analysis/`." That is no longer true.
Two fixer agents have applied these findings to `data/analysis/`. Read the analysis documents as they
now stand rather than assuming they still contain the claims quoted in the audits.

The audits have themselves been audited. `_falsifier/verify.py` found 6 failures across the two audit
documents; all 6 are corrected inline in them, marked `CORRECTED 2026-08-27`, with the wrong claim left
visible next to the right value.

## Contents

**The audits**

- `2026-08-27-experiment-vs-hypothesis-audit.md`: the main audit. 122 verified claims: 115 pass,
  5 fail, 2 uncheckable. Failures corrected inline.
- `2026-08-27-addendum-human-ratings.md`: what the audit missed about the human rating passes.
  27 verified claims: 26 pass, 1 fails. Corrected inline.

**The independent check**

- `verify.py`: a 181-check suite written from the raw artifacts only, without reading, importing or
  executing `recompute.py` or the honesty scripts. Re-run with `python3 _falsifier/verify.py`; it
  **exits 1 while any failure remains**. Read the exit code together with the caveat on the next
  bullet: some of its checks are pinned to the pre-fix state of the repo, so a nonzero exit is not
  by itself evidence that something is broken.
- `verify_result.json`: full per-check output plus eleven recorded cross-document inconsistencies
  (INC-1 through INC-11), including one in an upstream document that neither audit flagged.
  **This file is a snapshot at repo state `8c43273`, before the fixes were applied upstream.** Some
  checks assert the pre-fix state of `data/analysis/` (for example, that four documents contain zero
  occurrences of "withdraw"), so re-running the suite against the current tree flips five of them from
  pass to fail. Those five are the suite correctly noticing that the documents were repaired, not new
  defects. A re-run reports 11 failures rather than 6, and it **overwrites this file in place**.

**The follow-up experiments the audits asked for**

- `recompute.py`, `recompute_result.json`, `recompute_result.md`: the fixed-baseline re-analysis
  (audit finding 2) and the dose-response rerun without the withdrawn `anti_top` arm (finding 3).
- `honesty_blind.py`, `honesty_blind.jsonl`, `honesty_key.json`, `honesty_score.py`,
  `honesty_verdicts.json`, `honesty_result.json`, `honesty_result.md`: the blind honesty re-judge of
  the 400 cached continuations (finding 1). Result: the audit's design criticism stands, its empirical
  prediction fails.
- `_show.py`, `_append.py`: small helpers used while producing the above.

**Fix manifests**, where the fixer agents wrote them, record which upstream lines each finding was
applied to. They are produced by the fix pass, not by the audit, so they may or may not be present.

## How to read this folder

Read an audit together with the document it criticises, and together with `verify_result.json`, which
says which of its claims survived independent recomputation and which did not. An audit that has itself
been checked is worth more than one that has not, and the failures it turned up are part of the record
on purpose.

`_advocate/` is the counterpart folder: it makes the case **for** the project's claims. Reading only
this folder gives a systematically pessimistic picture, and reading only that one gives the opposite.
