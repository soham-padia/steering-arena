# Steering Arena documentation

```bash
curl -s https://sohampadianeu-steering-arena.hf.space/health
```

```
{"status":"ok","season":5,"season_name":"Season 3","model":"allenai/Olmo-3-1125-32B"}
```

That is the live board. This folder is how to understand what it measures and whether the
measurement means anything.

## What this project is

Players submit token sequences. A server scores how far each one shifts a frozen
OLMo-3-32B's residual stream along a direction fitted to separate pro-human from
anti-human text. The scoring never looks at what the model writes. It runs on free tiers
end to end.

Two questions, and keeping them apart is the whole discipline of the project:

1. **Can we curate a token sequence that pushes the residual stream along that direction**,
   at a chosen depth, read at the last token? Answered, at scale — 721 submissions across
   two seasons, and a Season 3 search reaching +0.16395.
2. **Does prepending that sequence make the model behave differently?** This is the one
   that turned a game back into research. Partly yes, with two of its own headline claims
   withdrawn within days of publication.

## How to read this set

Pages come in four modes and the mode is in the path. No page mixes two.

| mode | what it is for |
|---|---|
| `tutorials/` | learning by doing. Numbered steps that complete on a clean machine. |
| `how-to/` | one task. The title starts with "How to". Assumes you know why. |
| `reference/` | lookup. Identical structure per entry, no teaching. |
| `explanation/` | the why. Tradeoffs, alternatives rejected, history. |

Four files sit at the root of `docs/` in UPPERCASE and are **not** part of this set. They
are the dated record — a deploy runbook, an extraction methodology from June, an
agent-to-agent handoff with a known and corrected error in §3, and the rater protocol. They
keep their names and paths because a public page, a script's printed output, and three
analysis documents cite them, and because a frozen handoff is worth more quotable at its
original path than tidied.

## Start here

**I want to play.** `reference/scoring.md` for what is being measured →
`explanation/what-the-behavioural-tests-establish.md` for whether it means anything.

**I want to check your numbers.** `reference/withdrawn-claims.md` for what has already
failed → any `data/analysis/*.md`, each of which sits beside the JSON it was computed from
→ `python3 _falsifier/verify.py`, which recomputes 218 published numbers from their
artifacts and exits non-zero if one has drifted.

**I want to run the research.** `how-to/move-a-searched-string-without-corrupting-it.md`
first, because that trap has corrupted five live board rows → then the analysis documents
for whichever experiment you are extending.

## Reading order for the whole story

1. `explanation/what-the-behavioural-tests-establish.md` — the two questions and the
   controls.
2. `reference/scoring.md` — the metric, the aggregates, and the two traps that cost the
   most.
3. `data/analysis/season3_directions.md` and `season3_band_select.md` — where the direction
   came from, and the band chosen *against* its own best margin.
4. `data/analysis/season3_gcg_aggregate_asymmetry.md` and `season3_k3_control.md` — why the
   harder objective is harder, and what actually fixed it.
5. `data/analysis/prefix_eval_s3.md` — the behavioural result, with its full caveat list.
6. `SESSION_REPORT.md` — Season 2's behavioural experiments in full. It pre-dates two later
   corrections: the fixed-baseline recomputation and the loop control.
7. `data/analysis/REVISIONS_2026-09-05.md` — the most recent accounting of what survived.
8. `reference/withdrawn-claims.md` — the register, twelve entries.
9. `explanation/the-case-for-the-metric.md` **and** `_falsifier/README.md` — the advocate
   case and its counterpart. **Read both or neither.** Either alone is systematically
   distorted, and the repo is built that way on purpose.

## The research record

Four folders, deliberately adversarial. Read at least two before believing anything.

| folder | what it holds |
|---|---|
| `data/analysis/` | results, each `.md` beside the `.json` a script produced |
| `_falsifier/` | adversarial audits *of* those results, plus `verify.py` |
| `_advocate/` | the counterpart case *for* them |
| `_communication/` | append-only collaborator log; never edit an existing message |

## Conventions

- **Every number names its source.** A result is stated beside the artifact that produced
  it, and one number has one home. Where a page quotes a number from elsewhere it says
  which file.
- **If a page and a JSON disagree, the JSON wins** and the page is a bug.
- **The withdrawal goes next to the claim**, in the same paragraph, with a date. Nothing is
  silently rewritten and nothing is deleted.
- **A page that says "I did not run this"** was reconstructed from a committed artifact
  without the analysis being re-run. Believe its transcription, not its atmosphere.

## Local-only files

`CLAUDE.md` and `STRUCTURE.md` are **gitignored**. If you cloned this repo you do not have
them, and any reference to them will not resolve. That is also why this page exists: until
now the project's only entry points were files that never left the maintainer's machine.

## Not written yet

This set is partial, and it is more useful to say so than to imply otherwise. `tutorials/`
is empty. `how-to/` has one page of five planned — a cluster-job guide, a GCG-search guide,
a blind-eval guide and a new-season guide are outstanding. `reference/` is missing
`seasons.md`, `artifacts.md` and `environments.md`. `explanation/` is missing
`why-the-metric-is-shaped-this-way.md`, which is where the rejected alternatives belong —
including the geometric mean that would have ranked the most anti-human text in the corpus
as strongly pro-human, a measurement that currently survives only in a commit message.

Nine `data/analysis/` JSONs also still have no companion write-up: `layer_concept_profile`,
`normalization_check`, `causal_layer_curve`, `direction_purity`, `meandiff_ablation` and
four smaller ones. `REVISIONS_2026-09-05.md` records what those results *mean*; what is
missing is a page per artifact recording what each one *contains*.
