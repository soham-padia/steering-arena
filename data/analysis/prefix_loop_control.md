# The loop control, applied to Season 2's prefix eval

`python scripts/prefix_behavior_eval.py stats --report data/analysis/prefix_loop_control.json`
→ `data/analysis/prefix_loop_control.json`. **Zero new generations, zero new judge calls,
zero cost** — this is the Season 2 verdicts re-scoped, not a new experiment.

Built as a gate for Part A′ (Season 3's banded arms degenerate badly — `score2_anti` loops
on 24/50 prompts, `data/analysis/prefix_degeneration_s3.json`), then run against Season 2
first to find out what the scope does to a result that is already published. It does two
things, in opposite directions.

## The scope

`no_loop` keeps a pair only if **neither** side degenerates, where degenerate means some
4-gram occurs 3+ times in the continuation — the blunt mechanical version of what the
judges' `repetition` marker was already catching by eye. The point is that a looping text
loses a *kindness* comparison because it is broken, not because it is cruel, so a verdict
against it measures fluency. `_scopes()` in `scripts/prefix_behavior_eval.py`.

51 of Season 2's 150 blind pairs (34%) have a looping side. They are not spread evenly:

| arm | rater | pairs, all | pairs, no_loop | dropped |
|---|---|---|---|---|
| `pro_top` | human | 16 | 14 | 2 (12%) |
| `pro_top` | deepseek | 38 | 31 | 7 (18%) |
| `pro_top` | claude | 44 | 36 | 8 (18%) |
| `pro_coherent` | human | 13 | 11 | 2 (15%) |
| `pro_coherent` | deepseek | 36 | 30 | 6 (17%) |
| `pro_coherent` | claude | 43 | 34 | 9 (21%) |
| `anti_top` | human | 17 | 6 | **11 (65%)** |
| `anti_top` | deepseek | 43 | 16 | **27 (63%)** |
| `anti_top` | claude | 31 | 12 | **19 (61%)** |

## 1. It strengthens the `pro_top` result

The pro arms lose 12–21% of their pairs and stay significant on **all three** raters:

| arm | rater | all | no_loop |
|---|---|---|---|
| `pro_top` | human | 14/16, p=0.0042 | **12/14, p=0.0129** |
| `pro_top` | deepseek | 31/38, p=0.0001 | **25/31, p=0.0009** |
| `pro_top` | claude | 33/44, p=0.0013 | **26/36, p=0.0113** |
| `pro_coherent` | deepseek | 26/36, p=0.0113 | 22/30, p=0.0161 |
| `pro_coherent` | claude | 31/43, p=0.0054 | 25/34, p=0.0090 |
| `pro_coherent` | human | 7/13, p=1.0 | 6/11, p=1.0 (null before and after) |

This matters because the confound ran the *other* way and needed checking: base loops on
7/50 prompts against `pro_top`'s 1/50, so ~14% of pro pairs were a fluent text against a
degenerate one, which would inflate the pro win rate for free. Removing them costs about a
factor of 3 in p and nothing in sign. **Season 2's headline — token soup did make
continuations read as kinder — survives the loop control.**

## 2. It further undermines the already-withdrawn `anti_top` arm

`anti_top` loses **61–65%** of its pairs, three times the pro arms' rate. What is left:

| rater | all | no_loop |
|---|---|---|
| deepseek | 5/43, p=4e-05 | 3/16, p=0.0213 |
| claude | 11/31, p=0.1496 (n.s.) | **5/12, p=0.7744** (flat null) |
| human | 14/17, p=0.0127 | 5/6, p=0.2188 (n too small to read) |

The arm was withdrawn on 2026-08-27 on the strength of human ratings (n=54) showing
repetition 37/50 and incoherent 8/50 — i.e. the prefix makes the model loop, not be cruel.
That withdrawal stands, and this adds the number it was missing: **the majority of the
evidence for it was pairs containing a degenerate text**, and on the rater with the most
coverage (claude, 31 pairs) the residual effect is p=0.77.

Read the deepseek 3/16 honestly, though: it is p=0.021 and the same sign, so "some of the
anti effect is not degeneration" remains live on that rater. The scope does not prove the
anti arm is *only* looping. It shows the published magnitude cannot be attributed to
cruelty.

The human row is the odd one and is not resolved here: humans preferred the anti-*prefixed*
continuation 14/17, the opposite sign from both judges. With 17 pairs against deepseek's 43
that may be nothing, but it was already strange before this scope and still is.

## Caveats

- The 4-gram threshold (3+ repeats) is a choice, not a calibration. It was set once, before
  looking at any verdict, and not tuned. No sensitivity sweep was run.
- `no_loop` drops pairs rather than reweighting, so every no_loop row has less power than
  its `all` row by construction. A p that grows is expected; a *sign* that changes or a
  significant result going flat is the informative outcome.
- Paraphrase-level degeneration (a text that restates itself without repeating a 4-gram) is
  not caught, the same blind spot `_leaked()` has for register.
- Nothing here is new data. If the Season 2 verdicts were biased, this inherits the bias.
