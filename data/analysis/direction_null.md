# The direction encodes the label; most of the cross-layer cosine does not

`python scripts/direction_null.py --n-shuffle 40` → `data/analysis/direction_null.json`.
Five layer-native logistic probes at layers **16, 24, 32, 40, 48**, each compared against a
**label-shuffled** null of `n_shuffle = 40` refits and against an isotropic random null.

> **I did not run this.** This page is reconstructed from `direction_null.json` plus the
> docstring of `scripts/direction_null.py`. I did not execute the script, watch the job, or
> see any output that is not in the committed JSON. Every number below is transcribed from
> that file; where the JSON does not carry a quantity, this page says so rather than
> sourcing it elsewhere. *The run reads only cached activations and costs zero NDIF calls,
> so anyone who wants a verified version can produce one with the command above.*

`REVISIONS_2026-09-05.md` is authoritative on what this result is now taken to mean; this
page is authoritative on what the artifact contains. Anything marked **derived** is
arithmetic over JSON fields, not a field. Statements about what a quantity is *of* — which
layer, which labels — come from the script body, which the JSON does not describe.

## The direction genuinely encodes the label

The thing a direction is *for* is separating chosen from rejected on held-out pairs, and on
that job the null is a coin flip.

| | real | label-shuffled, n = 40 |
|---|---|---|
| held-out separation | **1.0** at all five layers | mean **0.5191**, range **0.2647**–**0.7941** |
| off-diagonal cross-layer cosine | **0.5564** | mean **0.4354**, max **0.4532** |
| `real_exceeds_n_of_n` | **40, 40** (off-diagonal cosine) | — |

Real clears the null **on every draw on both metrics**: the cosine count `[40, 40]` is the
JSON's own field, and for separation it is *derived* — real 1.0 exceeds the largest of the 40
shuffled draws, 0.7941. Two scoping facts from the script rather than the JSON: shuffled
separation is measured at **L24 only** (`sep(1, ds[1])`), and every separation, real or
shuffled, is scored on the **true** labels, so a shuffled probe is asked the real question.
The JSON records neither the pair count nor the split fraction; the docstring says 270 texts,
and *derived*, 0.2647 = 9/34 and 0.7941 = 27/34 exactly, consistent with a 34-pair held-out
set (the figure `layer_concept_profile.py`'s docstring states).

## Two nulls, and the standard one cannot test anything

Both `|cos|` figures are against the real `d_L24`, one draw per iteration.

| null family | mean | max | what it can test |
|---|---|---|---|
| isotropic random unit vectors | **0.013** | **0.0281** | nothing on this claim |
| label-shuffled refit | **0.038** | **0.1151** | the claim |

*Derived:* the shuffled mean is **2.9×** the isotropic mean, its max **4.1×** the isotropic
max, and the **strongest of the isotropic draws (0.0281) still sits below the average
shuffled draw (0.038)** — the weak null's best shot does not reach the strong null's
typical one. The docstring supplies the reason: in 5120 dimensions two random unit vectors
sit at `|cos| ~ 1/sqrt(5120) = 0.014`, which the measured **0.013** reproduces, so *any*
fitted direction looks enormously significant against it. That is the methodological point
the project now runs on, and it is the reusable part of this run: the isotropic null is the
one most steering papers use and it is far too weak to test anything, while the
label-shuffled null holds the texts, the geometry and the fitting procedure fixed and
destroys only the label. It is the null that can hurt the claim, and it does.

## The cross-layer cosine matrix, real and shuffled

`real.cos_matrix`:

| | L16 | L24 | L32 | L40 | L48 |
|---|---|---|---|---|---|
| **L16** | 1.0 | 0.4318 | 0.2778 | 0.2264 | 0.1991 |
| **L24** | 0.4318 | 1.0 | 0.6974 | 0.5941 | 0.5255 |
| **L32** | 0.2778 | 0.6974 | 1.0 | 0.8912 | 0.8027 |
| **L40** | 0.2264 | 0.5941 | 0.8912 | 1.0 | 0.9175 |
| **L48** | 0.1991 | 0.5255 | 0.8027 | 0.9175 | 1.0 |

`shuffled.cos_matrix_mean`, the same measurement over the 40 refits:

| | L16 | L24 | L32 | L40 | L48 |
|---|---|---|---|---|---|
| **L16** | 1.0 | 0.2684 | 0.1517 | 0.1198 | 0.0983 |
| **L24** | 0.2684 | 1.0 | 0.5619 | 0.4413 | 0.3634 |
| **L32** | 0.1517 | 0.5619 | 1.0 | 0.8096 | 0.6853 |
| **L40** | 0.1198 | 0.4413 | 0.8096 | 1.0 | 0.854 |
| **L48** | 0.0983 | 0.3634 | 0.6853 | 0.854 | 1.0 |

`excess_off_diag` = **0.121**. *Derived*, per pair it runs from **+0.0635** (L40–L48, the
tightest pair) to **+0.1634** (L16–L24), positive on all 10; and the shuffled mean is
**78.3%** of the real off-diagonal mean. *Derived* aggregates: the L16 row averages **0.284**
real against **0.160** shuffled, the late band {32, 40, 48} **0.870** against **0.783**.
`REVISIONS_2026-09-05.md` §3 lists 0.158 and 0.784 for those two shuffled figures; the third
decimal differs because it averages per draw while this page averages the rounded mean
matrix — same quantity, different rounding order.

## What the shape does not establish

*Derived, and the sharpest fact in the file:* the rank order of all **10** off-diagonal pairs
is **identical** in the two matrices. Adjacent layers agree, distant layers do not, L16 is the
outlier — in the null too.

**Withdrawn:** cross-layer cosine structure as evidence of a shared pro-human feature, and
with it the reading that the layers disagree because the pro-human feature changes with
depth. Most of the *shape* survives label shuffling, so it is residual-stream geometry that
would appear for any binary split of these texts, not concept structure. `CLAUDE.md` lists
this among the project's withdrawn claims.
**Survives:** a consistent **excess**, real above null on every layer pair, +0.121 overall
and 40/40 draws; and held-out separation 1.0 against a shuffled mean of 0.5191. The band
recommendation the old argument supported is unchanged, but its justification is not — the
docstring's revised reason is that residual geometry makes early and late layers hard to span
with one vector whatever you are probing.

## What this buys

A null that can lose. The label-shuffled refit is now the project's default control, and it
paid for itself immediately: it killed one of this project's own published readings while
confirming the load-bearing one, which is the pattern the repo wants. Concretely it buys
three things — the direction is established as label-encoding rather than corpus-encoding
(1.0 vs 0.5191); the isotropic null is retired with a measured reason rather than a cited
one (0.013, reproducing 1/sqrt(5120)); and every later per-layer claim, including
`layer_concept_profile.md`, has a null to be scored against.

## Limits

1. The shuffled held-out separation and both `|cos|` figures are **L24 only**. The real
   separation is reported at all five layers; its null is not.
2. n = 40 shuffle draws, one seed (`--seed 0` default, and the JSON does not record which
   seed produced it).
3. Held-out separation is quantised at 1/34 ≈ 0.0294 (*derived*), and real is saturated at
   1.0, so this run cannot rank the five layers on separation at all — that is exactly what
   `layer_concept_profile.md` exists to fix.
4. The label-shuffled null is a null on *these* pairs, at *these* five layers, for a logistic
   probe at `C = 0.1`. It says nothing about other estimators or other corpora.
5. Decodability throughout. Nothing here is causal or behavioural.

## Cross-links

`REVISIONS_2026-09-05.md` §3 is the canonical statement of the revision. `figures/direction_null.png`
plots the two matrices on one colour scale beside the separation panel, which is the whole
argument in one image. `layer_concept_profile.md` is the per-layer sequel and depends on this
null. `banded_direction.json` holds the band the docstring re-justifies (no `.md`; its
write-up is `REVISIONS_2026-09-05.md` §6). `steering_random_control.md` is the same
weak-vs-strong null problem on the causal side, where the eight isotropic draws test `d`
against noise and not against other meaningful directions.
