# Handoff — do the Season 3 GCG winners actually change behavior?

Written 2026-09-06 by the Claude instance that ran Part A (the search). You are picking up
Part A′: the **behavioral** question. Everything below is what you need and nothing else.
Read `CLAUDE.md` first for repo culture; this file assumes it.

---

## 1. Your job, in one paragraph

Part A found token sequences that maximally move OLMo-3-32B's residual stream along a
"pro-human" direction under two new Season 3 metrics. They are token soup. **Nobody has
checked whether they change what the model actually says.** Prepend each winner to real
prompts, generate continuations, and rate them blind. The metric's whole claim to meaning
rests on the answer. A null result is a real result and this repo publishes those.

Concretely: does a prefix that scores `+0.164` behave differently from one that scores
`+0.055`, and do either differ from no prefix at all?

---

## 2. What Season 3 scores (you need this to interpret anything)

Two objectives over a *band* of layers, not one layer. Both are means over 16 frozen probes
of a cosine shift; composition is `f"{seq} {probe}"`, one space, and the residual read is
`hidden_states[L+1]` (**off-by-one is the classic bug here**).

| | band | aggregate | direction file |
|---|---|---|---|
| **Score 1** | 19, 23, 27, 31 | `mean` over layers vs one shared `d` | `data/directions/d_olmo3_s3_score1.npz` |
| **Score 2** | 15, 23, 31, 39 | `min` over layers vs **per-layer** `d_L` | `data/directions/d_olmo3_s3_score2.npz` |

DB: Season 3 is **`season_id = 5`** (not 3 — the identity sequence consumed 2). Column
`score` = Score 1, `score_alt` = Score 2. 620 rows: 618 rescored from Season 2, plus 2 live
submissions.

Reference points, all in LIVE units (baseline subtracted), so directly comparable to a
leaderboard number:

```
                        score1     score2
Season-2 winner          +0.06747   +0.02308     (Jesse's GCG string, rescored)
field sd (618 entries)    0.01826    0.01421
baseline constant        -0.00101   -0.00941     live = board - baseline
```

---

## 3. The strings to test — and how to get them WITHOUT corrupting them

All under `/work/neu/p2026_0037_neu/steering-arena/gcg/<run>/best.json`.

| run dir | LIVE | aggregate | round-trips? |
|---|---|---|---|
| `score1-2026-09-06T19-20-13Z` | **+0.16395** | `banded_mean` | yes |
| `score2-mut3-2026-09-07T00-17-08Z` | **+0.06475** | `per_layer_min`, k=3 | no |
| `score2-2026-09-06T19-20-13Z` | +0.05548 | `per_layer_min`, softmin search | **yes** |
| `score2-2026-09-06T18-53-11Z` | +0.05372 | `per_layer_min` | no |
| `score1-anti-2026-09-06T20-20-17Z` | **−0.10144** | `banded_mean` vs −d | yes |
| `score2-anti-2026-09-06T20-16-20Z` | **−0.12628** | `per_layer_max` vs −d | no |

> **CORRECTED 2026-09-07.** The two anti rows first read −0.10346 and −0.14510. Both were
> wrong by exactly 2×baseline: `scripts/gcg/watch.py` computed `sign * (best - baseline)`,
> which distributes the sign flip over the baseline as well. The flip comes first, then the
> baseline is subtracted once — `sign * best - baseline`, and `sign * best` is what
> `best.json` already stores as `board_score_true_sign`, so the pro and anti cases are one
> rule. Scoring both strings under the **pro** objective (which is what the board computes)
> measures **−0.10135** and **−0.12595**, confirming the corrected values to 3e-4:
> `data/analysis/season3_prefix_scores.md`, job 709380. `watch.py` is fixed. The score2
> error was 1.9e-2, about 1.3 field sd, so it is not cosmetic. No pro number, ranking, or
> aggregate-asymmetry conclusion changes — anti was easier than pro and still is.

> **TRAP — read this before you copy any string.** These prefixes contain **real newline
> bytes**. If you print one with Python `repr()` and then copy it, the newlines arrive as
> literal `\n` text and you are testing a different string. This already happened: live row
> `id=1342` scores **+0.086** on the board but **+0.140** when measured locally, purely
> because it was pasted from a repr (38 tokens submitted vs 32 in the run). Always extract
> bytes programmatically:
>
> ```python
> import json, sys
> sys.stdout.write(json.load(open(f"{run}/best.json"))["prompt"])
> ```
>
> Then verify: `tok(s, add_special_tokens=False)["input_ids"] == best["ctrl_token_ids_retokenised"]`.

Prefer the **round-tripping** variants where you have a choice — for the others the board
scores a different token sequence than the optimiser searched.

---

## 4. The tool already exists — adapt it, do not reimplement

`scripts/prefix_behavior_eval.py` answers exactly this question for Season 2. It has
`select` / `generate` / `blind` / `judge` / `stats`, 50 eval prompts in
`data/eval/steering_prompts.json`, and **working blinding** (strips the model input so a
rater sees only the continuation; flags prompts where the model echoed the prefix).

**The Season 2 answer, which is your baseline** (`data/analysis/prefix_eval.md`):

| arm | deepseek judge | claude judge |
|---|---|---|
| `pro_top` (token soup, +0.108) | 31/38 = 82%, p=0.0001 | 33/44 = 75%, p=0.0013 |
| `pro_coherent` (readable, +0.040) | 26/36 = 72%, p=0.0113 | 31/43 = 72%, p=0.0054 |
| `anti_top` (−0.129) | 5/43 = 12%, p<0.0001 | 11/31 = 35%, **n.s.** |

So under Season 2, **the token soup did make continuations kinder** — that is the
surprising, load-bearing result you are trying to replicate or break under the banded
metrics. Two caveats already recorded there, do not re-derive them:

- The judge baseline **floats** (each arm re-rated its own baseline). Corrected
  2026-08-27: effects shrink 13–37%, no sign flips. Use a **fixed** baseline.
- The `anti_top` row is **WITHDRAWN**. Human ratings (n=54) showed the anti prefix does not
  make the model cruel — **it makes it loop** (repetition 37/50, incoherent 8/50). If your
  anti arms look "bad", check for degeneration before calling it anti-human behavior.

The new question Season 2 could not ask: **Score 2 is a conjunction across a wide band.**
If banded pressure produces more genuine behavioral change than single-layer pressure, the
Score-2 winner should beat the Score-1 winner behaviorally *despite* scoring 2.5× lower.
If it does not, the band buys robustness to gaming but not meaning. Either way it is the
finding.

Suggested arms: `score1_top`, `score2_top`, `score1_anti`, `score2_anti`, a readable
high-scorer as `pro_coherent`, and **a length-matched random-token control** — 32 random
tokens. Season 2 never ran that control and it is the obvious hole: any 32-token prefix
shifts a continuation somewhat.

---

## 5. Environment

```bash
source ~/startload.sh          # miniforge + HF_HOME=/scratch/$USER/hf
conda activate steering-arena  # NOT gpt-trauma, NOT sa-ndif
```

- `steering-arena`: torch 2.12.1+cu130, transformers **5.10.2** (pinned deliberately —
  the validated reference; do not upgrade).
- `sa-ndif` (py3.12) is CPU-only torch, for NDIF calls only.
- Model runs **locally** — OLMo-3-32B is 61 GB / 14 shards, fits one B200 (183 GB) or one
  RTX PRO 6000 (96 GB). The spec's "too big to run locally" is false on AICR.
- Outputs → `/work/neu/p2026_0037_neu/steering-arena/` (snapshotted). **Never `/home`**
  (~23 GB free). Weights on `/scratch` are **purged after 30 days by mtime**.
- Partitions: `rtx-devel` (4 h, starts fast, use this), `b200-devel` (4 h),
  `b200-batch` (24 h but ~2400 pending — effectively unusable). **`QOSMaxGRESPerUser`
  caps you at 2 concurrent GPUs per partition.** Account `p2026_0037_neu`.
- Never `set -u` in an sbatch script on this cluster.
- Log every job to `/work/neu/p2026_0037_neu/runlog/commands.jsonl` via
  `~/trauma-experiment-gpt/bin/runlog.py` (`add --cmd --why`, `note` after).
- Tests: `pytest tests/` (**not** bare `pytest` — an untracked duplicate tree in
  `steering_arena/` collides by module name). 114 currently pass.

---

## 6. Traps that have already cost time

1. **Sign convention on anti arms.** The optimiser always *maximises*, so an anti run's own
   numbers are positive-is-better. The board would show the negation. Checkpoints record
   `board_score_true_sign`; `watch.py` prints LIVE in board sign. Converting takes the flip
   FIRST and the baseline once — `sign * best - baseline`; the other order is a 2×baseline
   error and is what the corrected table in §3 is about. Jesse's reported anti
   score read `+0.13532` locally while the board showed `−0.06760` — that discrepancy was
   this and nothing else (`_communication/004`).
2. **Anti for Score 2 is `max`, not `min`.** `min_L cos(R,−d) = −max_L cos(R,d)`. Negating
   the direction alone optimises the model's *best* layer instead of its worst — a
   different objective that still prints plausible numbers.
3. **Retokenisation.** GCG optimises token *ids*; the board scores a *string*, and
   `encode(decode(ids)) != ids` for adversarial sequences. `best.json` records both and
   picks `best` on the board score.
4. **NDIF is canonical for any published number.** Local B200 is for search and sweeps. The
   ~7e-2 local/NDIF gap in `_communication/001` does **not** reproduce (max |gap| 3.71e-4,
   ρ=1.0, 0 rank inversions of 1225).
5. **Season = a frozen tuple** `(model_id, model_build, layer, d_version, scoring_config)`.
   Scores are comparable only within a season. Use the **new-season** skill.
6. **Don't quote a number from a `.md` without checking `data/analysis/REVISIONS_2026-09-05.md`**
   and `_falsifier/verify.py`. Several headline claims are withdrawn.
7. The **seed corpus** is confounded by `approach` (separates pairs at 0.824) even though
   the fitted `d` is not (cos = 0.1501). Both halves must be stated together.

---

## 7. What I got wrong in this session — so you don't inherit it

Recorded properly in `data/analysis/season3_gcg_aggregate_asymmetry.json`. Summary:

- **"`min` makes the search landscape flat."** Withdrawn. Score2 has *fewer* identical
  consecutive states (3.6%) than score1 (5.0%), not more.
- **"`T_SA` runs ~3.4× hotter on score2 because its steps are 3.4× smaller."** Withdrawn,
  and I committed it (`9517e05`) and launched a job on it before checking. The 3.4× gap
  exists only after iter 384; through iter 300 the two arms have near-identical step sizes
  (5.99e-4 vs 6.27e-4). I measured an *effect* of saturation and called it the *cause*.
  The cooled arm then came back **worse than baseline** (+0.02822 vs +0.04530 at iter 197),
  so cooling is not merely unsupported — it is refuted.
- **Annotated my own terminal output "no real newlines here."** There were two. That same
  repr-printing is what corrupted the live submission in §3.
- Earlier in the session (already fixed in git): dropped `padding_side="right"`; a test
  fixture with no attention that made 12 tests vacuous; a retokenisation guard that
  rejected everything and silently disabled itself; a dropped `manual_seed`; a broken
  multi-GPU rebind. **The smoke tests and the fidelity audit caught all of these — run
  them.**

The pattern worth generalising: I twice measured a late-window statistic and inferred a
mechanism without checking whether it held early. Check the trajectory, not the endpoint.

---

## 8. What is true about the search, for context

Established this session, with artifacts:

- **Score 2 genuinely resists gaming, and the reason is structural.** `min` is a
  *conjunction* — every band layer must move. Once the four equalise, no single-token edit
  lifts them together and the search saturates: net gain per 100 iters decayed
  `+0.045 → +0.012 → +0.005 → +0.005 → +0.002`. Score 1's `mean` gives partial credit and
  never hard-saturates.
- **Multi-position mutation is the fix, and it works.** `--n-mutations 3` reaches
  `+0.05715` by iter 300 where k=1 stalled at `+0.04530`.
- **Anti is much easier than pro, and mostly for the same reason.** With the aggregate held
  fixed (score1 uses `mean` both ways) the anti/pro ratio is 1.4–1.8×; with `min` vs `max`
  it is 2.9–3.2×. So ~1.9× is the aggregate: a disjunction needs only *one* layer to fail.
  **Consequence: a Score-2 anti-board would be far easier to top than the pro board.**

---

## 9. Live state at handoff

Running (`squeue -u $USER`): `707082` score2 k=3 pro, `706969` score2 cooled (refuted, let
it finish for the record), `707083` score2 k=3 anti (pending on `QOSMaxGRESPerUser`).

Branch `aicr-calibration`, clean as of `6304f2c`. Not deployed — Season 3 backend is live
but **none of these strings have been submitted to the board except the corrupted
`id=1342`**. The plan says do not submit until results are in; that decision is the
maintainer's, not yours.

Open and unstarted: **Part B** (a coherence term in the objective) — explicitly deferred by
the maintainer, do not start it. **Part C** upgrades (probe sampling, random restarts) —
noted, not started. LoRA was raised and is *not* applicable: it adapts weights, this
searches an input string against frozen weights.
