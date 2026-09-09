# The strings are not probe-set artifacts, and the overfitting is where you would guess

`sbatch slurm/heldout_probes.sbatch` → `heldout_baseline.json`,
`heldout_probe_scores_insample.json`, `heldout_probe_scores.json`. One B200, bf16, 53
seconds, no NDIF, $0. Job 726483 (726428 failed first — see Limits).

`REVISIONS_2026-09-05.md` is authoritative on what results now mean; this page is
authoritative on what these three artifacts contain.

## Why this exists

Every score this project has published is measured on the 16 prompts in
`data/probes/season3.json`, and GCG optimised against those same 16. So every number is
**in-sample by construction**, and nobody had checked what happens on prompts the optimiser
never saw. The question is not ours. Jesse Li — who took **both** Season 2 headline slots,
`pro_top` (#653) and `anti_top` (#632), out of 616 submissions — asked it in his write-up:

> "Steering Arena only tests the results on 16 prompts. How overfitted is the prompt prefix
> to those exact prompts?"

`data/probes/heldout_v1.json` is 16 held-out prompts, genre-matched to `season3.json`
(short neutral everyday requests, mixed questions and statements, nothing touching kindness),
zero exact overlap, written 2026-09-08 **after** every Season 3 string was frozen.

**The baseline is recomputed, and that is not optional.** A probe set carries its own
baseline — the probes' own alignment with `d` before any prefix. Season 3's is `−0.00101`
(Score 1) and `−0.00941` (Score 2); the held-out set's is **`+0.00032`** and **`−0.00757`**.
Reusing the old constant would have shifted every held-out score by 0.0013 to 0.0019, which
is 3-6× the fidelity tolerance and would have read as a real drop.
`score_banded_local.py` now refuses a `--probes` / `--baselines` mismatch outright.

## The control ran first, and it reproduces

Re-scoring the same eight arms on the **season3** probes returns every recorded value: max
|gap| **7.0e-04** (`pro_coherent`), five arms under 3.3e-04, `score1_top` exactly 0.0. So the
apparatus is sound and the held-out numbers below are a measurement, not drift.

## Result

Each arm on its own objective — Score 1 for `score1_top`, `pro_coherent`, `score1_anti`,
`random32`; Score 2 for the four score2 arms.

| arm | in-sample | held-out | retained |
|---|---|---|---|
| `score1_top` | +0.16395 | +0.13719 | 83.7% |
| `score2_top` | +0.06518 | +0.04859 | **74.5%** |
| `score2_top_final` | +0.06997 | +0.05034 | **71.9%** |
| `pro_coherent` | +0.03976 | +0.03482 | **87.6%** |
| `random32` | −0.00335 | −0.00217 | *(null either way)* |
| `score1_anti` | −0.10135 | −0.09471 | 93.5% |
| `score2_anti` | −0.12595 | −0.13045 | **103.6%** |
| `score2_anti_final` | −0.16288 | −0.16156 | 99.2% |

**The answer to Jesse's question is: yes, somewhat, and less than you would fear.** Nothing
collapses. The worst case keeps 71.9% of its score on prompts it has never seen, and the
metric's ordering survives — **Score 1's ranking of all eight arms is identical held-out**,
and Score 2's differs by a single adjacent swap between `score1_anti` (−0.116) and
`score2_anti` (−0.130), two arms separated by 0.004 in-sample.

**The overfitting is not uniform, and its shape is the interesting part.**

- **Pro arms lose 12-28%. Anti arms lose 0-7%, and one gains.** Whatever the anti strings
  found transfers almost perfectly; the pro strings are more tied to the specific probes.
- **The hardest-optimised arms generalise worst.** The two Score-2 arms — a conjunction
  objective, 3-position mutation, the most aggressive search in the season — retain 71.9%
  and 74.5%, the two lowest figures in the table.
- **The one string nobody optimised generalises best of the pro arms.** `pro_coherent` is a
  hand-written English sentence submitted by a human player (#1181, `toast`); it never saw a
  gradient. It retains **87.6%**, above `score1_top`'s 83.7% and well above both GCG Score-2
  arms. MINE, and stated as a reading rather than a measurement: optimisation against a fixed
  probe set buys in-sample score partly at the cost of transfer, and a string that was never
  optimised has no probe-specific component to lose.

## Limits

1. **One held-out set, n=16, written by one author in one session** (the assistant, on
   2026-09-08). Genre-matching is a judgement, not a measurement. A second set written by
   someone else would test whether 72-104% is a property of the strings or of this particular
   16.
2. **Nothing here is behavioural.** Every number is a banded cosine. Whether a string that
   keeps 72% of its score also keeps its effect on what the model writes is unasked.
3. **`random32`'s ratio is meaningless** — −0.00335 → −0.00217 is null on both sets, which is
   the correct control behaviour and not a 64.6% retention.
4. **Job 726428 failed first**, at the write in step 1: `baseline_const.py`'s argparse
   namespace `a` was shadowed by the band loop's activation tensor `a`. Renamed to `args`.
   The baselines had already computed correctly; no result was affected.
5. The comparison is bf16 local, not NDIF. `calibration/local_vs_ndif_tf5.10.2_sdpa_695054.json`
   puts local-vs-NDIF |gap| at max 3.71e-4, an order of magnitude below the effects here, but
   any published figure should be re-scored on NDIF.

## Cross-links

`data/probes/heldout_v1.json` (the probe set and why it is not a season set) ·
`scripts/gcg/baseline_const.py --probes` · `scripts/score_banded_local.py --probes/--baselines` ·
`season3_prefix_scores.md` (the in-sample numbers this is measured against)
