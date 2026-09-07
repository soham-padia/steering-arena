# Removing `approach` from `d` leaves every gate unchanged, and the seed corpus is confounded by `approach` regardless

`python scripts/direction_purity.py` → `data/analysis/direction_purity.json`. Three direction variants against the
135 seed pairs and **6** value-flip control pairs (`n_control_pairs` = 6). Zero NDIF, zero GPU, **$0** — cache only.

> **I did not run this.** This page is reconstructed from `direction_purity.json` plus the
> docstring of `scripts/direction_purity.py`. I did not execute the script, watch the job, or
> see any output that is not in the committed JSON. Every number below is transcribed from
> that file; where the JSON does not carry a quantity, this page says so rather than sourcing
> it elsewhere. The run reads only cached activations and costs zero NDIF calls, so anyone who
> wants a verified version can produce one with the command above.

`REVISIONS_2026-09-05.md` is authoritative on what this result is now taken to mean; this page is authoritative on
what the artifact contains. The distinction below is the one `CLAUDE.md` says keeps getting collapsed.

## The three variants

| variant | held-out separation | all-135 accuracy | control kind>cruel | control gap |
|---|---|---|---|---|
| `d` (shipped) | 1.0 | 1.0 | **6** / 6 | **0.2075** |
| `d`, `approach` removed | 1.0 | 1.0 | **6** / 6 | **0.2067** |
| `approach` alone | 0.8235 | 0.837 | 4 / 6 | 0.0206 |

Confound cosines: `cos(d, approach)` = **0.1501**; `cos(d, d_perp)` = **0.9887**, `d_perp` being `d` with the
`approach` component projected out.

## Half one: the fitted direction does not ride the confound

Project `approach` out and nothing moves — separation stays 1.0, all-135 accuracy stays 1.0, the control test stays
6/6, and the control gap goes 0.2075 → 0.2067, a change of 0.0008 or **0.4%** (*derived*). Whatever `d` reads to
separate the pairs is not the `approach` component, because that component deletes at no cost.

The 6 value-flip control pairs are what make this discriminate: both sides of each pair are active and assertive,
so approach is held constant and only human impact flips. `d` gets 6/6 at a gap of 0.2075, `approach` alone gets
4/6 at 0.0206 — smaller by 10.1× (*derived*). Caveats in the same breath: 6 pairs is a small instrument and 6/6
against 4/6 is the entire discriminating comparison here; and `cos(d, d_perp)` = 0.9887 is a rotation of only
about **8.6°** (*derived*, arccos), so "the gates do not move" is a weak test almost by construction — a test
would have to be brittle to notice an 8.6° nudge. Evidence that `d` is not an approach detector, not proof.

## Half two, which must be stated with half one: the corpus *is* confounded

`approach` alone separates held-out pairs at **0.8235** and all 135 at **0.837**. Not chance: a probe fit on this
corpus *could* have come out an approach detector, and the rows above are the only reason we know this one did not.
**Corpus problem, not direction problem** — and quoting either half alone misstates the result in opposite directions.

## Not in this artifact: the "pro-human is a family" figures

`REVISIONS_2026-09-05.md` §4 reports a mutual-cosine mean of about **+0.529**, range **+0.327 to +0.677**, **0 of 105**
pairs negative, and a top eigenvalue at **56.4%**. None of those is in `direction_purity.json`, and
`scripts/direction_purity.py` does not compute them — it fits three variants and tests them, it never builds a cosine
matrix over a family of directions. One command, `grep -rl "0.529\|56.4" data/analysis/*.json`, returns nine files —
`layer_concept_profile`, `behavioral_cache_repair`, `layer_sweep_prefix`, `season3_gcg_aggregate_asymmetry`,
`steering_ablation_measure`, `season3_gcg_ablation`, `steering_ablation`, `transfer_report`, `steering_ablation_check`
— and **not** `direction_purity.json`; those are substring matches on numeric literals, not evidence that any carries
the §4 quantities. Attribute the family figures to REVISIONS or recompute them, never to this artifact.

## What this buys

One sentence that can be said about the shipped `d` without hedging: its separation of the seed pairs does not depend
on the `approach` direction, and deleting that direction costs 0.4% of the control gap. One that must be said with it:
the corpus is separable by `approach` at 0.8235, so the next direction fit on it needs this check rerun, not assumed.

## Limits

1. Six control pairs; the 6/6-against-4/6 contrast carries the whole discrimination, with no interval on it.
2. The rotation is 8.6°, so invariance under it is a low bar, and one confound is all that is projected out here.
3. The JSON carries no layer, no seed, no pair listing and no per-pair scores, only aggregate gates and cosines.
4. Everything is geometric. No generation, no judge, no behavior.

## Cross-links

- `REVISIONS_2026-09-05.md` §4 — standing of the direction-family reading.
- `CLAUDE.md`, "The distinction that keeps getting collapsed" — the rule this page instantiates.
- `season3_directions.md` — Season 3 removes `approach` by construction; the 0.4% is why that was cheap.
- `layer_profile_all64.md` — layer dependence of the separation this page holds fixed.
