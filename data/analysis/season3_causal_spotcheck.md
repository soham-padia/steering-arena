# Season 3 causal spot check — what it does and does not establish

**Date:** 2026-09-06 · **Direction:** `d_olmo3_s3_score1.npz` (cache tag `c3ff96cd5077`)
**Model:** `allenai/Olmo-3-1125-32B` via NDIF · **Injected at:** layer 27 · **n:** 10 prompts × 3 arms

## Why this is a spot check and not the gate

`docs/EXTRACTION.md` requires a causal steering check before a direction ships: if adding
`α·d` does not move generations, `d` is bad and no scoring config rescues it. The full
version of that check is the Season 2 behavioural study — ~650 generations across eight
arms, two LLM judges, a human blind pass and norm-matched random controls.

That was **deliberately descoped** for Season 3, on one number:

```
cos(d_season3_score1, d_season2_L24) = +0.8688      (29.7°, chance floor 0.0140)
```

Season 2's direction already passed the full causal battery. At 0.87 the Season 3
direction is substantially the same vector, so re-running the whole study would mostly
re-establish a transferred result. This is the cheap residual check on what does not
transfer.

## Result

| quantity | value |
|---|---|
| identical continuations, base vs `+1·d` | **0 / 10** |
| looping, base | 2 / 10 |
| looping, `+1·d` | 3 / 10 |

**The direction is causally active.** Zero identical continuations is the bar: had the
edit done nothing, base and steered text would have matched. Compare the ablation arms in
`steering_ablation.md`, where removing `d` left 33/50 continuations byte-identical — that
is what a null looks like, and this is not it.

**No sign of degenerate steering at α = 1.0.** 3 loops against a base rate of 2, at n=10,
is noise. Worth restating because over-steering is the standard failure of a large
coefficient and REVISIONS section 6 found the Season 2 anti arm looping 39/50.

Qualitatively, of three pairs read by hand, one is a clear pro-social shift
("was hesitant… afraid of being taken advantage of" → "was happy to help"), one is
neutral, and one repeats a phrase. Three hand-read examples are an illustration, not
evidence.

## What this does NOT establish

1. **It does not show the direction moves behaviour in the *pro-human* direction.** It
   shows the text changes. Direction-of-effect needs a blind judged comparison, which was
   not run. Do not cite this as behavioural validation.
2. **n = 10, no judge, no controls.** No norm-matched random direction was run against
   this `d`, so "the change is caused by `d` rather than by a perturbation of that size"
   is inherited from Season 2's controls, not re-measured here.
3. **Only `score1` was checked.** `score2`'s per-layer directions were not steered with,
   and are not steerable by construction — that is why they rank nothing.
4. **The 0.87 transfer argument is an argument, not a measurement.** A 29.7° rotation is
   small but not nothing; the two directions are not the same vector.

## Standing recommendation

Before Season 3's results are written up as behavioural claims, run the full battery on
`d_olmo3_s3_score1.npz` — `scripts/behavioral_eval.py` and
`scripts/steering_random_control.py` both take `--d` now, and the generation cache is
direction-namespaced, so it can run without disturbing the Season 2 record.

Provenance: `scripts/behavioral_eval.py generate --d data/directions/d_olmo3_s3_score1.npz
--limit 10 --mults 1.0`, logged in the runlog. Cached under
`data/cache/behavioral/` keyed with the direction tag.
