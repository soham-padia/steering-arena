# All 64 layers: attention type does not predict probe quality

Companion to `layer_profile_all64.json`, produced by
`/work/neu/p2026_0037_neu/steering-arena/bin/layer_profile_all64.py`. Figure:
`figures/layer_all64.{png,svg}` from `scripts/plot_layer_all64.py`.

**One-line summary.** The full-attention hypothesis is a **null**: depth-matched, full and
sliding layers are indistinguishable as probe sites (excess 0.22823 vs 0.22888, with sliding
marginally ahead). What the sweep did turn up is that `cos(d, approach)` never reaches zero at
any depth, which settles that the approach confound belongs to the seed corpus and cannot be
fixed by choosing a layer.

---

## 1. Why this was run

Every layer sweep this project has run sampled multiples of 8 — 16, 24, 32, 40, 48. OLMo-3
places `full_attention` at layers **[3, 7, 11, …, 63]** (period 4, offset 3) and
`sliding_attention` everywhere else, 16 full against 48 sliding. So all five previously studied
layers are `sliding_attention`, and **no full-attention layer had ever been evaluated as a probe
site**. That is a systematic hole rather than a random one: the sampling grid and the
architectural period were both even, so the two never intersected.

Season 3 intends a multi-layer `d` built on full-attention layers. That premise had never been
measured, so it was measured before anything was built on it.

## 2. THE HEADLINE, AND IT IS A NULL

Depth-matched over L23–55, the region any band would actually use:

| layer set | n | excess margin | Cohen's d | cos(d, approach) |
|---|---|---|---|---|
| `full_attention` | 9 | 0.22823 | 3.133 | 0.17923 |
| `sliding_attention` | 24 | **0.22888** | **3.138** | 0.17915 |

**Sliding is marginally ahead.** The difference in excess margin is 0.00065, which is far inside
the spread across adjacent layers and should be read as zero, not as a sliding advantage.

The raw all-layer means do favour full attention — 0.20071 against 0.19615, Cohen 2.816 against
2.749 — but **that is an artifact and should not be quoted.** The 16 full layers sit at
[3,7,…,63], an even spread across the whole stack; the 48 sliding layers are the complement,
which includes the very shallow layers where the signal has not yet formed (L0 excess 0.01770).
The two sets therefore have different depth distributions, and depth is the variable that
actually matters (§3). The depth-matched comparison in the table above is the honest one.

**Adjacent-layer evidence, which needs no aggregation at all:**

| layer | type | excess |
|---|---|---|
| L23 | `full_attention` | 0.27040 |
| L24 | `sliding_attention` | 0.27064 |
| L25 | `sliding_attention` | 0.27088 |

Three consecutive layers spanning a type boundary, ordered by depth and not by type.

Quantifying the same thing across the whole stack: for each of the 16 full-attention layers,
the deviation of its excess margin from the mean of its immediate neighbours. If attention type
produced a step, full layers would sit systematically off the local trend. The largest
|deviation| is **0.01520**, at L63 — and L63 is the final layer, so it has only one neighbour
and its "deviation" is really the local slope of a steeply falling curve. Over the 14 interior
full layers (L7–L59) the largest |deviation| is **0.01004**, the deviations change sign
(8 positive, 6 negative), and their mean is **+0.00051**. There is no step at any type boundary.

**Status: the hypothesis that full-attention layers are better probe sites is RULED OUT** for
linear decodability on this corpus.

### Why this is unsurprising in hindsight

`sliding_window = 4096`, and the seed-pair texts are two orders of magnitude shorter than that,
so the window never binds and the attention masks are numerically identical between the two
layer types for this input distribution. The only remaining difference is that the two groups
were trained in different roles, and that turns out not to affect linear decodability.

This was raised as a caveat **before** the run, not derived from it, so it is a consistent
explanation rather than a prediction that was confirmed. It was not stated in advance as a
reason to expect a null of this size, and the run would have been worth doing either way.

## 3. Depth does drive it

A smooth unimodal curve: **0.01770 at L0 → peak 0.27088 at L25 → 0.13040 at L63.**

The shipped direction sits at L24, whose excess margin is 0.27064 — **0.00024 below the peak**,
i.e. indistinguishable from it. This retroactively supports L24 on a better criterion than the
one that chose it. `extract_direction.py` selects on held-out separation with a strict `>`, and
held-out accuracy saturates at 1.000 across the whole mid-stack, so under a tie it simply keeps
the first layer. Margin against a per-layer null does not saturate, and it independently puts
the optimum at L24–25.

This also contradicts the "expect layer ≈ 38–45" guidance from the external research report
reviewed this session: L40's excess margin is 0.23215 against L25's 0.27088. That guidance was
extrapolated from 7–8B models and does not hold here.

## 4. The finding that matters most: `cos(d, approach)` never reaches zero

| confound | behaviour across depth |
|---|---|
| `cos(d, valence)` | 0.1934 at L0, collapses to **0.0006** at L16, stays under 0.093 thereafter |
| `cos(d, length)` | 0.3568 at L0, collapses below 0.06 by L4, stays there |
| **`cos(d, approach)`** | **0.301 at L0, minimum 0.1175 at L16, back to 0.174–0.185 across L40–63** |

Valence and length behave the way a confound behaves when it is an artifact of the shallow
layers: they fall away as depth builds a task representation. Approach does not. It has a
minimum, not a collapse, and it never comes near zero at any of the 64 layers.

**This is decisive that approach is a property of the seed corpus, not of any representation
depth. No band selection can remove it.** The remedy is the corpus, and only the corpus.

**The distinction that must not be collapsed, restated because this result touches it.** The
fitted `d` does **not** ride approach: `cos(d, approach) = 0.1501` at the shipped layer, and
projecting approach out leaves held-out separation, all-135 accuracy and the kind>cruel gap
essentially unchanged (`direction_purity.json`). But the **corpus is** confounded: approach
alone separates the pairs at 0.824. Corpus problem, not direction problem. This document adds
that the corpus problem is also depth-invariant; it says nothing new about the fitted direction.

## 5. A margin-versus-purity tradeoff nobody has exploited

| layer | excess margin | cos(d, approach) | cos(d, valence) |
|---|---|---|---|
| **L16** | 0.23223 | **0.1175** | **0.0006** |
| L19 | 0.26552 | 0.1445 | 0.0339 |
| L25 (peak) | 0.27088 | 0.1598 | 0.0428 |

**The peak-margin region is also the most confounded region.** L16 gives up 14.3% of the peak
excess margin and buys a 26% lower approach cosine and a valence cosine that is essentially
zero. Nobody has previously chosen a layer on this axis, because until this sweep the confound
cosines had only ever been computed at one layer.

**Status: OPEN.** Which side of this trade to take is a decision, not a result, and it depends
on whether Season 3 weights concept signal or concept purity more highly. Nothing here
establishes that the cleaner layers produce a *better* direction — only a less contaminated one
by these three measures.

## 6. One control is now exhausted

The kind>cruel value-flip control (6 pairs where both options are active and assertive, so
approach is held constant and only human impact flips) scores **6/6 at every layer from L3
upward except L5, which is 5/6.** The shallow layers are the only ones that fail it: L0 3/6,
L1 4/6, L2 5/6.

**A saturated control is not evidence of quality.** It cannot discriminate between candidate
layers, and it should not be cited in support of one band over another. It retains its original
value as a floor — a direction that failed it would be disqualified — but it has no
discriminating power left in the range where any real choice will be made.

## 7. Method

Lifted verbatim from `scripts/layer_concept_profile.py` so the numbers are directly comparable
to the published five-layer profile:

- Logistic probe, `C=0.1`, `max_iter=4000`, fit on **raw** train activations; coefficient
  unit-normalised → `u`.
- `gap = (unit(chosen_val) @ u) − (unit(rejected_val) @ u)`, measured on **unit-normalised**
  validation activations so magnitude cannot inflate anything.
- `acc = mean(gap > 0)`; `margin = gap.mean()`; `cohen = gap.mean() / gap.std()`.
- **Null:** identical procedure with train labels randomly flipped, always scored against the
  **true** validation labels. `excess = margin − null_margin`.
- 25% validation, **20 splits + 20 shuffles, seed 0**. Splits are pre-generated once and reused
  across all 64 layers, so every layer sees identical splits.

Composition is `f"{prompt} {completion}"`, one space. The residual is the output of decoder
block `L`, i.e. `hidden_states[L + 1]` — index 0 is the embedding output. Model
`allenai/Olmo-3-1125-32B`, bfloat16, `sdpa`, transformers 5.10.2, single B200.

Confound directions are mean-differences over the reference corpora imported from
`scripts/confound_audit.py` (`APPROACH`/`AVOID`, `POS`/`NEG`) and `scripts/extract_direction.py`
(`LONG`/`SHORT`), captured at all 64 layers so the cosines are computed against exactly the
corpora the original single-layer audit used.

## 8. Limitations

1. **This measures DECODABILITY, not causal use.** Same caveat `layer_concept_profile.py`
   carries: a concept can be linearly present at a layer that never drives behaviour. "Where is
   it legible" is not "where does it act". **No causal or steering check was run at any of these
   64 layers.** The project has causal evidence only at L24.
2. **Every number here is provisional.** It is computed on the current, approach-confounded
   corpus. The seed-pair rewrite is expected to change §4 substantially and may shift §3. The
   activations are cached, so regenerating the whole profile after the rewrite costs one GPU job
   and about 30 seconds of CPU.
3. **135 pairs in 5120 dimensions is noisy**, and the 34-pair validation set is small. Repeated
   splits mitigate this but do not remove it.
4. The full-vs-sliding null is a null on **this** corpus, **this** task, and **these** sequence
   lengths. It does not establish that attention type is irrelevant at lengths where the 4096
   window binds.

## 9. What this implies for Season 3 — recommendation, not result

Full-attention layers remain a defensible choice of band, but **for a different reason than the
one that motivated them.** [3, 7, …, 63] is an evenly spaced grid, so it gives uniform depth
coverage by construction and permanently retires the "every sweep only sampled multiples of 8"
criticism. That is a sampling-scaffold argument and it costs nothing.

**The writeup must not claim full-attention layers are better probe sites.** That measured as a
null, and §2 is the citation for it. Claiming otherwise would be exactly the kind of overclaim
this repository exists to catch.

If the band is chosen for peak margin it lands near {19, 23, 27, 31}; if chosen for confound
purity, nearer {11, 15, 19, 23}. §5 is the tradeoff; the decision is open.

## 10. Provenance

| artifact | source |
|---|---|
| `cache/seedpair_acts_all64_695903.npz` (165 MB) | Slurm **695903**, `capture_all_layers.py` |
| `cache/confound_acts_all64_695948.npz` (29 MB) | Slurm **695948**, `capture_confounds_all_layers.py` |
| `runs/layer_profile_all64.json` | Slurm **697791**, `layer_profile_all64.py`, 32-core CPU, 32 s |

Caches live under `/work/neu/p2026_0037_neu/steering-arena/`. The three scripts were written on
the cluster, against cluster paths, in that tree's `bin/`; identical copies now also sit in
`scripts/`, both untracked at the time of writing. They carry absolute `/home` and `/work`
defaults, so they will need path arguments before they run anywhere else.

**Ledger gaps, recorded rather than papered over.** `runlog.py add --why` rows exist for the
capture (695903) and profile (697791) jobs. **Job 695948 was submitted without a ledger row**,
and **no `note --outcome` rows were written for any of 695903, 695948 or 697791** — only the
earlier calibration job 695054 has its outcome recorded. The convention in the home `CLAUDE.md`
is `add` before and `note` after; the `note` half was skipped here.
