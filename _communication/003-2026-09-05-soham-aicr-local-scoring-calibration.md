author: Soham Padia
agent: Claude Code (Opus 5)
date: 2026-09-05
re: 001 (NDIF vs local divergence), 002 (activation snapshot)

# The 7e-2 gap does not reproduce: two suspects ruled out, and the local side measured

Since 001 the work has moved onto **AICR** (`login.aicr.ai`), a multi-institution GPU
cluster. The relevant fact for this thread: `allenai/Olmo-3-1125-32B` is **61 GB across
14 shards** and fits on a single B200 (180 GB), so the model 001 could only reach through
NDIF now runs locally at will. Every remaining suspect in 001 is directly testable, and
two of them are settled below without spending a single GPU-hour.

Sections 1-4 required no forward pass at all — two suspects fall to a tokenizer call and an
HTTP request against the model repo. Section 5 is the measurement, and it is the headline:
against 50 Season-2 submissions with canonical NDIF scores, local scoring agrees to
**3.71e-4** with **Spearman rho = 1.0** and **zero rank inversions in 1225 pairs**. The
divergence 001 reported is not reproducible here, which relocates it from NDIF to the
reporting environment. Local GCG search transfers.

---

## 1. Tokenizer version: RULED OUT

001 named this "the cheapest decisive check" and published token ids for 8 debug prompts
plus 1 composed example under transformers 5.10.2. All 9 were re-tokenized under both
**5.10.2** and **5.12.1**:

| | result |
|---|---|
| strings checked | 9 (8 `jesse_prompts` + 1 `composed_example`) |
| ids identical, 5.10.2 vs published reference | **9 / 9** |
| ids identical, 5.12.1 vs published reference | **9 / 9** |

Environment facts confirmed alongside, all matching 001's correction #4:

- `bos_token = None`, `add_bos_token = False` — the OLMo-3 tokenizer prepends **nothing**.
- `vocab_size = 100278`, backend class `TokenizersBackend`.

So a tokenizer difference between 5.10.2 and 5.12.1 cannot explain any part of the gap on
these strings. Note the scope honestly: this rules out the tokenizer **across those two
versions**, on **those nine strings**. It does not speak to whatever version NDIF serves,
which we still cannot observe directly.

**Artifacts:** `/work/neu/p2026_0037_neu/steering-arena/calibration/tokids_transformers-5.10.2.json`
and `…/tokids_transformers-5.12.1.json` (per-string expected vs got, both files 4166 bytes).

## 2. Weight snapshot revision: RULED OUT

001 listed "weight snapshot revision (if allenai updated the repo after NDIF deployed)" as
a suspect. The HF repo's commit history says no:

| date | commit | title |
|---|---|---|
| 2025-12-03 | `c2b61dae89a1` | Remove chat_template.jinja (base model, not instruct) |
| 2025-11-25 | `c571abe3fdbe` | Extract evaluation results from README (#6) |
| 2025-11-24 | (9 commits) | README.md edits, olmo-base.png uploads |

38 commits total; the most recent **content** change is 2025-12-03 and it only deleted a
chat template. There is **no weight change in 2026 at all**, so allenai cannot have moved
the weights out from under an NDIF deployment that served this project from June 2026
onward. The snapshot in local use is `c2b61dae89a1ad10e4ad5653d0e46b590902607b`, i.e. repo
HEAD.

## 3. Sliding-window attention: NARROWED, probably not the mechanism

001's leading suspect was the Olmo3 modeling code, specifically that "its modeling code
mixes sliding-window and full attention per layer (`config.layer_types`)". The config:

- 64 entries: **48 `sliding_attention`, 16 `full_attention`**.
- `full_attention` at layers **[3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]**
  — period 4, offset 3.
- `sliding_window = 4096`.

The season-2 probes are **8 to 11 tokens** and the composed strings are a few dozen. At
those lengths a 4096-token window never binds, so sliding and full attention see the same
causal mask and are numerically equivalent. A version-to-version difference in the
*sliding-window* logic therefore has nothing to act on here. What is not excluded is a
difference in how the two implementations *construct* the mask (dtype, fill value, additive
vs boolean), which can still perturb bf16 softmax numerics even when the mask is logically
identical.

**Incidental finding, unrelated to this thread but worth recording where it will be found:**
the project's layer sweep sampled layers 16/24/32/40/48. Under a period-4 pattern at offset
3, every one of those is `sliding_attention` — the sweep has never evaluated a
`full_attention` layer, and the candidate band {32, 40, 48} is entirely sliding. Sliding and
full layers are trained differently, so "which layer holds the concept" has so far been
asked of only one of the two layer classes.

## 4. What is left

| suspect | status |
|---|---|
| prompt text mismatch | ruled out (001) |
| batched vs unbatched forward | ruled out (001), 4.9e-5 |
| hook-point off-by-one | ruled out (001), cosine 1.00000000 |
| BOS handling | ruled out (001 correction #4), reconfirmed above |
| tokenizer version | **ruled out (this message)** |
| weight snapshot revision | **ruled out (this message)** |
| sliding/full attention masking | **narrowed (this message)** — window never binds |
| **bf16 kernels / attention backend / NDIF multi-GPU sharding** | **the remaining live hypothesis** |

This is an uncomfortable place to arrive at and I want to state it plainly rather than
tidy it. 001 estimated kernel/backend/sharding effects at **~1e-3** on a cosine shift. The
observed gap is up to **~7e-2** on scores of **~3e-2** — the gap is larger than the signal.
With the other suspects gone, either that ~1e-3 estimate is wrong by one to two orders of
magnitude, or something is happening that none of the hypotheses so far describes. I do not
think we can pick between those two from the armchair, and I am not going to guess.

*Added after the run: it is neither. See §5 — on this stack the 7e-2 does not exist.*

## 5. The decisive test: run, and the gap does not reproduce

Result first: **scoring 50 stratified Season-2 submissions locally on one B200 reproduces
their canonical NDIF scores to within 3.71e-4.** There is no ~7e-2 divergence on this stack.

| quantity | value |
|---|---|
| n | 50, spanning NDIF scores -0.129120 to +0.107693 |
| `|gap|` max | **3.712e-04** |
| `|gap|` median | 9.06e-05 |
| gap mean +/- sd | -3.30e-05 +/- 1.26e-04 |
| every `|gap|` < 5e-4 | yes |
| **Spearman rho** | **1.0000000** |
| Pearson r | 0.9999921 |
| **pairwise rank inversions** | **0 of 1225** |
| relative error at the top score | 0.286% |

Environment: `transformers 5.10.2`, `torch 2.12.1+cu130`, `sdpa`, `bfloat16`, single NVIDIA
B200, layer 24, `d_olmo3_L24_logistic.npz`, the 16 frozen season2 probes, composition
`f"{seq} {probe}"`, residual read at `hidden_states[25]`. Artifact:
`local_vs_ndif_tf5.10.2_sdpa_695054.json` (Slurm job 695054).

**What this settles.** The residual 3.7e-4 sits inside the ~1e-3 band 001 predicted for bf16
kernel and reduction-order differences. So 001's *estimate* was right; what was wrong was the
inference that a structural NDIF-vs-local divergence explained the 7e-2. It does not, because
here there is no 7e-2 to explain. Taken with section 1 (token ids identical) and section 2 (no
weight change since 2025-12-03), the gap reported in 001 is **environment-side on the reporting
stack**, not a property of NDIF or of remote execution.

**What this does NOT settle.** We have not identified what in that environment produced 7e-2,
and this run cannot tell us — it can only say the difference is not where 001 was looking.
Candidates that remain open, all cheap to test against this baseline now that it exists:

- a different attention implementation (`eager` or `flash_attention_2` rather than `sdpa`);
- `float16` rather than `bfloat16`;
- a transformers release outside {5.10.2, 5.12.1};
- a reimplemented hook reading `hidden_states[L]` rather than `hidden_states[L+1]`.

The last is worth checking first, only because it is the exact off-by-one `score_local.py`
warns about in its own docstring and it is the one candidate that is a logic error rather than
a numeric one. I am not predicting its magnitude here; that would need the run.

**The practical consequence.** rho = 1.0 with zero rank inversions across 1225 pairs means a
local objective ranks identically to the board. **Local GCG search transfers.** The dose sweep
on k and the placebo control can both run on local GPUs, with finalists re-scored on NDIF so
the published number lives in the canonical space. NDIF stays canonical; it is simply no longer
a throughput constraint on search.

### Still available and still unrun: the layer-by-layer diff from 002

`data/analysis/activation_snapshot_20260721.npz` holds NDIF-side last-token residuals for 24
texts at layers [1, 4, 8, 12, 16, 20, 24]. 002 laid out the decision tree: mismatch at layer 1
means tokenization; agreement early with drift growing by depth means numeric divergence;
agreement at 24 with a differing score means the direction file or the cosine. Section 1 closes
the first branch. Given section 5, this diagnostic is no longer needed to explain *our* numbers
— but it is the sharpest available tool for characterising *someone else's* stack against a
known-good reference, so it stays on the list.

*(That snapshot was an unfetched 132-byte LFS pointer in the cluster checkout and has been
restored to its real 1.6 MB. It is intact: 52 keys, `acts_0(7, 5120)`.)*

The runner reuses `load_direction`, `load_probes`, `cosine` and `compose` from
`scripts/score_local.py` unmodified, so the only difference from the published local reference
is the driver loop — no chance of introducing a fresh discrepancy while measuring an old one.
