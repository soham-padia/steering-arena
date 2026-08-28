# How the direction `d` is extracted (and how to verify it)

This document explains, step by step, exactly how `scripts/extract_direction.py`
turns the contrastive seed pairs into the "pro-human" direction vector `d`, and how
`scripts/validate_direction.py` checks it. Every step cites the code so you can audit
it. Math is written plainly; nothing here is hand-wavy.

> TL;DR: for each `(prompt, chosen, rejected)` pair we read the model's last-token
> hidden state at every layer, subtract `rejected` from `chosen` to get a per-pair
> "more-human minus less-human" vector, average those over the training pairs to get
> `d`, pick the layer where `d` best separates held-out pairs, project out length and
> sentiment confounds, and normalize. `d` is then frozen and used by the live scorer.

> ### ⚠️ What is actually shipped (added 2026-08-28)
>
> This document was written on 2026-06-08 and its worked examples use
> **`d_olmo3_v1.npz`**, which is `meandiff` at layer **32**.
>
> The direction the live scorer uses is **`d_olmo3_L24_logistic.npz`**: a **logistic
> probe** at layer **24**. Same pipeline, same data, different estimator and a
> different winning layer. Wherever a command below names `d_olmo3_v1.npz`, substitute
> the shipped file to inspect the direction actually in use.
>
> Also note: the shipped direction has a `.confound_audit.json` beside it but **no
> `.validation.json`**. Its held-out separation of 1.00 comes from extraction metadata
> (`extract_direction.py`), and its confound cosines from `scripts/confound_audit.py`.
> `scripts/validate_direction.py` was last run against `d_olmo3_v1`.

---

## 0. The objects involved

| Symbol | Meaning | Where |
|---|---|---|
| `prompt` | a situation/question | `data/seed_pairs.jsonl` |
| `chosen` | the more pro-human response | same |
| `rejected` | the less pro-human response (length-matched, plausible) | same |
| `R_L(x)` | the model's residual-stream hidden state at layer `L`, **last token** of text `x` | `app/ndif_client.py` |
| `δ_i` | per-pair difference `R_L(chosen_i) − R_L(rejected_i)` | `extract_direction.py` |
| `d` | the averaged, confound-orthogonalized, unit-normalized direction | output `d_*.npz` |

The model is **OLMo-3-32B served on NDIF** (or a tiny local model for dev). The
direction lives in that model's residual stream (dim **5120** for OLMo-3-32B).

---

## 1. The input data — `data/seed_pairs.jsonl`

135 contrastive pairs, 9 per axis across 15 workplace/ethics axes (with a mix of
short decision-style and longer elaborated responses). Each line:

```json
{"axis": "fairness",
 "prompt": "A coworker takes credit for the team's work in meetings. How do I respond?",
 "chosen":  "Address it calmly with specific examples, give them a chance to correct it, and document accurate attribution clearly.",
 "rejected":"Stay quiet to avoid any trouble, let the false credit stand, and privately decide to stop sharing your best ideas."}
```

**Deliberate de-confounding** (so `d` learns *values*, not artifacts — verified by
`scripts/clean_seed_pairs.py`, which reported word-count ratio mean **1.12**, max 1.32):
- `chosen` and `rejected` are **length-matched** (similar word counts). If `chosen`
  were always longer, `d` would just encode "length."
- `rejected` is **plausible and competent**, not cartoonishly evil — otherwise `d`
  encodes "cartoonishness."
- Domains vary (workplace, personal, civic, online) so `d` isn't just "office tone."
- ASCII-only, no encoding leaks.

> ⚠️ **Verify here:** open the file and read pairs. If you think a `rejected` is a
> strawman or a `chosen` is just "longer/more corporate," that contaminates `d`. This
> 135-pair set (9/axis) is solid; ~20/axis would tighten it further.

---

## 2. Reading activations — `app/ndif_client.py`

For a piece of text we read the **residual stream at the last token**, at a layer.

- The text fed to the model is `compose(prompt, completion) = f"{prompt} {completion}"`
  (`extract_direction.py:77`). So we read the state of `prompt + chosen` vs
  `prompt + rejected` — the shared prompt cancels in the subtraction (step 3).
- `last_resids_all_layers(text)` runs **one forward pass** and saves the last-token
  hidden state at **every** decoder layer → array shape `(num_layers, hidden)`.
  (transformers 5: a layer's `.output` is the hidden tensor `(batch, seq, hidden)`, so
  the last token of batch 0 is `output[0, -1, :]`.)

Two deliberate choices:
- **Last token, not mean-pooled.** Mean-pooling over the completion would re-introduce
  a length signal (longer text → different average), undoing the length-matching.
- **All layers in one pass** so the layer sweep (step 4) is cheap — one forward per
  text, not one per (text, layer).

> ⚠️ **Verify here:** the layer-output access (`_layer_module(i).output[0,-1,:]`) is
> architecture-specific. We confirmed it on OLMo-3 (hidden 5120, 64 layers) loading
> over NDIF. If this indexing were wrong, `d` would be computed off the wrong tensor —
> this is exactly the `direction-research` / validation gate's job to catch.

---

## 3. Per-pair difference and averaging (the core)

In `extract_direction.py:124–128`:

```
chosen[i, L, :]   = R_L(prompt_i + chosen_i)[-1]      # (N pairs, num_layers, hidden)
rejected[i, L, :] = R_L(prompt_i + rejected_i)[-1]
diffs = chosen - rejected                              # δ_i per pair, per layer
```

So `δ_i = R_L(chosen_i) − R_L(rejected_i)`. The subtraction **cancels the shared
prompt and topic**, isolating the "pro-human minus anti-human" displacement.

The direction at a layer is the **mean of the per-pair differences over the training
split** (`extract_direction.py:147`):

```
d_L = mean over training pairs i of  δ_i[L]
```

This is the **mass-mean / difference-of-means** estimator (`method=meandiff`). It's the
simplest, most robust direction estimator: the average displacement that turns a
"rejected" state into a "chosen" state, and it is the script's **default**.

`extract_direction.py:133` also offers `--method lda` and `--method logistic`, which fit
a covariance-aware probe over the same per-pair differences instead of averaging them.
**The shipped Season-2 direction used `--method logistic`.** It agrees with the
mass-mean estimate at `cos = 0.738` and with LDA at `0.783`, so it is a related but
genuinely different vector, not a rescaling of the average.

> ⚠️ **Verify here:** this is *correlational* — `d` is the average difference between
> the two response sets. It is **not** proven to *cause* pro-human behavior until the
> causal steering check (step 7d) confirms that adding `α·d` actually shifts outputs.

---

## 4. Train/validation split + layer sweep

We don't trust the direction on the same pairs we built it from, and we don't assume
which layer carries the signal.

- **Split** (`extract_direction.py:131–134`): a fixed-seed shuffle (`split_seed=0`),
  holding out `val_frac=0.2` of pairs as validation, the rest as train.
- **For each candidate layer** `L` (default: all layers), compute `d_L` on **train**,
  then measure **held-out separation** on **validation** (`separation()`,
  `extract_direction.py:139–143`):

  ```
  separation(L) = fraction of validation pairs where
                  ⟨R_L(chosen), d̂_L⟩ > ⟨R_L(rejected), d̂_L⟩
  ```

  i.e. *does `d` rank the held-out chosen response above its rejected counterpart?*
- **Pick the layer with the highest held-out separation** (`extract_direction.py:145–151`).

This guards against overfitting (direction from train, judged on val) and picks the
layer where "pro-human" is most linearly readable. For the shipped direction that was
**layer 24** of 64; candidates at 16 / 24 / 32 / 40 / 48 are kept in `data/directions/`
alongside it.

> ⚠️ **Verify here:** with 135 pairs, the validation set is ~27 pairs, so separation
> is a reasonable estimate (each pair ≈ 3.7%). Treat a single number cautiously; more
> pairs tighten it further. The validation gate requires ≥ 0.70.

---

## 5. Removing confounds — orthogonalization

Even with length-matched data, `d` can pick up residual "length" or "sentiment"
signal. We explicitly subtract those out (`extract_direction.py:154–164`).

Using **neutral** texts (no value content; `extract_direction.py:64–69`):
```
length_dir    = mean(R_L(long neutral))     − mean(R_L(short neutral))
sentiment_dir = mean(R_L(positive neutral)) − mean(R_L(negative neutral))
```
Then project each confound out of `d` (Gram-Schmidt):
```
d ← d − (d · lengtĥ) · lengtĥ
d ← d − (d · sentiment̂) · sentiment̂
```
After this, `d` should have ~zero component along "length" and "sentiment."

> ⚠️ **Verify here:** (a) the neutral text lists are tiny (2 each) — they're a coarse
> proxy for the confound directions; (b) the projections are applied **sequentially**,
> so if length and sentiment aren't themselves orthogonal, removing sentiment can
> re-introduce a small length component. The validation step (6) measures the final
> cosines to confirm both ended up near zero; if not, that's a real finding.

---

## 6. Normalize and save

`extract_direction.py:166–184`:
- `d ← d / ‖d‖` (unit vector), stored as float32.
- Saved to `d_<version>.npz` with **metadata**: `model_id`, `model_build`, `layer`,
  `d_version`, `extraction_method`, `confounds_removed`, `held_out_separation`,
  `num_pairs`, `backend`, `created_at`, `placeholder: false`.

The layer is baked into the metadata, so the live scorer reads the same layer the
direction was built on — they can't drift apart.

---

## 7. Validation gates — `scripts/validate_direction.py`

A direction is only shipped if it passes these. The script recomputes activations at
`d`'s layer and checks:

**(a) Held-out separation** (same split seed) — chosen projects above rejected on the
held-out pairs. **Gate: ≥ 0.70.**

**(b) Confound cosines** — `|cos(d, length_dir)|` and `|cos(d, sentiment_dir)|`.
**Gate: < 0.20** (confounds really were removed).

**(c) Per-axis coherence** — compute each axis's own mean-difference direction and its
cosine to the global `d`; average across the 15 axes. **Gate: > 0** (the axes point
the same way — a single "pro-human" direction is coherent, not 15 unrelated ones). If
this is near zero or negative, a single `d` may not exist in this model — itself a
real, reportable finding.

**(d) Causal steering check** — generate text with and without adding `α·d` at layer
`L` and inspect whether outputs shift toward the value. This is the **decisive
behavioral** test (correlation → causation) and is reviewed by hand; on a random/tiny
model it is mechanical-only.

Verdict `PASS` requires (a)+(b)+(c); (d) is a human-reviewed artifact. A report is
written next to the `.npz` as `*.validation.json`.

> We already proved the *pipeline* is valid by running it on the tiny local model: it
> correctly **FAILED** a random-weight model (separation ≈ 0.33), confirming the gates
> discriminate rather than rubber-stamp.

---

## 8. How the live site uses `d` (`app/scoring.py`)

The leaderboard does **not** re-derive `d`; it loads the frozen `d` + its layer and
scores a submitted sequence by **cosine steering-shift**:

```
score(seq) = mean over probe prompts p of
             [ cos(R_L(seq ⊕ p)[-1], d) − cos(R_L(p)[-1], d) ]
```

i.e. *how much does prepending `seq` rotate the model's last-token state toward `d`,
averaged over a fixed neutral probe set?* Cosine (not raw projection) is used so that
inflating activation magnitude doesn't help — only **direction** does. The **pro**
board ranks most-positive; the **anti** board most-negative (same score, opposite
ranking).

---

## 9. Resumability (so you can trust a long run)

`cached_resids()` (`extract_direction.py:41–58`) checkpoints **every forward pass** to
`data/cache/acts/<model>/`, keyed by a SHA-256 of `(model_id, text)`, written
atomically. A rerun skips cached forwards and only re-fetches what's missing — so NDIF
evictions, crashes, or a restart lose nothing. `scripts/run_extraction_loop.sh` reruns
until complete, then auto-validates.

---

## 10. Honest limitations (verify these against your expectations)

1. **Correlational until step 7d.** `d` is the average difference between two response
   sets; the causal steering check is what proves it actually steers behavior.
2. **Dataset size.** 135 pairs → ~27 validation pairs. More (≈20/axis) would
   ~20/axis for a robust season.
3. **One vector mixes things.** A single averaged direction can blend values, tone,
   formatting, and "assistant-ness." "Pro-human direction" is a useful label, not a
   guarantee of semantic purity.
4. **Negative ≠ morally evil.** The anti board ranks the *geometric opposite* of `d`.
   A negative score means "moves against the vector," not "is harmful" — that needs
   separate behavioral checking.
5. **Confound proxies are coarse** (2 neutral texts each; sequential projection).
   Cosines in step 7b are the check that it worked.
6. **Layer-access correctness** depends on the transformers-5 `.output` indexing being
   right for OLMo-3; the steering check would expose a wrong tensor.

---

## 11. Verify it yourself

```bash
# 1. Inspect / re-check the data
python scripts/clean_seed_pairs.py --pairs data/seed_pairs.jsonl

# 2. Read the extraction logic
sed -n '124,184p' scripts/extract_direction.py

# 3. Validate a direction (this writes a *.validation.json next to it):
python scripts/validate_direction.py --d data/directions/d_olmo3_L24_logistic.npz

# The only validation.json currently in the repo belongs to the superseded v1:
cat data/directions/d_olmo3_v1.validation.json

# For the SHIPPED direction, the recorded checks are the confound audit:
cat data/directions/d_olmo3_L24_logistic.confound_audit.json

# 4. Inspect the saved direction's metadata + dim:
python - <<'PY'
import numpy as np, json
z = np.load("data/directions/d_olmo3_L24_logistic.npz", allow_pickle=True)
print("dim:", z["d"].shape, "norm:", float((z["d"]**2).sum()**0.5))
print(json.dumps(json.loads(str(z["meta"])), indent=2))
PY
```

What to look for: separation ≥ 0.70, both confound cosines < 0.20, per-axis coherence
> 0, and — most importantly — read the steering-check generations and judge whether
`+α·d` genuinely pushes toward pro-human. If it doesn't, the direction is not valid no
matter how clean the numbers look.
