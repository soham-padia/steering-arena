# Tutorial: compute a steering score yourself

You will install the dependencies, run the scorer on `gpt2` against a placeholder direction,
get a number, and then break that number three ways on purpose — once for each of the three
reproduction traps the scorer is built to avoid. Total wall time after the install is under
two minutes on a CPU.

**The number you produce is not meaningful.** A placeholder direction is a random unit
vector (`scripts/make_placeholder_direction.py`, `extraction_method: placeholder-random`),
so a shift along it measures nothing about the model and nothing about being pro-human. The
point is the mechanics: composition, the residual index, and the cosine's dtype. For what
each quantity means, read `docs/reference/scoring.md`.

Every command and every output on this page was executed. Where a command behaved
differently from what the repository's own docs say, the page says so.

---

## Step 0 — what you need

- A clone of this repository, and a shell in its root.
- Python 3.11 or newer. This page was run on **Python 3.11.15**.
- **No GPU, no NDIF key, no cluster account.** Steps 1-7 use none of them.
- About **6 GB of free disk**: 5.1 GB for the virtual environment (Step 1) and 526 MB for
  the `gpt2` weights, which download to `$HF_HOME` (or `~/.cache/huggingface` if `HF_HOME`
  is unset) on first use.
- Network access on the first run only, for PyPI and the Hugging Face hub.

After Step 1 you will have torch 2.12.0, transformers 5.10.2, numpy 2.2.1.

---

## Step 1 — install the dependencies

Create a fresh virtual environment and install the pinned requirements:

```bash
python -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

Confirm what you got:

```bash
./.venv/bin/python -c "import torch,transformers,numpy;print(torch.__version__,transformers.__version__,numpy.__version__)"
```

```
2.12.0+cu130 5.10.2 2.2.1
```

Two things about this install that the pin does not tell you.

**It downloads the CUDA build even with no GPU.** `requirements.txt` pins `torch==2.12.0`
with no index URL, so pip resolves the default wheel — `torch-2.12.0-cp311-cp311-manylinux_2_28_x86_64.whl (532.2 MB)`
plus twenty `nvidia-*` packages. The finished environment is **5.1 GB**. Everything on this
page still runs on the CPU. If you want the small install instead, `requirements.txt`'s own
comment gives it:

```bash
./.venv/bin/python -m pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
```

Install that *first*, then the requirements file, so pip keeps the CPU wheel.

**Do not run this command inside a pre-built project environment.** On the AICR cluster the
`steering-arena` conda env already carries torch 2.12.1+cu130 and numpy 2.4.6, and
`requirements.txt` pins lower. A dry run there reports:

```bash
python -m pip install --dry-run -r requirements.txt
```

```
Would install numpy-2.2.1 pytest-8.3.4 python-dotenv-1.0.1 torch-2.12.0 triton-3.7.0
```

That is a downgrade of the validated environment. Use the existing env as-is
(`conda activate steering-arena`), or a fresh venv, and never `pip install -r` into one from
the other.

The rest of this page writes `python`. Read it as `./.venv/bin/python` if you made the venv,
or as the interpreter of your activated project env.

---

## Step 2 — run the scoring tests

Before you trust a number the scorer gives you, run the tests that pin its behaviour:

```bash
python -m pytest tests/test_scoring_determinism.py -q
```

```
..............                                                           [100%]
14 passed in 0.10s
```

Fourteen checks, a tenth of a second, no network, no GPU and no model — the fixtures are
synthetic. If these fail, stop here; nothing later on this page will mean anything.

Run `pytest tests/` and not bare `pytest`. See **What went wrong for you** for why.

---

## Step 3 — make a placeholder direction

`gpt2` has a hidden size of 768. The shipped directions are 5120-dimensional, extracted on
OLMo-3-32B, and will not load against `gpt2`. Generate a 768-dimensional stand-in:

```bash
python scripts/make_placeholder_direction.py \
    --dim 768 --layer 6 --d-version dev --model-id gpt2 \
    --out data/directions/d_dev.npz
```

```
wrote data/directions/d_dev.npz  (dim=768, placeholder)
```

The file is 4736 bytes: the vector, plus a metadata blob carrying `"placeholder": true`.
That flag is what makes the server refuse to treat it as a shipped direction.
`data/directions/d_dev.npz` is listed in `.gitignore` — check with
`git check-ignore -v data/directions/d_dev.npz` — so you cannot commit it by accident.

The four flags above are the real names; confirm with
`python scripts/make_placeholder_direction.py --help`.

---

## Step 4 — score a sequence

```bash
python scripts/score_local.py "be honest and own mistakes" \
    --model gpt2 --layer 6 --d data/directions/d_dev.npz \
    --probes data/probes/season3.json --device cpu --dtype float32
```

```
model=gpt2 layer=6 dtype=float32 probes=16
score (mean cosine steering-shift): -0.001462
note: server (NDIF) score is canonical for the leaderboard; local may differ slightly by precision/hardware.
```

18.8 seconds wall on one CPU. You will also see two lines of transformers noise on stderr —
a `torch_dtype` deprecation notice and a `generation flags are not valid` notice. Both are
harmless; the script passes `torch_dtype` deliberately so it works on transformers 4.x and
5.x alike.

`-0.001462` is your reference number. It is 32 forward passes: 16 probe baselines, 16
composed strings, and the arithmetic in between. The same command in a clean venv on
torch 2.12.0 / numpy 2.2.1 and in the cluster's conda env on torch 2.12.1 / numpy 2.4.6
returns the identical `-0.001462`.

Now break it three times.

---

## Step 5 — break the residual read

The score reads the *output of decoder block `L`*. With `output_hidden_states=True` that is
`hidden_states[L + 1]`, because index 0 is the embedding output. Read `hidden_states[L]` and
you are scoring block `L−1`.

Run both in one process:

```bash
python3 - <<'PY'
import json, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

d = np.asarray(np.load("data/directions/d_dev.npz", allow_pickle=True)["d"], dtype=np.float64)
probes = json.load(open("data/probes/season3.json"))["prompts"]
seq = "be honest and own mistakes"
tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32,
                                             output_hidden_states=True).eval()

def cos(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

@torch.no_grad()
def resid(text, idx=7):
    return model(**tok(text, return_tensors="pt")).hidden_states[idx][0, -1, :].float().numpy()

def score(idx):
    base = {p: cos(resid(p, idx), d) for p in probes}
    return float(np.mean([cos(resid(f"{seq} {p}", idx), d) - base[p] for p in probes]))

right, wrong = score(7), score(6)
print(f"hidden_states[L + 1] (correct): {right:+.8f}")
print(f"hidden_states[L]     (wrong)  : {wrong:+.8f}")
print(f"difference                    : {wrong - right:+.8f}")
PY
```

```
hidden_states[L + 1] (correct): -0.00146172
hidden_states[L]     (wrong)  : -0.00147751
difference                    : -0.00001578
```

Note that the correct value, `-0.00146172`, is Step 4's `-0.001462` — you have reimplemented
the scorer from scratch and matched it.

Now look at what the trap costs you: **1.1%**. The sign is unchanged, the magnitude is
unchanged to three digits, and nothing about `-0.00147751` looks broken. That is the whole
problem with this bug. It does not announce itself; it quietly puts you on a different
layer, and on a real direction at a real layer the error is a rank change on the board rather
than a visibly silly number.

---

## Step 6 — break the composition

Composition is `f"{seq} {probe}"` — sequence, one space, probe. Change the join and you
change the tokenisation:

```bash
python3 - <<'PY'
import json, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

d = np.asarray(np.load("data/directions/d_dev.npz", allow_pickle=True)["d"], dtype=np.float64)
probes = json.load(open("data/probes/season3.json"))["prompts"]
seq = "be honest and own mistakes"
tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32,
                                             output_hidden_states=True).eval()

def cos(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

@torch.no_grad()
def resid(text):
    return model(**tok(text, return_tensors="pt")).hidden_states[7][0, -1, :].float().numpy()

base = {p: cos(resid(p), d) for p in probes}

def score(join):
    return float(np.mean([cos(resid(join(seq, p)), d) - base[p] for p in probes]))

for label, join in [("one space  (correct)", lambda s, p: f"{s} {p}"),
                    ("two spaces (wrong)  ", lambda s, p: f"{s}  {p}"),
                    ("no space   (wrong)  ", lambda s, p: f"{s}{p}"),
                    ("newline    (wrong)  ", lambda s, p: f"{s}\n{p}")]:
    print(f"{label}: {score(join):+.8f}   tokens={len(tok(join(seq, probes[0]))['input_ids'])}")
PY
```

```
one space  (correct): -0.00146172   tokens=13
two spaces (wrong)  : +0.00103860   tokens=14
no space   (wrong)  : -0.01047485   tokens=13
newline    (wrong)  : -0.00421494   tokens=14
```

This trap is not subtle. A second space **flips the sign**. Dropping the space multiplies
the magnitude by 7.2×. The extra space is one extra token in the input; the missing space
keeps the token count but glues `mistakes` to the probe's first word, producing a different
token entirely at the boundary. Both leave the last token — the one you read — sitting in a
different context.

If you are searching against this objective, the join is the first thing to check when your
local number and the board's disagree.

---

## Step 7 — break the cosine's precision

The cosine is computed in float64 whatever dtype the model runs in. Compute it in float32
instead, holding the residuals fixed so precision is the only variable:

```bash
python3 - <<'PY'
import json, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

raw = np.load("data/directions/d_dev.npz", allow_pickle=True)["d"]
probes = json.load(open("data/probes/season3.json"))["prompts"]
seq = "be honest and own mistakes"
tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32,
                                             output_hidden_states=True).eval()

def cos(a, b, dtype):
    a = np.asarray(a, dtype=dtype); b = np.asarray(b, dtype=dtype)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

@torch.no_grad()
def resid(text):
    return model(**tok(text, return_tensors="pt")).hidden_states[7][0, -1, :].float().numpy()

R = {p: resid(p) for p in probes}
R.update({f"{seq} {p}": resid(f"{seq} {p}") for p in probes})

def score(dtype):
    return float(np.mean([cos(R[f"{seq} {p}"], raw, dtype) - cos(R[p], raw, dtype)
                          for p in probes]))

f64, f32 = score(np.float64), score(np.float32)
print(f"cosine in float64 (correct): {f64:+.14f}")
print(f"cosine in float32 (wrong)  : {f32:+.14f}")
print(f"difference                 : {f32 - f64:+.3e}   relative: {(f32 - f64) / abs(f64):.3e}")
PY
```

```
cosine in float64 (correct): -0.00146172478090
cosine in float32 (wrong)  : -0.00146172571658
difference                 : -9.357e-10   relative: -6.401e-07
```

State this one accurately: on `gpt2`, at 768 dimensions, with float32 residuals, the float32
cosine costs you **9.4e-10** — six orders of magnitude smaller than Step 5's error and nine
smaller than Step 6's. You could not detect it by reading the number.

That is exactly why the rule is a rule rather than a judgement call. The error grows with
hidden size and with a lower-precision residual, and the score is a *difference of two
cosines of similar size*, so the accumulated error does not cancel — it is the leading term
of what is left. The scorer casts to float64 unconditionally so this can never become a term
in an argument about why two implementations disagree.

---

## Step 8 — the real model (optional, and it costs something)

Nothing above needed a GPU or a key. Both of these do. Neither is required to have finished
this tutorial.

**Locally, on a GPU.** With no arguments, `score_local.py` uses the live season's tuple —
`allenai/Olmo-3-1125-32B`, layer 24, `data/directions/d_olmo3_L24_logistic.npz`,
`data/probes/season2.json`:

```bash
python scripts/score_local.py "be honest and own mistakes"
```

*Cost:* **61 GB of weights** across 14 shards, downloaded to `$HF_HOME`, and enough VRAM to
hold them in bfloat16. Not run on this page — the machine it was written on reports
`NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver`, which is what
you will see too if you try it without a GPU. On the AICR cluster, get a node first
(`sh ~/gpu.sh`) and check the weights are present by bytes on disk, not by an exit code.

**Through NDIF, the canonical scorer.**

```bash
python -m app.scoring "be honest and own mistakes"
```

*Cost:* an `NDIF_API_KEY` in `.env`, and it **spends the maintainer's shared quota** — one
run of this is a run somebody else does not get. Do not run it to see whether it works. It
also has a hard prerequisite that the error message states plainly; on the Python 3.11 env
used for this page it ends in:

```
ConnectionError: Client python version 3.11 is incompatible with the server. The minimum supported version is 3.12. Please update your python version and try again.
```

NDIF requires **Python >= 3.12** client-side. On the cluster that is the `sa-ndif` env
(Python 3.12.14), not `steering-arena` (3.11.15).

---

## What went wrong for you

**`direction dim 5120 != model hidden size 768 - wrong model for this direction?`**
You pointed a shipped direction at `gpt2`. Reproduce it with
`--d data/directions/d_olmo3_s3_score1.npz --model gpt2`. This is a `SystemExit` raised
*after* the model has loaded, so you wait for the load before you see it. It means the
direction and the model are from different seasons; go back to Step 3 and make a placeholder
at the model's own hidden size. The same guard exists on the server path
(`app/scoring.py`), phrased as `Regenerate the placeholder at --dim 5120 ... or ship a real d
extracted on this model.`

**`[warning] using a PLACEHOLDER direction — scores are not meaningful yet.`**
Exactly what it says: the `.npz` you loaded has `"placeholder": true` in its metadata, so
the number is a shift along a random vector. **`scripts/score_local.py` does not print this
warning** — only `python -m app.scoring` does, from `app/scoring.py:_main`. Reproduce it with
`D_FILE=data/directions/d_dev.npz python -m app.scoring "be honest and own mistakes"` (the
setting is `D_FILE`; there is no env prefix). So on the local path, the absence of a warning
tells you nothing — Step 4 prints no warning and its number is still meaningless. Check the
metadata yourself:
`python3 -c "import json,numpy as np; print(json.loads(str(np.load('data/directions/d_dev.npz',allow_pickle=True)['meta'])))"`.

**`ValueError: cosine of a zero vector is undefined`**
The cosine refuses a zero-norm argument rather than returning 0 or NaN, so a silently empty
read cannot become a plausible-looking score. In practice the zero comes from **the
direction**, not the text: a `.npz` written from an uninitialised or all-zero array. An
*empty sequence* does **not** trigger it — `python scripts/score_local.py "" --model gpt2
--layer 6 --d data/directions/d_dev.npz --probes data/probes/season3.json --device cpu
--dtype float32` returns `+0.001142`, because composition makes the string `" {probe}"`,
which still tokenises to a non-empty sequence. To see the error, save a zero vector and
score against it:

```bash
python3 -c "import numpy as np,json; np.savez('/tmp/d_zero.npz', d=np.zeros(768,dtype=np.float32), meta=np.array(json.dumps({'placeholder':True})))"
python scripts/score_local.py "hi" --model gpt2 --layer 6 --d /tmp/d_zero.npz \
    --probes data/probes/season3.json --device cpu --dtype float32
```

```
  File "/home/padia_so_neu/steering-arena/scripts/score_local.py", line 60, in cosine
    raise ValueError("cosine of a zero vector is undefined")
ValueError: cosine of a zero vector is undefined
```

**`import file mismatch` on nine test files, and `Interrupted: 9 errors during collection`**
You ran bare `pytest`. An untracked duplicate of the whole project sits in `steering_arena/`
(unpacked from `steering_arena.zip`), its test modules have the same basenames as the real
ones, and pytest cannot hold both:

```
import file mismatch:
imported module 'tests.test_scoring_determinism' has this __file__ attribute:
  /home/padia_so_neu/steering-arena/steering_arena/tests/test_scoring_determinism.py
which is not the same as the test file we want to collect:
  /home/padia_so_neu/steering-arena/tests/test_scoring_determinism.py
```

```
!!!!!!!!!!!!!!!!!!! Interrupted: 9 errors during collection !!!!!!!!!!!!!!!!!!!!
============================== 9 errors in 7.05s ===============================
```

Nothing is broken. Pass the directory — `pytest tests/`, or a single file as in Step 2 — and
collection succeeds. Do not "fix" it by deleting `__pycache__`, which is what pytest's own
hint suggests; the duplicate tree is the cause and it is not yours to remove.
