# Environment reference

Every environment this project runs in, one entry each, then one entry per error string you
are likely to hit. This page is for lookup. For the scoring math these environments have to
agree on, read `docs/reference/scoring.md`.

Run this first. It prints the versions the local GPU work is validated against:

```bash
source ~/startload.sh
conda activate steering-arena
python -c "import torch, transformers, numpy; print(torch.__version__, transformers.__version__, numpy.__version__)"
```

```
2.12.1+cu130 5.10.2 2.4.6
```

Same environment, `python -V` → `Python 3.11.15`, and `nnsight.__version__` → `0.7.0`.

`~/startload.sh` must be **sourced**, not executed. It loads `miniforge3/25.3.0-3`, sets
`HF_HOME=/scratch/$USER/hf`, and sources `~/.hf_token`. It then activates `gpt-trauma`, so
the `conda activate steering-arena` line is not optional.

---

## Entry format

Each entry has the same six fields, in this order: `Definition`, `Value`, `Where it is
set`, `Source of truth`, `Failure mode`, `Positive implication`.

---

## The live Hugging Face Space

**Definition:** the deployed web app. It serves the UI, the API and the canonical scorer.

**Value:** `https://sohampadianeu-steering-arena.hf.space`, Space
`sohampadianeu/steering-arena`. A Docker Space on `python:3.13-slim`, port 7860, with
`HF_HOME=/tmp/hf_home`. Dependencies are pinned: torch **2.12.0** CPU build (installed from
the PyTorch CPU index), transformers **5.10.2**, numpy **2.2.1**, fastapi 0.115.6,
supabase 2.31.0. **No model weights are bundled** — only OLMo-3's config and tokenizer are
fetched at runtime.

**Where it is set:** `Dockerfile` for the image and the CPU torch install,
`requirements.txt` for the pins, `README.md`'s YAML frontmatter (`sdk: docker`,
`app_port: 7860`) for the Space type.

**Source of truth:** `docs/DEPLOY.md`.

**Failure mode:** the Space is a **deployment target, not a mirror**. Its pre-receive hook
walks the entire pushed history and rejects plain binary blobs, so a normal push of `main`
is refused over the `.npz` files that were once committed without LFS. You push one squashed
orphan commit instead. Losing the README frontmatter also breaks the build outright, because
that frontmatter is what makes it a Docker Space.

**Positive implication:** the scorer, `/health` and `/season` all read the active season row
from Supabase, so a season change needs no redeploy. Deploy first, flip the season second —
the reverse order lands rows under the new season carrying the old metric, silently.

---

## A laptop

**Definition:** the environment `PROJECT_SPEC.md` was written for, and the one where "too
big to run locally" is still true.

**Value:** the full web app, the test suite and the API run with no GPU and no NDIF key. The
model does not. `allenai/Olmo-3-1125-32B` is **61 GB across 14 safetensors shards** —
measured with `du -sh` on `$HF_HOME/hub/models--allenai--Olmo-3-1125-32B`.

**Where it is set:** nothing to set. The `Dockerfile` installs the CPU torch build and
bundles no weights, which is the same shape a laptop is in.

**Source of truth:** the byte count on disk, and `Dockerfile`'s comment that scoring runs on
NDIF.

**Failure mode:** reading the spec's "far too big to run locally, therefore NDIF is the
oracle" as a fact about all local hardware. It is true on a laptop and on a free Space, and
**false on AICR**, where 61 GB of weights fit on one card with room to spare. A plan that
routes every forward pass through NDIF because of that sentence spends quota it does not
need to.

**Positive implication:** you can develop and test the entire web app on a laptop — `pytest
tests/` touches no GPU, no NDIF and no network.

---

## AICR conda env `steering-arena`

**Definition:** the local GPU environment. This is where GCG search, layer sweeps and
behavioural generation run.

**Value:** Python 3.11.15, torch **2.12.1+cu130**, transformers **5.10.2**, numpy 2.4.6,
nnsight 0.7.0. The web stack (fastapi, supabase, slowapi, pydantic-settings) is installed
too, because the test suite imports it.

**Where it is set:** `conda activate steering-arena` after `source ~/startload.sh`. Every
script in `slurm/` activates it by name, with a comment saying why it is not `sa-ndif`.

**Source of truth:** the opening command on this page.

**Failure mode:** using `gpt-trauma` because `startload.sh` left you in it. That env's
transformers is 5.12.1, which is a **variable under test**, not a baseline — 5.10.2 is the
validated reference (`_communication/003-2026-09-05-soham-aicr-local-scoring-calibration.md`,
and `/work/neu/p2026_0037_neu/steering-arena/calibration/tokids_transformers-5.12.1.json`
records the separate tokenizer check). Note also that these versions are **not**
`requirements.txt`'s: that pins torch 2.12.0 CPU and numpy 2.2.1 for the Space, and the two
sets are allowed to differ because only transformers and nnsight affect the numbers.

**Positive implication:** 61 GB of weights on one 183 GB B200, one GPU, no NDIF calls and no
quota spent — which is what makes an overnight GCG search free.

---

## AICR conda env `sa-ndif`

**Definition:** the CPU-only environment that makes NDIF calls.

**Value:** Python **3.12.14**, torch 2.12.0+**cpu**, transformers 5.10.2, numpy 2.2.1,
nnsight 0.7.0.

**Where it is set:** `conda activate sa-ndif`. It exists as a second env because the NDIF
client stack needs Python ≥ 3.12 and `steering-arena` is on 3.11.15.

**Source of truth:** `conda env list` plus `importlib.metadata` in the env itself.

**Failure mode:** running a GCG search here. Its torch is a CPU build, so it will not see
the GPU and will not error informatively about why the run is slow. The mirror mistake is
making an NDIF call from `steering-arena`, whose Python is below the client's floor. Every
`slurm/*.sbatch` says `steering-arena, NOT sa-ndif` in a comment for exactly this reason.

**Positive implication:** the two envs agree on transformers 5.10.2 and nnsight 0.7.0, so
the only difference between the local scorer and the served one is the accelerator — not the
library that reads the residual.

---

## NDIF

**Definition:** the hosted model service that produces the canonical score.

**Value:** **canonical for any published number.** Local B200 runs are for search and
sweeps; headline results and any "matched board score" get re-scored on NDIF. Local scoring
is trusted to transfer on the strength of a measured calibration: `|gap|` max **3.71e-4**,
Spearman **ρ = 0.9999999999999999**, over 50 stratified Season-2
submissions.

**Where it is set:** `app/ndif_client.py` for the calls, `app/config.py:ndif_api_key` and
`ndif_timeout_s` (60 s) and `score_concurrency` (2) for the budget.

**Source of truth:**
`/work/neu/p2026_0037_neu/steering-arena/calibration/local_vs_ndif_tf5.10.2_sdpa_695054.json`
for the measurement (`transformers` 5.10.2, `torch` 2.12.1+cu130, `attn` sdpa, `dtype`
bfloat16, `layer` 24, `n` 50), and
`_communication/003-2026-09-05-soham-aicr-local-scoring-calibration.md` for the write-up.

**Failure mode:** publishing a local number as a board score. The separate failure is
assuming the gap is large: `_communication/001` reported ~7e-2, and it **does not
reproduce** — tokenizer version and weight-snapshot revision were each ruled out
separately, and the original gap was an environment difference on the reporter's side. Do
not re-derive a correction factor from that memo.

**Positive implication:** the transfer is measured rather than assumed, with the artifact
committed, so a future drift is something you can detect instead of something you would
argue about.

---

## The tokenizer, and `PREPEND_BOS`

**Definition:** the one piece of model behaviour that has to be identical in every
environment above, because it decides what tokens get scored.

**Value:** OLMo-3 prepends **no BOS**: `bos_token=None`, `add_bos_token=False`. So
`PREPEND_BOS=true` in `.env` is a **no-op**. The tokenizer files ship with the weights
(`tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`) and are the small
public download the Space fetches at runtime.

**Where it is set:** `app/config.py:prepend_bos`, default `True`.

**Source of truth:** the OLMo-3 tokenizer itself.
`/work/neu/p2026_0037_neu/steering-arena/calibration/tokids_transformers-5.10.2.json` and
`…-5.12.1.json` record the two versions' token ids side by side, which is how tokenizer
version was ruled out as the source of the reported local/NDIF gap.

**Failure mode:** "fixing" the flag into something that actually prepends a token. That
changes tokenisation for every submission and invalidates the season — every existing score
was computed without one.

**Positive implication:** the setting is inert, so no environment on this page can disagree
with another about it, and the Space and the cluster tokenise identically at the same
transformers pin.

---

## Storage tiers

**Definition:** which filesystem a given kind of file belongs on. Getting this wrong loses
data.

**Value:** measured with `quota -s` and `df -h` on 2026-09-07.

| Tier | For | State | Caveat |
|---|---|---|---|
| `/home/$USER` | code, configs | 73846M of 95368M used (~77%) | **Never** put `HF_HOME` here |
| `/scratch/$USER` | model weights, `HF_HOME=/scratch/$USER/hf` | 807G of 9314G used | **purged after 30 days by mtime**, no snapshots |
| `/work/neu/p2026_0037_neu/steering-arena/` | run outputs | 354G of 1.0T used (project-wide) | snapshotted; shared with the trauma project |

`/work/neu/p2026_0037_neu/steering-arena/` holds `bin/`, `cache/`, `calibration/`, `gcg/`,
`logs/`, `runs/`.

**Where it is set:** `HF_HOME` by `~/startload.sh` and re-exported by every
`slurm/*.sbatch`; the `--output` and `--error` paths in each sbatch point at
`/work/neu/p2026_0037_neu/steering-arena/logs/`.

**Source of truth:** `quota -s`.

**Failure mode:** a run whose outputs land on `/scratch` looks fine and is gone 30 days
later, by mtime, with no warning. The reverse mistake fills `/home`, which is at 77% of a
93 GiB quota and cannot hold 61 GB of weights at all.

**Positive implication:** re-downloading purged weights is routine and costs only time,
because nothing irreplaceable is ever on `/scratch` — the outputs the internet cannot give
back are on the snapshotted tier.

---

## GPU partitions and the 2-GPU cap

**Definition:** which partition to ask for, and what limits you against.

**Value:** measured with `sinfo` and `sacctmgr show qos`.

| Partition | GPUs per node | Time limit | QOS `MaxTRESPU` |
|---|---|---|---|
| `b200-devel` | 8 × b200 | 4:00:00 | `gres/gpu=2` |
| `rtx-devel` | 8 × rtx_pro_6000 | 4:00:00 | `gres/gpu=2` |
| `b200-batch` | 8 × b200 | 1-00:00:00 | `gres/gpu=32` |
| `rtx-batch` | 8 × rtx_pro_6000 | 1-00:00:00 | `gres/gpu=32` |
| `cpu` | none | 1-00:00:00 | `cpu=64` |

A B200 has 183 GB and an RTX PRO 6000 has 96 GB, so OLMo-3-32B's 61 GB fits on **one** card
of either type. Account `p2026_0037_neu`.

**Where it is set:** `#SBATCH --partition` in `slurm/gcg.sbatch` (`b200-batch`),
`slurm/gcg_score2.sbatch` (`b200-batch`), `slurm/prefix_behavior_s3.sbatch` (`b200-devel`).
All three request `--gpus=1 --cpus-per-task=8 --mem=192G --time=04:00:00`.

**Source of truth:** `sacctmgr -nP show qos format=Name,MaxTRESPU` for the caps.
`slurm/prefix_behavior_s3.sbatch`'s header comment for the reason it picks `b200-devel`.

**Failure mode:** treating the 2-GPU cap as global and serialising your jobs. It is **per
QOS, and each partition has its own QOS**, so two jobs on two devel partitions do not
compete. The real congestion risk is the other direction: `b200-batch` has been observed
with ~2400 jobs pending, which makes its 24-hour limit useless in practice.

**Positive implication:** a `b200-devel` job does not queue behind an `rtx-devel` one, so
you can run a GCG search and a behavioural eval at the same time on separate partitions.

---

## The run ledger

**Definition:** the append-only record of why each job was run.

**Value:** `/work/neu/p2026_0037_neu/runlog/commands.jsonl`, written through
`~/trauma-experiment-gpt/bin/runlog.py`. `add --why` before the job, `note` with the outcome
after.

**Where it is set:** that path — **not** `~/bin/`, which holds `gh`.

**Source of truth:** the JSONL file.

**Failure mode:** relying on `sacct`. It records that a job ran, its exit code and its
runtime; it does not record what question the job was supposed to answer, and three weeks
later that is the only thing you need.

**Positive implication:** the ledger is one file in a snapshotted tier, so every job in this
project's history is greppable by intent.

---

## `CLAUDE.md` — gitignored and local-only

**Definition:** the two operating-instruction files this project's work depends on, neither
of which a reader who clones the repo will have.

**Value:** `/home/padia_so_neu/steering-arena/CLAUDE.md` is **gitignored** —
`.gitignore:3` lists `CLAUDE.md` and `.gitignore:2` lists `.claude/`, and `git status
--porcelain --ignored CLAUDE.md` reports `!! CLAUDE.md`. `/home/padia_so_neu/CLAUDE.md` is
not in the repository at all; `git check-ignore` on it returns `fatal: … is outside
repository`.

**Where it is set:** `.gitignore` lines 1-3, under the comment
`# --- Claude Code files/folders ---`.

**Source of truth:** `git check-ignore -v CLAUDE.md .claude/`.

**Failure mode:** writing a tracked document that says "see CLAUDE.md for the cluster
setup". For anyone outside this machine that reference resolves to nothing. The same applies
to `.claude/skills/`, which is ignored wholesale, so a tracked page must restate what a
skill enforces rather than point at it.

**Positive implication:** `docs/` is tracked, so a fact written here survives the clone —
which is the whole reason this page exists.

---

## `QOSMaxGRESPerUser`

**Definition:** a Slurm pending reason, shown in `squeue`'s `NODELIST(REASON)` column.

**Value:** your queued job is holding at the QOS GPU cap, which is `gres/gpu=2` on both
devel partitions.

**Where it is set:** the QOS attached to the partition you submitted to.

**Source of truth:** `sacctmgr -nP show qos format=Name,MaxTRESPU`.

**Failure mode:** waiting it out on the assumption that the cluster is busy. It is not a
capacity problem — it is your own two running jobs. `docs/HANDOFF_BEHAVIORAL_S3.md` records
two jobs pending on it at once. Submit the third job to a *different* partition instead: the
cap is per QOS and each partition has its own, so moving a job from `b200-devel` to
`rtx-devel` clears it immediately.

**Positive implication:** the reason string names the exact limit, so you never have to
guess whether you are blocked by policy or by hardware.

---

## `set -u` kills the sbatch shell

**Definition:** an sbatch job that exits immediately, with an empty or near-empty `.out`
file and no Python traceback.

**Value:** **never `set -u` in an sbatch script on this cluster.** Use `set -eo pipefail`.

**Where it is set:** all three scripts in `slurm/` use `set -eo pipefail` and carry the
comment `NOTE: never set -u here — it kills the batch shell on this cluster` immediately
above it.

**Source of truth:** those three files.

**Failure mode:** the batch shell is initialised with variables `set -u` considers unbound,
so the script dies before your first command runs. There is no error you can attribute to
your own code, which is what makes it expensive to diagnose. Delete `set -u` and guard
optional arguments explicitly instead — `NTOK="${1:-32}"`, as `slurm/gcg.sbatch` does.

**Positive implication:** `set -eo pipefail` still gives you fail-fast on any real command
error, so you lose nothing but the unbound-variable check.

---

## `hf download --exclude` silently downloads nothing

**Definition:** a model download that exits 0 and leaves you with no weights.

**Value:** `hf download --exclude "*.bin" "*.pt"` parses the second and later patterns as
**positional filenames**, so it downloads those two "files", finds nothing, and **exits 0**.

**Where it is set:** the `hf` CLI's argument parser. `huggingface-cli` is dead; use `hf`.

**Source of truth:** the bytes on disk. `du -sh
/scratch/$USER/hf/hub/models--allenai--Olmo-3-1125-32B` reads **61G** for a complete
OLMo-3-32B, across `model-00001-of-00014.safetensors` … `model-00014-of-00014.safetensors`.

**Failure mode:** trusting the exit code. The command succeeds, the job then fails much
later inside `from_pretrained`, and the error blames the model rather than the download.
Pass one `--exclude` flag per pattern, and **verify every download by bytes on disk**, never
by exit status.

**Positive implication:** the check is cheap and unambiguous — 61 GB and 14 shards, or the
download did not happen.

---

## `import file mismatch` from bare `pytest`

**Definition:** pytest collection errors that name files under `steering_arena/`.

**Value:** run **`pytest tests/`**, not bare `pytest`. Measured 2026-09-07:

```
$ python -m pytest tests/ --collect-only -q
114 tests collected in 1.52s

$ python -m pytest --collect-only -q
57 tests collected, 9 errors in 7.97s
```

The error text:

```
import file mismatch:
imported module 'tests.test_captcha' has this __file__ attribute:
  /home/padia_so_neu/steering-arena/steering_arena/tests/test_captcha.py
which is not the same as the test file we want to collect:
  /home/padia_so_neu/steering-arena/tests/test_captcha.py
```

**Where it is set:** there is no `pyproject.toml` or `pytest.ini`, so rootdir collection
sweeps the whole tree. `steering_arena/` is an untracked duplicate of the entire project,
unpacked from `steering_arena.zip`; both are gitignored (`.gitignore:73-74`).

**Source of truth:** the collection counts above.

**Failure mode:** two test modules with the same basename and no package boundary between
them. Bare `pytest` interrupts on 9 errors and collects 57 of the 114 tests, so a green-ish
partial run can look like a pass. Always scope to `tests/`, and note that the count is
**114**, not the 84 some older notes give.

**Positive implication:** the scoped invocation is fast (1.52 s to collect) and hermetic —
none of the 114 tests touches NDIF, a GPU or the network.

---

## `ValueError: cosine of a zero vector is undefined`

**Definition:** the scorer refusing to score an empty residual.

**Value:** raised, not returned as 0 or NaN.

**Where it is set:** three places that must agree — `app/scoring.py:61`,
`scripts/score_local.py:60`, `scripts/gcg/gcg_utils.py:95`.

**Source of truth:** `app/scoring.py:cosine`.

**Failure mode:** your residual read is empty. In practice that is an empty sequence, or a
tokenizer that produced no ids, reaching the cosine with a zero-length last token. Check
what you passed in before you touch the scorer — the exception is about the input, not about
the metric.

**Positive implication:** it fails loudly. A silent 0 would enter the mean as a real score
and move a leaderboard rank.

---

## What this buys you

Local search transfers to the canonical scorer — measured at `|gap|` ≤ 3.71e-4 and Spearman
ρ = 0.9999999999999999 over n=50, with the artifact recorded rather than asserted. That is what
makes a $0 GPU search usable at all: you can burn a university allocation overnight on
`b200-batch`, and the string that wins there is the string that wins on the board. Every
version, quota and cap above came from a command on this page, so a drift is something you
can measure instead of something you discover in a result.
