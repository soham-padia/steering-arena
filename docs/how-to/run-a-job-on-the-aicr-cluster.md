# How to run a Steering Arena job on the AICR cluster

Getting a GPU job — a GCG search, a prefix eval, a rescore — from a login shell onto a node,
with its output somewhere that survives and its purpose written down. Every step here has a
failure mode that has already cost a run.

**Prerequisites:** a login shell on `login.aicr.ai`, account `p2026_0037_neu`, and the
`steering-arena` conda env. No step needs a GPU on the login node; the model loads on the
compute node.

```bash
source ~/startload.sh                 # miniforge + HF_HOME=/scratch/$USER/hf
conda activate steering-arena         # NOT gpt-trauma, NOT sa-ndif
python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
python ~/trauma-experiment-gpt/bin/runlog.py add \
  --cmd "sbatch -p rtx-devel slurm/gcg.sbatch score1 32" --why "positive control"
sbatch -p rtx-devel slurm/gcg.sbatch score1 32
```

## 1. Load the environment and print the versions

```bash
source ~/startload.sh
conda activate steering-arena
python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
```

```
2.12.1+cu130 5.10.2
```

Those two numbers are the contract. `5.10.2` is the validated local reference against NDIF;
the `gpt-trauma` env's transformers is a different version and is not the baseline. `sa-ndif`
is CPU-only torch and cannot run a local search. See `docs/reference/environments.md` for
which env holds what.

`startload.sh` must be **sourced**, not executed, and it prints a warning if `$HF_HOME/hub`
is missing.

## 2. Choose the partition at submit time

Both sbatch scripts carry a default partition in their header, and you override it on the
command line:

```bash
sbatch -p rtx-devel  -J gcg-s2 slurm/gcg.sbatch score2 32
sbatch -p b200-devel          slurm/prefix_behavior_s3.sbatch --limit 3
```

- **`rtx-devel`** starts fast. Two nodes, a 4-hour wall, ~20 jobs pending.
- **`b200-devel`** is a **separate QOS pool** with its own per-user GPU cap, so a job here
  does **not** queue behind a job you already have running on `rtx-devel`. This is the reason
  `slurm/prefix_behavior_s3.sbatch` pins `b200-devel` in a comment: it was written to run
  alongside a GCG search. 183 GB per card, which holds the 61 GB of OLMo-3 weights with room
  to spare.
- **`b200-batch`** currently has **2243 jobs pending** and is effectively unusable. Do not
  submit there and wait.

The devel wall is 4 hours and it is enforced: job `707083` on `rtx-devel` shows `TIMEOUT` at
`04:00:26`. A search that needs longer must checkpoint and resume, not ask for more time.

## 3. Log the job before you submit it

```bash
python ~/trauma-experiment-gpt/bin/runlog.py add \
  --cmd "sbatch -p rtx-devel slurm/gcg.sbatch score2 32 --anti" \
  --why "does the anti arm find a mirror of the score2 winner, or does it loop"
```

The file is mode `644`, so invoke it through `python` — running the path directly gives
`Permission denied`. It appends to `/work/neu/p2026_0037_neu/runlog/commands.jsonl`.

## 4. Submit, then watch the queue

```bash
sbatch -p rtx-devel -J gcg-s2-anti slurm/gcg.sbatch score2 32 --anti
squeue -u $USER -o "%.10i %.12P %.10j %.8T %.14r %.10M"
```

The first argument to `slurm/gcg.sbatch` is the role (`score1` or `score2`), the second the
controlled-token count, and anything after that passes through to
`scripts/gcg/optimize_banded.py`.

## 5. Read the output from `/work`, not from `/home`

Both scripts write their stdout and stderr to
`/work/neu/p2026_0037_neu/steering-arena/logs/<name>_<jobid>.{out,err}`, and run artifacts to
`runs/`, `gcg/`, `cache/` and `calibration/` under the same directory. That tree is on the
1.0 TB snapshotted project quota.

```bash
head -3 /work/neu/p2026_0037_neu/steering-arena/logs/pfx_s3_711936.out
```

```
node=b0030 job=711936 extra=
NVIDIA B200, 183359 MiB
torch 2.12.1+cu130 transformers 5.10.2
```

Those three lines are the convention: which node, which GPU, which library versions. Check
them before you believe anything further down the log.

Weights are separate. Both scripts `export HF_HOME=/scratch/$USER/hf`, and `/scratch` is
**purged after 30 days by mtime with no snapshots**. Re-downloading 61 GB of OLMo-3 is routine
and is not a data loss; putting a run output there would be.

```bash
du -sh /scratch/$USER/hf/hub
```

## 6. Close the runlog entry with the outcome

```bash
sacct -j 711936 --format=JobID%12,JobName%12,Partition%12,Elapsed%10,State%12,ExitCode -X
python ~/trauma-experiment-gpt/bin/runlog.py note --job 711936 \
  --outcome "COMPLETED in 2m13s on b200-devel; 8 arms re-scored, all ok, max gap 7.0e-4"
```

`sacct` records that a job ran, its state and its wall time. It does not record why you ran
it or whether the result was usable. The `note` is the only place that exists.

## What goes wrong

**`QOSMaxGRESPerUser` as the pending reason.** `squeue` shows the job `PENDING` with that
reason and it does not move. It means you are at the 2-GPU cap for that partition's QOS, not
that the cluster is full. The caps are per-QOS and `rtx-devel` and `b200-devel` are separate
pools, so resubmit to the other partition rather than waiting:

```bash
scancel <jobid> && sbatch -p b200-devel slurm/gcg.sbatch score2 32
```

**`set -u` in an sbatch script.** It kills the batch shell on this cluster — the job exits in
seconds with an empty or near-empty `.out` file and no Python traceback, because the shell
died before it reached the `python` line. Both scripts use `set -eo pipefail` and carry a
comment saying so. Do not add `-u`.

**`hf download --exclude "*.bin" "*.pt"` downloading nothing.** `hf` parses the second and
later patterns as **positional filenames**, downloads nothing, and **exits 0**. There is no
error to catch. Verify every download by bytes on disk:

```bash
du -sh /scratch/$USER/hf/hub/models--allenai--Olmo-3-1125-32B
```

Expect ~61 GB across 14 shards. `huggingface-cli` no longer exists; use `hf`.

**`/home` filling up.** The quota is ~93 GiB and ~72 GiB is already used:

```bash
quota -s | head -5
```

A `Disk quota exceeded` from a job that was writing to `$HOME` leaves a truncated artifact
that looks like a successful short run. Nothing heavy goes in `/home` — checkpoints, caches
and generations go to `/work`, weights to `/scratch`.

**A purged `/scratch`.** The symptom is `startload.sh` warning that `$HF_HOME/hub` is missing,
and then a job failing at model load rather than at submit. The 30-day purge is by **mtime**,
so weights you have not read in a month are gone even though the directory was there last
time you looked. Re-download and verify by bytes.

## Verify it yourself

The prefix eval has a 3-prompt smoke mode. It loads the real model, so it exercises the env,
the partition, `$HF_HOME` and the output path in one submission:

```bash
sbatch -p b200-devel slurm/prefix_behavior_s3.sbatch --limit 3
squeue -u $USER -o "%.10i %.12P %.8T %.14r"
head -3 /work/neu/p2026_0037_neu/steering-arena/logs/pfx_s3_<jobid>.out
```

If those three header lines report a node, a GPU with its memory, and
`torch 2.12.1+cu130 transformers 5.10.2`, the whole path works and a full run will too.

## Cross-links

`docs/reference/environments.md` (which env, which versions, which storage tier) ·
`slurm/gcg.sbatch` (the GCG runner and its role/token arguments) ·
`slurm/prefix_behavior_s3.sbatch` (two-phase job, and the `b200-devel` QOS note) ·
`docs/how-to/search-a-banded-objective-with-gcg.md` (what to run once the job starts).
