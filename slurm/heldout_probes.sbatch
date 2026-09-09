#!/bin/bash
# Probe-set generalisation: do the Season 3 strings still score on probes they never saw?
#
#   sbatch slurm/heldout_probes.sbatch
#
# Every score this project has published is measured on data/probes/season3.json, and GCG
# optimised against those same 16 prompts, so every number is IN-SAMPLE by construction.
# Jesse Li asked the obvious question after taking both Season 2 headline slots:
# "Steering Arena only tests the results on 16 prompts. How overfitted is the prompt prefix
# to those exact prompts?" Nobody has answered it. This does.
#
# Two steps, one GPU, both cheap:
#   1. baseline_const.py on the held-out set -- a probe set carries its OWN baseline (the
#      probes' own alignment with d), so reusing season3's constant would offset every score.
#   2. score_banded_local.py with both the held-out probes and the held-out baseline.
#
# The fidelity gate is skipped on purpose: against a different probe set a gap from the
# recorded score IS the result, not a corrupted string.
#
#SBATCH --job-name=heldout-probes
#SBATCH --account=p2026_0037_neu
#SBATCH --partition=b200-devel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=01:00:00
#SBATCH --output=/work/neu/p2026_0037_neu/steering-arena/logs/heldout_%j.out
#SBATCH --error=/work/neu/p2026_0037_neu/steering-arena/logs/heldout_%j.err
# NOTE: never `set -u` here — it kills the batch shell on this cluster.
set -eo pipefail

module load miniforge3/25.3.0-3
eval "$(conda shell.bash hook)"
conda activate steering-arena
export HF_HOME=/scratch/$USER/hf
[ -f ~/.hf_token ] && source ~/.hf_token

echo "node=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch,transformers;print('torch',torch.__version__,'transformers',transformers.__version__)"

cd "$HOME/steering-arena"

echo "=== 1/3  baseline for the HELD-OUT probe set ==="
python -u scripts/gcg/baseline_const.py \
  --probes data/probes/heldout_v1.json \
  --out    data/analysis/heldout_baseline.json

echo "=== 2/3  re-score the arms on the SEASON 3 probes (control: must reproduce) ==="
python -u scripts/score_banded_local.py \
  --arms data/analysis/prefix_eval_arms_s3.json \
  --out  data/analysis/heldout_probe_scores_insample.json

echo "=== 3/3  score the same arms on the HELD-OUT probes ==="
python -u scripts/score_banded_local.py \
  --arms      data/analysis/prefix_eval_arms_s3.json \
  --probes    data/probes/heldout_v1.json \
  --baselines data/analysis/heldout_baseline.json \
  --out       data/analysis/heldout_probe_scores.json

echo "done."
