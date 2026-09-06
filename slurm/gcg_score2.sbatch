#!/bin/bash
# GCG against Season 3's SCORE 2 (per-layer min over 15/23/31/39) -- the hard objective.
#
# Run this one FIRST. Score 1 correlates with the old single-layer metric at rho=0.95 with
# the same optimised string still at rank 1, so beating it would mostly re-confirm what is
# already known. Score 2 is the one that separates and the one nobody has searched against.
# A failure to beat it is a result.
#
#   sbatch slurm/gcg_score2.sbatch [n_controlled_tokens] [resume_dir]
#
#SBATCH --job-name=gcg-s2
#SBATCH --account=p2026_0037_neu
#SBATCH --partition=b200-batch
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=/work/neu/p2026_0037_neu/steering-arena/logs/gcg_s2_%j.out
#SBATCH --error=/work/neu/p2026_0037_neu/steering-arena/logs/gcg_s2_%j.err
# NOTE: never `set -u` here — it kills the batch shell on this cluster.
set -eo pipefail

module load miniforge3/25.3.0-3
eval "$(conda shell.bash hook)"
# steering-arena, NOT sa-ndif: sa-ndif is CPU-only torch for NDIF calls. This is pure
# local GPU against the weights in $HF_HOME — no NDIF, no quota spent.
conda activate steering-arena
export HF_HOME=/scratch/$USER/hf
[ -f ~/.hf_token ] && source ~/.hf_token

NTOK="${1:-32}"
RESUME="${2:-}"

echo "node=$(hostname) job=$SLURM_JOB_ID tokens=$NTOK"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch,transformers;print('torch',torch.__version__,'transformers',transformers.__version__)"

cd "$HOME/steering-arena"
if [ -n "$RESUME" ]; then
  python -u scripts/gcg/optimize_banded.py --role score2 --n-controlled-tokens "$NTOK" --resume-from "$RESUME"
else
  python -u scripts/gcg/optimize_banded.py --role score2 --n-controlled-tokens "$NTOK"
fi
