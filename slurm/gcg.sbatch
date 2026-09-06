#!/bin/bash
# GCG against a Season 3 banded objective. Generic runner.
#
#   sbatch slurm/gcg.sbatch score2 32                    # the hard objective (hard min)
#   sbatch slurm/gcg.sbatch score2 32 --softmin-search   # same, smooth search surrogate
#   sbatch slurm/gcg.sbatch score1 32                    # positive control (banded mean)
#   sbatch slurm/gcg.sbatch score2 32 --anti             # ANTI-human: most negative score2
#
# Score 2 is the real question: Score 1 correlates with the old single-layer metric at
# rho=0.95 with the same optimised string still rank 1, so beating it mostly re-confirms
# what is known. A FAILURE to beat Score 2 is as much a result as beating it.
#
#SBATCH --job-name=gcg
#SBATCH --account=p2026_0037_neu
#SBATCH --partition=b200-batch
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=/work/neu/p2026_0037_neu/steering-arena/logs/gcg_%j.out
#SBATCH --error=/work/neu/p2026_0037_neu/steering-arena/logs/gcg_%j.err
# NOTE: never `set -u` here — it kills the batch shell on this cluster.
set -eo pipefail

module load miniforge3/25.3.0-3
eval "$(conda shell.bash hook)"
# steering-arena, NOT sa-ndif: sa-ndif is CPU-only torch for NDIF. This is pure local GPU
# against the weights in $HF_HOME — no NDIF calls, no quota spent.
conda activate steering-arena
export HF_HOME=/scratch/$USER/hf
[ -f ~/.hf_token ] && source ~/.hf_token

ROLE="${1:?usage: sbatch slurm/gcg.sbatch <score1|score2> [ntok] [extra flags...]}"
NTOK="${2:-32}"
shift 2 || true

echo "node=$(hostname) job=$SLURM_JOB_ID role=$ROLE tokens=$NTOK extra=$*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch,transformers;print('torch',torch.__version__,'transformers',transformers.__version__)"

cd "$HOME/steering-arena"
python -u scripts/gcg/optimize_banded.py --role "$ROLE" --n-controlled-tokens "$NTOK" "$@"
