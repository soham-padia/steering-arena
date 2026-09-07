#!/bin/bash
# Part A': do the Season 3 GCG winners change what the model SAYS, or only where its
# activations point? Two phases, one model load each.
#
#   sbatch slurm/prefix_behavior_s3.sbatch              # all 50 eval prompts
#   sbatch slurm/prefix_behavior_s3.sbatch --limit 3    # smoke: 3 prompts
#
# PHASE 1 verifies the arms before spending anything on them: re-score every frozen
# string under both banded objectives and check it reproduces the number the run
# recorded. A string that scores differently is a CORRUPTED string (the id=1342 repr
# bug), and it also fills in the random-token control, which has no recorded score.
# PHASE 2 generates the continuations, 50 prompts x 6 arms + base, greedy, 40 tokens.
#
# b200-devel, NOT rtx-devel: the 2-GPU cap is per-QOS and each partition has its own, so
# this does not queue behind the GCG search running on rtx-devel. 61 GB of weights on a
# 183 GB card, one GPU, no NDIF calls and no quota spent.
#
#SBATCH --job-name=pfx-s3
#SBATCH --account=p2026_0037_neu
#SBATCH --partition=b200-devel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=/work/neu/p2026_0037_neu/steering-arena/logs/pfx_s3_%j.out
#SBATCH --error=/work/neu/p2026_0037_neu/steering-arena/logs/pfx_s3_%j.err
# NOTE: never `set -u` here — it kills the batch shell on this cluster.
set -eo pipefail

module load miniforge3/25.3.0-3
eval "$(conda shell.bash hook)"
conda activate steering-arena
export HF_HOME=/scratch/$USER/hf
[ -f ~/.hf_token ] && source ~/.hf_token

ARMS=data/analysis/prefix_eval_arms_s3.json

echo "node=$(hostname) job=$SLURM_JOB_ID extra=$*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch,transformers;print('torch',torch.__version__,'transformers',transformers.__version__)"

cd "$HOME/steering-arena"

echo "=== PHASE 1: re-score the frozen arms (fidelity audit + the random control) ==="
python -u scripts/score_banded_local.py --arms "$ARMS" --update-arms

echo "=== PHASE 2: generate continuations (resumable; re-run to fill gaps) ==="
python -u scripts/prefix_behavior_eval.py --tag s3 generate --backend local --max-new 40 "$@"

echo "=== next (no GPU needed) ==="
echo "  python scripts/prefix_behavior_eval.py --tag s3 blind"
echo "  python scripts/prefix_behavior_eval.py --tag s3 claude-batches"
