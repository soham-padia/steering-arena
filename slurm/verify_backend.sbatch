#!/bin/bash
# Equivalence gate for the scoring-backend refactor.
#
#   sbatch slurm/verify_backend.sbatch
#
# score_banded_local.py used to score through a torch implementation adapted from a third
# party. It now scores through app/scoring.py -- the same pure functions the server uses --
# with residuals from ResidualReader on the local backend. Same metric, different code.
#
# This re-scores the eight Season 3 arms and compares against the values recorded by the
# runs. It MUST reproduce within TOL = 2e-3; the previous (adapted) path achieved max |gap|
# 7.0e-04, so anything materially worse means the two paths are not equivalent and nothing
# should be deleted until that is understood.
#
# Two differences from the old path are expected and benign: left-padding instead of a
# right-padded explicit gather (both take the last real token), and no truncate_to_layer
# (a full forward instead of one cut at max(band)+2 -- slower, identical residuals).
#
#SBATCH --job-name=verify-backend
#SBATCH --account=p2026_0037_neu
#SBATCH --partition=b200-devel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=01:00:00
#SBATCH --output=/work/neu/p2026_0037_neu/steering-arena/logs/verify_%j.out
#SBATCH --error=/work/neu/p2026_0037_neu/steering-arena/logs/verify_%j.err
# NOTE: never `set -u` here — it kills the batch shell on this cluster.
set -eo pipefail

module load miniforge3/25.3.0-3
eval "$(conda shell.bash hook)"
conda activate steering-arena
export HF_HOME=/scratch/$USER/hf
[ -f ~/.hf_token ] && source ~/.hf_token

echo "node=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch,transformers,nnsight;print('torch',torch.__version__,'transformers',transformers.__version__,'nnsight',nnsight.__version__)"

cd "$HOME/steering-arena"

echo "=== re-score the 8 arms through app/scoring.py + ResidualReader(local) ==="
python -u scripts/score_banded_local.py \
  --arms data/analysis/prefix_eval_arms_s3.json \
  --out  data/analysis/backend_equivalence_check.json

echo "=== verdict ==="
python - <<'EOF'
import json, pathlib
new = json.load(open("data/analysis/backend_equivalence_check.json"))["scores"]
old = json.load(open("data/analysis/heldout_probe_scores_insample.json"))["scores"]
worst_rec, worst_old = 0.0, 0.0
print(f"  {'arm':<18}{'new':>11}{'old path':>11}{'vs recorded':>13}{'vs old path':>13}")
for a, r in new.items():
    k = f"{(r.get('role') or 'score1')}_live"
    g_rec = abs(r.get("gap", 0.0) or 0.0)
    g_old = abs(r[k] - old[a][k])
    worst_rec, worst_old = max(worst_rec, g_rec), max(worst_old, g_old)
    print(f"  {a:<18}{r[k]:>+11.5f}{old[a][k]:>+11.5f}{g_rec:>13.2e}{g_old:>13.2e}")
print(f"\n  worst |gap| vs recorded : {worst_rec:.2e}   (TOL 2.0e-03, old path 7.0e-04)")
print(f"  worst |gap| vs old path : {worst_old:.2e}")
print("\n  VERDICT:", "PASS" if worst_rec <= 2e-3 else "FAIL — do not delete anything")
EOF
