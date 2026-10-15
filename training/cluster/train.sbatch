#!/bin/bash
# Train one neural CA per target on a SLURM cluster, one GPU each.
#
#   sbatch --array=0-4 training/cluster/train.sbatch        # every preset below
#   TARGETS="tree" ROTATION=full sbatch --array=0 training/cluster/train.sbatch
#
# Edit the #SBATCH lines for your cluster (partition, account, time limit). Jobs write
# checkpoints, so a preempted or timed-out job picks up where it stopped when resubmitted
# (--requeue does that automatically where the cluster allows it).
#SBATCH --job-name=nca
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

TARGETS=(${TARGETS:-bunny tree mushroom cactus torus})
TARGET=${TARGETS[$SLURM_ARRAY_TASK_ID]}
SIZE=${SIZE:-32}
ROTATION=${ROTATION:-yaw}
ITERATIONS=${ITERATIONS:-8000}
OUT=${OUT:-runs}

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p "$OUT" logs

# One-time setup on the cluster (or point VENV at an existing environment):
#   python -m venv ~/venvs/nca && ~/venvs/nca/bin/pip install -r training/requirements.txt "jax[cuda12]"
source "${VENV:-$HOME/venvs/nca}/bin/activate"

python training/train_nca.py "$TARGET" \
  --size "$SIZE" --rotation "$ROTATION" --iterations "$ITERATIONS" \
  --batch 8 --damage 3 --hidden 128 --min-steps 48 --max-steps 96 \
  --checkpoint "$OUT/$TARGET.ckpt" --out "$OUT/$TARGET.json" --export-every 200
