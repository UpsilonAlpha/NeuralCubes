#!/bin/bash
# Train several targets on one machine, spreading them over its GPUs (one target per GPU at
# a time). For SLURM clusters use cluster/train.sbatch instead.
#
#   training/train_all.sh                                  # every preset, all visible GPUs
#   TARGETS="tree cactus" ROTATION=full training/train_all.sh
#
# Settings (environment): TARGETS, SIZE (32), ROTATION (yaw), ITERATIONS (8000), OUT (runs),
# GPUS (default: every GPU nvidia-smi lists). Rerunning resumes from the checkpoints in OUT.

set -euo pipefail
cd "$(dirname "$0")/.."

TARGETS=(${TARGETS:-bunny tree mushroom cactus torus sphere})
SIZE=${SIZE:-32}
ROTATION=${ROTATION:-yaw}
ITERATIONS=${ITERATIONS:-8000}
OUT=${OUT:-runs}
GPUS=(${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || echo 0)})
mkdir -p "$OUT" logs

train() {  # target gpu
  CUDA_VISIBLE_DEVICES=$2 python training/train_nca.py "$1" \
    --size "$SIZE" --rotation "$ROTATION" --iterations "$ITERATIONS" \
    --batch 8 --damage 3 --hidden 128 --min-steps 64 --max-steps 96 \
    --checkpoint "$OUT/$1.ckpt" --out "$OUT/$1.json" --export-every 200 \
    > "logs/$1.log" 2>&1
}

# Each GPU works through its share of the targets in turn.
for g in "${!GPUS[@]}"; do
  (
    for ((i = g; i < ${#TARGETS[@]}; i += ${#GPUS[@]})); do
      echo "GPU ${GPUS[$g]}: ${TARGETS[$i]} (log: logs/${TARGETS[$i]}.log)"
      train "${TARGETS[$i]}" "${GPUS[$g]}" || echo "${TARGETS[$i]} failed, see logs/${TARGETS[$i]}.log"
    done
  ) &
done
wait
echo "done; weights are in $OUT/"
