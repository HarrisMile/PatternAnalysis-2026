#!/bin/bash
#SBATCH --job-name=hipmri-prior
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=hipmri_prior_%j.out
#SBATCH --error=hipmri_prior_%j.err

set -euo pipefail

source "$HOME/miniconda3/bin/activate"
conda activate comp3710lab2

PROJECT_DIR="${PROJECT_DIR:-$HOME/PatternAnalysis-2026}"
EXPERIMENT_DIR="$PROJECT_DIR/recognition/hipmri-vqvae"
PRIOR_DATA="$EXPERIMENT_DIR/outputs/prior-data/train_code_indices.npz"
PRIOR_SUMMARY="$EXPERIMENT_DIR/outputs/prior-data/summary.json"
RUN_DIR="$EXPERIMENT_DIR/outputs/formal-prior"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LEARNING_RATE="${LEARNING_RATE:-0.0002}"
SEED="${SEED:-3710}"

cd "$EXPERIMENT_DIR"
export PYTHONUNBUFFERED=1

test -f "$PRIOR_DATA"
test -f "$PRIOR_SUMMARY"

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi
echo "Prior-data archive hash"
sha256sum "$PRIOR_DATA"

TRAIN_ARGS=(
  --data "$PRIOR_DATA"
  --data-summary "$PRIOR_SUMMARY"
  --output "$RUN_DIR"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --learning-rate "$LEARNING_RATE"
  --embedding-dim 64
  --hidden-channels 64
  --residual-blocks 6
  --validation-subjects 6
  --num-workers "$NUM_WORKERS"
  --seed "$SEED"
  --device cuda
)

if [[ -f "$RUN_DIR/last.pt" ]]; then
  echo "Resuming from $RUN_DIR/last.pt"
  TRAIN_ARGS+=(--resume "$RUN_DIR/last.pt")
fi

python train_prior.py "${TRAIN_ARGS[@]}"
cat "$RUN_DIR/summary.json"
echo "Job $SLURM_JOB_ID finished $(date)"
