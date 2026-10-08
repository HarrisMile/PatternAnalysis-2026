#!/bin/bash
#SBATCH --job-name=hipmri-vae
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=hipmri_vae_%j.out
#SBATCH --error=hipmri_vae_%j.err

set -euo pipefail

source "$HOME/miniconda3/bin/activate"
conda activate comp3710lab2

PROJECT_DIR="${PROJECT_DIR:-$HOME/PatternAnalysis-2026}"
EXPERIMENT_DIR="$PROJECT_DIR/recognition/hipmri-vqvae"
DATA_ROOT="${HIPMRI_ROOT:-/home/groups/comp3710/HipMRI_Study_open/keras_slices_data}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LEARNING_RATE="${LEARNING_RATE:-0.0002}"
SEED="${SEED:-3710}"
MANIFEST="$EXPERIMENT_DIR/data/manifest.csv"
RUN_DIR="$EXPERIMENT_DIR/outputs/formal-vae"

cd "$EXPERIMENT_DIR"
export PYTHONUNBUFFERED=1

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi
python dataset.py --data-root "$DATA_ROOT" --output "$MANIFEST"

TRAIN_ARGS=(
  --model vae
  --manifest "$MANIFEST"
  --output "$RUN_DIR"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --learning-rate "$LEARNING_RATE"
  --num-workers "$NUM_WORKERS"
  --seed "$SEED"
  --device cuda
)

if [[ -f "$RUN_DIR/last.pt" ]]; then
  echo "Resuming from $RUN_DIR/last.pt"
  TRAIN_ARGS+=(--resume "$RUN_DIR/last.pt")
fi

python train.py "${TRAIN_ARGS[@]}"

python predict.py \
  --checkpoint "$RUN_DIR/best.pt" \
  --manifest "$MANIFEST" \
  --split validation \
  --output predictions/formal-vae-validation \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --num-visualisations 6 \
  --device cuda \
  --overwrite

echo "Job $SLURM_JOB_ID finished $(date)"
