#!/bin/bash
#SBATCH --job-name=hipmri-vqvae-v2
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:30:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=hipmri_vqvae_v2_%j.out
#SBATCH --error=hipmri_vqvae_v2_%j.err

set -euo pipefail

source "$HOME/miniconda3/bin/activate"
conda activate comp3710lab2

PROJECT_DIR="${PROJECT_DIR:-$HOME/PatternAnalysis-2026}"
EXPERIMENT_DIR="$PROJECT_DIR/recognition/hipmri-vqvae"
DATA_ROOT="${HIPMRI_ROOT:-/home/groups/comp3710/HipMRI_Study_open/keras_slices_data}"
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LEARNING_RATE="${LEARNING_RATE:-0.0002}"
SEED="${SEED:-3710}"
CODEBOOK_SIZE="${CODEBOOK_SIZE:-128}"
EMBEDDING_DIM="${EMBEDDING_DIM:-64}"
COMMITMENT_COST="${COMMITMENT_COST:-0.25}"
EMA_DECAY="${EMA_DECAY:-0.99}"
DEAD_CODE_PATIENCE="${DEAD_CODE_PATIENCE:-100}"
SSIM_WEIGHT="${SSIM_WEIGHT:-0.10}"
RUN_NAME="${RUN_NAME:-vqvae-v2-ema-k128-ssim010}"
MANIFEST="$EXPERIMENT_DIR/data/manifest.csv"
RUN_DIR="$EXPERIMENT_DIR/outputs/$RUN_NAME"
PREDICTION_DIR="$EXPERIMENT_DIR/predictions/${RUN_NAME}-validation"

cd "$EXPERIMENT_DIR"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
echo "Validation-only VQ-VAE v2 protocol: $RUN_NAME"
echo "Acceptance criteria: validation SSIM > 0.6000 and active codes >= 32/$CODEBOOK_SIZE"
nvidia-smi
python dataset.py --data-root "$DATA_ROOT" --output "$MANIFEST"

TRAIN_ARGS=(
  --model vqvae
  --manifest "$MANIFEST"
  --output "$RUN_DIR"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --learning-rate "$LEARNING_RATE"
  --num-workers "$NUM_WORKERS"
  --seed "$SEED"
  --device cuda
  --vq-num-embeddings "$CODEBOOK_SIZE"
  --vq-embedding-dim "$EMBEDDING_DIM"
  --vq-commitment-cost "$COMMITMENT_COST"
  --vq-codebook-update ema
  --vq-ema-decay "$EMA_DECAY"
  --vq-dead-code-patience "$DEAD_CODE_PATIENCE"
  --vq-ssim-weight "$SSIM_WEIGHT"
  --checkpoint-metric ssim
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
  --output "$PREDICTION_DIR" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --num-visualisations 6 \
  --device cuda \
  --overwrite

python - "$PREDICTION_DIR/summary.json" "$CODEBOOK_SIZE" <<'PY'
import json
import pathlib
import sys

summary_path = pathlib.Path(sys.argv[1])
codebook_size = int(sys.argv[2])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
ssim = float(summary["ssim"])
active_codes = int(summary["active_codes"])
minimum_active = max(32, codebook_size // 4)
accepted = ssim > 0.6 and active_codes >= minimum_active
print(
    "VQ-VAE v2 validation decision: "
    f"ssim={ssim:.4f}, active_codes={active_codes}/{codebook_size}, "
    f"accepted={str(accepted).lower()}"
)
PY

echo "Job $SLURM_JOB_ID finished $(date)"
