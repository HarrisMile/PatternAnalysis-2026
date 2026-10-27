#!/bin/bash
#SBATCH --job-name=hipmri-vqvae-v2-test
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=hipmri_vqvae_v2_test_%j.out
#SBATCH --error=hipmri_vqvae_v2_test_%j.err

set -euo pipefail

source "$HOME/miniconda3/bin/activate"
conda activate comp3710lab2

PROJECT_DIR="${PROJECT_DIR:-$HOME/PatternAnalysis-2026}"
EXPERIMENT_DIR="$PROJECT_DIR/recognition/hipmri-vqvae"
RUN_NAME="${RUN_NAME:-vqvae-v2-ema-k128-ssim010}"
MANIFEST="$EXPERIMENT_DIR/data/manifest.csv"
CHECKPOINT="$EXPERIMENT_DIR/outputs/$RUN_NAME/best.pt"
PREDICTION_DIR="$EXPERIMENT_DIR/predictions/${RUN_NAME}-test"

cd "$EXPERIMENT_DIR"
export PYTHONUNBUFFERED=1

test -f "$MANIFEST"
test -f "$CHECKPOINT"
test ! -e "$PREDICTION_DIR"

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
echo "One-time post-hoc VQ-VAE v2 test evaluation: $RUN_NAME"
echo "The v2 configuration was selected using validation data after the v1 test result had been observed."
echo "Frozen checkpoint hash"
sha256sum "$CHECKPOINT"
nvidia-smi

python predict.py \
  --checkpoint "$CHECKPOINT" \
  --manifest "$MANIFEST" \
  --split test \
  --output "$PREDICTION_DIR" \
  --batch-size 64 \
  --num-workers 4 \
  --num-visualisations 6 \
  --device cuda

python - "$PREDICTION_DIR/summary.json" <<'PY'
import json
import pathlib
import sys

summary = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(
    "VQ-VAE v2 post-hoc test result: "
    f"ssim={float(summary['ssim']):.4f}, "
    f"mse={float(summary['mse']):.6f}, "
    f"active_codes={int(summary['active_codes'])}, "
    f"perplexity={float(summary['codebook_perplexity']):.2f}"
)
PY

echo "Job $SLURM_JOB_ID finished $(date)"
