#!/bin/bash
#SBATCH --job-name=hipmri-vq-indices
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=hipmri_vq_indices_%j.out
#SBATCH --error=hipmri_vq_indices_%j.err

set -euo pipefail

source "$HOME/miniconda3/bin/activate"
conda activate comp3710lab2

PROJECT_DIR="${PROJECT_DIR:-$HOME/PatternAnalysis-2026}"
EXPERIMENT_DIR="$PROJECT_DIR/recognition/hipmri-vqvae"
MANIFEST="$EXPERIMENT_DIR/data/manifest.csv"
CHECKPOINT="$EXPERIMENT_DIR/outputs/formal-vqvae/best.pt"
RESULT_DIR="$EXPERIMENT_DIR/outputs/prior-data"

cd "$EXPERIMENT_DIR"
export PYTHONUNBUFFERED=1

test -f "$MANIFEST"
test -f "$CHECKPOINT"
test ! -e "$RESULT_DIR"

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi
echo "Frozen VQ-VAE checkpoint hash"
sha256sum "$CHECKPOINT"

python export_code_indices.py \
  --checkpoint "$CHECKPOINT" \
  --manifest "$MANIFEST" \
  --output "$RESULT_DIR" \
  --batch-size 64 \
  --num-workers 4 \
  --device cuda

cat "$RESULT_DIR/summary.json"
echo "Job $SLURM_JOB_ID finished $(date)"
