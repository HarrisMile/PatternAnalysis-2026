#!/bin/bash
#SBATCH --job-name=hipmri-test
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=hipmri_test_%j.out
#SBATCH --error=hipmri_test_%j.err

set -euo pipefail

source "$HOME/miniconda3/bin/activate"
conda activate comp3710lab2

PROJECT_DIR="${PROJECT_DIR:-$HOME/PatternAnalysis-2026}"
EXPERIMENT_DIR="$PROJECT_DIR/recognition/hipmri-vqvae"
MANIFEST="$EXPERIMENT_DIR/data/manifest.csv"
VAE_CHECKPOINT="$EXPERIMENT_DIR/outputs/formal-vae/best.pt"
VQVAE_CHECKPOINT="$EXPERIMENT_DIR/outputs/formal-vqvae/best.pt"

cd "$EXPERIMENT_DIR"
export PYTHONUNBUFFERED=1

test -f "$MANIFEST"
test -f "$VAE_CHECKPOINT"
test -f "$VQVAE_CHECKPOINT"
test ! -e predictions/formal-vae-test
test ! -e predictions/formal-vqvae-test

echo "Frozen checkpoint hashes"
sha256sum "$VAE_CHECKPOINT" "$VQVAE_CHECKPOINT"

python predict.py \
  --checkpoint "$VAE_CHECKPOINT" \
  --manifest "$MANIFEST" \
  --split test \
  --output predictions/formal-vae-test \
  --batch-size 64 \
  --num-workers 4 \
  --num-visualisations 6 \
  --device cuda

python predict.py \
  --checkpoint "$VQVAE_CHECKPOINT" \
  --manifest "$MANIFEST" \
  --split test \
  --output predictions/formal-vqvae-test \
  --batch-size 64 \
  --num-workers 4 \
  --num-visualisations 6 \
  --device cuda

echo "Job $SLURM_JOB_ID finished $(date)"
