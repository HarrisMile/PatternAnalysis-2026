#!/bin/bash
#SBATCH --job-name=hipmri-latency
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=hipmri_latency_%j.out
#SBATCH --error=hipmri_latency_%j.err

set -euo pipefail

source "$HOME/miniconda3/bin/activate"
conda activate comp3710lab2

PROJECT_DIR="${PROJECT_DIR:-$HOME/PatternAnalysis-2026}"
EXPERIMENT_DIR="$PROJECT_DIR/recognition/hipmri-vqvae"
MANIFEST="$EXPERIMENT_DIR/data/manifest.csv"
VAE_CHECKPOINT="$EXPERIMENT_DIR/outputs/formal-vae/best.pt"
VQVAE_CHECKPOINT="$EXPERIMENT_DIR/outputs/formal-vqvae/best.pt"
RESULT_DIR="$EXPERIMENT_DIR/report_assets/inference_latency"

cd "$EXPERIMENT_DIR"
export PYTHONUNBUFFERED=1

test -f "$MANIFEST"
test -f "$VAE_CHECKPOINT"
test -f "$VQVAE_CHECKPOINT"
test ! -e "$RESULT_DIR"

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi
echo "Frozen checkpoint hashes"
sha256sum "$VAE_CHECKPOINT" "$VQVAE_CHECKPOINT"

python benchmark_latency.py \
  --vae-checkpoint "$VAE_CHECKPOINT" \
  --vqvae-checkpoint "$VQVAE_CHECKPOINT" \
  --manifest "$MANIFEST" \
  --output "$RESULT_DIR" \
  --batch-size 64 \
  --num-workers 4 \
  --warmup-batches 10 \
  --repetitions 20 \
  --device cuda

cat "$RESULT_DIR/summary.csv"
echo "Job $SLURM_JOB_ID finished $(date)"
