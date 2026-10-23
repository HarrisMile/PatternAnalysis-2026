#!/bin/bash
#SBATCH --job-name=hipmri-prior-sample
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=hipmri_prior_sample_%j.out
#SBATCH --error=hipmri_prior_sample_%j.err

set -euo pipefail

source "$HOME/miniconda3/bin/activate"
conda activate comp3710lab2

PROJECT_DIR="${PROJECT_DIR:-$HOME/PatternAnalysis-2026}"
EXPERIMENT_DIR="$PROJECT_DIR/recognition/hipmri-vqvae"
PRIOR_CHECKPOINT="$EXPERIMENT_DIR/outputs/formal-prior/best.pt"
VQVAE_CHECKPOINT="$EXPERIMENT_DIR/outputs/formal-vqvae/best.pt"
RESULT_DIR="$EXPERIMENT_DIR/predictions/formal-prior-samples"

cd "$EXPERIMENT_DIR"
export PYTHONUNBUFFERED=1

test -f "$PRIOR_CHECKPOINT"
test -f "$VQVAE_CHECKPOINT"
test ! -e "$RESULT_DIR"

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi
echo "Frozen checkpoint hashes"
sha256sum "$PRIOR_CHECKPOINT" "$VQVAE_CHECKPOINT"

python sample_prior.py \
  --prior-checkpoint "$PRIOR_CHECKPOINT" \
  --vqvae-checkpoint "$VQVAE_CHECKPOINT" \
  --output "$RESULT_DIR" \
  --num-samples 32 \
  --batch-size 8 \
  --temperature 1.0 \
  --seed 3710 \
  --device cuda

cat "$RESULT_DIR/summary.json"
echo "Job $SLURM_JOB_ID finished $(date)"
