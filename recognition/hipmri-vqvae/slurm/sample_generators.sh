#!/bin/bash
#SBATCH --job-name=hipmri-gen-audit
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=hipmri_gen_audit_%j.out
#SBATCH --error=hipmri_gen_audit_%j.err

set -euo pipefail

source "$HOME/miniconda3/bin/activate"
conda activate comp3710lab2

PROJECT_DIR="${PROJECT_DIR:-$HOME/PatternAnalysis-2026}"
EXPERIMENT_DIR="$PROJECT_DIR/recognition/hipmri-vqvae"
PRIOR_CHECKPOINT="$EXPERIMENT_DIR/outputs/formal-prior/best.pt"
VQVAE_CHECKPOINT="$EXPERIMENT_DIR/outputs/formal-vqvae/best.pt"
VAE_CHECKPOINT="$EXPERIMENT_DIR/outputs/formal-vae/best.pt"
VQ_SAMPLE_DIR="$EXPERIMENT_DIR/predictions/formal-prior-multiseed"
VAE_SAMPLE_DIR="$EXPERIMENT_DIR/predictions/formal-vae-multiseed"
VQ_AUDIT_DIR="$EXPERIMENT_DIR/report_assets/generative_evaluation_multiseed"
COMPARISON_DIR="$EXPERIMENT_DIR/report_assets/generative_baseline_comparison"

cd "$EXPERIMENT_DIR"
export PYTHONUNBUFFERED=1

test -f "$PRIOR_CHECKPOINT"
test -f "$VQVAE_CHECKPOINT"
test -f "$VAE_CHECKPOINT"
test -f outputs/prior-data/train_code_indices.npz
test -f outputs/prior-data/train_samples.csv
test -f outputs/prior-data/summary.json
test -f data/manifest.csv
test ! -e "$VQ_SAMPLE_DIR"
test ! -e "$VAE_SAMPLE_DIR"
test ! -e "$VQ_AUDIT_DIR"
test ! -e "$COMPARISON_DIR"

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi
echo "Frozen checkpoint hashes"
sha256sum "$PRIOR_CHECKPOINT" "$VQVAE_CHECKPOINT" "$VAE_CHECKPOINT"

python sample_prior.py \
  --prior-checkpoint "$PRIOR_CHECKPOINT" \
  --vqvae-checkpoint "$VQVAE_CHECKPOINT" \
  --output "$VQ_SAMPLE_DIR" \
  --seeds 3710 3711 3712 \
  --samples-per-seed 16 \
  --batch-size 8 \
  --temperature 1.0 \
  --device cuda

python sample_vae.py \
  --checkpoint "$VAE_CHECKPOINT" \
  --output "$VAE_SAMPLE_DIR" \
  --seeds 3710 3711 3712 \
  --samples-per-seed 16 \
  --batch-size 16 \
  --device cuda

python evaluate_generation.py \
  --samples "$VQ_SAMPLE_DIR/samples.npz" \
  --sample-summary "$VQ_SAMPLE_DIR/summary.json" \
  --train-codes outputs/prior-data/train_code_indices.npz \
  --train-metadata outputs/prior-data/train_samples.csv \
  --train-summary outputs/prior-data/summary.json \
  --prior-summary outputs/formal-prior/summary.json \
  --vqvae-checkpoint "$VQVAE_CHECKPOINT" \
  --manifest data/manifest.csv \
  --output "$VQ_AUDIT_DIR" \
  --failure-cases 5 \
  --interpolation-pairs 4 \
  --interpolation-steps 7 \
  --batch-size 64 \
  --num-workers 4 \
  --device cuda

python compare_generators.py \
  --vae-samples "$VAE_SAMPLE_DIR/samples.npz" \
  --vae-summary "$VAE_SAMPLE_DIR/summary.json" \
  --vq-samples "$VQ_SAMPLE_DIR/samples.npz" \
  --vq-summary "$VQ_SAMPLE_DIR/summary.json" \
  --manifest data/manifest.csv \
  --train-metadata outputs/prior-data/train_samples.csv \
  --output "$COMPARISON_DIR" \
  --batch-size 64 \
  --num-workers 4

cat "$VQ_AUDIT_DIR/REPORT_RESULTS.md"
cat "$COMPARISON_DIR/REPORT_RESULTS.md"
echo "Job $SLURM_JOB_ID finished $(date)"
