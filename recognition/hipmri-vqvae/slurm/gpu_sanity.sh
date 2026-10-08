#!/bin/bash
#SBATCH --job-name=hipmri-sanity
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=hipmri_sanity_%j.out
#SBATCH --error=hipmri_sanity_%j.err

set -euo pipefail

source "$HOME/miniconda3/bin/activate"
conda activate comp3710lab2

PROJECT_DIR="${PROJECT_DIR:-$HOME/PatternAnalysis-2026}"
EXPERIMENT_DIR="$PROJECT_DIR/recognition/hipmri-vqvae"
DATA_ROOT="${HIPMRI_ROOT:-/home/groups/comp3710/HipMRI_Study_open/keras_slices_data}"
MANIFEST="$EXPERIMENT_DIR/data/manifest.csv"

cd "$EXPERIMENT_DIR"
export PYTHONUNBUFFERED=1

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

python dataset.py --data-root "$DATA_ROOT" --output "$MANIFEST"

VAE_RUN="outputs/rangpur-sanity-vae-$SLURM_JOB_ID"
VQVAE_RUN="outputs/rangpur-sanity-vqvae-$SLURM_JOB_ID"

python train.py \
  --model vae \
  --manifest "$MANIFEST" \
  --output "$VAE_RUN" \
  --epochs 1 \
  --batch-size 16 \
  --num-workers 4 \
  --max-train-batches 20 \
  --max-validation-batches 10 \
  --device cuda

python predict.py \
  --checkpoint "$VAE_RUN/best.pt" \
  --manifest "$MANIFEST" \
  --split validation \
  --output "predictions/rangpur-sanity-vae-$SLURM_JOB_ID" \
  --batch-size 16 \
  --num-workers 4 \
  --max-batches 10 \
  --num-visualisations 4 \
  --device cuda

python train.py \
  --model vqvae \
  --manifest "$MANIFEST" \
  --output "$VQVAE_RUN" \
  --epochs 5 \
  --batch-size 16 \
  --num-workers 4 \
  --max-train-batches 20 \
  --max-validation-batches 10 \
  --device cuda

python predict.py \
  --checkpoint "$VQVAE_RUN/best.pt" \
  --manifest "$MANIFEST" \
  --split validation \
  --output "predictions/rangpur-sanity-vqvae-$SLURM_JOB_ID" \
  --batch-size 16 \
  --num-workers 4 \
  --max-batches 10 \
  --num-visualisations 4 \
  --device cuda

echo "Job $SLURM_JOB_ID finished $(date)"
