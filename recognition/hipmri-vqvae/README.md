# HipMRI 2D Reconstruction with VAE and VQ-VAE

## Project status

Stage 3 experiment pipeline. The downloaded HipMRI slices have been inspected, the official patient-level splits have been verified, and end-to-end training, validation, checkpoint loading, metric export, and failure-case visualisation have been tested with both model architectures. No final trained-model results are claimed yet.

## Research question

Under the same patient-level data split, preprocessing, training budget, and evaluation protocol, does a VQ-VAE provide a better reconstruction-quality and representation-efficiency trade-off than a continuous-latent convolutional VAE on 2D HipMRI slices?

## Models

- **Baseline:** convolutional VAE with a continuous spatial latent tensor.
- **Hard model:** VQ-VAE with a learned discrete codebook and straight-through estimator.

The encoder and decoder capacity are intentionally similar so that the comparison focuses on continuous versus vector-quantised latent representations.
The VQ codebook is initialised once from diverse encoder vectors in the first
training batch using deterministic farthest-point sampling. This avoids the
single-code collapse observed with the original narrow random initialisation;
after that first batch, the embeddings are learned by the standard codebook and
commitment losses. Batch normalisation before quantisation keeps encoder scales
stable while the codebook and encoder co-adapt.

## Verified dataset inventory

The source data are the course-provided HipMRI 2D NIfTI slices. Raw data are local-only and ignored by Git.

| Split | MRI slices | Masks | Patient IDs |
| --- | ---: | ---: | --- |
| Train | 11,460 | 11,460 | 004–033 and 035 |
| Validation | 660 | 660 | 036–039 |
| Test | 540 | 540 | 040–042 |

The filename key `(patient, week, slice)` pairs every MRI with exactly one segmentation mask. The patient sets are disjoint across all three splits. Of the 12,660 MRI slices, 12,600 have shape `256 x 128`; the 60 slices from patient 019 have shape `256 x 144`. MRI values are non-negative `float32`; masks are `uint8` labels with observed values 0–5.

## Deterministic preprocessing

`dataset.py` implements the following fixed pipeline:

1. Discover the official train, validation, and test directories.
2. Pair `case_...nii.gz` and `seg_...nii.gz` files by patient, week, and slice.
3. Reject missing pairs, duplicate keys, malformed filenames, or patient overlap.
4. Load each 2D NIfTI slice without modifying the source file.
5. Clip each MRI at the 99.5th percentile of its positive pixels and scale it to `[0, 1]`.
6. Centre-pad with zeros to `256 x 144`; no image content is cropped.
7. Preserve the integer segmentation mask for later region-aware evaluation.

The per-slice normalisation is deterministic and does not use labels or statistics from other patients. Its trade-off is that absolute scanner-intensity differences between slices are removed; this limitation must be considered when interpreting results.

## Environment and manifest

From this directory:

```bash
conda activate 3710torch
python -m pip install -r requirements.txt

python dataset.py \
  --data-root data/keras_slices_data \
  --output data/manifest.csv
```

The generated manifest contains relative MRI/mask paths plus patient, week, slice, and split identifiers. Both the raw data and manifest remain under the ignored `data/` directory.

## Training and validation

`train.py` evaluates the validation split after every epoch and writes:

- `best.pt`: checkpoint with the lowest validation loss;
- `last.pt`: checkpoint from the most recent epoch;
- `history.csv`: per-epoch loss, reconstruction metrics, timings, and VQ codebook statistics;
- `config.json`: data, optimiser, reproducibility, software, device, and parameter-count settings;
- `summary.json`: best epoch, runtime, parameter count, device, and peak CUDA memory when available.

Example local sanity run:

```bash
python train.py \
  --model vae \
  --manifest data/manifest.csv \
  --output outputs/sanity-vae \
  --epochs 2 \
  --batch-size 4 \
  --max-train-batches 6 \
  --max-validation-batches 3 \
  --device cpu
```

The batch-limit options mark the run as limited in `summary.json`. They are for pipeline checks only and must not be used for final model claims.

## Checkpoint evaluation

During development, evaluate only the validation split:

```bash
python predict.py \
  --checkpoint outputs/sanity-vae/best.pt \
  --manifest data/manifest.csv \
  --split validation \
  --output predictions/sanity-vae \
  --max-batches 3 \
  --device cpu
```

The command saves per-sample MSE, MAE, PSNR, and windowed SSIM to `metrics.csv`, aggregated values to `summary.json`, and a panel of the highest-MSE cases to `worst_reconstructions.png`. VQ-VAE evaluations additionally report codebook perplexity, active codes, dead codes, and active-code fraction. The untouched test split should be evaluated only after both final models and the comparison protocol have been frozen.

## Rangpur execution

The Rangpur workflow reuses the `comp3710lab2` Conda environment and the course account/partition pattern that was verified in earlier COMP3710 work. A live check on 8 October 2026 confirmed that the `comp3710` partition is available with A100 GPUs, the user account is associated with the `comp3710` Slurm account under normal QOS, and the existing environment contains CUDA-enabled PyTorch 2.14.0. `nibabel` was the only required package missing before installing this project's requirements. Because cluster configuration can change, run `sinfo` before every new submission and compare it with the current [EAIT Compute documentation](https://student.eait.uq.edu.au/infrastructure/compute/).

The complete login, setup, submission, monitoring, and log-inspection sequence is in `RANGPUR_COMMANDS.txt`. Three Slurm scripts are provided:

- `slurm/gpu_sanity.sh`: short validation-only ConvVAE and VQ-VAE GPU pipeline check;
- `slurm/train_vae.sh`: formal ConvVAE baseline training and validation evaluation;
- `slurm/train_vqvae.sh`: formal VQ-VAE training and validation evaluation under the same defaults.

Formal jobs default to 40 epochs, batch size 64, learning rate `2e-4`, seed 3710, four data workers, and one GPU. These values remain provisional until the short GPU run confirms runtime, memory use, and codebook behaviour. They can be overridden through exported Slurm environment variables without editing the scripts.

Every epoch writes `last.pt`; the best validation checkpoint remains in `best.pt`. If Slurm requeues a job, the scripts detect `last.pt` and pass `--resume`. The trainer restores the model, optimiser, epoch history, and PyTorch random state. A local interrupted-versus-continuous test produced identical metric histories and exactly equal final model weights.

## Smoke tests

Software-only model test:

```bash
python train.py --smoke-test
```

The data-loader verification produced batches with shape `[N, 1, 256, 144]`, `float32` MRI values in `[0, 1]`, and integer masks. Both VAE and VQ-VAE returned reconstructions of the same shape.

An intentionally limited CPU run also verified the full artifact pipeline. The ConvVAE completed two epochs over 6 training and 3 validation batches per epoch, selected epoch 2 as the best checkpoint, and reloaded it using PyTorch's safe `weights_only=True` mode. The first A100 feasibility run completed both models and all evaluation outputs in 65 seconds, but exposed a collapsed VQ codebook with only 1/512 active entries. That diagnostic result motivated the deterministic data-dependent codebook initialisation and pre-quantisation normalisation above and must not be presented as final model-quality evidence. The revised sanity script uses five limited VQ-VAE epochs so that codebook adaptation is assessed rather than judged from its first epoch alone. A second GPU sanity run is required before either formal job is launched.

## Planned evaluation

- Reconstruction: MSE, MAE, PSNR, and SSIM on the fixed test split.
- VQ-VAE representation: codebook perplexity, active-code fraction, and dead codes.
- Resources: parameter count, training time, inference time, and peak accelerator memory.
- Qualitative analysis: original/reconstruction/error-map panels for 3–5 representative lower-performing test cases.
- Region-aware analysis using the supplied masks without training on the test labels.
- Recommendation based on quality, stability, resource use, and observed failure modes.

## Required files

- `modules.py`: ConvVAE, vector quantiser, VQ-VAE, and loss functions.
- `dataset.py`: NIfTI discovery, pairing, leakage checks, manifest generation, and loading.
- `metrics.py`: shared MSE, MAE, PSNR, and windowed SSIM implementation.
- `train.py`: training, validation, best-checkpoint selection, configuration, and history logging.
- `predict.py`: safe checkpoint loading, per-sample evaluation, and failure-case visualisation.
- `slurm/`: Rangpur sanity, baseline, and VQ-VAE batch scripts with automatic resume.
- `RANGPUR_COMMANDS.txt`: copyable Rangpur setup, monitoring, and evidence commands.
- `README.md`: experiment protocol, commands, evidence, and findings.

## Immediate next steps

1. Re-run the short GPU sanity experiment and confirm that the VQ-VAE uses multiple codes.
2. Freeze a justified epoch, batch, and codebook budget from the sanity evidence.
3. Train the ConvVAE baseline under the frozen budget.
4. Train the VQ-VAE using the same data, hardware, and budget while monitoring codebook collapse.
5. Evaluate both frozen checkpoints once on the untouched test patients.
6. Record failure cases, resource statistics, and final engineering recommendations.
