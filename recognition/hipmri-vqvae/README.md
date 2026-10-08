# HipMRI 2D Reconstruction with VAE and VQ-VAE

## Project status

Stage 2 data pipeline. The downloaded HipMRI slices have been inspected, the official patient-level splits have been verified, and the NIfTI loader has been tested with both model architectures. No trained-model results are claimed yet.

## Research question

Under the same patient-level data split, preprocessing, training budget, and evaluation protocol, does a VQ-VAE provide a better reconstruction-quality and representation-efficiency trade-off than a continuous-latent convolutional VAE on 2D HipMRI slices?

## Models

- **Baseline:** convolutional VAE with a continuous spatial latent tensor.
- **Hard model:** VQ-VAE with a learned discrete codebook and straight-through estimator.

The encoder and decoder capacity are intentionally similar so that the comparison focuses on continuous versus vector-quantised latent representations.

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

## Smoke tests

Software-only model test:

```bash
python train.py --smoke-test
```

The data-loader verification performed during Stage 2 produced batches with shape `[N, 1, 256, 144]`, `float32` MRI values in `[0, 1]`, and integer masks. Both VAE and VQ-VAE returned reconstructions of the same shape. These checks verify wiring only; they are not experimental results.

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
- `train.py`: common training entry point and random-input smoke test.
- `predict.py`: checkpoint loading and test reconstruction export.
- `README.md`: experiment protocol, commands, evidence, and findings.

## Immediate next steps

1. Add validation loss, checkpoint selection, metrics, and configuration logging.
2. Train the ConvVAE baseline under a fixed budget.
3. Train the VQ-VAE using the same data and compute budget.
4. Evaluate both checkpoints on the untouched test patients.
5. Record failure cases, resource statistics, and final engineering recommendations.
