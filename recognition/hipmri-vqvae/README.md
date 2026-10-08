# HipMRI 2D Reconstruction with VAE and VQ-VAE

## Project status

Stage 1 scaffold. The model interfaces and a leakage-aware manifest contract are implemented. No experimental results are claimed yet. Raw HipMRI preprocessing will be implemented only after the downloaded dataset structure and intensity conventions have been inspected.

## Research question

Under the same patient-level data split, preprocessing, training budget, and evaluation protocol, does a VQ-VAE provide a better reconstruction-quality and representation-efficiency trade-off than a continuous-latent convolutional VAE on 2D HipMRI slices?

## Models

- **Baseline:** convolutional VAE with a continuous spatial latent tensor.
- **Hard model:** VQ-VAE with a learned discrete codebook and straight-through estimator.

The encoder and decoder capacity are intentionally similar so that the comparison focuses on continuous versus vector-quantised latent representations.

## Required files

- `modules.py`: ConvVAE, vector quantiser, VQ-VAE, and loss functions.
- `dataset.py`: fixed-manifest 2D slice loader with subject-leakage checks.
- `train.py`: common training entry point and random-input smoke test.
- `predict.py`: checkpoint loading and test reconstruction export.
- `README.md`: experiment protocol, commands, evidence, and findings.

## Stage 1 data contract

The current loader expects preprocessed single-channel slices saved as `.npy` arrays with shape `[H, W]` or `[1, H, W]`. A CSV manifest must contain:

```text
path,subject_id,split
slices/sub-001_slice-040.npy,sub-001,train
slices/sub-014_slice-037.npy,sub-014,validation
slices/sub-021_slice-052.npy,sub-021,test
```

The manifest reader rejects any subject appearing in more than one split. The final preprocessing procedure, slice-selection rule, spatial resolution, and intensity normalisation will be documented after the raw data are inspected.

## Smoke test

From this directory:

```bash
python train.py --smoke-test
```

The command performs forward and backward passes for both models using random `64 x 64` single-channel images. It verifies software wiring only; it is not experimental evidence.

## Planned evaluation

- Reconstruction: MSE, MAE, PSNR, and SSIM on the fixed test split.
- VQ-VAE representation: codebook perplexity, active-code fraction, and dead codes.
- Resources: parameter count, training time, inference time, and peak accelerator memory.
- Qualitative analysis: original/reconstruction/error-map panels for 3–5 representative lower-performing test cases.
- Recommendation: an engineering judgement based on quality, stability, resource use, and observed failure modes.

## Immediate next steps

1. Download and inspect HipMRI without changing the raw files.
2. Identify the subject/volume keys that prevent slice-level leakage.
3. Implement deterministic raw-data preprocessing and generate the split manifest.
4. Run the ConvVAE baseline before tuning VQ-VAE.
5. Add validation, metrics, reproducible configuration logging, and figures.
