# Frozen experiment results for the final report

This note is generated from the saved formal-run artifacts. It does not contain new
training, checkpoint selection, or test-set tuning.

## Core test comparison

| Model | MSE | MAE | PSNR | SSIM | Foreground MSE | Foreground PSNR | Foreground SSIM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ConvVAE | 0.000500 | 0.013314 | 33.256 | 0.9185 | 0.000664 | 32.035 | 0.9441 |
| VQ-VAE | 0.007269 | 0.051884 | 21.585 | 0.5897 | 0.009578 | 20.400 | 0.5294 |

The ConvVAE achieved a 14.52-fold lower MSE than the VQ-VAE, an
11.67 dB higher PSNR, and a 0.329 higher SSIM. The
foreground-only comparison reaches the same conclusion, so the advantage is not an
artefact of padded background pixels.

## Resource and selection evidence

| Model | Parameters | Training time (s) | Peak GPU memory (MiB) | Selected epoch |
| --- | --- | --- | --- | --- |
| ConvVAE | 476,513 | 1888.9 | 1348.9 | 40 |
| VQ-VAE | 501,153 | 1923.7 | 1348.9 | 38 |

The two models used nearly identical training time and peak GPU memory. The VQ-VAE
therefore did not obtain a reconstruction-quality benefit in exchange for its slightly
higher parameter count. Its discrete representation is compact at the index level, but
only 13/512 codebook entries were active on test
(2.54%), with perplexity
10.46.

## Patient-level robustness

| Model | Patient | Slices | MSE | PSNR | SSIM | Foreground MSE | Foreground SSIM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ConvVAE | 040 | 420 | 0.000503 | 33.239 | 0.9198 | 0.000667 | 0.9455 |
| ConvVAE | 041 | 60 | 0.000513 | 33.048 | 0.9149 | 0.000686 | 0.9411 |
| ConvVAE | 042 | 60 | 0.000470 | 33.583 | 0.9133 | 0.000620 | 0.9367 |
| VQ-VAE | 040 | 420 | 0.007576 | 21.404 | 0.5912 | 0.009976 | 0.5303 |
| VQ-VAE | 041 | 60 | 0.006533 | 21.940 | 0.5914 | 0.008629 | 0.5351 |
| VQ-VAE | 042 | 60 | 0.005856 | 22.498 | 0.5770 | 0.007735 | 0.5169 |

The ConvVAE is better for every held-out patient, so the aggregate conclusion is not
driven by one patient. Patient 040 contributes more slices than patients 041 and 042;
the patient-level table is therefore an important robustness check alongside the
slice-weighted aggregate metrics.

## Qualitative selection protocol

| Selection | Patient | Week | Slice | VQ-VAE MSE |
| --- | --- | --- | --- | --- |
| Representative (subject median VQ-VAE MSE) | 040 | 3 | 9 | 0.006795 |
| Challenging (subject maximum VQ-VAE MSE) | 040 | 7 | 58 | 0.012125 |
| Representative (subject median VQ-VAE MSE) | 041 | 0 | 12 | 0.006074 |
| Challenging (subject maximum VQ-VAE MSE) | 041 | 0 | 58 | 0.010106 |
| Representative (subject median VQ-VAE MSE) | 042 | 0 | 47 | 0.005982 |
| Challenging (subject maximum VQ-VAE MSE) | 042 | 0 | 34 | 0.008047 |

For each test patient, one representative slice is selected as the slice closest to that
patient's median VQ-VAE MSE, and one challenging slice is selected as that patient's
maximum VQ-VAE MSE. Both frozen models are shown on exactly the same slices, with a
shared error-map scale within each row. This avoids comparing different model-specific
worst cases.

## Interpretation and recommendation

The ConvVAE preserves anatomical boundaries and fine texture substantially better. The
VQ-VAE reconstructions are visibly blurred and their errors are concentrated around
tissue interfaces and fine structures. The VQ-VAE avoided total single-code collapse,
but it still learned a heavily under-utilised codebook. Under the controlled 40-epoch
budget, the ConvVAE is the recommended model for HipMRI reconstruction quality and
stability. The VQ-VAE result should be reported as a valid negative result and a clear
limitation, not tuned further after test evaluation.
