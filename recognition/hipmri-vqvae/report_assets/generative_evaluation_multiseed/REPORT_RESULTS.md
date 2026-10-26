# Unconditional Generation Audit

## Scope

This audit compares 48 unconditional samples with all 11,460 training slices. Only the training split is used as the memorisation reference.
Samples were drawn from seeds [3710, 3711, 3712] at 203.90 ms/image on NVIDIA A100-PCIE-40GB.

## Memorisation check

- Exact pixel matches: 0
- Exact VQ code-grid matches: 0
- Nearest-training full-resolution SSIM: median 0.256, range 0.133 to 0.330
- Nearest-training VQ mismatch fraction: median 0.672, range 0.582 to 0.807

No direct copy of a training image or training code grid was detected. This is evidence against exact memorisation, but it is not a formal privacy guarantee.

## Diversity

- Exact duplicate generated images: 0
- Exact duplicate generated code grids: 0
- Pairwise generated-image MSE: median 0.0687, minimum 0.0135
- Pairwise generated-code mismatch fraction: median 0.770, minimum 0.328

The sample set is not a collection of exact duplicates, although visual quality must be assessed separately from numerical diversity.

## Distribution coverage

The samples use 14 of the 15 codes active in the training export (93.3% coverage).
Generated-code perplexity is 10.308, compared with 10.510 for training codes; code-frequency Jensen-Shannon divergence is 0.0444 nats.

Image-space coverage is substantially weaker. The generated near-zero fraction is 2.49 training standard deviations above the training mean, while tissue fraction is 1.40 standard deviations below it. The vertical centre of mass is shifted by 3.93 standard deviations. These shifts agree with the large black bands and fragmented anatomy visible in the sample panel.

## Deterministically selected generation failures

The 5 cases below are the largest predefined feature-deviation scores, not visually cherry-picked examples.

- Case 1: seed 3712, sample 11; dominant deviation `vertical_centre_of_mass` = 19.41 training standard deviations, nearest-training SSIM 0.133.
- Case 2: seed 3712, sample 6; dominant deviation `vertical_centre_of_mass` = -17.30 training standard deviations, nearest-training SSIM 0.202.
- Case 3: seed 3712, sample 1; dominant deviation `vertical_centre_of_mass` = -15.16 training standard deviations, nearest-training SSIM 0.272.
- Case 4: seed 3712, sample 3; dominant deviation `vertical_centre_of_mass` = -15.13 training standard deviations, nearest-training SSIM 0.243.
- Case 5: seed 3710, sample 8; dominant deviation `vertical_centre_of_mass` = -13.71 training standard deviations, nearest-training SSIM 0.206.

These cases require visual interpretation alongside the recorded quantitative trigger; the score is an audit selection rule, not a clinical abnormality score.

## Latent interpolation diagnostic

4 generated/training-neighbour pairs were selected at fixed nearest-neighbour-distance ranks. Each pair was decoded at 7 evenly spaced points between its training and generated codebook-embedding grids.
Adjacent interpolation-frame SSIM has median row mean 0.823 and worst observed value 0.775.

The interpolation is a decoder-continuity diagnostic, not a new sampling method: intermediate embedding tensors generally do not correspond to discrete codebook entries and may therefore be off the learned VQ support.

## Conclusion

The learned prior enables reproducible unconditional sampling and shows numerical diversity without detected exact training-set copies. However, code-frequency coverage does not translate into anatomically coherent images. The generated cohort should therefore be reported as a technically successful but low-quality negative result, not as clinically realistic synthetic MRI.

## Limitations

- Pixel-space nearest-neighbour search is sensitive to padding and intensity alignment.
- No exact or close neighbour is evidence against direct copying, not a privacy guarantee.
- 48 samples across 3 seeds support an engineering audit but not clinical validation.
- Linear embedding interpolation probes decoder continuity but intermediate latents can lie outside the discrete codebook support.
