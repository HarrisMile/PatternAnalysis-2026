# Unconditional Generation Audit

## Scope

This audit compares 32 unconditional samples with all 11,460 training slices. Only the training split is used as the memorisation reference.

## Memorisation check

- Exact pixel matches: 0
- Exact VQ code-grid matches: 0
- Nearest-training full-resolution SSIM: median 0.258, range 0.206 to 0.330
- Nearest-training VQ mismatch fraction: median 0.657, range 0.516 to 0.807

No direct copy of a training image or training code grid was detected. This is evidence against exact memorisation, but it is not a formal privacy guarantee.

## Diversity

- Exact duplicate generated images: 0
- Exact duplicate generated code grids: 0
- Pairwise generated-image MSE: median 0.0656, minimum 0.0215
- Pairwise generated-code mismatch fraction: median 0.757, minimum 0.389

The sample set is not a collection of exact duplicates, although visual quality must be assessed separately from numerical diversity.

## Distribution coverage

The samples use 14 of the 15 codes active in the training export (93.3% coverage).
Generated-code perplexity is 10.365, compared with 10.510 for training codes; code-frequency Jensen-Shannon divergence is 0.0404 nats.

Image-space coverage is substantially weaker. The generated near-zero fraction is 2.39 training standard deviations above the training mean, while tissue fraction is 1.24 standard deviations below it. The vertical centre of mass is shifted by 4.66 standard deviations. These shifts agree with the large black bands and fragmented anatomy visible in the sample panel.

## Latent interpolation diagnostic

4 generated/training-neighbour pairs were selected at fixed nearest-neighbour-distance ranks. Each pair was decoded at 7 evenly spaced points between its training and generated codebook-embedding grids.
Adjacent interpolation-frame SSIM has median row mean 0.796 and worst observed value 0.646.

The interpolation is a decoder-continuity diagnostic, not a new sampling method: intermediate embedding tensors generally do not correspond to discrete codebook entries and may therefore be off the learned VQ support.

## Conclusion

The learned prior enables reproducible unconditional sampling and shows numerical diversity without detected exact training-set copies. However, code-frequency coverage does not translate into anatomically coherent images. The generated cohort should therefore be reported as a technically successful but low-quality negative result, not as clinically realistic synthetic MRI.

## Limitations

- Pixel-space nearest-neighbour search is sensitive to padding and intensity alignment.
- No exact or close neighbour is evidence against direct copying, not a privacy guarantee.
- Thirty-two samples support an initial audit but not clinical validation.
- Linear embedding interpolation probes decoder continuity but intermediate latents can lie outside the discrete codebook support.
