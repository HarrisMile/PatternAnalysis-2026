# Generative Baseline Comparison

ConvVAE and VQ-VAE use the same seeds and sample count. Image-space metrics are computed against the complete training cohort; nearest-neighbour SSIM is a similarity and memorisation diagnostic, not the paired-reconstruction SSIM target.

| Model | Samples / seeds | Pairwise MSE median | Mean feature shift | Nearest-train SSIM | ms/image | Peak memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ConvVAE | 48 / 3 | 0.0614 | 1.114 | 0.065 | 36.63 | 222.6 MiB |
| VQ-VAE + PixelCNN | 48 / 3 | 0.0687 | 1.429 | 0.256 | 203.90 | 114.6 MiB |

Lower feature shift indicates closer agreement with the seven predefined training image statistics. Pairwise MSE measures numerical diversity but does not establish anatomical plausibility. The uncurated panel must be interpreted together with the VQ-VAE failure autopsy and nearest-neighbour audit.
