"""Model definitions for the HipMRI VAE, VQ-VAE, and learned prior."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Shared three-stage convolutional encoder for 2D MRI slices."""

    def __init__(self, in_channels: int = 1, hidden_channels: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels * 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out_channels = hidden_channels * 4

    def forward(self, images: Tensor) -> Tensor:
        return self.network(images)


class Decoder(nn.Module):
    """Shared decoder that reconstructs intensities normalised to [0, 1]."""

    def __init__(self, latent_channels: int, out_channels: int = 1, hidden_channels: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, hidden_channels * 4, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, latents: Tensor) -> Tensor:
        return self.network(latents)


class ConvVAE(nn.Module):
    """Continuous spatial-latent convolutional VAE baseline."""

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 32,
        latent_channels: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_channels)
        self.mu_head = nn.Conv2d(self.encoder.out_channels, latent_channels, 1)
        self.logvar_head = nn.Conv2d(self.encoder.out_channels, latent_channels, 1)
        self.decoder = Decoder(latent_channels, in_channels, hidden_channels)

    @staticmethod
    def reparameterise(mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, images: Tensor) -> Dict[str, Tensor]:
        features = self.encoder(images)
        mu = self.mu_head(features)
        logvar = self.logvar_head(features)
        latents = self.reparameterise(mu, logvar) if self.training else mu
        return {
            "reconstruction": self.decoder(latents),
            "mu": mu,
            "logvar": logvar,
            "latents": latents,
        }


class VectorQuantizer(nn.Module):
    """Nearest-neighbour codebook with data-dependent initialisation."""

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)
        self.register_buffer("codebook_initialized", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def _initialise_from_batch(self, flat_inputs: Tensor) -> None:
        """Seed the codebook with diverse vectors from the first encoder batch."""

        if flat_inputs.shape[0] == 0:
            raise ValueError("Cannot initialise a codebook from an empty latent batch.")

        candidates = flat_inputs.detach()
        centroids = torch.empty(
            self.num_embeddings,
            self.embedding_dim,
            device=candidates.device,
            dtype=candidates.dtype,
        )
        mean = candidates.mean(dim=0, keepdim=True)
        next_index = (candidates - mean).pow(2).sum(dim=1).argmax()
        min_distances = torch.full(
            (candidates.shape[0],),
            float("inf"),
            device=candidates.device,
            dtype=candidates.dtype,
        )

        for code_index in range(self.num_embeddings):
            centroid = candidates[next_index]
            centroids[code_index].copy_(centroid)
            distances = (candidates - centroid).pow(2).sum(dim=1)
            min_distances = torch.minimum(min_distances, distances)
            next_index = min_distances.argmax()

        self.embedding.weight.copy_(centroids)
        self.codebook_initialized.fill_(True)

    def forward(self, inputs: Tensor) -> Dict[str, Tensor]:
        if inputs.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected {self.embedding_dim} latent channels, received {inputs.shape[1]}."
            )

        channels_last = inputs.permute(0, 2, 3, 1).contiguous()
        flat_inputs = channels_last.view(-1, self.embedding_dim)
        if self.training and not bool(self.codebook_initialized.item()):
            self._initialise_from_batch(flat_inputs)
        embedding_weight = self.embedding.weight
        distances = (
            flat_inputs.pow(2).sum(dim=1, keepdim=True)
            + embedding_weight.pow(2).sum(dim=1)
            - 2.0 * flat_inputs @ embedding_weight.t()
        )
        indices = distances.argmin(dim=1)
        quantized = self.embedding(indices).view_as(channels_last)

        codebook_loss = F.mse_loss(quantized, channels_last.detach())
        commitment_loss = self.commitment_cost * F.mse_loss(quantized.detach(), channels_last)
        quantizer_loss = codebook_loss + commitment_loss
        quantized_st = channels_last + (quantized - channels_last).detach()

        counts = torch.bincount(indices, minlength=self.num_embeddings).to(inputs.dtype)
        probabilities = counts / counts.sum().clamp_min(1.0)
        perplexity = torch.exp(-torch.sum(probabilities * torch.log(probabilities + 1e-10)))
        active_codes = (counts > 0).sum()

        return {
            "quantized": quantized_st.permute(0, 3, 1, 2).contiguous(),
            "quantizer_loss": quantizer_loss,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "perplexity": perplexity,
            "active_codes": active_codes,
            "encoding_indices": indices.view(inputs.shape[0], inputs.shape[2], inputs.shape[3]),
        }


class VQVAE(nn.Module):
    """Convolutional VQ-VAE for 2D MRI reconstruction."""

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 32,
        embedding_dim: int = 64,
        num_embeddings: int = 512,
        commitment_cost: float = 0.25,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_channels)
        self.pre_quantizer = nn.Sequential(
            nn.Conv2d(self.encoder.out_channels, embedding_dim, 1),
            nn.BatchNorm2d(embedding_dim),
        )
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(embedding_dim, in_channels, hidden_channels)

    def forward(self, images: Tensor) -> Dict[str, Tensor]:
        encoded = self.pre_quantizer(self.encoder(images))
        quantizer_output = self.quantizer(encoded)
        return {
            "reconstruction": self.decoder(quantizer_output["quantized"]),
            "encoded": encoded,
            **quantizer_output,
        }


class MaskedConv2d(nn.Conv2d):
    """Convolution masked to preserve raster-order autoregressive causality."""

    def __init__(self, mask_type: str, *args: object, **kwargs: object) -> None:
        if mask_type not in {"A", "B"}:
            raise ValueError("mask_type must be 'A' or 'B'.")
        super().__init__(*args, **kwargs)
        kernel_height, kernel_width = self.kernel_size
        if kernel_height % 2 == 0 or kernel_width % 2 == 0:
            raise ValueError("MaskedConv2d requires odd kernel dimensions.")

        mask = torch.ones_like(self.weight)
        centre_row = kernel_height // 2
        centre_column = kernel_width // 2
        mask[:, :, centre_row + 1 :, :] = 0
        first_hidden_column = centre_column if mask_type == "A" else centre_column + 1
        mask[:, :, centre_row, first_hidden_column:] = 0
        self.mask_type = mask_type
        self.register_buffer("mask", mask)

    def forward(self, inputs: Tensor) -> Tensor:
        return F.conv2d(
            inputs,
            self.weight * self.mask,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class PixelCNNResidualBlock(nn.Module):
    """Residual block whose spatial convolution can only use past pixels."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        bottleneck_channels = max(channels // 2, 1)
        self.network = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, bottleneck_channels, 1),
            nn.ReLU(inplace=False),
            MaskedConv2d(
                "B",
                bottleneck_channels,
                bottleneck_channels,
                3,
                padding=1,
            ),
            nn.ReLU(inplace=False),
            nn.Conv2d(bottleneck_channels, channels, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return features + self.network(features)


class PixelCNNPrior(nn.Module):
    """Autoregressive categorical prior over compact VQ code indices."""

    def __init__(
        self,
        num_tokens: int,
        embedding_dim: int = 64,
        hidden_channels: int = 64,
        residual_blocks: int = 6,
    ) -> None:
        super().__init__()
        if num_tokens <= 1:
            raise ValueError("num_tokens must be greater than one.")
        if embedding_dim <= 0 or hidden_channels <= 0:
            raise ValueError("PixelCNN channel dimensions must be positive.")
        if residual_blocks <= 0:
            raise ValueError("residual_blocks must be positive.")

        self.num_tokens = num_tokens
        self.embedding_dim = embedding_dim
        self.hidden_channels = hidden_channels
        self.residual_blocks = residual_blocks
        self.embedding = nn.Embedding(num_tokens, embedding_dim)
        self.input_conv = MaskedConv2d(
            "A",
            embedding_dim,
            hidden_channels,
            7,
            padding=3,
        )
        self.residual_stack = nn.Sequential(
            *(PixelCNNResidualBlock(hidden_channels) for _ in range(residual_blocks))
        )
        self.output_head = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_channels, num_tokens, 1),
        )

    def forward(self, token_grids: Tensor) -> Tensor:
        if token_grids.ndim != 3:
            raise ValueError(
                f"Expected token grids with shape [batch, height, width], got {token_grids.shape}."
            )
        if token_grids.dtype != torch.long:
            raise TypeError("PixelCNN token grids must use torch.long indices.")
        if token_grids.numel() == 0:
            raise ValueError("PixelCNN received an empty token grid.")
        if int(token_grids.min()) < 0 or int(token_grids.max()) >= self.num_tokens:
            raise ValueError("A PixelCNN token is outside the compact vocabulary.")

        embeddings = self.embedding(token_grids).permute(0, 3, 1, 2).contiguous()
        features = self.input_conv(embeddings)
        features = self.residual_stack(features)
        return self.output_head(features)

    @torch.inference_mode()
    def sample(
        self,
        batch_size: int,
        grid_shape: tuple[int, int],
        device: torch.device,
        temperature: float = 1.0,
        top_k: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Draw categorical grids one position at a time in raster order."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if len(grid_shape) != 2 or min(grid_shape) <= 0:
            raise ValueError("grid_shape must contain two positive dimensions.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if top_k is not None and not 1 <= top_k <= self.num_tokens:
            raise ValueError("top_k must lie within the compact vocabulary size.")

        samples = torch.zeros((batch_size, *grid_shape), dtype=torch.long, device=device)
        was_training = self.training
        self.eval()
        try:
            for row in range(grid_shape[0]):
                for column in range(grid_shape[1]):
                    logits = self(samples)[:, :, row, column] / temperature
                    if top_k is not None:
                        top_values, top_indices = logits.topk(top_k, dim=1)
                        probabilities = F.softmax(top_values, dim=1)
                        selected = torch.multinomial(
                            probabilities,
                            1,
                            generator=generator,
                        )
                        next_tokens = top_indices.gather(1, selected).squeeze(1)
                    else:
                        probabilities = F.softmax(logits, dim=1)
                        next_tokens = torch.multinomial(
                            probabilities,
                            1,
                            generator=generator,
                        ).squeeze(1)
                    samples[:, row, column] = next_tokens
        finally:
            self.train(was_training)
        return samples


def vae_loss(outputs: Dict[str, Tensor], targets: Tensor, beta: float = 1e-4) -> Dict[str, Tensor]:
    reconstruction_loss = F.mse_loss(outputs["reconstruction"], targets)
    kl_loss = -0.5 * torch.mean(1.0 + outputs["logvar"] - outputs["mu"].pow(2) - outputs["logvar"].exp())
    return {
        "loss": reconstruction_loss + beta * kl_loss,
        "reconstruction_loss": reconstruction_loss,
        "regularisation_loss": kl_loss,
    }


def vqvae_loss(outputs: Dict[str, Tensor], targets: Tensor) -> Dict[str, Tensor]:
    reconstruction_loss = F.mse_loss(outputs["reconstruction"], targets)
    return {
        "loss": reconstruction_loss + outputs["quantizer_loss"],
        "reconstruction_loss": reconstruction_loss,
        "regularisation_loss": outputs["quantizer_loss"],
    }
