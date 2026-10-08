"""Model definitions for the HipMRI VAE and VQ-VAE comparison."""

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
    """Nearest-neighbour codebook with the straight-through estimator."""

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, inputs: Tensor) -> Dict[str, Tensor]:
        if inputs.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected {self.embedding_dim} latent channels, received {inputs.shape[1]}."
            )

        channels_last = inputs.permute(0, 2, 3, 1).contiguous()
        flat_inputs = channels_last.view(-1, self.embedding_dim)
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
        self.pre_quantizer = nn.Conv2d(self.encoder.out_channels, embedding_dim, 1)
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
