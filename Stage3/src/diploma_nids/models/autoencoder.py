"""Autoencoder and VAE for semi-supervised anomaly detection.

Trained on normal traffic only; anomaly score = reconstruction MSE
(AE) or negative ELBO (VAE). Threshold = configured quantile of the
score on a clean validation set.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .base import DLDetector, ModelOutput, register


def _build_mlp(dims: list[int], dropout: float, activation: nn.Module) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers += [nn.Linear(dims[i], dims[i + 1]), activation, nn.Dropout(dropout)]
    return nn.Sequential(*layers[:-1])  # drop last dropout for cleanliness


@register("autoencoder")
class Autoencoder(DLDetector):
    """Window is mean-pooled to a flow-vector before encoding."""

    expects_windows = False

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        arch = config["architecture"]
        d_in = int(arch["input_features"])
        enc_dims = [d_in, *[int(d) for d in arch["encoder_dims"]]]
        dec_dims = [enc_dims[-1], *[int(d) for d in arch["decoder_dims"]], d_in]
        drop = float(arch.get("dropout", 0.0))
        act = nn.ReLU()
        self.encoder = _build_mlp(enc_dims, drop, act)
        self.decoder = _build_mlp(dec_dims, drop, act)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        if x.ndim == 3:
            x = x.mean(dim=1)
        z = self.encoder(x)
        recon = self.decoder(z)
        err = ((x - recon) ** 2).mean(dim=-1)   # per-sample reconstruction MSE
        return ModelOutput(logits=err, probs=err, extras={"recon": recon, "z": z})


@register("vae")
class VAE(DLDetector):
    expects_windows = False

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        arch = config["architecture"]
        d_in = int(arch["input_features"])
        enc_dims = [d_in, *[int(d) for d in arch["encoder_dims"]]]
        latent = int(arch["latent_dim"])
        dec_dims = [latent, *[int(d) for d in arch["decoder_dims"]], d_in]
        drop = float(arch.get("dropout", 0.0))
        act = nn.ReLU()

        self.encoder = _build_mlp(enc_dims, drop, act)
        self.mu = nn.Linear(enc_dims[-1], latent)
        self.logvar = nn.Linear(enc_dims[-1], latent)
        self.decoder = _build_mlp(dec_dims, drop, act)
        self.kl_weight = float(arch.get("kl_weight", 0.1))

    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> ModelOutput:
        if x.ndim == 3:
            x = x.mean(dim=1)
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = self._reparameterize(mu, logvar)
        recon = self.decoder(z)

        recon_err = ((x - recon) ** 2).sum(dim=-1)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        # Negative ELBO acts as anomaly score
        score = recon_err + self.kl_weight * kl
        return ModelOutput(
            logits=score,
            probs=score,
            extras={"recon": recon, "mu": mu, "logvar": logvar, "kl": kl, "recon_err": recon_err},
        )
