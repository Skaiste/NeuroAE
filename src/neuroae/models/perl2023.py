from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Model used in:
Sanz Perl Y, Pallavicini C, Piccinini J, Demertzi A, Vonhomme V, Martial C, et al. 
Low-dimensional organization of global brain states of reduced consciousness. 
Cell Press. 2023;42(5):112491.
https://github.com/yonisanzperl/Low_dimensional_organization/tree/LDR
"""
class Perl2023(nn.Module):
    """Fully-connected VAE used for FC latent-space perturbation analyses."""

    def __init__(
        self,
        input_dim: int = 8100,
        intermediate_dim: int = 1028,
        latent_dim: int = 2,
        output_activation: str = "sigmoid",
    ):
        super().__init__()
        if output_activation not in ("sigmoid", "identity"):
            raise ValueError("output_activation must be 'sigmoid' or 'identity'")

        self.input_dim = int(input_dim)
        self.intermediate_dim = int(intermediate_dim)
        self.latent_dim = int(latent_dim)
        self.output_activation = output_activation

        self.encoder_backbone = nn.Sequential(
            nn.Linear(self.input_dim, self.intermediate_dim),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(self.intermediate_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.intermediate_dim, self.latent_dim)

        self.decoder_backbone = nn.Sequential(
            nn.Linear(self.latent_dim, self.intermediate_dim),
            nn.ReLU(inplace=True),
        )
        self.fc_out = nn.Linear(self.intermediate_dim, self.input_dim)

        self.loss_fn_params = {}

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"Expected x shape (B, {self.input_dim}), got {tuple(x.shape)}")
        h = self.encoder_backbone(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        # Stabilize KL term under noisy training.
        logvar = torch.clamp(logvar, -10.0, 10.0)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(f"Expected z shape (B, {self.latent_dim}), got {tuple(z.shape)}")
        h = self.decoder_backbone(z)
        logits = self.fc_out(h)
        if self.output_activation == "sigmoid":
            return torch.sigmoid(logits)
        return logits

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z

    def set_loss_fn_params(self, params):
        if params is not None:
            self.loss_fn_params = {k: v for p in params for k, v in p.items()}
        else:
            self.loss_fn_params = {}

    def loss(self, x, model_output):
        x_hat, mu, logvar, _ = model_output
        error_per_feature = self.loss_fn_params.get("loss_per_feature", True)
        recon_distribution = self.loss_fn_params.get("recon_distribution", "bernoulli")
        beta = float(self.loss_fn_params.get("beta", 1.0))

        if recon_distribution == "bernoulli":
            # use averaging across all features
            if error_per_feature:
                recon = F.binary_cross_entropy(x_hat, x, reduction="mean")
            else:
                # Match Keras pattern: per-sample BCE summed across features, then batch mean.
                recon = F.binary_cross_entropy(x_hat, x, reduction="none").sum(dim=1).mean()
        elif recon_distribution == "gaussian":
            if error_per_feature:
                recon = F.mse_loss(x_hat, x, reduction="mean")
            else:
                recon = F.mse_loss(x_hat, x, reduction="none").sum(dim=1).mean()
        else:
            raise ValueError("recon_distribution must be 'bernoulli' or 'gaussian'")

        kld = 0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar, dim=1)
        kld = kld.mean()

        return {
            "loss": recon + beta * kld,
            "recon": recon,
            "kld": kld,
        }

    @torch.no_grad()
    def project_latent(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        """Project inputs to latent space.

        Args:
            x: Tensor with shape ``(B, input_dim)``.
            sample: If ``True``, returns sampled z; else returns deterministic mean.
        """
        mu, logvar = self.encode(x)
        if sample:
            return self.reparameterize(mu, logvar)
        return mu


def project_to_latent(
    model: Perl2023,
    x: torch.Tensor,
    *,
    sample: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Utility wrapper for latent projection outside training loops."""
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    was_training = model.training
    model.eval()
    z = model.project_latent(x.to(device), sample=sample)
    if was_training:
        model.train()
    return z
