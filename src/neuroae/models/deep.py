"""Deep timepoint-wise autoencoders.

``DAE`` behaves as a deterministic autoencoder when ``loss_fn_params.beta`` is
zero (the default), and as a variational autoencoder otherwise.  The legacy
``VAE`` names are retained below as aliases for backwards compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ModelBase
from .head import PredHeadAvg, PredHeadConv, PredHeadTemporalPool, PredHeadGatedTemporalPool


class DAE(ModelBase):
    class Encoder(nn.Module):
        def __init__(self, region_dim, hidden_dims, latent_dim):
            super().__init__()
            layers, last_dim = [], region_dim
            for dim in hidden_dims:
                layers.extend([nn.Linear(last_dim, dim), nn.GELU()])
                last_dim = dim
            self.fc = nn.Sequential(*layers)
            self.fc_mean = nn.Linear(last_dim, latent_dim)
            self.fc_logvar = nn.Linear(last_dim, latent_dim)

        def forward(self, x, variational=True):
            h = self.fc(x)
            mean = self.fc_mean(h)
            if not variational:
                return mean
            return mean, torch.clamp(self.fc_logvar(h), -10.0, 10.0)

    class Decoder(nn.Module):
        def __init__(self, latent_dim, hidden_dims, region_dim):
            super().__init__()
            layers, last_dim = [], latent_dim
            for dim in reversed(hidden_dims):
                layers.extend([nn.Linear(last_dim, dim), nn.GELU()])
                last_dim = dim
            layers.append(nn.Linear(last_dim, region_dim))
            self.fc = nn.Sequential(*layers)

        def forward(self, z):
            return self.fc(z)

    def __init__(self, region_dim, timepoint_dim, latent_dim, hidden_dims,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.region_dim, self.timepoint_dim, self.latent_dim = map(int, (region_dim, timepoint_dim, latent_dim))
        self.hidden_dims = [hidden_dims] if isinstance(hidden_dims, int) else list(hidden_dims)
        self.device = device
        if self.region_dim <= 0 or self.timepoint_dim <= 0 or self.latent_dim <= 0:
            raise ValueError("region_dim, timepoint_dim, and latent_dim must be > 0.")
        if not self.hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer size.")
        self.feature_dim = self.region_dim
        self.input_dim = self.region_dim * self.timepoint_dim
        self.latent_per_timepoint = self.latent_dim
        self.latent_flat_dim = self.timepoint_dim * self.latent_dim
        self.encoder = self.Encoder(self.region_dim, self.hidden_dims, self.latent_dim).to(device)
        self.decoder = self.Decoder(self.latent_dim, self.hidden_dims, self.region_dim).to(device)

    def _beta(self):
        return float(getattr(self, "loss_fn_params", {}).get("beta", 0.0))

    def _variational_enabled(self):
        return self._beta() != 0.0

    def _reshape_input(self, x):
        if x.ndim != 3 or x.shape[1:] != (self.timepoint_dim, self.region_dim):
            raise ValueError(f"Expected x shape (B, {self.timepoint_dim}, {self.region_dim}), got {tuple(x.shape)}")
        return x

    def _flatten_latent(self, z_time):
        return z_time.reshape(z_time.shape[0], self.latent_flat_dim)

    def _reshape_latent(self, z):
        if z.ndim != 2 or z.shape[1] != self.latent_flat_dim:
            raise ValueError(f"Expected z shape (B, {self.latent_flat_dim}), got {tuple(z.shape)}")
        return z.reshape(z.shape[0], self.timepoint_dim, self.latent_dim)

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def reset_decoder(self):
        self.decoder = self.Decoder(self.latent_dim, self.hidden_dims, self.region_dim).to(self.device)

    @staticmethod
    def reparameterize(mean, std):
        return mean + std * torch.randn_like(std)

    def forward(self, x):
        x_time = self._reshape_input(x)
        if not self._variational_enabled():
            z_time = self.encoder(x_time, variational=False)
            return self.decoder(z_time), self._flatten_latent(z_time)
        mean_time, log_var_time = self.encoder(x_time, variational=True)
        mean, log_var = self._flatten_latent(mean_time), self._flatten_latent(log_var_time)
        z = self.reparameterize(mean, torch.exp(0.5 * log_var)) if self.training else mean
        z_time = self._reshape_latent(z)
        return self.decoder(z_time), mean, log_var, z_time.transpose(2, 1)

    def loss(self, x, model_output):
        if len(model_output) == 2:
            x_hat, _ = model_output
            recon = self.reconstruction_loss(x_hat, x)
            loss = {"loss": recon, "recon": recon, "kld": torch.zeros((), device=x.device, dtype=x.dtype)}
        else:
            x_hat, mu, log_var, _ = model_output
            recon = self.reconstruction_loss(x_hat, x)
            kld = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
            kld = kld.sum(dim=1).mean() / log_var.size(1)
            loss = {"loss": recon + self._beta() * kld, "recon": recon, "kld": kld}
        return self.add_weighted_reconstruction_losses(loss, x, x_hat)


class DAEPredHeads(DAE):
    """DAE with temporal prediction heads for biomarker prediction."""

    def __init__(self, region_dim, timepoint_dim, pred_head_type="gated_temp_pool",
                 pred_head_num=1, hidden_dims=(1024, 512, 256, 128), latent_dim=32,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(region_dim, timepoint_dim, latent_dim, hidden_dims, device)
        head_types = {"avg": PredHeadAvg, "conv": PredHeadConv,
                      "conv_no_hidden": lambda l, r: PredHeadConv(l, r, with_hidden=False),
                      "temp_pool": PredHeadTemporalPool, "gated_temp_pool": PredHeadGatedTemporalPool}
        if pred_head_type not in head_types:
            raise ValueError(f"Selected prediction head type - '{pred_head_type}' is not available.")
        self.heads = nn.ModuleList([head_types[pred_head_type](latent_dim, self.region_dim) for _ in range(pred_head_num)])

    def forward(self, x):
        output = super().forward(x)
        if len(output) == 2:
            x_hat, z = output
            z_heads = [head(self._reshape_latent(z).transpose(2, 1)) for head in self.heads]
            return x_hat, z_heads, z
        x_hat, mean, log_var, z = output
        return x_hat, mean, log_var, [head(z) for head in self.heads], z

    def loss(self, x, x_heads, model_output):
        if len(model_output) == 3:
            x_hat, z_heads, _ = model_output
            mu = log_var = None
        else:
            x_hat, mu, log_var, z_heads, _ = model_output
        recon = self.reconstruction_loss(x_hat, x)
        loss = {"loss": recon, "recon": recon, "kld": torch.zeros((), device=x.device, dtype=x.dtype)}
        if mu is not None:
            kld = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
            kld = kld.sum(dim=1).mean() / log_var.size(1)
            loss.update(kld=kld, loss=loss["loss"] + self._beta() * kld)
        assert len(x_heads) == len(z_heads), "label heads and predicted heads must have the same length"
        head_losses = []
        for index, label in enumerate(x_heads):
            head_loss = F.smooth_l1_loss(z_heads[index], x_heads[label], reduction="mean", beta=1.0)
            head_losses.append(head_loss)
            loss[f"{label}_loss"] = head_loss
        if head_losses:
            loss["loss"] += float(getattr(self, "loss_fn_params", {}).get("pred_heads_delta", 0.0)) * sum(head_losses) / len(head_losses)
        return self.add_weighted_reconstruction_losses(loss, x, x_hat)


class VDAE(DAE):
    """Variational DAE with a default KL weight of one."""

    def _beta(self):
        return float(getattr(self, "loss_fn_params", {}).get("beta", 1.0))


class VAE(VDAE):
    """Legacy alias for :class:`VDAE`."""


class VAEPredHeads(DAEPredHeads):
    """Legacy prediction-head variant, preserving its historical ``beta`` default."""

    def _beta(self):
        return float(getattr(self, "loss_fn_params", {}).get("beta", 0.5))
