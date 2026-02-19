import math
from dataclasses import dataclass
from typing import Sequence, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Perl2023(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 4,
        hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.0,
        use_layernorm: bool = True,
        recon_distribution: str = "gaussian",  # "gaussian" or "bernoulli"
    ):
        super().__init__()
        if recon_distribution not in ("gaussian", "bernoulli"):
            raise ValueError("recon_distribution must be 'gaussian' or 'bernoulli'")

        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.recon_distribution = recon_distribution

        def mlp_block(in_f: int, out_f: int) -> nn.Sequential:
            layers = [nn.Linear(in_f, out_f)]
            if use_layernorm:
                layers.append(nn.LayerNorm(out_f))
            layers.append(nn.GELU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            return nn.Sequential(*layers)

        # -------- Encoder backbone --------
        enc = []
        prev = self.input_dim
        for h in self.hidden_dims:
            enc.append(mlp_block(prev, h))
            prev = h
        self.encoder = nn.Sequential(*enc)

        self.fc_mu = nn.Linear(prev, self.latent_dim)
        self.fc_logvar = nn.Linear(prev, self.latent_dim)

        # -------- Decoder backbone --------
        dec = []
        prev = self.latent_dim
        for h in reversed(self.hidden_dims):
            dec.append(mlp_block(prev, h))
            prev = h
        self.decoder = nn.Sequential(*dec)
        self.fc_out = nn.Linear(prev, self.input_dim)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # std = exp(0.5*logvar)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"Expected x shape (B, {self.input_dim}), got {tuple(x.shape)}")
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(f"Expected z shape (B, {self.latent_dim}), got {tuple(z.shape)}")
        h = self.decoder(z)
        x_logits_or_mean = self.fc_out(h)
        if self.recon_distribution == "bernoulli":
            # For Bernoulli likelihood you typically output logits
            return x_logits_or_mean
        # For Gaussian likelihood interpret as mean
        return x_logits_or_mean

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar#, z

    def set_loss_fn_params(self, params):
        if params is not None:
            self.loss_fn_params = {k:v for p in params for k,v in p.items()}
        else:
            self.loss_fn_params = {}

    def loss(self, x, model_output):
        x_hat, mu, log_var = model_output
        beta = float(self.loss_fn_params.get("beta", 1.0))
        recon_distribution = self.loss_fn_params.get("recon_distribution", "gaussian")

        # per-sample reconstruction loss
        if recon_distribution == "gaussian":
            # mean over features, keep batch
            recon = F.mse_loss(x_hat, x, reduction="none").mean(dim=1)
        elif recon_distribution == "bernoulli":
            recon = F.binary_cross_entropy_with_logits(x_hat, x, reduction="none").mean(dim=1)
        else:
            raise ValueError("recon_distribution must be 'gaussian' or 'bernoulli'")
        
        # kl divergence: KL(q(z|x) || N(0, I)) per-sample.
        kld = 0.5 * torch.sum(torch.exp(log_var) + mu**2 - 1.0 - log_var, dim=1)

        return {
            'loss': (recon + beta * kld).mean(),
            'recon': recon.mean(), 
            'kld': kld.mean()
        }