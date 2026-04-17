import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ModelBase
from .head import PredHeadAvg, PredHeadConv, PredHeadTemporalPool, PredHeadGatedTemporalPool


class VAE(ModelBase):
    class Encoder(nn.Module):
        def __init__(self, region_dim, hidden_dims, latent_dim):
            super().__init__()
            if type(hidden_dims) == int:
                hidden_dims = [hidden_dims]

            layers = []
            last_dim = region_dim
            for dim in hidden_dims:
                layers.append(nn.Linear(last_dim, dim))
                layers.append(nn.LeakyReLU(0.2))
                last_dim = dim
            self.fc = nn.Sequential(*layers)
            self.fc_mean = nn.Linear(last_dim, latent_dim)
            self.fc_logvar = nn.Linear(last_dim, latent_dim)

        def forward(self, x):
            # x shape: (B, T, F)
            h = self.fc(x)
            mean = self.fc_mean(h)
            log_var = self.fc_logvar(h)
            log_var = torch.clamp(log_var, -10.0, 10.0)
            return mean, log_var

    class Decoder(nn.Module):
        def __init__(self, latent_dim, hidden_dims, region_dim):
            super().__init__()

            if type(hidden_dims) == int:
                hidden_dims = [hidden_dims]

            layers = []
            last_dim = latent_dim
            for dim in hidden_dims[::-1]:
                layers.append(nn.Linear(last_dim, dim))
                layers.append(nn.LeakyReLU(0.2))
                last_dim = dim
            layers.append(nn.Linear(last_dim, region_dim))
            self.fc = nn.Sequential(*layers)

        def forward(self, z):
            # z shape: (B, T, L)
            return self.fc(z)

    def __init__(
        self,
        region_dim,
        timepoint_dim,
        latent_dim,
        hidden_dims,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super(VAE, self).__init__()
        self.region_dim = int(region_dim)
        self.timepoint_dim = int(timepoint_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dims = hidden_dims
        self.device = device

        if self.region_dim <= 0:
            raise ValueError("region_dim must be > 0.")
        if self.timepoint_dim <= 0:
            raise ValueError("timepoint_dim must be > 0.")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be > 0.")
        if isinstance(self.hidden_dims, int):
            self.hidden_dims = [self.hidden_dims]
        if len(self.hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one layer size.")

        # Derived compatibility attributes for existing code paths.
        self.feature_dim = self.region_dim
        self.input_dim = self.region_dim * self.timepoint_dim
        self.latent_per_timepoint = self.latent_dim
        self.latent_flat_dim = self.timepoint_dim * self.latent_dim

        self.encoder = self.Encoder(
            region_dim=self.region_dim,
            hidden_dims=self.hidden_dims,
            latent_dim=self.latent_dim,
        ).to(device)
        self.decoder = self.Decoder(
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims,
            region_dim=self.region_dim,
        ).to(device)

    def _reshape_input(self, x):
        expected_shape = (self.timepoint_dim, self.region_dim)
        if x.ndim != 3 or x.shape[1:] != expected_shape:
            raise ValueError(
                f"Expected x shape (B, {self.timepoint_dim}, {self.region_dim}), got {tuple(x.shape)}"
            )
        return x

    def _flatten_latent(self, z_time):
        # (B, T, L) -> (B, T*L)
        return z_time.reshape(z_time.shape[0], self.latent_flat_dim)

    def _reshape_latent(self, z):
        if z.ndim != 2 or z.shape[1] != self.latent_flat_dim:
            raise ValueError(f"Expected z shape (B, {self.latent_flat_dim}), got {tuple(z.shape)}")
        return z.reshape(z.shape[0], self.timepoint_dim, self.latent_dim)

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def reset_decoder(self):
        self.decoder = self.Decoder(
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims,
            region_dim=self.region_dim,
        ).to(self.device)

    def reparameterize(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)
        z = mean + var * epsilon
        return z

    def forward(self, x):
        x_time = self._reshape_input(x)
        mean_time, log_var_time = self.encoder(x_time)
        log_var_time = torch.clamp(log_var_time, -10.0, 10.0)

        mean = self._flatten_latent(mean_time)
        log_var = self._flatten_latent(log_var_time)
        z = self.reparameterize(mean, torch.exp(0.5 * log_var))

        z_time = self._reshape_latent(z)
        x_hat_time = self.decoder(z_time)
        return x_hat_time, mean, log_var, z_time.transpose(2, 1)

    def loss(self, x, model_output):
        x_hat, mu, log_var, _ = model_output
        error_per_feature = self.loss_fn_params.get("loss_per_feature", True)
        beta = float(self.loss_fn_params.get("beta", 1.0))
        if error_per_feature:
            recon = F.mse_loss(x_hat, x, reduction="mean")
            kld = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
            kld = kld.sum(dim=1).mean() / log_var.size(1)
        else:
            recon = F.mse_loss(x_hat, x, reduction="none")
            recon = recon.sum(dim=1).mean()
            kld = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
            kld = kld.sum(dim=1).mean() / log_var.size(1)

        loss = {
            'loss': recon + beta * kld,
            'recon': recon, 
            'kld': kld
        }

        if self.swfcd is not None:
            swfcd = self.swfcd.apply(x, x_hat)
            swfcd_beta = self.loss_fn_params.get("swfcd_beta", 1.0)
            loss['swfcd_rmse'] = swfcd['rmse']
            loss['loss'] += swfcd_beta * swfcd['rmse']

        return loss


class VAEPredHeads(VAE):
    """ VAE + linear temporal models for ABeta and Tau level prediction 
            Assumes latent dimension preserves the time dimension
    """
    def __init__(
        self,
        region_dim,
        timepoint_dim,
        pred_head_type: str = "gated_temp_pool",
        pred_head_num: int = 1,
        hidden_dims=[1024, 512, 256, 128], 
        latent_dim=32,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ) -> None:
        super().__init__(
            region_dim=region_dim,
            timepoint_dim=timepoint_dim,
            hidden_dims=hidden_dims, 
            latent_dim=latent_dim,
            device=device,
        )
        pred_head_idx = {
            "avg": PredHeadAvg,
            "conv": PredHeadConv,
            "conv_no_hidden": lambda l, r: PredHeadConv(l, r, with_hidden=False),
            "temp_pool": PredHeadTemporalPool,
            "gated_temp_pool": PredHeadGatedTemporalPool
        }
        if pred_head_type not in pred_head_idx:
            raise ValueError(f"Selected prediction head type - '{pred_head_type}' is not available.")
        
        self.heads = [pred_head_idx[pred_head_type](latent_dim, self.region_dim) for i in range(pred_head_num)]

    def to(self, device):
        self.heads = [h.to(device) for h in self.heads]
        return super().to(device)

    def forward(self, x):
        x_hat, mean, log_var, z = super().forward(x)
        z_heads = [h(z) for h in self.heads]
        return x_hat, mean, log_var, z_heads, z
    
    def loss(self, x, x_heads, model_output):
        x_hat, mu, log_var, z_heads, _ = model_output
        beta = self.loss_fn_params.get("beta", 0.5)
        error_per_feature = self.loss_fn_params.get("loss_per_feature", True)
        pred_heads_delta = float(self.loss_fn_params.get("pred_heads_delta", 0.0))
        
        # if selected error per feature, we are averaging everything
        if error_per_feature:
            # recon: mean mse loss
            recon = F.mse_loss(x_hat, x, reduction="mean")

            # KL: mean over batch, then mean over latent dims
            kld = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
            kld = kld.sum(dim=1).mean() / log_var.size(1)

        # if selected error per sample, we are summing everything
        else:
            # recon: sum over features per sample, then mean over batch
            recon = F.mse_loss(x_hat, x, reduction="none")  # [B, D]
            recon = recon.sum(dim=1).mean()

            # kld: sum over latent dims per sample, then mean over batch
            kld = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
            kld = kld.sum(dim=1).mean() / log_var.size(1)

        loss = {
            'loss': recon + beta * kld,
            'recon': recon, 
            'kld': kld
        }

        # calculate loss from prediction heads
        assert len(x_heads) == len(z_heads), f"label heads ({len(x_heads)}) is not the same as predicted heads ({len(z_heads)})"
        pred_head_loss = []
        for i, bl in enumerate(x_heads):
            head_loss = F.smooth_l1_loss(z_heads[i], x_heads[bl], reduction="mean", beta=1.0)
            pred_head_loss.append(head_loss)
            loss[f"{bl}_loss"] = head_loss

        if len(pred_head_loss) > 0:
            loss['loss'] += pred_heads_delta * sum(pred_head_loss) / len(pred_head_loss)

        if self.swfcd is not None:
            swfcd = self.swfcd.apply(x, x_hat)
            swfcd_beta = self.loss_fn_params.get("swfcd_beta", 1.0)
            loss['swfcd_rmse'] = swfcd['rmse']
            loss['loss'] += swfcd_beta * swfcd['rmse']

        return loss
