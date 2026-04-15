import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ModelBase


class _ConvEncoder(nn.Module):
    def __init__(self, hidden_channels, latent_dim, kernel_size):
        super().__init__()
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels]
        if len(hidden_channels) == 0:
            hidden_channels = [32, 64]

        padding = kernel_size // 2
        layers = []
        in_channels = 1
        for channels in hidden_channels:
            layers.append(nn.Conv1d(in_channels, channels, kernel_size=kernel_size, padding=padding))
            layers.append(nn.GELU())
            in_channels = channels
        self.features = nn.Sequential(*layers)
        self.to_latent = nn.Conv1d(in_channels, 1, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool1d(latent_dim)

    def forward(self, x):
        h = self.features(x)
        h = self.to_latent(h)
        return self.pool(h)


class _ConvDecoder(nn.Module):
    def __init__(self, hidden_channels, output_dim, kernel_size):
        super().__init__()
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels]
        if len(hidden_channels) == 0:
            hidden_channels = [32, 64]

        padding = kernel_size // 2
        decoder_channels = list(hidden_channels[::-1])

        self.expand = nn.Conv1d(1, decoder_channels[0], kernel_size=1)
        self.upsample = nn.Upsample(size=output_dim, mode="linear", align_corners=False)

        layers = []
        in_channels = decoder_channels[0]
        for channels in decoder_channels[1:]:
            layers.append(nn.Conv1d(in_channels, channels, kernel_size=kernel_size, padding=padding))
            layers.append(nn.GELU())
            in_channels = channels
        layers.append(nn.Conv1d(in_channels, 1, kernel_size=kernel_size, padding=padding))
        self.reconstruction = nn.Sequential(*layers)

    def forward(self, z):
        h = self.expand(z)
        h = self.upsample(h)
        return self.reconstruction(h)


class ConvAE(ModelBase):
    """
    Convolutional autoencoder for inputs shaped (B, T, R).

    The model applies the same 1D convolutional encoder/decoder to each timepoint:
    input  : (B, T, R)
    latent : (B, T, L)
    recon  : (B, T, R)

    If ``variational=True``, ``forward`` returns ``(x_hat, mu, log_var, z)``.
    Otherwise it returns ``(x_hat, z)``.
    """

    def __init__(
        self,
        regions: int,
        timepoints: int,
        latent_dim: int,
        hidden_channels=(32, 64),
        kernel_size: int = 3,
        variational: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        if regions <= 0:
            raise ValueError("regions must be > 0.")
        if timepoints <= 0:
            raise ValueError("timepoints must be > 0.")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be > 0.")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")

        self.regions = int(regions)
        self.timepoints = int(timepoints)
        self.latent_dim = int(latent_dim)
        self.hidden_channels = tuple(hidden_channels) if not isinstance(hidden_channels, int) else (hidden_channels,)
        self.kernel_size = int(kernel_size)
        self.variational = bool(variational)
        self.device = device

        self.encoder = _ConvEncoder(
            hidden_channels=self.hidden_channels,
            latent_dim=self.latent_dim,
            kernel_size=self.kernel_size,
        ).to(device)

        if self.variational:
            self.mu_head = nn.Conv1d(1, 1, kernel_size=1).to(device)
            self.logvar_head = nn.Conv1d(1, 1, kernel_size=1).to(device)
        else:
            self.mu_head = None
            self.logvar_head = None

        self.decoder = _ConvDecoder(
            hidden_channels=self.hidden_channels,
            output_dim=self.regions,
            kernel_size=self.kernel_size,
        ).to(device)

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        if self.mu_head is not None:
            for param in self.mu_head.parameters():
                param.requires_grad = False
        if self.logvar_head is not None:
            for param in self.logvar_head.parameters():
                param.requires_grad = False

    def reset_decoder(self):
        self.decoder = _ConvDecoder(
            hidden_channels=self.hidden_channels,
            output_dim=self.regions,
            kernel_size=self.kernel_size,
        ).to(self.device)

    def _check_input(self, x):
        if x.ndim != 3:
            raise ValueError(f"Expected x shape (B, T, R), got {tuple(x.shape)}")
        if x.shape[1] != self.timepoints:
            raise ValueError(f"Expected T={self.timepoints}, got {x.shape[1]}")
        if x.shape[2] != self.regions:
            raise ValueError(f"Expected R={self.regions}, got {x.shape[2]}")

    def _to_conv_input(self, x):
        self._check_input(x)
        return x.reshape(x.shape[0] * x.shape[1], 1, x.shape[2])

    def _from_conv_output(self, x, batch_size):
        return x.reshape(batch_size, self.timepoints, -1)

    def reparameterize(self, mean, std):
        epsilon = torch.randn_like(std).to(mean.device)
        return mean + std * epsilon

    def encode(self, x):
        batch_size = x.shape[0]
        x_conv = self._to_conv_input(x)
        latent_base = self.encoder(x_conv)

        if not self.variational:
            return self._from_conv_output(latent_base.squeeze(1), batch_size)

        mu = self.mu_head(latent_base).squeeze(1)
        log_var = self.logvar_head(latent_base).squeeze(1)
        log_var = torch.clamp(log_var, -10.0, 10.0)
        return (
            self._from_conv_output(mu, batch_size),
            self._from_conv_output(log_var, batch_size),
        )

    def decode(self, z):
        if z.ndim != 3:
            raise ValueError(f"Expected z shape (B, T, L), got {tuple(z.shape)}")
        if z.shape[1] != self.timepoints:
            raise ValueError(f"Expected T={self.timepoints}, got {z.shape[1]}")
        if z.shape[2] != self.latent_dim:
            raise ValueError(f"Expected L={self.latent_dim}, got {z.shape[2]}")

        batch_size = z.shape[0]
        z_conv = z.reshape(batch_size * self.timepoints, 1, self.latent_dim)
        x_hat = self.decoder(z_conv).squeeze(1)
        return self._from_conv_output(x_hat, batch_size)

    def forward(self, x):
        if not self.variational:
            z = self.encode(x)
            x_hat = self.decode(z)
            return x_hat, z

        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, torch.exp(0.5 * log_var))
        x_hat = self.decode(z)
        return x_hat, mu, log_var, z

    def loss(self, x, model_output):
        loss_fn_params = getattr(self, "loss_fn_params", {})
        error_per_feature = loss_fn_params.get("loss_per_feature", True)

        if not self.variational:
            x_hat, _ = model_output
            if error_per_feature:
                recon = F.mse_loss(x_hat, x, reduction="mean")
            else:
                recon = F.mse_loss(x_hat, x, reduction="none")
                recon = recon.flatten(1).sum(dim=1).mean()

            loss = {
                "loss": recon,
                "recon": recon,
            }
        else:
            x_hat, mu, log_var, _ = model_output
            beta = float(loss_fn_params.get("beta", 1.0))
            if error_per_feature:
                recon = F.mse_loss(x_hat, x, reduction="mean")
            else:
                recon = F.mse_loss(x_hat, x, reduction="none")
                recon = recon.flatten(1).sum(dim=1).mean()

            mu_flat = mu.reshape(mu.shape[0], -1)
            log_var_flat = log_var.reshape(log_var.shape[0], -1)
            kld = -0.5 * (1 + log_var_flat - mu_flat.pow(2) - log_var_flat.exp())
            kld = kld.sum(dim=1).mean() / log_var_flat.size(1)

            loss = {
                "loss": recon + beta * kld,
                "recon": recon,
                "kld": kld,
            }

        if self.swfcd is not None:
            swfcd = self.swfcd.apply(x, model_output[0])
            swfcd_beta = loss_fn_params.get("swfcd_beta", 1.0)
            loss["swfcd_rmse"] = swfcd["rmse"]
            loss["loss"] += swfcd_beta * swfcd["rmse"]

        return loss


class ConvVAE(ConvAE):
    def __init__(self, *args, **kwargs):
        kwargs["variational"] = True
        super().__init__(*args, **kwargs)
