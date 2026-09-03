import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ModelBase
from .head import PredHeadAvg, PredHeadConv, PredHeadTemporalPool, PredHeadGatedTemporalPool


def _region_reduction_params(regions, latent_dim):
    """Return convolution parameters that map a length of ``regions`` to ``latent_dim``."""
    if latent_dim > regions:
        raise ValueError(
            "latent_dim must be <= regions when using learned regional downsampling."
        )

    stride = regions // latent_dim
    kernel_size = regions - stride * (latent_dim - 1)
    return kernel_size, stride


class _ConvEncoder(nn.Module):
    def __init__(self, hidden_channels, latent_dim, kernel_size, region_kernel_size, region_stride):
        super().__init__()
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels]

        padding = kernel_size // 2
        layers = []
        in_channels = 1
        for channels in hidden_channels:
            layers.append(nn.Conv1d(in_channels, channels, kernel_size=kernel_size, padding=padding))
            layers.append(nn.GELU())
            in_channels = channels
        self.features = nn.Sequential(*layers)
        # Reduce the region axis with a separate learned filter for each
        # hidden channel: (N, H, R) -> (N, H, L).
        self.reduce_regions = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size=region_kernel_size,
                stride=region_stride,
                groups=in_channels,
            ),
            nn.GELU(),
        )
        # Mix the H features at each latent position: (N, H, L) -> (N, 1, L).
        self.to_latent = nn.Conv1d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        h = self.features(x)
        h = self.reduce_regions(h)
        return self.to_latent(h)


class _ConvDecoder(nn.Module):
    def __init__(self, hidden_channels, kernel_size, region_kernel_size, region_stride):
        super().__init__()
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels]

        padding = kernel_size // 2
        decoder_channels = list(hidden_channels[::-1]) or [1]

        self.expand = nn.Conv1d(1, decoder_channels[0], kernel_size=1)
        # Restore the encoder's regional length exactly:
        # (L - 1) * stride + kernel_size == R.
        self.expand_regions = nn.Sequential(
            nn.ConvTranspose1d(
                decoder_channels[0],
                decoder_channels[0],
                kernel_size=region_kernel_size,
                stride=region_stride,
                groups=decoder_channels[0],
            ),
            nn.GELU(),
        )

        if hidden_channels:
            layers = []
            in_channels = decoder_channels[0]
            for channels in decoder_channels[1:]:
                layers.append(nn.Conv1d(in_channels, channels, kernel_size=kernel_size, padding=padding))
                layers.append(nn.GELU())
                in_channels = channels
            layers.append(nn.Conv1d(in_channels, 1, kernel_size=kernel_size, padding=padding))
            self.reconstruction = nn.Sequential(*layers)
        else:
            self.reconstruction = nn.Identity()

    def forward(self, z):
        h = self.expand(z)
        h = self.expand_regions(h)
        return self.reconstruction(h)


class ConvAE(ModelBase):
    """
    Convolutional autoencoder for inputs shaped (B, T, R).

    The model applies the same 1D convolutional encoder/decoder to each timepoint:
    input  : (B, T, R)
    latent : (B, T, L)
    recon  : (B, T, R)

    Pass ``hidden_channels=[]`` to omit hidden convolutions. If
    ``variational=True``, ``forward`` returns ``(x_hat, mu, log_var, z)``.
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
        self.region_kernel_size, self.region_stride = _region_reduction_params(
            self.regions, self.latent_dim
        )

        self.encoder = _ConvEncoder(
            hidden_channels=self.hidden_channels,
            latent_dim=self.latent_dim,
            kernel_size=self.kernel_size,
            region_kernel_size=self.region_kernel_size,
            region_stride=self.region_stride,
        ).to(device)

        if self.variational:
            self.mu_head = nn.Conv1d(1, 1, kernel_size=1).to(device)
            self.logvar_head = nn.Conv1d(1, 1, kernel_size=1).to(device)
        else:
            self.mu_head = None
            self.logvar_head = None

        self.decoder = _ConvDecoder(
            hidden_channels=self.hidden_channels,
            kernel_size=self.kernel_size,
            region_kernel_size=self.region_kernel_size,
            region_stride=self.region_stride,
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
            kernel_size=self.kernel_size,
            region_kernel_size=self.region_kernel_size,
            region_stride=self.region_stride,
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
        z = self.reparameterize(mu, torch.exp(0.5 * log_var)) if self.training else mu
        x_hat = self.decode(z)
        return x_hat, mu, log_var, z

    def loss(self, x, model_output):
        loss_fn_params = getattr(self, "loss_fn_params", {})

        if not self.variational:
            x_hat, _ = model_output
            recon = self.reconstruction_loss(x_hat, x)

            loss = {
                "loss": recon,
                "recon": recon,
            }
        else:
            x_hat, mu, log_var, _ = model_output
            beta = float(loss_fn_params.get("beta", 1.0))
            recon = self.reconstruction_loss(x_hat, x)

            mu_flat = mu.reshape(mu.shape[0], -1)
            log_var_flat = log_var.reshape(log_var.shape[0], -1)
            kld = -0.5 * (1 + log_var_flat - mu_flat.pow(2) - log_var_flat.exp())
            kld = kld.sum(dim=1).mean() / log_var_flat.size(1)

            loss = {
                "loss": recon + beta * kld,
                "recon": recon,
                "kld": kld,
            }

        return self.add_weighted_reconstruction_losses(loss, x, model_output[0])


class VConvAE(ConvAE):
    """Variational ConvAE with a default KL weight of one."""

    def __init__(self, *args, **kwargs):
        kwargs["variational"] = True
        super().__init__(*args, **kwargs)


# Backwards-compatible name for saved configurations and external imports.
ConvVAE = VConvAE



def _build_pred_heads(pred_head_type, pred_head_num, latent_dim, output_dim):
    pred_head_idx = {
        "avg": PredHeadAvg,
        "conv": PredHeadConv,
        "conv_no_hidden": lambda l, r: PredHeadConv(l, r, with_hidden=False),
        "temp_pool": PredHeadTemporalPool,
        "gated_temp_pool": PredHeadGatedTemporalPool,
    }
    if pred_head_type not in pred_head_idx:
        raise ValueError(f"Selected prediction head type - '{pred_head_type}' is not available.")

    return nn.ModuleList(
        [pred_head_idx[pred_head_type](latent_dim, output_dim) for _ in range(pred_head_num)]
    )


class ConvAEPredHeads(ConvAE):
    """ConvAE + temporal prediction heads for biomarker prediction."""

    def __init__(
        self,
        regions,
        timepoints,
        latent_dim=32,
        pred_head_type: str = "gated_temp_pool",
        pred_head_num: int = 1,
        hidden_channels=(32, 64),
        kernel_size: int = 3,
        variational: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        super().__init__(
            regions=regions,
            timepoints=timepoints,
            latent_dim=latent_dim,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            variational=variational,
            device=device,
        )
        self.heads = _build_pred_heads(
            pred_head_type=pred_head_type,
            pred_head_num=pred_head_num,
            latent_dim=self.latent_dim,
            output_dim=self.regions,
        )

    def forward(self, x):
        model_output = super().forward(x)
        if self.variational:
            x_hat, mean, log_var, z = model_output
        else:
            x_hat, z = model_output
        z_heads_input = z.transpose(1, 2)
        z_heads = [head(z_heads_input) for head in self.heads]
        if self.variational:
            return x_hat, mean, log_var, z_heads, z
        return x_hat, z_heads, z

    def loss(self, x, x_heads, model_output):
        loss_fn_params = getattr(self, "loss_fn_params", {})
        pred_heads_delta = float(loss_fn_params.get("pred_heads_delta", 0.0))

        if self.variational:
            x_hat, mu, log_var, z_heads, _ = model_output
            beta = loss_fn_params.get("beta", 0.5)
            recon = self.reconstruction_loss(x_hat, x)
            kld = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
            kld = kld.sum(dim=(1, 2)).mean() / (log_var.size(1) * log_var.size(2))

            loss = {
                "loss": recon + beta * kld,
                "recon": recon,
                "kld": kld,
            }
        else:
            x_hat, z_heads, _ = model_output
            recon = self.reconstruction_loss(x_hat, x)
            loss = {
                "loss": recon,
                "recon": recon,
            }

        assert len(x_heads) == len(z_heads), (
            f"label heads ({len(x_heads)}) is not the same as predicted heads ({len(z_heads)})"
        )
        pred_head_loss = []
        for i, bl in enumerate(x_heads):
            head_loss = F.smooth_l1_loss(z_heads[i], x_heads[bl], reduction="mean", beta=1.0)
            pred_head_loss.append(head_loss)
            loss[f"{bl}_loss"] = head_loss

        if pred_head_loss:
            loss["loss"] += pred_heads_delta * sum(pred_head_loss) / len(pred_head_loss)

        return self.add_weighted_reconstruction_losses(loss, x, x_hat)


class ConvVAEPredHeads(ConvAEPredHeads):
    def __init__(self, *args, **kwargs):
        kwargs["variational"] = True
        super().__init__(*args, **kwargs)
