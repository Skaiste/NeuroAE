"""Two-dimensional convolutional autoencoder with a time-preserving latent space."""

import torch
import torch.nn as nn

from . import ModelBase
from .convAE import _region_reduction_params


class Conv2dAEConfigurationError(ValueError):
    """A Conv2dAE hyperparameter combination that cannot produce its requested latent shape."""


class _Conv2dEncoder(nn.Module):
    def __init__(self, hidden_channels, hidden_kernel_size, hidden_stride, region_kernel_size, region_stride):
        super().__init__()
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels]

        layers = []
        in_channels = 1
        for channels in hidden_channels:
            layers.extend([
                nn.Conv2d(
                    in_channels, channels, kernel_size=hidden_kernel_size,
                    stride=hidden_stride,
                    padding=((hidden_kernel_size[0] - 1) // 2, (hidden_kernel_size[1] - 1) // 2),
                ),
                nn.GELU(),
            ])
            in_channels = channels
        self.features = nn.Sequential(*layers)
        # Downsample regions only; the time length is retained exactly.
        self.reduce_regions = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=(1, region_kernel_size),
                stride=(1, region_stride),
                groups=in_channels,
            ),
            nn.GELU(),
        )
        self.to_latent = nn.Conv2d(in_channels, 1, kernel_size=(1, 1))

    def forward(self, x):
        return self.to_latent(self.reduce_regions(self.features(x)))


class _Conv2dDecoder(nn.Module):
    def __init__(
        self, hidden_channels, hidden_kernel_size, hidden_stride, hidden_region_widths,
        region_kernel_size, region_stride,
    ):
        super().__init__()
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels]
        # Without hidden layers the latent channel is expanded directly back
        # into the single input/reconstruction channel.
        decoder_channels = list(reversed(hidden_channels)) or [1]

        self.expand = nn.Conv2d(1, decoder_channels[0], kernel_size=(1, 1))
        # (L - 1) * stride + kernel_size == R, while the time length remains T.
        self.expand_regions = nn.Sequential(
            nn.ConvTranspose2d(
                decoder_channels[0],
                decoder_channels[0],
                kernel_size=(1, region_kernel_size),
                stride=(1, region_stride),
                groups=decoder_channels[0],
            ),
            nn.GELU(),
        )

        if hidden_channels:
            layers = []
            channels = [1, *hidden_channels]
            padding = ((hidden_kernel_size[0] - 1) // 2, (hidden_kernel_size[1] - 1) // 2)
            for index in range(len(hidden_channels) - 1, -1, -1):
                source_width = hidden_region_widths[index + 1]
                target_width = hidden_region_widths[index]
                base_width = (source_width - 1) * hidden_stride[1] - 2 * padding[1] + hidden_kernel_size[1]
                output_padding = target_width - base_width
                layers.append(nn.ConvTranspose2d(
                    channels[index + 1], channels[index],
                    kernel_size=hidden_kernel_size,
                    stride=hidden_stride,
                    padding=padding,
                    output_padding=(0, output_padding),
                ))
                if index > 0:
                    layers.append(nn.GELU())
            self.reconstruction = nn.Sequential(*layers)
        else:
            self.reconstruction = nn.Identity()

    def forward(self, z):
        return self.reconstruction(self.expand_regions(self.expand(z)))


class Conv2dAE(ModelBase):
    """2D convolutional AE for inputs ``(B, T, R)``.

    The hidden convolutions operate across both time and regions, whereas the
    learned downsampling operates only across regions.  The resulting latent
    space therefore retains one ``latent_dim`` vector per input timepoint:
    ``(B, T, L)``.  Pass ``hidden_channels=[]`` to omit the hidden
    convolutions entirely.  ``hidden_kernel_size`` and ``hidden_stride``
    apply only to the optional hidden convolutions.
    """

    def __init__(
        self,
        regions: int,
        timepoints: int,
        latent_dim: int,
        hidden_channels=(32, 16),
        hidden_kernel_size=(3, 3),
        hidden_stride=(1, 1),
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.regions = int(regions)
        self.timepoints = int(timepoints)
        self.latent_dim = int(latent_dim)
        self.hidden_channels = (hidden_channels,) if isinstance(hidden_channels, int) else tuple(hidden_channels)
        self.hidden_kernel_size = self._as_pair(hidden_kernel_size, "hidden_kernel_size")
        self.hidden_stride = self._as_pair(hidden_stride, "hidden_stride")
        self.device = device
        if self.regions <= 0 or self.timepoints <= 0 or self.latent_dim <= 0:
            raise ValueError("regions, timepoints, and latent_dim must be > 0.")
        if any(value <= 0 for value in self.hidden_kernel_size + self.hidden_stride):
            raise ValueError("hidden_kernel_size and hidden_stride values must be > 0.")
        if any(value % 2 == 0 for value in self.hidden_kernel_size):
            raise ValueError("hidden_kernel_size values must be odd to preserve dimensions predictably.")
        if self.hidden_stride[0] != 1:
            raise Conv2dAEConfigurationError(
                "hidden_stride must have time stride 1 to retain a (T, L) latent space."
            )

        self.hidden_region_widths = [self.regions]
        padding_regions = (self.hidden_kernel_size[1] - 1) // 2
        for _ in self.hidden_channels:
            current_width = self.hidden_region_widths[-1]
            width = (current_width + 2 * padding_regions - self.hidden_kernel_size[1]) // self.hidden_stride[1] + 1
            self.hidden_region_widths.append(width)

        if self.hidden_region_widths[-1] < self.latent_dim:
            raise Conv2dAEConfigurationError(
                "Conv2dAE hidden-layer strides reduce the regional width to "
                f"{self.hidden_region_widths[-1]}, which is smaller than "
                f"latent_dim={self.latent_dim}."
            )

        self.region_kernel_size, self.region_stride = _region_reduction_params(
            self.hidden_region_widths[-1], self.latent_dim
        )
        self.encoder = _Conv2dEncoder(
            self.hidden_channels, self.hidden_kernel_size, self.hidden_stride,
            self.region_kernel_size, self.region_stride,
        ).to(device)
        self.decoder = _Conv2dDecoder(
            self.hidden_channels, self.hidden_kernel_size, self.hidden_stride,
            self.hidden_region_widths, self.region_kernel_size, self.region_stride,
        ).to(device)

    @staticmethod
    def _as_pair(value, name):
        if isinstance(value, int):
            return (value, value)
        if len(value) != 2:
            raise ValueError(f"{name} must be an integer or a pair (time, regions).")
        return tuple(int(item) for item in value)

    def _check_input(self, x):
        if x.ndim != 3 or x.shape[1:] != (self.timepoints, self.regions):
            raise ValueError(
                f"Expected x shape (B, {self.timepoints}, {self.regions}), got {tuple(x.shape)}"
            )

    def encode(self, x):
        self._check_input(x)
        # Conv2d layout is (batch, channels, time, regions).
        return self.encoder(x.unsqueeze(1)).squeeze(1)

    def decode(self, z):
        if z.ndim != 3 or z.shape[1:] != (self.timepoints, self.latent_dim):
            raise ValueError(
                f"Expected z shape (B, {self.timepoints}, {self.latent_dim}), got {tuple(z.shape)}"
            )
        return self.decoder(z.unsqueeze(1)).squeeze(1)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

    def loss(self, x, model_output):
        x_hat, _ = model_output
        recon = self.reconstruction_loss(x_hat, x)
        return self.add_weighted_reconstruction_losses(
            {"loss": recon, "recon": recon}, x, x_hat
        )

    def freeze_encoder(self):
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def reset_decoder(self):
        self.decoder = _Conv2dDecoder(
            self.hidden_channels, self.hidden_kernel_size, self.hidden_stride,
            self.hidden_region_widths, self.region_kernel_size, self.region_stride,
        ).to(self.device)


class VConv2dAE(Conv2dAE):
    """Variational Conv2dAE with a default KL weight of one."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu_head = nn.Conv2d(1, 1, kernel_size=1).to(self.device)
        self.logvar_head = nn.Conv2d(1, 1, kernel_size=1).to(self.device)

    def forward(self, x):
        self._check_input(x)
        latent_base = self.encoder(x.unsqueeze(1))
        mu = self.mu_head(latent_base).squeeze(1)
        log_var = torch.clamp(self.logvar_head(latent_base).squeeze(1), -10.0, 10.0)
        z = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu) if self.training else mu
        return self.decode(z), mu, log_var, z

    def loss(self, x, model_output):
        x_hat, mu, log_var, _ = model_output
        recon = self.reconstruction_loss(x_hat, x)
        kld = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
        kld = kld.sum(dim=(1, 2)).mean() / (log_var.size(1) * log_var.size(2))
        beta = float(getattr(self, "loss_fn_params", {}).get("beta", 1.0))
        return self.add_weighted_reconstruction_losses(
            {"loss": recon + beta * kld, "recon": recon, "kld": kld}, x, x_hat
        )
