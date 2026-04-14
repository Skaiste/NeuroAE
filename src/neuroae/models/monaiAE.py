import math

import torch
import torch.nn.functional as F
from monai.networks.nets.autoencoderkl import AutoencoderKL


class MonaiAEKL(AutoencoderKL):
    """
    1D MONAI AutoencoderKL wrapper with a ConvAE-like public interface.

    Input  : (B, T, R)
    Latent : (B, T, L)
    Recon  : (B, T, R)

    Internally, each timepoint is treated as an independent 1D sample for MONAI's
    `AutoencoderKL` with `spatial_dims=1`.
    """

    def __init__(
        self,
        regions: int,
        timepoints: int,
        latent_dim: int,
        hidden_channels=(32, 64),
        kernel_size: int = 3,
        num_res_blocks=1,
        attention_levels=None,
        latent_channels: int = 3,
        norm_num_groups= None,
        norm_eps: float = 1e-6,
        with_encoder_nonlocal_attn: bool = True,
        with_decoder_nonlocal_attn: bool = True,
        use_checkpoint: bool = False,
        use_convtranspose: bool = False,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        if regions <= 0:
            raise ValueError("regions must be > 0.")
        if timepoints <= 0:
            raise ValueError("timepoints must be > 0.")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be > 0.")
        if latent_channels <= 0:
            raise ValueError("latent_channels must be > 0.")
        if kernel_size != 3:
            raise ValueError("MONAI AutoencoderKL uses fixed kernel_size=3.")

        channels = self._normalize_channels(hidden_channels)
        if attention_levels is None:
            attention_levels = tuple(False for _ in channels)
        else:
            attention_levels = tuple(bool(level) for level in attention_levels)
        if len(attention_levels) != len(channels):
            raise ValueError("attention_levels must match hidden_channels length.")

        num_res_blocks = self._normalize_num_res_blocks(num_res_blocks, len(channels))
        norm_num_groups = self._resolve_norm_num_groups(channels, norm_num_groups)

        self.regions = int(regions)
        self.timepoints = int(timepoints)
        self.latent_dim = int(latent_dim)
        self.hidden_channels = channels
        self.kernel_size = int(kernel_size)
        self.num_res_blocks = num_res_blocks
        self.attention_levels = attention_levels
        self.latent_channels = int(latent_channels)
        self.norm_num_groups = int(norm_num_groups)
        self.norm_eps = float(norm_eps)
        self.with_encoder_nonlocal_attn = bool(with_encoder_nonlocal_attn)
        self.with_decoder_nonlocal_attn = bool(with_decoder_nonlocal_attn)
        self.use_checkpoint = bool(use_checkpoint)
        self.use_convtranspose = bool(use_convtranspose)
        self.include_fc = bool(include_fc)
        self.use_combined_linear = bool(use_combined_linear)
        self.use_flash_attention = bool(use_flash_attention)
        self.device = device
        self.swfcd = None

        self._downsample_factor = 2 ** max(len(self.hidden_channels) - 1, 0)
        self._monai_latent_length = self._compute_monai_latent_length(self.regions)
        self._monai_latent_flat_dim = self.latent_channels * self._monai_latent_length

        super().__init__(
            spatial_dims=1,
            in_channels=1,
            out_channels=1,
            num_res_blocks=self.num_res_blocks,
            channels=self.hidden_channels,
            attention_levels=self.attention_levels,
            latent_channels=self.latent_channels,
            norm_num_groups=self.norm_num_groups,
            norm_eps=self.norm_eps,
            with_encoder_nonlocal_attn=self.with_encoder_nonlocal_attn,
            with_decoder_nonlocal_attn=self.with_decoder_nonlocal_attn,
            use_checkpoint=self.use_checkpoint,
            use_convtranspose=self.use_convtranspose,
            include_fc=self.include_fc,
            use_combined_linear=self.use_combined_linear,
            use_flash_attention=self.use_flash_attention,
        )

        self._init_kwargs = {
            "regions": self.regions,
            "timepoints": self.timepoints,
            "latent_dim": self.latent_dim,
            "hidden_channels": self.hidden_channels,
            "kernel_size": self.kernel_size,
            "num_res_blocks": self.num_res_blocks,
            "attention_levels": self.attention_levels,
            "latent_channels": self.latent_channels,
            "norm_num_groups": self.norm_num_groups,
            "norm_eps": self.norm_eps,
            "with_encoder_nonlocal_attn": self.with_encoder_nonlocal_attn,
            "with_decoder_nonlocal_attn": self.with_decoder_nonlocal_attn,
            "use_checkpoint": self.use_checkpoint,
            "use_convtranspose": self.use_convtranspose,
            "include_fc": self.include_fc,
            "use_combined_linear": self.use_combined_linear,
            "use_flash_attention": self.use_flash_attention,
            "device": self.device,
        }

    @staticmethod
    def _normalize_channels(hidden_channels):
        if isinstance(hidden_channels, int):
            hidden_channels = (hidden_channels,)
        channels = tuple(int(channel) for channel in hidden_channels)
        if not channels:
            raise ValueError("hidden_channels must contain at least one channel value.")
        if any(channel <= 0 for channel in channels):
            raise ValueError("hidden_channels must contain positive integers.")
        return channels

    @staticmethod
    def _normalize_num_res_blocks(num_res_blocks, depth):
        if isinstance(num_res_blocks, int):
            if num_res_blocks <= 0:
                raise ValueError("num_res_blocks must be > 0.")
            return tuple(num_res_blocks for _ in range(depth))

        blocks = tuple(int(block) for block in num_res_blocks)
        if len(blocks) != depth:
            raise ValueError("num_res_blocks must be an int or match hidden_channels length.")
        if any(block <= 0 for block in blocks):
            raise ValueError("num_res_blocks entries must be > 0.")
        return blocks

    @staticmethod
    def _resolve_norm_num_groups(channels, norm_num_groups):
        if norm_num_groups is not None:
            norm_num_groups = int(norm_num_groups)
            if norm_num_groups <= 0:
                raise ValueError("norm_num_groups must be > 0.")
            if any(channel % norm_num_groups != 0 for channel in channels):
                raise ValueError("All hidden_channels must be divisible by norm_num_groups.")
            return norm_num_groups

        gcd_value = channels[0]
        for channel in channels[1:]:
            gcd_value = math.gcd(gcd_value, channel)

        for candidate in range(min(32, gcd_value), 0, -1):
            if gcd_value % candidate == 0:
                return candidate
        return 1

    def _compute_monai_latent_length(self, regions):
        latent_length = regions
        for _ in range(len(self.hidden_channels) - 1):
            latent_length = max(1, latent_length // 2)
        return latent_length

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

    def _match_region_dim(self, x):
        if x.shape[-1] == self.regions:
            return x
        return F.interpolate(x, size=self.regions, mode="linear", align_corners=False)

    def _flatten_monai_latent(self, z):
        return z.reshape(z.shape[0], 1, -1)

    def _unflatten_monai_latent(self, z):
        return z.reshape(z.shape[0], self.latent_channels, self._monai_latent_length)

    def _project_to_public_latent(self, z):
        z = self._flatten_monai_latent(z)
        z = F.adaptive_avg_pool1d(z, self.latent_dim)
        return z.squeeze(1)

    def _project_to_monai_latent(self, z):
        z = z.unsqueeze(1)
        if z.shape[-1] != self._monai_latent_flat_dim:
            z = F.interpolate(z, size=self._monai_latent_flat_dim, mode="linear", align_corners=False)
        return self._unflatten_monai_latent(z)

    def freeze_encoder(self):
        for module in (self.encoder, self.quant_conv_mu, self.quant_conv_log_sigma):
            for param in module.parameters():
                param.requires_grad = False

    def reset_decoder(self):
        fresh_model = type(self)(**self._init_kwargs).to(next(self.parameters()).device)
        self.decoder = fresh_model.decoder
        self.post_quant_conv = fresh_model.post_quant_conv

    def encode(self, x):
        batch_size = x.shape[0]
        x_conv = self._to_conv_input(x)
        z_mu, z_sigma = super().encode(x_conv)
        log_var = torch.log(z_sigma.pow(2).clamp_min(1e-8))
        return (
            self._from_conv_output(self._project_to_public_latent(z_mu), batch_size),
            self._from_conv_output(self._project_to_public_latent(log_var), batch_size),
        )

    def decode(self, z):
        if z.ndim != 3:
            raise ValueError(f"Expected z shape (B, T, L), got {tuple(z.shape)}")
        if z.shape[1] != self.timepoints:
            raise ValueError(f"Expected T={self.timepoints}, got {z.shape[1]}")
        if z.shape[2] != self.latent_dim:
            raise ValueError(f"Expected L={self.latent_dim}, got {z.shape[2]}")

        batch_size = z.shape[0]
        z_conv = z.reshape(batch_size * self.timepoints, self.latent_dim)
        z_conv = self._project_to_monai_latent(z_conv)
        x_hat = super().decode(z_conv)
        x_hat = self._match_region_dim(x_hat).squeeze(1)
        return self._from_conv_output(x_hat, batch_size)

    def reparameterize(self, mean, std):
        epsilon = torch.randn_like(std).to(mean.device)
        return mean + std * epsilon

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, torch.exp(0.5 * log_var))
        x_hat = self.decode(z)
        return x_hat, mu, log_var, z

    def set_loss_fn_params(self, params):
        self.loss_fn_params = params or {}

    def set_swfcd(self, swfcd):
        self.swfcd = swfcd

    def loss(self, x, model_output):
        x_hat, mu, log_var, _ = model_output
        loss_fn_params = getattr(self, "loss_fn_params", {})
        error_per_feature = loss_fn_params.get("loss_per_feature", True)
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
            swfcd = self.swfcd.apply(x, x_hat)
            swfcd_beta = loss_fn_params.get("swfcd_beta", 1.0)
            loss["swfcd_rmse"] = swfcd["rmse"]
            loss["loss"] += swfcd_beta * swfcd["rmse"]

        return loss
