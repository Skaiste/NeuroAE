import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from . import ModelBase


from .head import PredHeadAvg, PredHeadConv, PredHeadTemporalPool, PredHeadGatedTemporalPool


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels, kernel_size=1):
        super().__init__()
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels]
        if len(hidden_channels) == 0:
            hidden_channels = [max(8, latent_channels * 2)]

        layers = []
        last_channels = in_channels
        for channels in hidden_channels:
            layers.append(nn.Conv1d(last_channels, channels, kernel_size=kernel_size))
            # layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.GELU())
            last_channels = channels

        layers.append(nn.Conv1d(last_channels, latent_channels, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_channels, hidden_channels, out_channels, kernel_size=1):
        super().__init__()
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels]
        if len(hidden_channels) == 0:
            hidden_channels = [max(8, latent_channels * 2)]

        layers = []
        last_channels = latent_channels
        for channels in hidden_channels[::-1]:
            layers.append(nn.Conv1d(last_channels, channels, kernel_size=kernel_size))
            # layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.GELU())
            last_channels = channels

        layers.append(nn.Conv1d(last_channels, out_channels, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class BasicConvAE(ModelBase):
    """
    Deterministic convolutional autoencoder for inputs shaped (B, R, T).
    Latent representation is shaped (B, latent_channels, T).
    """

    def __init__(
        self,
        regions: int,
        timepoints: int,
        hidden_channels = [64],
        latent_dim: int = 8,
        kernel_size=1,
    ):
        super().__init__()
        if regions <= 0:
            raise ValueError("regions must be > 0")
        if timepoints <= 0:
            raise ValueError("timepoints must be > 0")
        
        self.regions = int(regions)
        self.timepoints = int(timepoints)
        self.hidden_channels = hidden_channels
        self.latent_dim = int(latent_dim)
        self.kernel_size = int(kernel_size)

        self.encoder = Encoder(
            in_channels=self.regions,
            hidden_channels=hidden_channels,
            latent_channels=self.latent_dim,
            kernel_size=kernel_size
        )
        self.decoder = Decoder(
            latent_channels=self.latent_dim,
            hidden_channels=hidden_channels,
            out_channels=self.regions,
            kernel_size=kernel_size
        )

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def reset_decoder(self):
        self.decoder = Decoder(
            latent_channels=self.latent_dim,
            hidden_channels=self.hidden_channels,
            out_channels=self.regions,
            kernel_size=self.kernel_size
        ).to(next(self.parameters()).device)

    def encode(self, x):
        if x.ndim != 3:
            raise ValueError(f"Expected x shape (B, R, T), got {tuple(x.shape)}")
        if x.shape[1] != self.regions:
            raise ValueError(f"Expected R={self.regions}, got {x.shape[1]}")
        if x.shape[2] != self.timepoints:
            raise ValueError(f"Expected T={self.timepoints}, got {x.shape[2]}")
        return self.encoder(x)

    def decode(self, z):
        if z.ndim != 3:
            raise ValueError(f"Expected z shape (B, L, T), got {tuple(z.shape)}")
        if z.shape[1] != self.latent_dim:
            raise ValueError(f"Expected latent channels L={self.latent_dim}, got {z.shape[1]}")
        if z.shape[2] != self.timepoints:
            raise ValueError(f"Expected T={self.timepoints}, got {z.shape[2]}")
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def loss(self, x, model_output):
        x_hat = model_output[0]
        error_per_feature = self.loss_fn_params.get("loss_per_feature", True)

        if error_per_feature:
            recon = F.mse_loss(x_hat, x, reduction="mean")
        else:
            recon = F.mse_loss(x_hat, x, reduction="none")
            recon = recon.flatten(1).sum(dim=1).mean()

        loss = {
            "loss": recon,
            "recon": recon,
        }

        if self.swfcd is not None:
            swfcd = self.swfcd.apply(x, x_hat)
            swfcd_beta = self.loss_fn_params.get("swfcd_beta", 1.0)
            loss["swfcd_rmse"] = swfcd["rmse"]
            loss["loss"] += swfcd_beta * swfcd["rmse"]

        return loss



class BasicConvAEPredHeads(BasicConvAE):
    """ BasicConvAE + linear temporal models for ABeta and Tau level prediction """
    def __init__(
        self,
        regions: int,
        timepoints: int,
        pred_head_type: str = "gated_temp_pool",
        pred_head_num: int = 1,
        hidden_channels=[64],
        latent_dim: int = 8,
        kernel_size=1,
    ) -> None:
        super().__init__(
            regions,
            timepoints,
            hidden_channels,
            latent_dim,
            kernel_size,
        )
        pred_head_idx = {
            "avg": PredHeadAvg,
            "conv": PredHeadConv,
            "conv_no_hidden": lambda l,r: PredHeadConv(l,r, with_hidden=False),
            "temp_pool": PredHeadTemporalPool,
            "gated_temp_pool": PredHeadGatedTemporalPool
        }
        if pred_head_type not in pred_head_idx:
            raise ValueError(f"Selected prediction head type - '{pred_head_type}' is not available.")
        
        self.heads = [pred_head_idx[pred_head_type](latent_dim, regions) for _ in range(pred_head_num)]

    def to(self, device):
        self.heads = [h.to(device) for h in self.heads]
        return super().to(device)

    def forward(self, x):
        recon, z = super().forward(x)
        z_heads = [h(z) for h in self.heads]
        return recon, z_heads, z
    
    def loss(self, x, x_heads, model_output):
        x_hat, z_heads, _ = model_output
        error_per_feature = self.loss_fn_params.get("loss_per_feature", True)
        pred_heads_delta = float(self.loss_fn_params.get("pred_heads_delta", 0.0))
        # if selected error per feature, we are averaging everything
        if error_per_feature:
            # recon: mean mse loss
            recon = F.mse_loss(x_hat, x, reduction="mean")

        # if selected error per sample, we are summing everything
        else:
            # recon: sum over features per sample, then mean over batch
            recon = F.mse_loss(x_hat, x, reduction="none")  # [B, D]
            recon = recon.sum(dim=1).mean()

        loss = {
            'loss': recon,
            'recon': recon
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