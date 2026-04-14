import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import ModelBase
from ..head import PredHeadAvg, PredHeadConv, PredHeadTemporalPool, PredHeadGatedTemporalPool


class LinearAE(ModelBase):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        
        self.encoder = nn.Linear(input_dim, latent_dim, bias=True)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=True)
        
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def loss(self, x, model_output):
        loss = {
            'loss': F.mse_loss(model_output[0], x)
        }

        if self.swfcd is not None:
            swfcd = self.swfcd.apply(x, model_output[0])
            swfcd_beta = self.loss_fn_params.get("swfcd_beta", 1.0)
            loss['swfcd_rmse'] = swfcd['rmse']
            loss['loss'] += swfcd_beta * swfcd['rmse']

        return loss

class LinearAETrad(ModelBase):
    def __init__(self, input_dim, latent_dim, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device
        
        self.encoder = nn.Linear(input_dim, latent_dim, bias=True)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=True)
        
    def forward(self, x):
        # expect a 2D input (R, T)
        assert len(x.shape) > 2, "The input features cannot be flattened"
        # apply the same linear map independently at each timepoint
        z_steps = []
        x_recon_steps = []

        for t in range(x.shape[2]):
            z_t = self.encoder(x[:, :, t])
            x_recon_t = self.decoder(z_t)
            z_steps.append(z_t)
            x_recon_steps.append(x_recon_t)

        z = torch.stack(z_steps, dim=2)
        x_recon = torch.stack(x_recon_steps, dim=2)
        return x_recon, z

    def loss(self, x, model_output):
        mse_loss = F.mse_loss(model_output[0], x)
        loss = {
            'loss': mse_loss,
            'mse': mse_loss
        }

        if self.swfcd is not None:
            swfcd = self.swfcd.apply(x, model_output[0])
            swfcd_beta = self.loss_fn_params.get("swfcd_beta", 1.0)
            loss['swfcd_rmse'] = swfcd['rmse']
            loss['loss'] += swfcd_beta * swfcd['rmse']

        return loss


class LinearAETimeShared(ModelBase):
    def __init__(
        self,
        input_dim=784,
        timepoint_dim=1,
        latent_dim=8,  # per timepoint
        input_layout="feature_time",
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.timepoint_dim = int(timepoint_dim)
        if self.timepoint_dim <= 0:
            raise ValueError("timepoint_dim must be > 0.")
        if self.input_dim % self.timepoint_dim != 0:
            raise ValueError(
                f"input_dim ({self.input_dim}) must be divisible by timepoint_dim ({self.timepoint_dim})."
            )

        self.feature_dim = self.input_dim // self.timepoint_dim
        self.latent_per_timepoint = int(latent_dim)
        self.latent_dim = self.latent_per_timepoint * self.timepoint_dim

        self.input_layout = str(input_layout)
        if self.input_layout not in {"feature_time", "time_feature"}:
            raise ValueError("input_layout must be one of {'feature_time', 'time_feature'}.")

        # shared across timepoints
        self.encoder = nn.Linear(self.feature_dim, self.latent_per_timepoint, bias=True)
        self.decoder = nn.Linear(self.latent_per_timepoint, self.feature_dim, bias=True)

    def _reshape_input(self, x):
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"Expected x shape (B, {self.input_dim}), got {tuple(x.shape)}")
        if self.input_layout == "feature_time":
            return x.reshape(x.shape[0], self.feature_dim, self.timepoint_dim).transpose(1, 2)  # (B,T,F)
        return x.reshape(x.shape[0], self.timepoint_dim, self.feature_dim)  # (B,T,F)

    def _flatten_recon(self, x_time):
        if self.input_layout == "feature_time":
            return x_time.transpose(1, 2).reshape(x_time.shape[0], self.input_dim)
        return x_time.reshape(x_time.shape[0], self.input_dim)

    def _flatten_latent(self, z_time):
        if self.input_layout == "feature_time":
            return z_time.transpose(1, 2).reshape(z_time.shape[0], self.latent_dim)
        return z_time.reshape(z_time.shape[0], self.latent_dim)  # (B, T*L)

    def forward(self, x):
        x_time = self._reshape_input(x)           # (B,T,F)
        z_time = self.encoder(x_time)             # (B,T,L), shared linear map over F
        x_hat_time = self.decoder(z_time)         # (B,T,F)
        z = self._flatten_latent(z_time)          # (B,T*L)
        x_hat = self._flatten_recon(x_hat_time)   # (B,D)
        return x_hat, z

    def loss(self, x, model_output):
        x_hat, _ = model_output
        loss = {"loss": F.mse_loss(x_hat, x)}
        if self.swfcd is not None:
            swfcd = self.swfcd.apply(x, x_hat)
            swfcd_beta = self.loss_fn_params.get("swfcd_beta", 1.0)
            loss["swfcd_rmse"] = swfcd["rmse"]
            loss["loss"] += swfcd_beta * swfcd["rmse"]
        return loss



class LinearAEPredHeads(LinearAE):
    """ LinearAE + linear temporal models for ABeta and Tau level prediction 
            Assumes latent dimension preserves the time dimension
    """
    def __init__(
        self,
        input_dim,
        timepoints,
        pred_head_type: str = "gated_temp_pool",
        pred_head_num: int = 1,
        latent_dim=32
    ) -> None:
        super().__init__(
            input_dim=input_dim, 
            latent_dim=latent_dim
        )
        self.timepoints = timepoints
        self.regions = input_dim // timepoints
        self.latent_regions = latent_dim // timepoints
        pred_head_idx = {
            "avg": PredHeadAvg,
            "conv": PredHeadConv,
            "conv_no_hidden": lambda l, r: PredHeadConv(l, r, with_hidden=False),
            "temp_pool": PredHeadTemporalPool,
            "gated_temp_pool": PredHeadGatedTemporalPool
        }
        if pred_head_type not in pred_head_idx:
            raise ValueError(f"Selected prediction head type - '{pred_head_type}' is not available.")
        
        self.heads = [pred_head_idx[pred_head_type](self.latent_regions, self.regions) for i in range(pred_head_num)]

    def to(self, device):
        self.heads = [h.to(device) for h in self.heads]
        return super().to(device)

    def forward(self, x):
        x_hat, z = super().forward(x)
        # reshape latent space to 2d
        z_2d = z.reshape(z.shape[0],-1,self.timepoints)
        z_heads = [h(z_2d) for h in self.heads]
        return x_hat, z_heads, z
    
    def loss(self, x, x_heads, model_output):
        x_hat, z_heads, _ = model_output
        pred_heads_delta = float(self.loss_fn_params.get("pred_heads_delta", 0.0))
        
        loss = {
            'loss': F.mse_loss(x_hat, x)
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
    


class LinearAETimeSharedPredHeads(LinearAETimeShared):
    """ LinearAETimeShared + linear temporal models for ABeta and Tau level prediction 
            Assumes latent dimension preserves the time dimension
    """
    def __init__(
        self,
        input_dim,
        timepoint_dim,
        pred_head_type: str = "gated_temp_pool",
        pred_head_num: int = 1,
        latent_dim=32
    ) -> None:
        super().__init__(
            input_dim=input_dim, 
            latent_dim=latent_dim,
            timepoint_dim=timepoint_dim
        )
        self.regions = input_dim // timepoint_dim
        self.latent_regions = latent_dim
        pred_head_idx = {
            "avg": PredHeadAvg,
            "conv": PredHeadConv,
            "conv_no_hidden": lambda l, r: PredHeadConv(l, r, with_hidden=False),
            "temp_pool": PredHeadTemporalPool,
            "gated_temp_pool": PredHeadGatedTemporalPool
        }
        if pred_head_type not in pred_head_idx:
            raise ValueError(f"Selected prediction head type - '{pred_head_type}' is not available.")
        
        self.heads = [pred_head_idx[pred_head_type](self.latent_regions, self.regions) for i in range(pred_head_num)]

    def to(self, device):
        self.heads = [h.to(device) for h in self.heads]
        return super().to(device)

    def forward(self, x):
        x_hat, z = super().forward(x)
        # reshape latent space to 2d
        z_2d = z.reshape(z.shape[0], -1, self.timepoint_dim)
        z_heads = [h(z_2d) for h in self.heads]
        return x_hat, z_heads, z
    
    def loss(self, x, x_heads, model_output):
        x_hat, z_heads, _ = model_output
        pred_heads_delta = float(self.loss_fn_params.get("pred_heads_delta", 0.0))
        
        loss = {
            'loss': F.mse_loss(x_hat, x)
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
    
