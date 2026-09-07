import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ModelBase
from .head import (
    ClsHeadLinear,
    ClsHeadMLP,
    PredHeadAvg,
    PredHeadConv,
    PredHeadTemporalPool,
    PredHeadGatedTemporalPool,
)


class LAE(ModelBase):
    def __init__(
        self,
        region_dim,
        timepoint_dim,
        latent_dim,
    ):
        super().__init__()
        self.region_dim = int(region_dim)
        self.timepoint_dim = int(timepoint_dim)
        self.latent_dim = int(latent_dim)

        if self.region_dim <= 0:
            raise ValueError("region_dim must be > 0.")
        if self.timepoint_dim <= 0:
            raise ValueError("timepoint_dim must be > 0.")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be > 0.")

        # Derived compatibility attributes for existing code paths.
        self.feature_dim = self.region_dim
        self.input_dim = self.region_dim * self.timepoint_dim
        self.latent_per_timepoint = self.latent_dim
        self.latent_flat_dim = self.timepoint_dim * self.latent_dim

        # shared across timepoints
        self.encoder = nn.Linear(self.region_dim, self.latent_dim, bias=True)
        self.encoder_logvar = nn.Linear(self.region_dim, self.latent_dim, bias=True)
        self.decoder = nn.Linear(self.latent_dim, self.region_dim, bias=True)

    def _reshape_input(self, x):
        expected_shape = (self.timepoint_dim, self.region_dim)
        if x.ndim != 3 or x.shape[1:] != expected_shape:
            raise ValueError(
                f"Expected x shape (B, {self.timepoint_dim}, {self.region_dim}), got {tuple(x.shape)}"
            )
        return x

    def _flatten_latent(self, z_time):
        return z_time.reshape(z_time.shape[0], self.latent_flat_dim)

    def _beta(self):
        return float(getattr(self, "loss_fn_params", {}).get("beta", 0.0))

    def _variational_enabled(self):
        return self._beta() != 0.0

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mean + std * epsilon

    def forward(self, x):
        x_time = self._reshape_input(x)           # (B,T,F)
        mu_time = self.encoder(x_time)            # (B,T,L), shared linear map over F

        if self._variational_enabled():
            log_var_time = self.encoder_logvar(x_time)
            log_var_time = torch.clamp(log_var_time, -10.0, 10.0)
            z_time = self.reparameterize(mu_time, log_var_time) if self.training else mu_time
        else:
            log_var_time = torch.zeros_like(mu_time)
            z_time = mu_time

        x_hat_time = self.decoder(z_time)         # (B,T,F)
        mu = self._flatten_latent(mu_time)
        log_var = self._flatten_latent(log_var_time)
        z = self._flatten_latent(z_time)          # (B,T*L)

        if self._variational_enabled():
            return x_hat_time, mu, log_var, z
        return x_hat_time, z

    def loss(self, x, model_output):
        beta = self._beta()

        if len(model_output) == 4:
            x_hat, mu, log_var, _ = model_output
        else:
            x_hat, _ = model_output
            mu = None
            log_var = None

        recon = self.reconstruction_loss(x_hat, x)

        loss = {"loss": recon, "recon": recon}

        if beta != 0.0 and mu is not None and log_var is not None:
            kld = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
            kld = kld.sum(dim=1).mean() / log_var.size(1)
            loss["kld"] = kld
            loss["loss"] = loss["loss"] + beta * kld
        else:
            loss["kld"] = torch.zeros((), device=x.device, dtype=x.dtype)

        return self.add_weighted_reconstruction_losses(loss, x, x_hat)

    def _classification_logits(self, z):
        if isinstance(self.cls_head, ClsHeadLinear):
            # Flattening preserves (time, latent); pool over time, not latent.
            z = z.reshape(z.shape[0], self.timepoint_dim, self.latent_dim).transpose(1, 2)
        return self.cls_head(z)

    def classification_loss(self, logits, classes):
        """Cross-entropy using class weights fitted on the training split."""
        weights = self.cls_class_weights.to(device=logits.device, dtype=logits.dtype)
        return F.cross_entropy(logits, classes.long(), weight=weights, label_smoothing=0.05)

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.encoder_logvar.parameters():
            param.requires_grad = False

    def reset_decoder(self):
        device = next(self.parameters()).device
        self.decoder = nn.Linear(self.latent_dim, self.region_dim, bias=True).to(device)


class VLAE(LAE):
    """Variational LAE with a default KL weight of one."""

    def _beta(self):
        return float(getattr(self, "loss_fn_params", {}).get("beta", 1.0))



class LAEPredHeads(LAE):
    """ LAE + linear temporal models for ABeta and Tau level prediction 
            Assumes latent dimension preserves the time dimension
    """
    def __init__(
        self,
        region_dim,
        timepoint_dim,
        pred_head_type: str = "gated_temp_pool",
        pred_head_num: int = 1,
        latent_dim=32
    ) -> None:
        super().__init__(
            region_dim=region_dim,
            latent_dim=latent_dim,
            timepoint_dim=timepoint_dim
        )
        self.regions = self.region_dim
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
        
        # Register the heads as submodules so their parameters are included in
        # ``model.parameters()`` and are consequently optimized and saved.
        self.heads = nn.ModuleList(
            [pred_head_idx[pred_head_type](self.latent_regions, self.regions) for _ in range(pred_head_num)]
        )

    def to(self, device):
        return super().to(device)

    def forward(self, x):
        output = super().forward(x)

        if len(output) == 4:
            x_hat, mean, log_var, z = output
        else:
            x_hat, z = output
            mean = None
            log_var = None

        # reshape latent space to 2d
        z_2d = z.reshape(z.shape[0], -1, self.timepoint_dim)
        z_heads = [h(z_2d) for h in self.heads]
        if mean is not None and log_var is not None:
            return x_hat, mean, log_var, z_heads, z
        return x_hat, z_heads, z
    
    def loss(self, x, x_heads, model_output):
        beta = self._beta()
        pred_heads_delta = float(self.loss_fn_params.get("pred_heads_delta", 0.0))

        if len(model_output) == 5:
            x_hat, mu, log_var, z_heads, _ = model_output
        else:
            x_hat, z_heads, _ = model_output
            mu = None
            log_var = None

        recon = self.reconstruction_loss(x_hat, x)

        loss = {
            "loss": recon,
            "recon": recon,
        }

        if beta != 0.0 and mu is not None and log_var is not None:
            kld = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
            kld = kld.sum(dim=1).mean() / log_var.size(1)
            loss["kld"] = kld
            loss["loss"] = loss["loss"] + beta * kld
        else:
            loss["kld"] = torch.zeros((), device=x.device, dtype=x.dtype)

        # calculate loss from prediction heads
        assert len(x_heads) == len(z_heads), f"label heads ({len(x_heads)}) is not the same as predicted heads ({len(z_heads)})"
        pred_head_loss = []
        for i, bl in enumerate(x_heads):
            head_loss = F.smooth_l1_loss(z_heads[i], x_heads[bl], reduction="mean", beta=1.0)
            pred_head_loss.append(head_loss)
            loss[f"{bl}_loss"] = head_loss

        if len(pred_head_loss) > 0:
            loss['loss'] = loss['loss'] + pred_heads_delta * sum(pred_head_loss) / len(pred_head_loss)

        return self.add_weighted_reconstruction_losses(loss, x, x_hat)


class LAEClsHead(LAE):
    """LAE with a supervised classifier attached to its flattened latent space."""
    def __init__(
        self,
        region_dim,
        timepoint_dim,
        latent_dim=32,
        num_classes=2,
        cls_head_type: str = "linear",
        cls_head_hidden_dim=None,
        class_labels=None,
        cls_head_dropout: float = 0.0,
    ):
        super().__init__(region_dim=region_dim, timepoint_dim=timepoint_dim, latent_dim=latent_dim)
        if int(num_classes) < 2:
            raise ValueError("num_classes must be at least 2.")
        head_types = {"linear": ClsHeadLinear, "mlp": ClsHeadMLP}
        if cls_head_type not in head_types:
            raise ValueError("cls_head_type must be either 'linear' or 'mlp'.")
        self.class_labels = list(class_labels) if class_labels is not None else list(range(int(num_classes)))
        if len(self.class_labels) != int(num_classes):
            raise ValueError("class_labels length must match num_classes.")
        self.class_to_idx = {label: idx for idx, label in enumerate(self.class_labels)}
        # Derived from the training split; keep old weight checkpoints compatible.
        self.register_buffer("cls_class_weights", torch.ones(int(num_classes)), persistent=False)
        if len(self.class_to_idx) != len(self.class_labels):
            raise ValueError("class_labels must be unique.")
        self.cls_head = head_types[cls_head_type](
            self.latent_flat_dim, int(num_classes), cls_head_hidden_dim, dropout=cls_head_dropout
        ) if cls_head_type == "mlp" else head_types[cls_head_type](self.latent_dim, int(num_classes), dropout=cls_head_dropout)

    def forward(self, x):
        output = super().forward(x)
        if len(output) == 4:
            x_hat, mu, log_var, z = output
            return x_hat, mu, log_var, self._classification_logits(z), z
        x_hat, z = output
        return x_hat, self._classification_logits(z), z

    def loss(self, x, classes, model_output):
        if len(model_output) == 5:
            x_hat, mu, log_var, logits, _ = model_output
        else:
            x_hat, logits, _ = model_output
            mu = log_var = None

        recon = self.reconstruction_loss(x_hat, x)
        loss = {"loss": recon, "recon": recon}
        beta = self._beta()
        if beta != 0.0 and mu is not None and log_var is not None:
            kld = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
            kld = kld.sum(dim=1).mean() / log_var.size(1)
            loss["kld"] = kld
            loss["loss"] = loss["loss"] + beta * kld
        else:
            loss["kld"] = torch.zeros((), device=x.device, dtype=x.dtype)

        cls_loss = self.classification_loss(logits, classes)
        loss["cls_loss"] = cls_loss
        cls_weight = self.loss_fn_params.get(
            "cls_head_weight", self.loss_fn_params.get("cls_head_delta", 1.0)
        )
        loss["loss"] = loss["loss"] + float(cls_weight) * cls_loss

        return self.add_weighted_reconstruction_losses(loss, x, x_hat)


class LAEPredClsHeads(LAEPredHeads):
    """LAE with both regression prediction heads and a latent classifier.

    The regression heads receive the time-preserving latent representation,
    while the classification head receives the flattened latent vector.  Its
    :meth:`loss` accepts both regression targets and class indices so the
    reconstruction, regression, classification, and optional KL terms
    can be optimized together.
    """

    def __init__(
        self,
        region_dim,
        timepoint_dim,
        pred_head_type: str = "gated_temp_pool",
        pred_head_num: int = 1,
        latent_dim=32,
        num_classes=2,
        cls_head_type: str = "linear",
        cls_head_hidden_dim=None,
        class_labels=None,
        cls_head_dropout: float = 0.0,
    ):
        super().__init__(
            region_dim=region_dim,
            timepoint_dim=timepoint_dim,
            pred_head_type=pred_head_type,
            pred_head_num=pred_head_num,
            latent_dim=latent_dim,
        )
        if int(num_classes) < 2:
            raise ValueError("num_classes must be at least 2.")
        head_types = {"linear": ClsHeadLinear, "mlp": ClsHeadMLP}
        if cls_head_type not in head_types:
            raise ValueError("cls_head_type must be either 'linear' or 'mlp'.")

        self.class_labels = (
            list(class_labels) if class_labels is not None else list(range(int(num_classes)))
        )
        if len(self.class_labels) != int(num_classes):
            raise ValueError("class_labels length must match num_classes.")
        self.class_to_idx = {label: idx for idx, label in enumerate(self.class_labels)}
        # Derived from the training split; keep old weight checkpoints compatible.
        self.register_buffer("cls_class_weights", torch.ones(int(num_classes)), persistent=False)
        if len(self.class_to_idx) != len(self.class_labels):
            raise ValueError("class_labels must be unique.")

        self.cls_head = (
            head_types[cls_head_type](self.latent_flat_dim, int(num_classes), cls_head_hidden_dim, dropout=cls_head_dropout)
            if cls_head_type == "mlp"
            else head_types[cls_head_type](self.latent_dim, int(num_classes), dropout=cls_head_dropout)
        )

    def forward(self, x):
        output = super().forward(x)
        if len(output) == 5:
            x_hat, mu, log_var, z_heads, z = output
            return x_hat, mu, log_var, z_heads, self._classification_logits(z), z
        x_hat, z_heads, z = output
        return x_hat, z_heads, self._classification_logits(z), z

    def loss(self, x, x_heads, classes, model_output):
        """Return the joint loss for regression targets and class indices.

        ``x_heads`` maps regression-head names to their target tensors.
        ``classes`` must contain encoded class indices in ``[0, num_classes)``.
        """
        if len(model_output) == 6:
            x_hat, mu, log_var, z_heads, logits, _ = model_output
        else:
            x_hat, z_heads, logits, _ = model_output
            mu = log_var = None

        recon = self.reconstruction_loss(x_hat, x)
        loss = {"loss": recon, "recon": recon}

        beta = self._beta()
        if beta != 0.0 and mu is not None and log_var is not None:
            kld = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
            kld = kld.sum(dim=1).mean() / log_var.size(1)
            loss["kld"] = kld
            loss["loss"] = loss["loss"] + beta * kld
        else:
            loss["kld"] = torch.zeros((), device=x.device, dtype=x.dtype)

        if len(x_heads) != len(z_heads):
            raise ValueError(
                f"label heads ({len(x_heads)}) is not the same as predicted heads ({len(z_heads)})"
            )
        pred_losses = []
        for prediction, (name, target) in zip(z_heads, x_heads.items()):
            head_loss = F.smooth_l1_loss(prediction, target, reduction="mean", beta=1.0)
            pred_losses.append(head_loss)
            loss[f"{name}_loss"] = head_loss
        if pred_losses:
            loss["loss"] = loss["loss"] + float(self.loss_fn_params.get("pred_heads_delta", 0.0)) * (
                sum(pred_losses) / len(pred_losses)
            )

        cls_loss = self.classification_loss(logits, classes)
        loss["cls_loss"] = cls_loss
        cls_weight = self.loss_fn_params.get(
            "cls_head_weight", self.loss_fn_params.get("cls_head_delta", 1.0)
        )
        loss["loss"] = loss["loss"] + float(cls_weight) * cls_loss

        return self.add_weighted_reconstruction_losses(loss, x, x_hat)
    
