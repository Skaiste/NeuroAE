import torch
import torch.nn as nn

class ModelBase(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_loss_fn_params(self, params):
        self.loss_fn_params = params or {}

    def add_weighted_fc_and_std_losses(self, loss, x, x_hat):
        """Add optional FC and temporal standard-deviation matching terms.

        Inputs are expected to have time on their penultimate axis and regions
        on their final axis, i.e. ``(batch, time, regions)``.  Both weights
        default to zero so these terms add no compute unless explicitly
        enabled in ``loss_params``.
        """
        loss_params = getattr(self, "loss_fn_params", {}) or {}
        fc_weight = float(loss_params.get("fc_weight", 0.0))
        std_weight = float(loss_params.get("std_weight", 0.0))

        if fc_weight == 0.0 and std_weight == 0.0:
            return loss
        if x.ndim < 3 or x_hat.ndim < 3:
            raise ValueError(
                "FC and standard-deviation losses require inputs shaped "
                "(..., time, regions)."
            )

        if fc_weight != 0.0:
            x_centered = x - x.mean(dim=-2, keepdim=True)
            x_hat_centered = x_hat - x_hat.mean(dim=-2, keepdim=True)
            x_fc = torch.nn.functional.normalize(x_centered, p=2, dim=-2, eps=1e-12)
            x_hat_fc = torch.nn.functional.normalize(x_hat_centered, p=2, dim=-2, eps=1e-12)
            fc_loss = torch.nn.functional.mse_loss(
                x_fc.transpose(-1, -2) @ x_fc,
                x_hat_fc.transpose(-1, -2) @ x_hat_fc,
            )
            loss["fc_loss"] = fc_loss
            loss["loss"] = loss["loss"] + fc_weight * fc_loss

        if std_weight != 0.0:
            std_loss = torch.nn.functional.mse_loss(
                x.std(dim=-2, unbiased=False),
                x_hat.std(dim=-2, unbiased=False),
            )
            loss["std_loss"] = std_loss
            loss["loss"] = loss["loss"] + std_weight * std_loss

        return loss
