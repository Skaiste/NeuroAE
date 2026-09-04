import torch
import torch.nn as nn

class ModelBase(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_loss_fn_params(self, params):
        self.loss_fn_params = params or {}

    def reconstruction_loss(self, x_hat, x):
        """Return the configured reconstruction objective (``mse`` by default).

        Set ``loss_params.recon_loss`` to ``"mse"``, ``"rmse"``, ``"mae"``,
        or ``"huber"``. Reconstruction error is always the mean across all
        input elements. Huber loss uses ``loss_params.huber_delta`` (default
        ``1.0``) as its transition point.
        """
        loss_params = getattr(self, "loss_fn_params", {}) or {}
        recon_loss = str(loss_params.get("recon_loss", "mse")).lower()
        valid_losses = {"mse", "rmse", "mae", "huber"}
        if recon_loss not in valid_losses:
            raise ValueError(
                "loss_params.recon_loss must be one of: "
                "'mse', 'rmse', 'mae', or 'huber'."
            )

        if recon_loss == "mae":
            return torch.nn.functional.l1_loss(x_hat, x, reduction="mean")
        if recon_loss == "huber":
            delta = float(loss_params.get("huber_delta", 1.0))
            if delta <= 0:
                raise ValueError("loss_params.huber_delta must be greater than zero.")
            return torch.nn.functional.huber_loss(
                x_hat, x, reduction="mean", delta=delta
            )

        mse = torch.nn.functional.mse_loss(x_hat, x, reduction="mean")
        return mse if recon_loss == "mse" else torch.sqrt(mse)

    @staticmethod
    def windowed_fc_variability_loss(x, x_hat, window_size, step):
        """Match the across-window variability of functional connectivity."""
        if window_size <= 1:
            raise ValueError("SWFC variability loss requires window_size > 1.")
        if step <= 0:
            raise ValueError("SWFC variability loss requires step > 0.")
        if x.shape[-2] < window_size or x_hat.shape[-2] < window_size:
            raise ValueError("SWFC variability loss window_size exceeds available timepoints.")

        # unfold returns (..., windows, regions, window_size); move time before regions.
        x_windows = x.unfold(-2, window_size, step).movedim(-1, -2)
        x_hat_windows = x_hat.unfold(-2, window_size, step).movedim(-1, -2)
        if x_windows.shape[-3] < 2 or x_hat_windows.shape[-3] < 2:
            raise ValueError("SWFC variability loss requires at least two windows.")

        def _windowed_fc(windows):
            centered = windows - windows.mean(dim=-2, keepdim=True)
            normalized = torch.nn.functional.normalize(centered, p=2, dim=-2, eps=1e-12)
            return normalized.transpose(-1, -2) @ normalized

        true_fc_windows = _windowed_fc(x_windows)
        pred_fc_windows = _windowed_fc(x_hat_windows)
        true_std = true_fc_windows.std(dim=-3, unbiased=False)
        pred_std = pred_fc_windows.std(dim=-3, unbiased=False)
        return torch.nn.functional.mse_loss(pred_std, true_std)

    def add_weighted_reconstruction_losses(self, loss, x, x_hat):
        """Add optional FC, SWFC-variability, and first-derivative terms.

        Inputs are expected to have time on their penultimate axis and regions
        on their final axis, i.e. ``(batch, time, regions)``.  Both weights
        default to zero so these terms add no compute unless explicitly
        enabled in ``loss_params``.
        """
        loss_params = getattr(self, "loss_fn_params", {}) or {}
        fc_weight = float(loss_params.get("fc_weight", 0.0))
        derivative_weight = float(loss_params.get("derivative_weight", 0.0))
        swfc_variability_weight = float(loss_params.get("swfc_variability_weight", 0.0))

        if (
            fc_weight == 0.0
            and derivative_weight == 0.0
            and swfc_variability_weight == 0.0
        ):
            return loss
        if x.ndim < 3 or x_hat.ndim < 3:
            raise ValueError(
                "Reconstruction regularization losses require inputs shaped "
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

        if swfc_variability_weight != 0.0:
            swfc_variability_loss = ModelBase.windowed_fc_variability_loss(
                x,
                x_hat,
                window_size=int(loss_params.get("swfc_window", 30)),
                step=int(loss_params.get("swfc_step", 3)),
            )
            loss["swfc_variability_loss"] = swfc_variability_loss
            loss["loss"] = loss["loss"] + swfc_variability_weight * swfc_variability_loss

        if derivative_weight != 0.0:
            if x.shape[-2] < 2 or x_hat.shape[-2] < 2:
                raise ValueError("Derivative loss requires at least two timepoints.")
            derivative_loss = torch.nn.functional.mse_loss(
                x.diff(dim=-2),
                x_hat.diff(dim=-2),
            )
            loss["derivative_loss"] = derivative_loss
            loss["loss"] = loss["loss"] + derivative_weight * derivative_loss

        return loss
