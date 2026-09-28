"""Gaussian training corruption scaled to each clean region's temporal SD."""

import math
from collections.abc import Mapping

import torch


def validate_denoising(config, dataset, model):
    if config is None:
        return
    if not isinstance(config, Mapping) or set(config) - {"percentage"}:
        raise ValueError("training.denoising must be a mapping containing percentage only.")
    percentage = config.get("percentage")
    if isinstance(percentage, bool) or not isinstance(percentage, (int, float)) or not math.isfinite(percentage) or percentage < 0:
        raise ValueError("denoising.percentage must be a finite nonnegative number.")
    if percentage == 0:
        return
    if getattr(dataset, "fc_input", False) or getattr(dataset, "timepoints_as_samples", False):
        raise ValueError("Gaussian denoising requires full BOLD timeseries (fc_input and timepoints_as_samples false).")
    if not getattr(model, "requires_optimizer", True) or hasattr(model, "fit_train_loader"):
        raise ValueError("Gaussian denoising requires minibatch optimizer training.")


def add_gaussian_noise(x, config, dataset, valid_mask=None):
    """Add independent zero-mean Gaussian noise, preserving targets and padding.

    percentage=10 gives noise SD = 0.1 * population temporal SD, independently
    for each region in each example, measured after dataset preprocessing.
    """
    if config is None or config["percentage"] == 0:
        return x
    shape = x.shape
    valid = torch.ones_like(x, dtype=torch.bool) if valid_mask is None else valid_mask.bool()
    if getattr(dataset, "flatten", False):
        original_shape = getattr(dataset, "original_shape", None)
        if original_shape is None or math.prod(original_shape) != x[0].numel():
            raise ValueError("Gaussian denoising needs original_shape matching flattened input.")
        x = x.reshape(x.shape[0], *original_shape)
        valid = valid.reshape_as(x)
    if x.ndim != 3:
        raise ValueError("Gaussian denoising requires a batch of 2D BOLD timeseries.")
    time_axis = 1 if getattr(dataset, "transpose", False) else 2
    count = valid.sum(dim=time_axis, keepdim=True).clamp_min(1)
    mean = x.masked_fill(~valid, 0).sum(dim=time_axis, keepdim=True) / count
    variance = (x - mean).square().masked_fill(~valid, 0).sum(dim=time_axis, keepdim=True) / count
    noise = torch.randn_like(x) * variance.sqrt() * (config["percentage"] / 100)
    return (x + noise.masked_fill(~valid, 0)).reshape(shape)
