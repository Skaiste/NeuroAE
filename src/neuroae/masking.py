"""Training-only BOLD corruption; reconstruction targets remain clean."""

import math
from collections.abc import Mapping

import torch


def validate_masking(config, dataset, model):
    if config is None:
        return
    if not isinstance(config, Mapping):
        raise ValueError("training.masking must be a mapping.")
    if set(config) - {"mode", "percentage", "segment_length"}:
        raise ValueError("Unknown training.masking option.")
    if config.get("mode") not in {"values", "timepoints", "segments", "regions"}:
        raise ValueError("masking.mode must be values, timepoints, segments, or regions.")
    percentage = config.get("percentage")
    if isinstance(percentage, bool) or not isinstance(percentage, (int, float)) or not math.isfinite(percentage) or not 0 <= percentage <= 100:
        raise ValueError("masking.percentage must be a number from 0 to 100.")
    length = config.get("segment_length", 5)
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise ValueError("masking.segment_length must be a positive integer.")
    if percentage == 0:
        return
    if getattr(dataset, "fc_input", False):
        raise ValueError("BOLD masking requires timeseries input, not FC input.")
    if not getattr(model, "requires_optimizer", True) or hasattr(model, "fit_train_loader"):
        raise ValueError("BOLD masking requires a model trained through the minibatch optimizer.")
    if config["mode"] != "values" and getattr(dataset, "timepoints_as_samples", False):
        raise ValueError("Temporal and region masking require timepoints_as_samples: false.")


def mask_bold(x, config, dataset, valid_mask=None):
    """Zero a fixed random budget per sample without modifying ``x``.

    Region mode masks whole region time courses.
    Temporal modes mask all regions together. Segments are randomly selected
    contiguous blocks of at most segment_length, with a random block offset.
    The final selected block is shortened to meet the rounded masking budget.
    """
    if config is None or config["percentage"] == 0:
        return x
    valid = torch.ones_like(x, dtype=torch.bool) if valid_mask is None else valid_mask.bool()
    shape = x.shape
    if getattr(dataset, "flatten", False) and not getattr(dataset, "timepoints_as_samples", False):
        original_shape = getattr(dataset, "original_shape", None)
        if original_shape is None or math.prod(original_shape) != x[0].numel():
            raise ValueError("BOLD masking needs original_shape matching the flattened input.")
        x = x.reshape(x.shape[0], *original_shape)
        valid = valid.reshape_as(x)
    masked = x.clone(memory_format=torch.contiguous_format)
    fraction = config["percentage"] / 100
    if config["mode"] == "values":
        for sample, allowed in zip(masked, valid):
            indices = allowed.reshape(-1).nonzero().flatten()
            count = int(len(indices) * fraction + 0.5)
            chosen = indices[torch.randperm(len(indices), device=x.device)[:count]]
            sample.reshape(-1)[chosen] = 0
    else:
        if x.ndim != 3:
            raise ValueError("Temporal and region masking require a batch of 2D BOLD timeseries.")
        # Dataset samples are (regions, time) unless transposed.
        time_axis = 1 if getattr(dataset, "transpose", False) else 2
        selection_axis = 3 - time_axis if config["mode"] == "regions" else time_axis
        grouped = masked.movedim(selection_axis, 1)
        grouped_valid = valid.movedim(selection_axis, 1).any(dim=-1)
        for sample, allowed in zip(grouped, grouped_valid):
            indices = allowed.nonzero().flatten()
            count = int(len(indices) * fraction + 0.5)
            if config["mode"] in {"timepoints", "regions"}:
                chosen = indices[torch.randperm(len(indices), device=x.device)[:count]]
                sample[chosen] = 0
            else:
                length = config.get("segment_length", 5)
                offset = int(torch.randint(min(length, max(len(indices), 1)), (), device=x.device))
                blocks = []
                if offset:
                    blocks.append(indices[:offset])
                blocks.extend(indices[offset:].split(length))
                for block_idx in torch.randperm(len(blocks)).tolist():
                    block = blocks[block_idx][:count]
                    sample[block] = 0
                    count -= len(block)
                    if count == 0:
                        break
    # Preserve padding even when an entire timepoint was selected.
    return torch.where(valid, masked, x).reshape(shape)
