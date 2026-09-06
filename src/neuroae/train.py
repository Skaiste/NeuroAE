import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score

from .metrics.swfcd_torch import SwFCD


def _dataset_valid_last_dim(dataset):
    if not getattr(dataset, "pad_features", False):
        return None
    original_shape = getattr(dataset, "original_shape", None)
    if original_shape is None or len(original_shape) == 0:
        return None
    valid_last_dim = int(original_shape[-1])
    if valid_last_dim <= 0:
        return None
    return valid_last_dim


def _build_valid_mask(x, valid_last_dim):
    if valid_last_dim is None or x.shape[-1] <= valid_last_dim:
        return None
    mask = torch.zeros_like(x)
    mask[..., :valid_last_dim] = 1.0
    return mask


def _apply_recon_mask(x, model_output, mask):
    if mask is None:
        return model_output

    def _mask_recon(recon):
        return recon * mask + x * (1.0 - mask)

    if isinstance(model_output, dict):
        out = dict(model_output)
        for key in ("x_hat", "recon", "reconstruction"):
            if key in out and torch.is_tensor(out[key]):
                out[key] = _mask_recon(out[key])
                break
        return out

    if isinstance(model_output, tuple):
        if len(model_output) == 0:
            return model_output
        return (_mask_recon(model_output[0]), *model_output[1:])

    if isinstance(model_output, list):
        if len(model_output) == 0:
            return model_output
        out = list(model_output)
        out[0] = _mask_recon(out[0])
        return out

    if torch.is_tensor(model_output):
        return _mask_recon(model_output)

    return model_output


def _masked_mse(x_hat, x, mask):
    if mask is None:
        return F.mse_loss(x_hat, x, reduction="mean")
    if x_hat.shape != x.shape:
        if x_hat.shape[:-1] != x.shape[:-1]:
            raise ValueError(
                f"Cannot align x_hat shape {tuple(x_hat.shape)} with x shape {tuple(x.shape)}."
            )
        common_last_dim = min(x_hat.shape[-1], x.shape[-1], mask.shape[-1])
        x_hat = x_hat[..., :common_last_dim]
        x = x[..., :common_last_dim]
        mask = mask[..., :common_last_dim]
    se = (x_hat - x).pow(2) * mask
    denom = mask.sum().clamp_min(1.0)
    return se.sum() / denom


def _extract_model_outputs(model_out):
    if isinstance(model_out, dict):
        recon_x = None
        for key in ("x_hat", "recon", "reconstruction"):
            value = model_out.get(key)
            if torch.is_tensor(value):
                recon_x = value
                break
        latent = None
        for key in ("z", "mu"):
            value = model_out.get(key)
            if torch.is_tensor(value):
                latent = value
                break
    elif isinstance(model_out, (tuple, list)):
        recon_x = model_out[0]
        latent = model_out[-1]
    else:
        recon_x = model_out
        latent = None

    return recon_x, latent


def _append_history_metric(history, split, metric_name, value):
    if metric_name not in history[split]:
        history[split][metric_name] = []
    history[split][metric_name].append(float(value) if value is not None else float("nan"))


def _metric_values(history, split, metric):
    split_metrics = history.get(split)
    if isinstance(split_metrics, dict):
        values = split_metrics.get(metric, [])
        return values if isinstance(values, list) else []
    return []


def _is_finite_number(value):
    return value is not None and np.isfinite(value)


def _compare_higher(candidate, best, min_delta=0.0):
    if _is_finite_number(candidate) and not _is_finite_number(best):
        return 1
    if not _is_finite_number(candidate) and _is_finite_number(best):
        return -1
    if not _is_finite_number(candidate) and not _is_finite_number(best):
        return 0
    if (candidate - best) > min_delta:
        return 1
    if (best - candidate) > min_delta:
        return -1
    return 0


def _compare_lower(candidate, best, min_delta=0.0):
    return _compare_higher(best, candidate, min_delta=min_delta)


def _joint_metric_score(swfcd, logreg, swfcd_weight=0.5, logreg_weight=0.5):
    score = 0.0
    total_weight = 0.0
    if _is_finite_number(swfcd):
        score += float(swfcd_weight) * float(swfcd)
        total_weight += float(swfcd_weight)
    if _is_finite_number(logreg):
        score += float(logreg_weight) * float(logreg)
        total_weight += float(logreg_weight)
    if total_weight == 0.0:
        return float("nan")
    return score / total_weight


def _head_loss_from_metrics(metrics, configured_key=None):
    """Return the validation loss produced by a supervised AE head.

    A classification head takes precedence.  Prediction-head losses are
    otherwise averaged so models with multiple biological targets use one
    comparable checkpoint signal.
    """
    if configured_key:
        value = metrics.get(configured_key)
        return float(value) if _is_finite_number(value) else float("nan")
    if _is_finite_number(metrics.get("cls_loss")):
        return float(metrics["cls_loss"])
    excluded = {"loss", "recon_loss", "fc_loss", "swfc_variability_loss", "derivative_loss"}
    values = [
        float(value)
        for name, value in metrics.items()
        if name.endswith("_loss") and name not in excluded and _is_finite_number(value)
    ]
    return float(np.mean(values)) if values else float("nan")


def _extract_cls_logits(model_out):
    """Extract logits from LAEClsHead/LAEPredClsHeads output tuples."""
    if isinstance(model_out, (tuple, list)) and len(model_out) >= 3:
        candidate = model_out[-2]
        if torch.is_tensor(candidate) and candidate.ndim == 2:
            return candidate
    return None


def _reject_removed_selection_metric(selection_metric):
    if selection_metric in {"swfcd_loss_joint", "swfcd_joint"}:
        raise ValueError(
            f"Checkpoint selection metric {selection_metric!r} has been removed. "
            "Use 'val_loss' for selection and early stopping based on validation loss."
        )


def select_best_checkpoint(
    history,
    selection_metric="val_loss",
    min_delta=0.0,
    swfcd_weight=0.5,
    classifier_weight=0.5,
):
    _reject_removed_selection_metric(selection_metric)
    val_losses = _metric_values(history, "val", "loss")
    val_swfcd = _metric_values(history, "val", "swfcd_pearson")
    val_head_loss = _metric_values(history, "val", "head_loss")
    val_cls_macro_f1 = _metric_values(history, "val", "cls_macro_f1")
    num_epochs = max(len(val_losses), len(val_swfcd), len(val_head_loss), len(val_cls_macro_f1))
    if num_epochs == 0:
        return None

    def _epoch_metrics(idx):
        loss = float(val_losses[idx]) if idx < len(val_losses) else float("nan")
        swfcd = float(val_swfcd[idx]) if idx < len(val_swfcd) else float("nan")
        head_loss = float(val_head_loss[idx]) if idx < len(val_head_loss) else float("nan")
        cls_macro_f1 = float(val_cls_macro_f1[idx]) if idx < len(val_cls_macro_f1) else float("nan")
        joint_score = (
            swfcd + cls_macro_f1
            if _is_finite_number(swfcd) and _is_finite_number(cls_macro_f1)
            else float("nan")
        )
        return loss, swfcd, head_loss, cls_macro_f1, joint_score

    # Once full auxiliary weight is reached, exclude warmup and ramp checkpoints.
    active = _metric_values(history, "val", "auxiliary_active")
    start_idx = next((idx for idx, value in enumerate(active) if value), 0)
    best_idx = start_idx
    best_loss, best_swfcd, best_head_loss, best_cls_macro_f1, best_joint_score = _epoch_metrics(start_idx)

    if selection_metric == "swfcd_cls_macro_f1_joint":
        for idx in range(start_idx + 1, num_epochs):
            loss, swfcd, head_loss, cls_macro_f1, joint_score = _epoch_metrics(idx)
            if _compare_higher(joint_score, best_joint_score, min_delta=min_delta) > 0:
                best_idx = idx
                best_loss, best_swfcd = loss, swfcd
                best_head_loss, best_cls_macro_f1, best_joint_score = head_loss, cls_macro_f1, joint_score
        return {
            "best_index": best_idx,
            "best_epoch": best_idx + 1,
            "loss": best_loss,
            "swfcd_pearson": best_swfcd,
            "head_loss": best_head_loss,
            "cls_macro_f1": best_cls_macro_f1,
            "selection_metric": selection_metric,
        }

    if selection_metric == "swfcd_head_loss_guarded":
        # This is deliberately sequential. SwFCD is the primary signal: an
        # improvement accepts the epoch regardless of head loss. Head loss is
        # used only when SwFCD is flat (inside its 0.01 guard band). A larger
        # SwFCD drop selects the preceding epoch.
        for idx in range(start_idx + 1, num_epochs):
            _, previous_swfcd, previous_head_loss, _, _ = _epoch_metrics(idx - 1)
            loss, swfcd, head_loss, _, joint_score = _epoch_metrics(idx)
            swfcd_dropped = (
                _is_finite_number(swfcd)
                and _is_finite_number(previous_swfcd)
                and swfcd < (previous_swfcd - 0.01)
            )
            swfcd_improved = _compare_higher(swfcd, previous_swfcd, min_delta=min_delta) > 0
            head_loss_improved = _compare_lower(head_loss, previous_head_loss, min_delta=min_delta) > 0
            if swfcd_dropped:
                break
            if swfcd_improved or head_loss_improved:
                best_idx = idx
                best_loss, best_swfcd, best_head_loss, best_joint_score = loss, swfcd, head_loss, joint_score

        return {
            "best_index": best_idx,
            "best_epoch": best_idx + 1,
            "loss": best_loss,
            "swfcd_pearson": best_swfcd,
            "head_loss": best_head_loss,
            "cls_macro_f1": best_cls_macro_f1,
            "selection_metric": selection_metric,
        }

    for idx in range(start_idx + 1, num_epochs):
        loss, swfcd, head_loss, cls_macro_f1, joint_score = _epoch_metrics(idx)

        if selection_metric in {"swfcd_classifier_joint", "swfcd_logreg_joint"}:
            comparisons = (
                _compare_higher(joint_score, best_joint_score, min_delta=min_delta),
                _compare_higher(swfcd, best_swfcd),
                _compare_lower(loss, best_loss),
            )
            is_better = next((comparison > 0 for comparison in comparisons if comparison != 0), False)
        elif selection_metric in {"swfcd", "swfcd_pearson"}:
            is_better = _compare_higher(swfcd, best_swfcd, min_delta=min_delta) > 0
        else:
            is_better = _compare_lower(loss, best_loss, min_delta=min_delta) > 0

        if is_better:
            best_idx = idx
            best_loss, best_swfcd, best_head_loss, best_cls_macro_f1, best_joint_score = loss, swfcd, head_loss, cls_macro_f1, joint_score

    return {
        "best_index": best_idx,
        "best_epoch": best_idx + 1,
        "loss": best_loss,
        "swfcd_pearson": best_swfcd,
        "head_loss": best_head_loss,
        "cls_macro_f1": best_cls_macro_f1,
        "selection_metric": selection_metric,
    }


def _should_display_loss(loss_name, loss_params):
    """Return whether a loss component contributes to the configured objective."""
    loss_params = loss_params or {}
    if loss_name == "kld":
        return float(loss_params.get("beta", 0.0)) != 0.0
    if loss_name == "fc_loss":
        return float(loss_params.get("fc_weight", 0.0)) != 0.0
    if loss_name == "swfc_variability_loss":
        return float(loss_params.get("swfc_variability_weight", 0.0)) != 0.0
    if loss_name == "derivative_loss":
        return float(loss_params.get("derivative_weight", 0.0)) != 0.0
    if loss_name == "cls_loss":
        return float(
            loss_params.get("cls_head_weight", loss_params.get("cls_head_delta", 1.0))
        ) != 0.0
    if loss_name.endswith("_loss"):
        return float(loss_params.get("pred_heads_delta", 0.0)) != 0.0
    return True


def loss_params2str(train_params, train_batches, val_params, val_batches, loss_params=None):
    def _format_loss_dict(params, type, batches):
        return " | ".join(
            f"{type} {k}: {float(v/batches):.4f}"
            for k, v in params.items()
            if _should_display_loss(k, loss_params)
        )

    train_pstr = _format_loss_dict(train_params, "Train", train_batches)
    val_pstr = _format_loss_dict(val_params, "Val", val_batches)
    return f"{train_pstr} | {val_pstr}"


def _train_only_loss_params_str(train_params, train_batches, loss_params=None):
    return " | ".join(
        f"Train {k}: {float(v/train_batches):.4f}"
        for k, v in train_params.items()
        if _should_display_loss(k, loss_params)
    )


def _batch_labels_to_list(batch_labels):
    if isinstance(batch_labels, torch.Tensor):
        return batch_labels.detach().cpu().tolist()
    if isinstance(batch_labels, np.ndarray):
        return batch_labels.tolist()
    if isinstance(batch_labels, (list, tuple)):
        return list(batch_labels)
    return [batch_labels]


def _collect_latents_and_labels(model, data_loader, device, use_pred_heads, valid_last_dim):
    latents = []
    labels = []
    model.eval()
    with torch.no_grad():
        for data, batch_labels in data_loader:
            x = data.to(device)
            valid_mask = _build_valid_mask(x, valid_last_dim)
            output = model(x)
            output = _apply_recon_mask(x, output, valid_mask)
            _, latent = _extract_model_outputs(output)
            if latent is None:
                continue
            latents.append(latent.detach().cpu())
            raw_labels = batch_labels[0] if use_pred_heads else batch_labels
            labels.extend(_batch_labels_to_list(raw_labels))
    if not latents:
        return None, []
    return torch.cat(latents, dim=0).numpy(), labels

def _optimizer_param_groups(model, weight_decay, aux_learning_rate=None):
    """Give auxiliary heads an optional learning rate and zero weight decay."""
    if aux_learning_rate is not None:
        aux_learning_rate = float(aux_learning_rate)
        if not np.isfinite(aux_learning_rate) or aux_learning_rate < 0:
            raise ValueError("aux_learning_rate must be a finite non-negative number or null.")
    auxiliary_ids = {
        id(param)
        for name in ("heads", "cls_head")
        for module in [getattr(model, name, None)]
        if module is not None
        for param in module.parameters()
    }
    ae_params, aux_params = [], []
    for param in model.parameters():
        (aux_params if id(param) in auxiliary_ids else ae_params).append(param)
    groups = []
    if ae_params:
        groups.append({"params": ae_params, "weight_decay": weight_decay})
    if aux_params:
        auxiliary_group = {"params": aux_params, "weight_decay": 0.0}
        if aux_learning_rate is not None:
            auxiliary_group["lr"] = aux_learning_rate
        groups.append(auxiliary_group)
    return groups

def _get_aux_loss(loss, use_pred_heads=False, use_cls_head=False):
    """Return the unweighted aggregate auxiliary loss."""
    if use_cls_head:
        return loss.get("cls_loss")

    if use_pred_heads:
        excluded = {
            "loss",
            "recon",
            "recon_loss",
            "kld",
            "fc_loss",
            "swfc_variability_loss",
            "derivative_loss",
        }

        aux_losses = [
            value
            for name, value in loss.items()
            if name.endswith("_loss") and name not in excluded
        ]

        if aux_losses:
            return sum(aux_losses) / len(aux_losses)

    return None


def _accumulate_loss_metrics(totals, loss, batch_size, cls_mass=None):
    """Detach batch statistics; CE means use target-weight mass, not batch count."""
    totals["_samples"] = totals.get("_samples", 0) + batch_size
    for name, value in loss.items():
        value = float(value.detach())
        totals[name] = totals.get(name, 0.0) + value * batch_size
        if name == "cls_loss" and cls_mass is not None:
            totals["_cls_sum"] = totals.get("_cls_sum", 0.0) + value * cls_mass
            totals["_cls_mass"] = totals.get("_cls_mass", 0.0) + cls_mass


def _mean_loss_metrics(totals, cls_weight=0.0):
    if not totals.get("_samples"):
        raise ValueError("Training and validation loaders must contain at least one batch.")
    metrics = {key: value / totals["_samples"] for key, value in totals.items() if not key.startswith("_")}
    if totals.get("_cls_mass"):
        cls_mean = totals["_cls_sum"] / totals["_cls_mass"]
        correction = cls_weight * (cls_mean - metrics["cls_loss"])
        metrics["cls_loss"] = cls_mean
        metrics["loss"] += correction
    return metrics


def _auxiliary_weight_scale(epoch, warmup_epochs, ramp_epochs):
    if epoch < warmup_epochs:
        return 0.0
    if ramp_epochs == 0:
        return 1.0
    return min(1.0, (epoch - warmup_epochs + 1) / ramp_epochs)


def _training_loss(model, *args, auxiliary_warmup=False, auxiliary_scale=1.0):
    """Scale auxiliary objectives while preserving configured target weights."""
    scale = 0.0 if auxiliary_warmup else auxiliary_scale
    if scale == 1.0:
        return model.loss(*args)
    original_params = model.loss_fn_params
    params = original_params or {}
    model.loss_fn_params = {
        **params,
        "pred_heads_delta": scale * float(params.get("pred_heads_delta", 0.0)),
        "cls_head_weight": scale * float(params.get("cls_head_weight", params.get("cls_head_delta", 1.0))),
        "cls_head_delta": scale * float(params.get("cls_head_delta", 1.0)),
    }
    try:
        return model.loss(*args)
    finally:
        model.loss_fn_params = original_params



def _dataset_class_labels(dataset):
    """Read actual split labels without iterating a shuffled/drop-last loader."""
    if isinstance(dataset, torch.utils.data.Subset):
        labels = _dataset_class_labels(dataset.dataset)
        return [labels[index] for index in dataset.indices]
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        return [label for part in dataset.datasets for label in _dataset_class_labels(part)]
    labels = getattr(dataset, "labels", None)
    if labels is None and isinstance(dataset, torch.utils.data.TensorDataset):
        labels = dataset.tensors[1]
    if labels is None:
        labels = [dataset[index][1] for index in range(len(dataset))]
    return [label.item() if torch.is_tensor(label) and label.ndim == 0 else label for label in labels]


def _configure_classifier_class_weights(model, dataset):
    """Fit inverse-frequency or square-root weights using training labels only."""
    setting = (getattr(model, "loss_fn_params", {}) or {}).get("cls_class_weights", "weighted")
    if isinstance(setting, str):
        setting = setting.lower().replace("-", "_")
    num_classes = len(model.class_to_idx)
    if setting is None or (isinstance(setting, str) and setting == "unweighted"):
        weights = torch.ones(num_classes)
    elif isinstance(setting, str) and setting in {"weighted", "balanced", "sqrt_balanced"}:
        counts = torch.zeros(num_classes)
        for label in _dataset_class_labels(dataset):
            if label not in model.class_to_idx:
                raise ValueError(f"Encountered class label not configured for cls_head: {label!r}")
            counts[model.class_to_idx[label]] += 1
        if (counts == 0).any():
            missing = [label for label, index in model.class_to_idx.items() if counts[index] == 0]
            raise ValueError(f"Balanced classifier loss requires training samples for every class; missing: {missing!r}")
        weights = counts.sum() / (num_classes * counts)
        if setting == "sqrt_balanced":
            weights = weights.sqrt()
    elif isinstance(setting, (list, tuple)):
        weights = torch.as_tensor(setting, dtype=torch.float32)
    else:
        raise ValueError("cls_class_weights must be 'weighted', 'unweighted', 'sqrt_balanced', "
            "or a list of positive weights in class_labels order ('balanced' and null are legacy aliases).")
    if weights.shape != (num_classes,) or not torch.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError("cls_class_weights must contain one finite positive weight per class.")
    model.cls_class_weights.copy_(weights.to(model.cls_class_weights))


def train_vae(
    model,
    train_loader,
    val_loader=None,
    num_epochs=100,
    learning_rate=1e-3,
    weight_decay=1e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    save_dir='./checkpoints',
    name='basicVAE_general',
    pca=None,
    noise=None,
    use_pred_heads=False,
    use_cls_head=False,
    convergence_patience=None,
    convergence_min_delta=0.0,
    convergence_warmup_epochs=0,
    checkpoint_selection_metric="val_loss",
    save_checkpoint=True,
    vectorize_val_reference=False,
    compute_swfcd_during_training=None,
    aux_head_warmup_epochs=0,
    aux_learning_rate=None,
    aux_head_ramp_epochs=20,
):
    if isinstance(aux_head_warmup_epochs, bool) or not isinstance(aux_head_warmup_epochs, int) or aux_head_warmup_epochs < 0:
        raise ValueError("aux_head_warmup_epochs must be a non-negative integer.")

    if isinstance(aux_head_ramp_epochs, bool) or not isinstance(aux_head_ramp_epochs, int) or aux_head_ramp_epochs < 0:
        raise ValueError("aux_head_ramp_epochs must be a non-negative integer.")

    auxiliary_modules = [
        module
        for name in ("heads", "cls_head")
        for module in [getattr(model, name, None)]
        if module is not None
    ]

    if not (use_pred_heads or use_cls_head):
        aux_head_warmup_epochs = 0
        aux_head_ramp_epochs = 0

    device = torch.device(device)
    model = model.to(device)

    if use_pred_heads and use_cls_head:
        raise ValueError("Prediction heads and a classification head cannot be trained together.")

    class_to_idx = getattr(model, "class_to_idx", None)
    if use_cls_head and not class_to_idx:
        raise ValueError("Classification-head training requires model.class_to_idx.")
    if use_cls_head:
        _configure_classifier_class_weights(model, train_loader.dataset)

    def _class_targets(batch_labels):
        raw_labels = batch_labels[0] if use_pred_heads else batch_labels
        try:
            labels = [label.item() if torch.is_tensor(label) and label.ndim == 0 else label for label in raw_labels]
            return torch.as_tensor([class_to_idx[label] for label in labels], device=device)
        except KeyError as exc:
            raise ValueError(f"Encountered class label not configured for cls_head: {exc.args[0]!r}") from exc

    if noise is not None:
        noise = {k: v for p in noise for k, v in p.items()}

    history = {"train": {}, "val": {}}
    best_model_losses = None
    epochs_without_improvement = 0

    requires_optimizer = bool(getattr(model, "requires_optimizer", True))
    optimizer = (
        optim.AdamW(
            _optimizer_param_groups(model, weight_decay, aux_learning_rate),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        if requires_optimizer
        else None
    )

    train_valid_last_dim = _dataset_valid_last_dim(train_loader.dataset)
    val_valid_last_dim = _dataset_valid_last_dim(val_loader.dataset) if val_loader is not None else None

    selection_metric = str(checkpoint_selection_metric or "val_loss")
    _reject_removed_selection_metric(selection_metric)
    selection_requires_joint_metrics = selection_metric in {"swfcd_classifier_joint", "swfcd_logreg_joint"}

    if selection_requires_joint_metrics:
        raise ValueError(
            "Training-time checkpoint_selection_metric values "
            "'swfcd_classifier_joint' and 'swfcd_logreg_joint' are no longer supported "
            "because classifier metrics are evaluation-only. Use 'swfcd' or 'val_loss' instead."
        )

    compute_head_loss_during_training = selection_metric == "swfcd_head_loss_guarded"
    compute_cls_macro_f1_during_training = use_cls_head

    if selection_metric == "swfcd_cls_macro_f1_joint" and not use_cls_head:
        raise ValueError("swfcd_cls_macro_f1_joint requires a model with a classification auxiliary head.")

    if compute_swfcd_during_training is None:
        compute_swfcd_during_training = selection_metric in {
            "swfcd",
            "swfcd_pearson",
            "swfcd_head_loss_guarded",
            "swfcd_cls_macro_f1_joint",
        }

    if val_loader is None:
        compute_swfcd_during_training = False

    val_swfcd = SwFCD(val_loader.dataset, 30, 3) if (compute_swfcd_during_training and val_loader is not None) else None
    val_reference_vec = None

    if (
        compute_swfcd_during_training
        and val_loader is not None
        and vectorize_val_reference
        and not getattr(val_loader.dataset, "fc_input", False)
    ):
        val_reference = torch.as_tensor(val_loader.dataset.data, dtype=torch.float32, device=device)
        val_reference_vec = val_swfcd.vectorize(val_reference, track_grad=False)

    max_training_epochs = getattr(model, "max_training_epochs", None)
    if max_training_epochs is not None:
        num_epochs = min(int(num_epochs), int(max_training_epochs))

    # Center validation only against its fixed warmup baseline. Training uses
    # the unshifted objective. Validation subtracts the scaled baseline.
    val_aux_delta = None
    full_aux_epoch = aux_head_warmup_epochs + max(aux_head_ramp_epochs - 1, 0)

    for epoch in range(num_epochs):
        auxiliary_warmup = epoch < aux_head_warmup_epochs
        auxiliary_scale = _auxiliary_weight_scale(epoch, aux_head_warmup_epochs, aux_head_ramp_epochs)
        capture_aux_delta = aux_head_warmup_epochs > 0 and epoch == aux_head_warmup_epochs - 1

        if full_aux_epoch and epoch == full_aux_epoch:
            best_model_losses = None
            epochs_without_improvement = 0

        if epoch == 0 and hasattr(model, "fit_train_loader"):
            model.fit_train_loader(train_loader, device=device)

        # =========================
        # Training
        # =========================
        train_loss_params = {}
        cls_weight = 0.0 if not use_cls_head else auxiliary_scale * float(
            model.loss_fn_params.get("cls_head_weight", model.loss_fn_params.get("cls_head_delta", 1.0))
        )

        model.train()
        if auxiliary_warmup:
            for module in auxiliary_modules:
                module.eval()

        for batch_idx, (data, labels) in enumerate(train_loader):
            x = data.to(device)
            valid_mask = _build_valid_mask(x, train_valid_last_dim)

            if noise is not None:
                if noise["type"] == "gaussian":
                    x += torch.randn_like(x) + float(noise["std"])
                elif noise["type"] == "mask":
                    x *= (torch.rand_like(x) > float(noise["ratio"])).float()

            if optimizer is not None:
                optimizer.zero_grad()

            output = model(x)
            output = _apply_recon_mask(x, output, valid_mask)

            if use_pred_heads:
                heads = {bl: h.to(device) for bl, h in labels[1].items()}
                loss = _training_loss(model, x, heads, output, auxiliary_warmup=auxiliary_warmup, auxiliary_scale=auxiliary_scale)
            elif use_cls_head:
                loss = _training_loss(model, x, _class_targets(labels), output, auxiliary_warmup=auxiliary_warmup, auxiliary_scale=auxiliary_scale)
            else:
                loss = model.loss(x, output)

            cls_mass = float(model.cls_class_weights[_class_targets(labels)].sum()) if use_cls_head else None
            _accumulate_loss_metrics(train_loss_params, loss, x.shape[0], cls_mass)

            if optimizer is not None:
                loss["loss"].backward()

                if auxiliary_warmup:
                    for module in auxiliary_modules:
                        for param in module.parameters():
                            param.grad = None

                optimizer.step()

        train_loss_params = _mean_loss_metrics(train_loss_params, cls_weight)
        num_batches = 1  # Metrics below are already normalized over the epoch.
        for key, value in train_loss_params.items():
            _append_history_metric(history, "train", key, value)

        val_metric_str = ""
        current_metrics = {
            "train": {p: train_loss_params[p] / num_batches for p in train_loss_params},
            "val": {},
        }

        # =========================
        # Validation
        # =========================
        if val_loader is not None:
            model.eval()

            val_loss_params = {}

            val_recons = [] if val_reference_vec is not None else None
            swfcd_pearson_sum = 0.0
            swfcd_pearson_count = 0
            val_cls_targets = []
            val_cls_predictions = []

            with torch.no_grad():
                for batch_idx, (data, labels) in enumerate(val_loader):
                    x = data.to(device)
                    valid_mask = _build_valid_mask(x, val_valid_last_dim)

                    output = model(x)
                    output = _apply_recon_mask(x, output, valid_mask)

                    if use_pred_heads:
                        heads = {bl: h.to(device) for bl, h in labels[1].items()}
                        loss = _training_loss(model, x, heads, output, auxiliary_warmup=auxiliary_warmup, auxiliary_scale=auxiliary_scale)
                    elif use_cls_head:
                        loss = _training_loss(model, x, _class_targets(labels), output, auxiliary_warmup=auxiliary_warmup, auxiliary_scale=auxiliary_scale)
                    else:
                        loss = model.loss(x, output)

                    val_aux_loss = _get_aux_loss(loss, use_pred_heads=use_pred_heads, use_cls_head=use_cls_head)

                    if not auxiliary_warmup and val_aux_delta is not None and val_aux_loss is not None:
                        if use_cls_head:
                            aux_weight = float(model.loss_fn_params.get("cls_head_weight", model.loss_fn_params.get("cls_head_delta", 1.0)))
                        else:
                            aux_weight = float(model.loss_fn_params.get("pred_heads_delta", 0.0))

                        loss["loss"] = loss["loss"] - auxiliary_scale * aux_weight * val_aux_delta

                    cls_mass = float(model.cls_class_weights[_class_targets(labels)].sum()) if use_cls_head else None
                    _accumulate_loss_metrics(val_loss_params, loss, x.shape[0], cls_mass)

                    if compute_cls_macro_f1_during_training:
                        logits = _extract_cls_logits(output)
                        if logits is None:
                            raise ValueError("Could not extract classification logits from the auxiliary-head model output.")

                        targets = _class_targets(labels)
                        val_cls_targets.extend(targets.detach().cpu().tolist())
                        val_cls_predictions.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())

                    recon_x, _ = _extract_model_outputs(output)
                    recon_x_detached = recon_x.detach()

                    if val_recons is not None:
                        val_recons.append(recon_x_detached)
                    elif compute_swfcd_during_training and not getattr(val_loader.dataset, "fc_input", False):
                        swfcd_results = val_swfcd.apply(x.detach(), recon_x_detached)

                        if swfcd_results is not None:
                            swfcd_pearson_sum += float(swfcd_results["pearson"].detach().cpu().item()) * data.shape[0]
                            swfcd_pearson_count += int(data.shape[0])

            val_loss_params = _mean_loss_metrics(val_loss_params, cls_weight)
            num_val_batches = 1
            if capture_aux_delta:
                val_aux_delta = _get_aux_loss(val_loss_params, use_pred_heads, use_cls_head)
                if val_aux_delta is not None:
                    print(f"Captured val auxiliary baseline: {val_aux_delta:.6f}", flush=True)
            for key, value in val_loss_params.items():
                _append_history_metric(history, "val", key, value)
            if aux_head_warmup_epochs or aux_head_ramp_epochs:
                _append_history_metric(history, "val", "auxiliary_active", auxiliary_scale == 1.0)

            swfcd_pearson = float("nan")

            if compute_swfcd_during_training:
                if val_reference_vec is not None and val_recons:
                    swfcd_results = val_swfcd.apply(None, torch.cat(val_recons, dim=0), x_vec=val_reference_vec)

                    if swfcd_results is not None:
                        swfcd_pearson = float(swfcd_results["pearson"].detach().cpu().item())

                elif swfcd_pearson_count > 0:
                    swfcd_pearson = swfcd_pearson_sum / swfcd_pearson_count

                _append_history_metric(history, "val", "swfcd_pearson", swfcd_pearson)

                val_metric_str += (
                    f" | Val swfcd_pearson: {swfcd_pearson:.4f}"
                    if np.isfinite(swfcd_pearson)
                    else " | Val swfcd_pearson: nan"
                )

            current_metrics["val"] = {p: val_loss_params[p] / num_val_batches for p in val_loss_params}

            if compute_swfcd_during_training:
                current_metrics["val"]["swfcd_pearson"] = history["val"]["swfcd_pearson"][-1]

            if compute_cls_macro_f1_during_training:
                cls_macro_f1 = float(
                    f1_score(
                        val_cls_targets,
                        val_cls_predictions,
                        labels=list(range(len(class_to_idx))),
                        average="macro",
                        zero_division=0,
                    )
                )

                _append_history_metric(history, "val", "cls_macro_f1", cls_macro_f1)
                current_metrics["val"]["cls_macro_f1"] = cls_macro_f1
                val_metric_str += f" | Val cls_macro_f1: {cls_macro_f1:.4f}"

            if compute_head_loss_during_training:
                previous_swfcd = (
                    history["val"]["swfcd_pearson"][-2]
                    if len(history["val"]["swfcd_pearson"]) > 1 and epoch != full_aux_epoch
                    else None
                )

                swfcd_dropped = (
                    _is_finite_number(swfcd_pearson)
                    and _is_finite_number(previous_swfcd)
                    and swfcd_pearson < previous_swfcd - 0.01
                )

                if swfcd_dropped:
                    head_loss = float("nan")
                    val_metric_str += " | Val head_loss: skipped (SwFCD drop > 0.01)"
                else:
                    head_loss = _head_loss_from_metrics(
                        current_metrics["val"],
                        configured_key=getattr(model, "loss_fn_params", {}).get("checkpoint_head_loss_key"),
                    )

                    if not _is_finite_number(head_loss):
                        raise ValueError(
                            "swfcd_head_loss_guarded requires a supervised head loss in validation metrics. "
                            "Use a model with cls_loss or prediction-head losses, or set "
                            "loss_params.checkpoint_head_loss_key."
                        )

                    val_metric_str += f" | Val head_loss: {head_loss:.4f}"

                _append_history_metric(history, "val", "head_loss", head_loss)
                current_metrics["val"]["head_loss"] = head_loss

        # =========================
        # Logging
        # =========================
        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"{loss_params2str(train_loss_params, num_batches, val_loss_params, num_val_batches, model.loss_fn_params) if val_loader is not None else _train_only_loss_params_str(train_loss_params, num_batches, model.loss_fn_params)}"
            f"{val_metric_str}",
            flush=True,
        )

        # =========================
        # Checkpoint selection
        # =========================
        if val_loader is None:
            improved = False
        elif best_model_losses is None:
            improved = True
        else:
            tmp_history = (
                history
                if selection_metric == "swfcd_head_loss_guarded"
                else {
                    "val": {
                        "loss": [
                            best_model_losses["val"].get("loss", float("nan")),
                            current_metrics["val"].get("loss", float("nan")),
                        ],
                        "swfcd_pearson": [
                            best_model_losses["val"].get("swfcd_pearson", float("nan")),
                            current_metrics["val"].get("swfcd_pearson", float("nan")),
                        ],
                        "head_loss": [
                            best_model_losses["val"].get("head_loss", float("nan")),
                            current_metrics["val"].get("head_loss", float("nan")),
                        ],
                        "cls_macro_f1": [
                            best_model_losses["val"].get("cls_macro_f1", float("nan")),
                            current_metrics["val"].get("cls_macro_f1", float("nan")),
                        ],
                    }
                }
            )

            selection = select_best_checkpoint(
                tmp_history,
                selection_metric=checkpoint_selection_metric,
                min_delta=convergence_min_delta,
            )

            latest_index = len(history["val"].get("loss", [])) - 1
            improved = selection is not None and selection["best_index"] == (
                latest_index if selection_metric == "swfcd_head_loss_guarded" else 1
            )

        if improved:
            best_model_losses = current_metrics
            epochs_without_improvement = 0

            if save_checkpoint:
                torch.save(model.state_dict(), f"{save_dir}/{name}_model.pt")

        elif val_loader is not None:
            epochs_without_improvement += 1

        # =========================
        # Early stopping
        # =========================
        if (
            val_loader is not None
            and convergence_patience is not None
            and convergence_patience > 0
            and epoch + 1 > max(convergence_warmup_epochs, full_aux_epoch)
            and epochs_without_improvement >= convergence_patience
        ):
            print(
                f"Converged: stopping early at epoch {epoch + 1} after "
                f"{epochs_without_improvement} epochs without validation-loss improvement."
            )
            break

    # =========================
    # PCA validation
    # =========================
    mse_pca = 0

    if pca is not None and val_loader is not None:
        total_mse_pca = 0
        num_batches = 0

        for batch_idx, (data, _) in enumerate(val_loader):
            x = data.to(device)
            valid_mask = _build_valid_mask(x, val_valid_last_dim)

            z_pca = pca.transform(x.detach().cpu().numpy())
            x_recon_pca = pca.inverse_transform(z_pca)
            x_recon_pca = torch.as_tensor(x_recon_pca, dtype=x.dtype, device=x.device)

            mse_pca = _masked_mse(x_recon_pca, x, valid_mask)
            total_mse_pca += mse_pca.item()
            num_batches += 1

        mse_pca = float(total_mse_pca / num_batches)

    if val_loader is None and save_checkpoint:
        torch.save(model.state_dict(), f"{save_dir}/{name}_model.pt")

    print("Training complete!")
    return history, mse_pca
