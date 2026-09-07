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


def _extract_cls_logits(model_out):
    """Extract logits from LAEClsHead/LAEPredClsHeads output tuples."""
    if isinstance(model_out, (tuple, list)) and len(model_out) >= 3:
        candidate = model_out[-2]
        if torch.is_tensor(candidate) and candidate.ndim == 2:
            return candidate
    return None


def _reject_removed_selection_metric(selection_metric):
    if selection_metric not in {"val_loss", "swfcd", "swfcd_pearson"}:
        raise ValueError("checkpoint_selection_metric must be 'val_loss', 'swfcd', or 'swfcd_pearson'.")


def select_best_checkpoint(history, selection_metric="val_loss", min_delta=0.0):
    _reject_removed_selection_metric(selection_metric)
    val_losses = _metric_values(history, "val", "loss")
    val_swfcd = _metric_values(history, "val", "swfcd_pearson")
    val_f1 = _metric_values(history, "val", "cls_macro_f1")
    count = max(len(val_losses), len(val_swfcd))
    if not count:
        return None
    def value(values, index):
        return float(values[index]) if index < len(values) else float("nan")
    values = val_losses if selection_metric == "val_loss" else val_swfcd
    compare = _compare_lower if selection_metric == "val_loss" else _compare_higher
    best = 0
    for index in range(1, count):
        if compare(value(values, index), value(values, best), min_delta) > 0:
            best = index
    return {
        "best_index": best,
        "best_epoch": best + 1,
        "loss": value(val_losses, best),
        "swfcd_pearson": value(val_swfcd, best),
        "cls_macro_f1": value(val_f1, best),
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
):
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
        optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if requires_optimizer
        else None
    )

    train_valid_last_dim = _dataset_valid_last_dim(train_loader.dataset)
    val_valid_last_dim = _dataset_valid_last_dim(val_loader.dataset) if val_loader is not None else None

    selection_metric = str(checkpoint_selection_metric or "val_loss")
    _reject_removed_selection_metric(selection_metric)
    compute_cls_macro_f1_during_training = use_cls_head
    if compute_swfcd_during_training is None:
        compute_swfcd_during_training = selection_metric in {"swfcd", "swfcd_pearson"}

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

    for epoch in range(num_epochs):
        if epoch == 0 and hasattr(model, "fit_train_loader"):
            model.fit_train_loader(train_loader, device=device)

        # =========================
        # Training
        # =========================
        train_loss_params = {}
        cls_weight = 0.0 if not use_cls_head else float(
            model.loss_fn_params.get("cls_head_weight", model.loss_fn_params.get("cls_head_delta", 1.0))
        )

        model.train()

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
                loss = model.loss(x, heads, output)
            elif use_cls_head:
                loss = model.loss(x, _class_targets(labels), output)
            else:
                loss = model.loss(x, output)

            cls_mass = float(model.cls_class_weights[_class_targets(labels)].sum()) if use_cls_head else None
            _accumulate_loss_metrics(train_loss_params, loss, x.shape[0], cls_mass)

            if optimizer is not None:
                loss["loss"].backward()

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
                        loss = model.loss(x, heads, output)
                    elif use_cls_head:
                        loss = model.loss(x, _class_targets(labels), output)
                    else:
                        loss = model.loss(x, output)

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
            for key, value in val_loss_params.items():
                _append_history_metric(history, "val", key, value)
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
            tmp_history = {"val": {
                key: [best_model_losses["val"].get(key, float("nan")), current_metrics["val"].get(key, float("nan"))]
                for key in ("loss", "swfcd_pearson", "cls_macro_f1")
            }}

            selection = select_best_checkpoint(
                tmp_history,
                selection_metric=checkpoint_selection_metric,
                min_delta=convergence_min_delta,
            )

            improved = selection is not None and selection["best_index"] == 1

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
        patience = convergence_patience
        if (
            val_loader is not None
            and patience is not None
            and patience > 0
            and epoch + 1 > convergence_warmup_epochs
            and epochs_without_improvement >= patience
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
