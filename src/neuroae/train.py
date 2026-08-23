import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from .metrics.classifier_accuracy import run_latent_braingnn_classifier
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


def select_best_checkpoint(
    history,
    selection_metric="swfcd_loss_joint",
    min_delta=0.0,
    swfcd_weight=0.5,
    classifier_weight=0.5,
):
    val_losses = _metric_values(history, "val", "loss")
    val_swfcd = _metric_values(history, "val", "swfcd_pearson")
    num_epochs = max(len(val_losses), len(val_swfcd))
    if num_epochs == 0:
        return None

    def _epoch_metrics(idx):
        loss = float(val_losses[idx]) if idx < len(val_losses) else float("nan")
        swfcd = float(val_swfcd[idx]) if idx < len(val_swfcd) else float("nan")
        joint_score = _joint_metric_score(swfcd, float("nan"), swfcd_weight=swfcd_weight, logreg_weight=classifier_weight)
        return loss, swfcd, joint_score

    best_idx = 0
    best_loss, best_swfcd, best_joint_score = _epoch_metrics(0)

    for idx in range(1, num_epochs):
        loss, swfcd, joint_score = _epoch_metrics(idx)

        if selection_metric in {"swfcd_classifier_joint", "swfcd_logreg_joint"}:
            comparisons = (
                _compare_higher(joint_score, best_joint_score, min_delta=min_delta),
                _compare_higher(swfcd, best_swfcd),
                _compare_lower(loss, best_loss),
            )
            is_better = next((comparison > 0 for comparison in comparisons if comparison != 0), False)
        elif selection_metric in {"swfcd_loss_joint", "swfcd_joint"}:
            comparisons = (
                _compare_higher(swfcd, best_swfcd, min_delta=min_delta),
                _compare_lower(loss, best_loss),
            )
            is_better = next((comparison > 0 for comparison in comparisons if comparison != 0), False)
        elif selection_metric in {"swfcd", "swfcd_pearson"}:
            is_better = _compare_higher(swfcd, best_swfcd, min_delta=min_delta) > 0
        else:
            is_better = _compare_lower(loss, best_loss, min_delta=min_delta) > 0

        if is_better:
            best_idx = idx
            best_loss, best_swfcd, best_joint_score = loss, swfcd, joint_score

    return {
        "best_index": best_idx,
        "best_epoch": best_idx + 1,
        "loss": best_loss,
        "swfcd_pearson": best_swfcd,
        "joint_score": best_joint_score,
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
    checkpoint_selection_metric="swfcd_loss_joint",
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

    def _class_targets(batch_labels):
        raw_labels = batch_labels[0] if use_pred_heads else batch_labels
        try:
            labels = [label.item() if torch.is_tensor(label) and label.ndim == 0 else label for label in raw_labels]
            return torch.as_tensor([class_to_idx[label] for label in labels], device=device)
        except KeyError as exc:
            raise ValueError(f"Encountered class label not configured for cls_head: {exc.args[0]!r}") from exc

    if noise is not None:
        noise = {k:v for p in noise for k,v in p.items()}

    history = {
        'train': {},
        'val': {}
    }
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
    selection_requires_joint_metrics = selection_metric in {"swfcd_classifier_joint", "swfcd_logreg_joint"}
    if selection_requires_joint_metrics:
        raise ValueError(
            "Training-time checkpoint_selection_metric values "
            "'swfcd_classifier_joint' and 'swfcd_logreg_joint' are no longer supported "
            "because classifier metrics are evaluation-only. Use 'swfcd_loss_joint', "
            "'swfcd', or 'val_loss' instead."
        )
    if compute_swfcd_during_training is None:
        compute_swfcd_during_training = selection_metric in {"swfcd_loss_joint", "swfcd_joint", "swfcd", "swfcd_pearson"}
    if val_loader is None:
        compute_swfcd_during_training = False

    val_swfcd = SwFCD(val_loader.dataset, 30, 3) if (compute_swfcd_during_training and val_loader is not None) else None
    # val_swfcd = SwFCD(val_loader.dataset, 2, 1) if (compute_swfcd_during_training and val_loader is not None) else None
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
        # Non-gradient models (e.g. PCAAE) may prepare their closed-form fit
        # from the whole training set before their first and only epoch.
        if epoch == 0 and hasattr(model, "fit_train_loader"):
            model.fit_train_loader(train_loader, device=device)
        train_loss_params = {}
        model.train()
        for batch_idx, (data, labels) in enumerate(train_loader):
            x = data.to(device)
            valid_mask = _build_valid_mask(x, train_valid_last_dim)

            if noise is not None:
                if noise['type'] == 'gaussian':
                    n = torch.randn_like(x) + float(noise['std'])
                    x += n
                elif noise['type'] == 'mask':
                    n = (torch.rand_like(x) > float(noise['ratio'])).float()
                    x *= n

            if optimizer is not None:
                optimizer.zero_grad()

            output = model(x)
            output = _apply_recon_mask(x, output, valid_mask)

            if use_pred_heads:
                heads = {bl:h.to(device) for bl,h in labels[1].items()}
                loss = model.loss(x, heads, output)
            elif use_cls_head:
                loss = model.loss(x, _class_targets(labels), output)
            else:
                loss = model.loss(x, output)

            for p in loss:
                if p not in train_loss_params:
                    train_loss_params[p] = 0
                train_loss_params[p] += loss[p]

            if optimizer is not None:
                loss['loss'].backward()
                optimizer.step()

        num_batches = batch_idx + 1
        for p in train_loss_params:
            train_loss_params[p] = float(train_loss_params[p].detach())
            if p not in history['train']:
                history['train'][p] = []
            history['train'][p].append(train_loss_params[p] / num_batches)

        val_metric_str = ""
        current_metrics = {
            "train": {p: train_loss_params[p] / num_batches for p in train_loss_params},
            "val": {},
        }
        if val_loader is not None:
            model.eval()
            val_loss_params = {}
            val_recons = [] if val_reference_vec is not None else None
            swfcd_pearson_sum = 0.0
            swfcd_pearson_count = 0
            with torch.no_grad():
                for batch_idx, (data, labels) in enumerate(val_loader):
                    x = data.to(device)
                    valid_mask = _build_valid_mask(x, val_valid_last_dim)
                    output = model(x)
                    output = _apply_recon_mask(x, output, valid_mask)

                    if use_pred_heads:
                        heads = {bl:h.to(device) for bl,h in labels[1].items()}
                        loss = model.loss(x, heads, output)
                    elif use_cls_head:
                        loss = model.loss(x, _class_targets(labels), output)
                    else:
                        loss = model.loss(x, output)

                    for p in loss:
                        if p not in val_loss_params:
                            val_loss_params[p] = 0
                        val_loss_params[p] += loss[p]

                    recon_x, _ = _extract_model_outputs(output)
                    recon_x_detached = recon_x.detach()
                    if val_recons is not None:
                        val_recons.append(recon_x_detached)
                    elif compute_swfcd_during_training and not getattr(val_loader.dataset, "fc_input", False):
                        swfcd_results = val_swfcd.apply(x.detach(), recon_x_detached)
                        if swfcd_results is not None:
                            swfcd_pearson_sum += float(swfcd_results["pearson"].detach().cpu().item()) * data.shape[0]
                            swfcd_pearson_count += int(data.shape[0])

            num_val_batches = batch_idx + 1
            for p in val_loss_params:
                val_loss_params[p] = float(val_loss_params[p].detach())
                _append_history_metric(history, 'val', p, val_loss_params[p] / num_val_batches)

            swfcd_pearson = float("nan")
            if val_reference_vec is not None and val_recons:
                swfcd_results = val_swfcd.apply(None, torch.cat(val_recons, dim=0), x_vec=val_reference_vec)
                if swfcd_results is not None:
                    swfcd_pearson = float(swfcd_results["pearson"].detach().cpu().item())
            elif swfcd_pearson_count > 0:
                swfcd_pearson = swfcd_pearson_sum / swfcd_pearson_count
            _append_history_metric(history, 'val', 'swfcd_pearson', swfcd_pearson)
            val_metric_str += (
                f" | Val swfcd_pearson: {swfcd_pearson:.4f}"
                if np.isfinite(swfcd_pearson)
                else " | Val swfcd_pearson: nan"
            )
            current_metrics["val"] = {p: val_loss_params[p] / num_val_batches for p in val_loss_params}
            current_metrics["val"]["swfcd_pearson"] = history["val"]["swfcd_pearson"][-1]

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"{loss_params2str(train_loss_params, num_batches, val_loss_params, num_val_batches, model.loss_fn_params) if val_loader is not None else _train_only_loss_params_str(train_loss_params, num_batches, model.loss_fn_params)}"
            f"{val_metric_str}", flush=True
        )

        if val_loader is None:
            improved = False
        elif best_model_losses is None:
            improved = True
        else:
            tmp_history = {
                "val": {
                    "loss": [
                        best_model_losses["val"].get("loss", float("nan")),
                        current_metrics["val"].get("loss", float("nan")),
                    ],
                    "swfcd_pearson": [
                        best_model_losses["val"].get("swfcd_pearson", float("nan")),
                        current_metrics["val"].get("swfcd_pearson", float("nan")),
                    ],
                }
            }
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
                torch.save(model.state_dict(), f'{save_dir}/{name}_model.pt')
        elif val_loader is not None:
            epochs_without_improvement += 1

        if (
            val_loader is not None
            and
            convergence_patience is not None
            and convergence_patience > 0
            and (epoch + 1) > convergence_warmup_epochs
            and epochs_without_improvement >= convergence_patience
        ):
            print(
                "Converged: stopping early at "
                f"epoch {epoch + 1} after {epochs_without_improvement} "
                "epochs without validation-loss improvement."
            )
            break

    # run validation set through pca
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
        torch.save(model.state_dict(), f'{save_dir}/{name}_model.pt')

    print("Training complete!")
    return history, mse_pca
