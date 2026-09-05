import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


from neurocls.eval import compute_classification_metrics

from .metrics.classifier_accuracy import run_latent_svm_classifier
from .utils.np_utils import to_numpy
from .metrics.fc_preservation import fc_preservation_score
from .metrics.silhouette import silhouette
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


def _build_valid_mask(x, dataset):
    valid_last_dim = _dataset_valid_last_dim(dataset)
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


def _masked_mse_torch(x_hat, x, mask):
    if mask is None:
        return float(F.mse_loss(x_hat, x, reduction="mean").item())
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
    return float((se.sum() / denom).item())


def _masked_mse_numpy(x_hat, x, mask):
    if mask is None:
        return float(np.mean((x_hat - x) ** 2))
    if x_hat.shape != x.shape:
        if x_hat.shape[:-1] != x.shape[:-1]:
            raise ValueError(
                f"Cannot align x_hat shape {x_hat.shape} with x shape {x.shape}."
            )
        common_last_dim = min(x_hat.shape[-1], x.shape[-1], mask.shape[-1])
        x_hat = x_hat[..., :common_last_dim]
        x = x[..., :common_last_dim]
        mask = mask[..., :common_last_dim]
    se = ((x_hat - x) ** 2) * mask
    denom = np.maximum(np.sum(mask), 1.0)
    return float(np.sum(se) / denom)


def _reconstruction_pearson(x_hat, x, mask=None):
    """Return the Pearson correlation across all valid reconstruction values."""
    x_hat_values = to_numpy(x_hat)
    x_values = to_numpy(x)
    if mask is not None:
        valid = to_numpy(mask).astype(bool)
        x_hat_values = x_hat_values[valid]
        x_values = x_values[valid]
    else:
        x_hat_values = x_hat_values.ravel()
        x_values = x_values.ravel()

    if x_hat_values.size < 2 or x_values.size < 2:
        return float("nan")
    if np.std(x_hat_values) == 0.0 or np.std(x_values) == 0.0:
        return float("nan")
    return float(np.corrcoef(x_hat_values, x_values)[0, 1])


def _to_scalar_metric(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    if isinstance(value, (list, tuple)):
        return float(np.mean(value)) if len(value) > 0 else float("nan")
    if isinstance(value, np.ndarray):
        return float(np.mean(value)) if value.size > 0 else float("nan")
    return float(value)


def _extract_model_outputs(model_out):
    """Return reconstruction and latent matrix from model outputs."""
    if isinstance(model_out, dict):
        recon_x = None
        latent = None
        for key in ("x_hat", "recon", "reconstruction"):
            if key in model_out:
                recon_x = model_out[key]
                break
        for key in ("z", "mu"):
            if key in model_out:
                latent = model_out[key]
                break
    elif isinstance(model_out, (tuple, list)):
        recon_x = model_out[0]
        latent = model_out[-1]
    else:
        recon_x = model_out
        latent = None

    if recon_x is None:
        raise ValueError("Could not extract reconstruction tensor from model output.")

    if latent is None:
        raise ValueError("Could not extract latent tensor from model output.")

    return recon_x, latent


def _batch_labels_to_list(batch_labels):
    if isinstance(batch_labels, torch.Tensor):
        return batch_labels.detach().cpu().tolist()
    if isinstance(batch_labels, np.ndarray):
        return batch_labels.tolist()
    if isinstance(batch_labels, (list, tuple)):
        return list(batch_labels)
    return [batch_labels]


def _collect_split_outputs(model, data_loader, device, use_pred_heads=False, include_recons=True):
    inputs = []
    recons = [] if include_recons else None
    latents = []
    labels = []
    masks = []

    model.eval()
    with torch.no_grad():
        for data, batch_labels in data_loader:
            x = data.to(device)
            valid_mask = _build_valid_mask(x, data_loader.dataset)
            model_out = model(x)
            model_out = _apply_recon_mask(x, model_out, valid_mask)
            recon_x, latent = _extract_model_outputs(model_out)

            inputs.append(x.detach().cpu())
            if recons is not None:
                recons.append(recon_x.detach().cpu())
            latents.append(latent.detach().cpu())
            if valid_mask is not None:
                masks.append(valid_mask.detach().cpu())

            raw_labels = batch_labels[0] if use_pred_heads else batch_labels
            labels.extend(_batch_labels_to_list(raw_labels))

    return {
        "inputs": torch.cat(inputs, dim=0) if inputs else None,
        "recons": torch.cat(recons, dim=0) if recons else None,
        "latents": torch.cat(latents, dim=0) if latents else None,
        "labels": np.asarray(labels, dtype=object),
        "valid_mask": torch.cat(masks, dim=0) if masks else None,
    }


def _prepare_classifier_latents(model, latents, split_name="unknown"):
    if latents is None:
        return None
    if not torch.is_tensor(latents):
        return latents
    if latents.ndim == 3:
        print(
            f"Evaluation: classifier latents for {split_name} already 3D with shape={tuple(latents.shape)}",
            flush=True,
        )
        return latents
    if latents.ndim != 2:
        raise ValueError(
            f"Classifier latents for {split_name} must be 3D latent timeseries or reshapeable 2D latents; "
            f"got shape={tuple(latents.shape)}"
        )

    timepoint_dim = getattr(model, "timepoint_dim", None)
    latent_per_timepoint = getattr(model, "latent_per_timepoint", None)
    latent_flat_dim = getattr(model, "latent_flat_dim", None)
    if (
        timepoint_dim is None
        or latent_per_timepoint is None
        or latent_flat_dim is None
        or latents.shape[1] != int(latent_flat_dim)
    ):
        raise ValueError(
            f"Classifier latents for {split_name} must be 3D. Received flattened shape={tuple(latents.shape)} "
            "but the model does not expose compatible timepoint/latent metadata for reshaping."
        )

    reshaped = latents.reshape(latents.shape[0], int(timepoint_dim), int(latent_per_timepoint))
    print(
        f"Evaluation: reshaped classifier latents for {split_name} "
        f"from shape={tuple(latents.shape)} to shape={tuple(reshaped.shape)}",
        flush=True,
    )
    return reshaped



def _classifier_metric_bundle(metrics):
    if not isinstance(metrics, dict):
        return {
            "classifier_accuracy": np.nan,
            "classifier_balanced_accuracy": np.nan,
            "classifier_macro_f1": np.nan,
            "classifier_per_class_f1": {},
        }
    per_class = metrics.get("per_class", {})
    per_class_f1 = {}
    if isinstance(per_class, dict):
        for class_label, class_metrics in per_class.items():
            if isinstance(class_metrics, dict):
                per_class_f1[str(class_label)] = float(class_metrics.get("f1", np.nan))
    return {
        "classifier_accuracy": float(metrics.get("accuracy", np.nan)),
        "classifier_balanced_accuracy": float(metrics.get("balanced_accuracy", np.nan)),
        "classifier_macro_f1": float(metrics.get("macro_f1", np.nan)),
        "classifier_per_class_f1": per_class_f1,
    }


def _compute_swfcd_metrics_in_batches(swfcd, inputs, recons, device, batch_size):
    """Compute dataset-level SwFCD metrics without materializing every subject on one device."""
    if inputs is None or recons is None or inputs.shape[0] == 0:
        return {"pearson": np.nan, "mad": np.nan, "rmse": np.nan}

    device = torch.device(device)
    batch_size = max(1, int(batch_size or inputs.shape[0]))
    samples_per_subject = 1
    if getattr(swfcd.dataset, "timepoints_as_samples", False):
        samples_per_subject = int(swfcd.dataset.original_shape[0])
        batch_size = max(samples_per_subject, batch_size // samples_per_subject * samples_per_subject)

    metric_sums = {"pearson": 0.0, "mad": 0.0, "rmse": 0.0}
    subject_count = 0
    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            end = min(start + batch_size, inputs.shape[0])
            # A timepoint-as-sample dataset must retain complete subjects in each chunk.
            if samples_per_subject > 1 and (end - start) % samples_per_subject:
                end -= (end - start) % samples_per_subject
                if end == start:
                    end = min(start + samples_per_subject, inputs.shape[0])

            input_batch = inputs[start:end].to(device)
            recon_batch = recons[start:end].to(device)
            results = swfcd.apply(input_batch, recon_batch)
            batch_subject_count = (end - start) // samples_per_subject
            if results is not None and batch_subject_count > 0:
                for name in metric_sums:
                    metric_sums[name] += _to_scalar_metric(results[name]) * batch_subject_count
                subject_count += batch_subject_count

            del input_batch, recon_batch, results

    if subject_count == 0:
        return {"pearson": np.nan, "mad": np.nan, "rmse": np.nan}
    return {name: value / subject_count for name, value in metric_sums.items()}


def _compute_model_metrics(
    sw_fcd,
    inputs,
    recons,
    latents,
    labels,
    dataset,
    classifier_metrics=None,
    valid_mask=None,
    device="cpu",
    swfcd_batch_size=None,
):
    mse = _masked_mse_torch(recons, inputs, valid_mask)
    recon_pearson = _reconstruction_pearson(recons, inputs, valid_mask)
    fc_preservation = fc_preservation_score(inputs, recons, dataset)

    swfcd_results = _compute_swfcd_metrics_in_batches(
        sw_fcd,
        inputs,
        recons,
        device=device,
        batch_size=swfcd_batch_size,
    )
    swfcd_pearson = _to_scalar_metric(swfcd_results["pearson"]) if swfcd_results else np.nan
    swfcd_mad = _to_scalar_metric(swfcd_results["mad"]) if swfcd_results else np.nan
    swfcd_rmse = _to_scalar_metric(swfcd_results["rmse"]) if swfcd_results else np.nan

    z_np = to_numpy(latents)
    label_array = np.asarray(labels)
    sil = silhouette(z_np, label_array)
    metrics = {
        "rmse": float(np.sqrt(mse)),
        "recon_pearson": recon_pearson,
        "fc_preservation": fc_preservation,
        "silhouette": sil,
        "swfcd_pearson": swfcd_pearson,
        "swfcd_mad": swfcd_mad,
        "swfcd_rmse": swfcd_rmse,
    }
    metrics.update(_classifier_metric_bundle(classifier_metrics))
    return metrics


def _slice_classifier_metrics(classifier_result, group_indices):
    if not isinstance(classifier_result, dict):
        return None
    if classifier_result.get("test_predictions") is None:
        return None
    encoded_true = classifier_result["label_payload"]["test"][group_indices]
    encoded_pred = classifier_result["test_predictions"][group_indices]
    encoded_proba = classifier_result["test_probabilities"][group_indices]
    return compute_classification_metrics(
        encoded_true,
        encoded_pred,
        classifier_result["label_payload"]["classes"],
        y_proba=encoded_proba,
    )


def _subset_tensor(values, indices):
    if values is None:
        return None
    return values[indices]


def _sorted_unique_labels(labels):
    label_array = np.asarray(labels, dtype=object)
    unique_values = pd.unique(label_array)
    cleaned = [value for value in unique_values if value is not None and not pd.isna(value)]
    return sorted((str(value) for value in cleaned), key=str)


def _print_metric_summary(title, metrics):
    print(title)
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(
        f"  Reconstruction Pearson: {metrics['recon_pearson']:.6f}"
        if np.isfinite(metrics["recon_pearson"])
        else "  Reconstruction Pearson: nan"
    )
    print(
        f"  FC preservation: {metrics['fc_preservation']:.6f}"
        if np.isfinite(metrics["fc_preservation"])
        else "  FC preservation: nan"
    )
    print(f"  Silhouette: {metrics['silhouette']:.6f}" if np.isfinite(metrics["silhouette"]) else "  Silhouette: nan")
    print(
        f"  Classifier accuracy: {metrics['classifier_accuracy']:.6f}"
        if np.isfinite(metrics["classifier_accuracy"])
        else "  Classifier accuracy: nan"
    )
    print(
        f"  Classifier balanced accuracy: {metrics['classifier_balanced_accuracy']:.6f}"
        if np.isfinite(metrics["classifier_balanced_accuracy"])
        else "  Classifier balanced accuracy: nan"
    )
    print(
        f"  Classifier macro F1: {metrics['classifier_macro_f1']:.6f}"
        if np.isfinite(metrics["classifier_macro_f1"])
        else "  Classifier macro F1: nan"
    )
    per_class_f1 = metrics.get("classifier_per_class_f1", {})
    if isinstance(per_class_f1, dict) and per_class_f1:
        per_class_parts = []
        for class_label, f1_value in per_class_f1.items():
            per_class_parts.append(
                f"{class_label}={f1_value:.6f}" if np.isfinite(f1_value) else f"{class_label}=nan"
            )
        print("  Classifier per-class F1: " + ", ".join(per_class_parts))
    print(
        f"  SwFCD Pearson: {metrics['swfcd_pearson']:.6f}"
        if np.isfinite(metrics["swfcd_pearson"])
        else "  SwFCD Pearson: nan"
    )
    print(
        f"  SwFCD Mean absolute difference: {metrics['swfcd_mad']:.6f}"
        if np.isfinite(metrics["swfcd_mad"])
        else "  SwFCD Mean absolute difference: nan"
    )
    print(
        f"  SwFCD RMSE: {metrics['swfcd_rmse']:.6f}"
        if np.isfinite(metrics["swfcd_rmse"])
        else "  SwFCD RMSE: nan"
    )


def eval_vae(
    model,
    train_loader,
    val_loader,
    data_loader,
    use_pred_heads=False,
    evaluation_scope="combined",
    classifier_train_loader=None,
    classifier_val_loader=None,
    group_transfer_target_group=None,
    device='cuda' if torch.cuda.is_available() else 'cpu',
):
    """
    Run inference-time evaluation focused on reconstruction and latent-space metrics.

    Metrics:
    - RMSE
    - FC preservation
    - Latent silhouette score
    - Latent classifier accuracy via BrainGNN trained on AE train latents and evaluated on test
    """
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    classifier_train_loader = classifier_train_loader or train_loader

    print("Evaluation: collecting train split latents", flush=True)
    swfcd = SwFCD(data_loader.dataset, 30, 3)
    # swfcd = SwFCD(data_loader.dataset, 2, 1)
    train_outputs = _collect_split_outputs(model, train_loader, device, use_pred_heads=use_pred_heads, include_recons=False)
    print("Evaluation: collecting evaluation split reconstructions and latents", flush=True)
    eval_outputs = _collect_split_outputs(model, data_loader, device, use_pred_heads=use_pred_heads, include_recons=True)
    use_classifier_overrides = classifier_train_loader is not train_loader
    if use_classifier_overrides:
        print("Evaluation: collecting classifier train split latents from override loader", flush=True)
        classifier_train_outputs = _collect_split_outputs(
            model,
            classifier_train_loader,
            device,
            use_pred_heads=use_pred_heads,
            include_recons=False,
        )
    else:
        classifier_train_outputs = train_outputs

    x_all = eval_outputs["inputs"]
    x_hat_all = eval_outputs["recons"]
    z_all = eval_outputs["latents"]
    valid_mask_all = eval_outputs["valid_mask"]
    labels = eval_outputs["labels"]
    scope = str(evaluation_scope or "combined")
    if scope not in {"combined", "per_group"}:
        raise ValueError(f"Unsupported evaluation_scope: {evaluation_scope}")

    train_classifier_latents = _prepare_classifier_latents(
        model,
        classifier_train_outputs["latents"],
        split_name="classifier_train" if use_classifier_overrides else "train",
    )
    eval_classifier_latents = _prepare_classifier_latents(model, eval_outputs["latents"], split_name="test")

    print("Evaluation: training latent classifier for model latents", flush=True)
    classifier_result = run_latent_svm_classifier(
        to_numpy(train_classifier_latents),
        classifier_train_outputs["labels"].tolist(),
        to_numpy(eval_classifier_latents),
        labels.tolist(),
        device=device,
    )
    print("Evaluation: latent classifier finished for model latents", flush=True)
    classifier_metrics = classifier_result.get("test_metrics")

    print("Evaluation: computing model metrics", flush=True)
    model_metrics = _compute_model_metrics(
        sw_fcd=swfcd,
        inputs=x_all,
        recons=x_hat_all,
        latents=z_all,
        labels=labels,
        dataset=data_loader.dataset,
        classifier_metrics=classifier_metrics,
        valid_mask=valid_mask_all,
        device=device,
        swfcd_batch_size=getattr(data_loader, "batch_size", None),
    )
    metrics = {
        "scope": scope,
        "model": model_metrics,
    }

    _print_metric_summary("Inference metrics (model):", model_metrics)

    if scope == "per_group":
        groups = {}
        for group_name in _sorted_unique_labels(labels):
            group_indices = np.flatnonzero(labels.astype(str) == group_name)
            if group_indices.size == 0:
                continue
            group_idx_tensor = torch.as_tensor(group_indices, dtype=torch.long)
            group_inputs = _subset_tensor(x_all if torch.is_tensor(x_all) else torch.as_tensor(x_all), group_idx_tensor)
            group_recons = _subset_tensor(x_hat_all, group_idx_tensor)
            group_latents = _subset_tensor(z_all, group_idx_tensor)
            group_valid_mask = _subset_tensor(valid_mask_all, group_idx_tensor) if valid_mask_all is not None else None
            group_labels = labels[group_indices]
            group_classifier_metrics = None
            skip_group_classification = (
                group_transfer_target_group is not None
                and str(group_name) == str(group_transfer_target_group)
            )
            if not skip_group_classification:
                group_classifier_metrics = _slice_classifier_metrics(classifier_result, group_indices)

            group_model_metrics = _compute_model_metrics(
                sw_fcd=swfcd,
                inputs=group_inputs,
                recons=group_recons,
                latents=group_latents,
                labels=group_labels,
                dataset=data_loader.dataset,
                classifier_metrics=group_classifier_metrics,
                valid_mask=group_valid_mask,
            )
            group_metrics = {"model": group_model_metrics}

            groups[group_name] = group_metrics
            _print_metric_summary(f"Inference metrics (model) [{group_name}]:", group_model_metrics)

        if groups:
            metrics["groups"] = groups

    return metrics
