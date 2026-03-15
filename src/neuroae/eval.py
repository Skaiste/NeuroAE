import numpy as np
import torch
import torch.nn.functional as F


from .utils.np_utils import to_numpy
from .metrics.fc_preservation import fc_preservation_score
from .metrics.silhouette import silhouette
from .metrics.logreg_accuracy import logreg_accuracy_cv
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
        recon_x = model_out.get("x_hat") or model_out.get("recon") or model_out.get("reconstruction")
        latent = model_out.get("z") or model_out.get("mu")
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



def _compute_pca_metrics(pca, swfcd, inputs, latents, labels, dataset, valid_mask=None):
    mse = np.nan
    fc = np.nan
    swfcd_results = {'pearson': np.nan, 'mad': np.nan, 'rmse': np.nan}
    if inputs.size > 0:
        recon = pca.inverse_transform(pca.transform(inputs))
        mse = _masked_mse_numpy(recon, inputs, valid_mask)
        fc = fc_preservation_score(inputs, recon, dataset)

        inputs_t = torch.as_tensor(inputs, dtype=torch.float32)
        recon_t = torch.as_tensor(recon, dtype=torch.float32)
        swfcd_results = swfcd.apply(inputs_t, recon_t)

    swfcd_pearson = _to_scalar_metric(swfcd_results['pearson']) if swfcd_results else np.nan
    swfcd_mad = _to_scalar_metric(swfcd_results['mad']) if swfcd_results else np.nan
    swfcd_rmse = _to_scalar_metric(swfcd_results['rmse']) if swfcd_results else np.nan

    sil = silhouette(latents, labels)
    logreg_acc = logreg_accuracy_cv(latents, labels)
    return {
        "mse": mse,
        "fc_preservation": fc,
        "silhouette": sil,
        "logreg_accuracy": logreg_acc,
        "swfcd_pearson": swfcd_pearson,
        "swfcd_mad": swfcd_mad,
        "swfcd_rmse": swfcd_rmse,
    }


def eval_vae(
    model,
    data_loader,
    pca=None,
    use_pred_heads=False,
    device='cuda' if torch.cuda.is_available() else 'cpu',
):
    """
    Run inference-time evaluation focused on reconstruction and latent-space metrics.

    Metrics:
    - MSE
    - FC preservation
    - Latent silhouette score
    - Latent logistic-regression accuracy (cross-validated)
    - PCA baseline comparison (if PCA object is provided)
    """
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    all_inputs = []
    all_recons = []
    all_latents = []
    all_masks = []

    swfcd = SwFCD(data_loader.dataset, 30, 3)

    with torch.no_grad():
        for data, _ in data_loader:
            x = data.to(device)
            valid_mask = _build_valid_mask(x, data_loader.dataset)
            model_out = model(x)
            model_out = _apply_recon_mask(x, model_out, valid_mask)

            recon_x, latent = _extract_model_outputs(model_out)

            all_inputs.append(x.detach().cpu())
            all_recons.append(recon_x.detach().cpu())
            all_latents.append(latent.detach().cpu())
            if valid_mask is not None:
                all_masks.append(valid_mask.detach().cpu())

    x_all = torch.cat(all_inputs, dim=0)
    x_hat_all = torch.cat(all_recons, dim=0)
    z_all = torch.cat(all_latents, dim=0)
    valid_mask_all = torch.cat(all_masks, dim=0) if all_masks else None

    mse = _masked_mse_torch(x_hat_all, x_all, valid_mask_all)
    fc_preservation = fc_preservation_score(x_all, x_hat_all, data_loader.dataset)

    swfcd_results = swfcd.apply(x_all, x_hat_all)
    swfcd_pearson = _to_scalar_metric(swfcd_results['pearson']) if swfcd_results else np.nan
    swfcd_mad = _to_scalar_metric(swfcd_results['mad']) if swfcd_results else np.nan
    swfcd_rmse = _to_scalar_metric(swfcd_results['rmse']) if swfcd_results else np.nan

    z_np = to_numpy(z_all)
    labels = np.asarray(getattr(data_loader.dataset, "labels", []))
    sil = silhouette(z_np, labels)
    logreg_acc = logreg_accuracy_cv(z_np, labels)

    metrics = {
        "model": {
            "mse": mse,
            "fc_preservation": fc_preservation,
            "silhouette": sil,
            "logreg_accuracy": logreg_acc,
            "swfcd_pearson": swfcd_pearson,
            "swfcd_mad": swfcd_mad,
            "swfcd_rmse": swfcd_rmse,
        }
    }

    print("Inference metrics (model):")
    print(f"  MSE: {mse:.6f}")
    print(f"  FC preservation: {fc_preservation:.6f}" if np.isfinite(fc_preservation) else "  FC preservation: nan")
    print(f"  Silhouette: {sil:.6f}" if np.isfinite(sil) else "  Silhouette: nan")
    print(f"  Logistic regression accuracy (CV): {logreg_acc:.6f}" if np.isfinite(logreg_acc) else "  Logistic regression accuracy (CV): nan")
    print(f"  SwFCD Pearson: {swfcd_pearson:.6f}" if np.isfinite(swfcd_pearson) else "  SwFCD Pearson: nan")
    print(f"  SwFCD Mean absolute difference: {swfcd_mad:.6f}" if np.isfinite(swfcd_mad) else "  SwFCD Mean absolute difference: nan")
    print(f"  SwFCD RMSE: {swfcd_rmse:.6f}" if np.isfinite(swfcd_rmse) else "  SwFCD RMSE: nan")

    if pca is not None:
        x_all = x_all.detach().cpu().numpy()
        valid_mask_np = to_numpy(valid_mask_all) if valid_mask_all is not None else None
        z_pca = pca.transform(x_all)

        pca_metrics = _compute_pca_metrics(
            pca=pca,
            swfcd=swfcd,
            inputs=x_all,
            latents=z_pca,
            labels=labels,
            dataset=data_loader.dataset,
            valid_mask=valid_mask_np,
        )

        metrics["pca"] = pca_metrics
        metrics["comparison"] = {
            "mse_delta_model_minus_pca": metrics["model"]["mse"] - pca_metrics["mse"],
            "fc_delta_model_minus_pca": metrics["model"]["fc_preservation"] - pca_metrics["fc_preservation"],
            "silhouette_delta_model_minus_pca": metrics["model"]["silhouette"] - pca_metrics["silhouette"],
            "logreg_delta_model_minus_pca": metrics["model"]["logreg_accuracy"] - pca_metrics["logreg_accuracy"],
            "swfcd_pearson_delta_model_minus_pca": metrics["model"]["swfcd_pearson"] - pca_metrics["swfcd_pearson"],
            "swfcd_mad_delta_model_minus_pca": metrics["model"]["swfcd_mad"] - pca_metrics["swfcd_mad"],
            "swfcd_rmse_delta_model_minus_pca": metrics["model"]["swfcd_rmse"] - pca_metrics["swfcd_rmse"],
        }

        # print("Inference metrics (PCA baseline):")
        # print(f"  MSE: {pca_metrics['mse']:.6f}" if np.isfinite(pca_metrics['mse']) else "  MSE: nan")
        # print(
        #     f"  FC preservation: {pca_metrics['fc_preservation']:.6f}"
        #     if np.isfinite(pca_metrics['fc_preservation'])
        #     else "  FC preservation: nan"
        # )
        # print(f"  Silhouette: {pca_metrics['silhouette']:.6f}" if np.isfinite(pca_metrics['silhouette']) else "  Silhouette: nan")
        # print(
        #     f"  Logistic regression accuracy (CV): {pca_metrics['logreg_accuracy']:.6f}"
        #     if np.isfinite(pca_metrics['logreg_accuracy'])
        #     else "  Logistic regression accuracy (CV): nan"
        # )
        # print(
        #     f"  SwFCD Pearson: {pca_metrics['swfcd_pearson']:.6f}"
        #     if np.isfinite(pca_metrics['swfcd_pearson'])
        #     else "  SwFCD Pearson: nan"
        # )
        # print(
        #     f"  SwFCD Mean absolute difference: {pca_metrics['swfcd_mad']:.6f}"
        #     if np.isfinite(pca_metrics['swfcd_mad'])
        #     else "  SwFCD Mean absolute difference: nan"
        # )
        # print(
        #     f"  SwFCD RMSE: {pca_metrics['swfcd_rmse']:.6f}"
        #     if np.isfinite(pca_metrics['swfcd_rmse'])
        #     else "  SwFCD RMSE: nan"
        # )

        # print("Model vs PCA deltas (model - PCA):")
        # print(f"  MSE delta: {metrics['comparison']['mse_delta_model_minus_pca']:.6f}")
        # print(
        #     f"  FC preservation delta: {metrics['comparison']['fc_delta_model_minus_pca']:.6f}"
        #     if np.isfinite(metrics['comparison']['fc_delta_model_minus_pca'])
        #     else "  FC preservation delta: nan"
        # )
        # print(
        #     f"  Silhouette delta: {metrics['comparison']['silhouette_delta_model_minus_pca']:.6f}"
        #     if np.isfinite(metrics['comparison']['silhouette_delta_model_minus_pca'])
        #     else "  Silhouette delta: nan"
        # )
        # print(
        #     f"  Logistic regression accuracy delta: {metrics['comparison']['logreg_delta_model_minus_pca']:.6f}"
        #     if np.isfinite(metrics['comparison']['logreg_delta_model_minus_pca'])
        #     else "  Logistic regression accuracy delta: nan"
        # )
        # print(
        #     f"  SwFCD Pearson delta: {metrics['comparison']['swfcd_pearson_delta_model_minus_pca']:.6f}"
        #     if np.isfinite(metrics['comparison']['swfcd_pearson_delta_model_minus_pca'])
        #     else "  SwFCD Pearson delta: nan"
        # )
        # print(
        #     f"  SwFCD Mean absolute difference delta: {metrics['comparison']['swfcd_mad_delta_model_minus_pca']:.6f}"
        #     if np.isfinite(metrics['comparison']['swfcd_mad_delta_model_minus_pca'])
        #     else "  SwFCD Mean absolute difference delta: nan"
        # )
        # print(
        #     f"  SwFCD RMSE delta: {metrics['comparison']['swfcd_rmse_delta_model_minus_pca']:.6f}"
        #     if np.isfinite(metrics['comparison']['swfcd_rmse_delta_model_minus_pca'])
        #     else "  SwFCD RMSE delta: nan"
        # )

    return metrics
