import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .train import loss_function

def denormalise(a_norm, normaliser):
    if len(a_norm.shape) == 1:
        a_norm = a_norm.reshape(1, -1)
    elif len(a_norm.shape) == 2 and a_norm.shape[-1] != normaliser.mean_.shape[-1]:
        a_norm = a_norm.reshape(1, a_norm.shape[0]*a_norm.shape[1])
    try:
        return normaliser.inverse_transform(a_norm)
    except Exception as e:
        breakpoint()
        raise e

def _is_tensor(data):
    # Helper function to check (PyTorch/NumPy) and convert to numpy
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        return data
    else:
        return np.array(data)

def _guess_and_reshape(array, loader):
    # Try to infer from loader the original shape (if possible)
    # loader may be a DataLoader or ADNIDataset - extract sample shape heuristically
    if hasattr(loader, 'dataset') and hasattr(loader.dataset, 'data'):
        orig_shape = None
        # See if loader.dataset.data is a list/ndarray of right shape
        sample = loader.dataset.data[0]
        if hasattr(loader.dataset, 'flatten') and not loader.dataset.flatten:
            # Already 2D
            orig_shape = sample.shape
        else:
            orig_shape = loader.dataset.original_shape
        if orig_shape is not None:
            return array.reshape(orig_shape)
    # Otherwise, leave as 1D
    return array


def get_rmse(x, x_hat):
    error = x - x_hat
    return np.sqrt(np.mean(error ** 2))


def mse_denormalised(x_norm, x_hat_norm, mean_val, std_val, per_feature=True):
    """
    Compute MSE on denormalised data.

    Args:
        x_norm: Original data (normalised), shape (n_samples, n_features)
        x_hat_norm: Reconstructed data (normalised)
        min_val: Minimum value used for normalisation
        max_val: Maximum value used for normalisation
        per_feature: If True, return MSE per feature (mean over samples), shape (n_features,).
                     If False, return scalar MSE over all.

    Returns:
        MSE in original (denormalised) units — array of shape (n_features,) or scalar
    """
    x = _is_tensor(x_norm)
    x_hat = _is_tensor(x_hat_norm)
    x_denorm = denormalise(x, mean_val, std_val)
    x_hat_denorm = denormalise(x_hat, mean_val, std_val)
    sq_err = (x_denorm - x_hat_denorm) ** 2
    if per_feature:
        return np.mean(sq_err, axis=0)  # shape (n_features,)
    return np.mean(sq_err)


def visualise_examples(examples, data_loader, show=True):
    """
    Visualise each example as a heatmap: original x (row) and reconstructed x (row) for n examples.
    """
    n_examples = len(examples)
    fig, axs = plt.subplots(n_examples, 3, figsize=(10, 3*n_examples))
    if n_examples == 1:
        axs = np.expand_dims(axs, 0)  # so we can index axs[i, ...]
    for idx, (x, x_hat) in enumerate(examples):
        # Convert tensor->numpy (detach, cpu, etc)
        x_np = _is_tensor(x.squeeze())
        x_hat_np = _is_tensor(x_hat.squeeze())
        # Try to reshape to (N_ROIs, T_timepoints) for heatmap
        x_2d = _guess_and_reshape(x_np, data_loader)
        x_hat_2d = _guess_and_reshape(x_hat_np, data_loader)
        # Plot original
        ax0 = axs[idx, 0]
        im0 = ax0.imshow(x_2d, aspect='auto', cmap='viridis')
        ax0.set_title(f'Example {idx+1} - Original')
        plt.colorbar(im0, ax=ax0, fraction=0.05)
        # Plot reconstruction
        ax1 = axs[idx, 1]
        im1 = ax1.imshow(x_hat_2d, aspect='auto', cmap='viridis')
        ax1.set_title(f'Example {idx+1} - Reconstructed')
        plt.colorbar(im1, ax=ax1, fraction=0.05)

        # Plot difference
        ax2 = axs[idx, 2]
        err = x_2d - x_hat_2d
        im2 = ax2.imshow(err, aspect='auto', cmap='viridis')
        ax2.set_title(f'Example {idx+1} - Error')
        plt.colorbar(im2, ax=ax2, fraction=0.05)

        for ax in (ax0, ax1, ax2):
            ax.set_xlabel('Time')
            ax.set_ylabel('ROI')
    plt.tight_layout()
    if show:
        plt.show()

def export_examples(examples, data_loader, sample_dir, denormalise=False):
    """
    Export examples to CSV files.
    """
    for idx, (x, x_hat) in enumerate(examples):
        x_np = _is_tensor(x.squeeze())
        x_hat_np = _is_tensor(x_hat.squeeze())

        if denormalise:
            x_2d = denormalise(x_np, data_loader.dataset.normaliser)
            x_hat_2d = denormalise(x_hat_np, data_loader.dataset.normaliser)
        else:
            x_2d = x_np
            x_hat_2d = x_hat_np

        # x_2d = denormalise(x_np, data_loader.dataset.normaliser)
        # x_hat_2d = denormalise(x_hat_np, data_loader.dataset.normaliser)

        x_2d = _guess_and_reshape(x_2d, data_loader)
        x_hat_2d = _guess_and_reshape(x_hat_2d, data_loader)
        # denormalise

        # get rmse error
        error = x_2d - x_hat_2d

        x_df = pd.DataFrame(x_2d.T)
        x_hat_df = pd.DataFrame(x_hat_2d.T)
        error_df = pd.DataFrame(error.T)
        x_df.to_csv(os.path.join(sample_dir, f'example_{idx}_original.csv'), index=False, header=False)
        x_hat_df.to_csv(os.path.join(sample_dir, f'example_{idx}_reconstructed.csv'), index=False, header=False)
        error_df.to_csv(os.path.join(sample_dir, f'example_{idx}_error.csv'), index=False, header=False)

def visualise_error(errors, data_loader, show=True):
    bias = np.mean(errors, axis=0)
    mse = np.mean(errors ** 2, axis=0)
    rmse = np.sqrt(mse)
    
    # breakpoint()
    bias_map = _guess_and_reshape(bias, data_loader)
    mse_map = _guess_and_reshape(mse, data_loader)
    rmse_map = _guess_and_reshape(rmse, data_loader)

    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    im0 = axs[0].imshow(bias_map, aspect='auto', cmap='viridis')
    axs[0].set_title('Bias')
    plt.colorbar(im0, ax=axs[0], fraction=0.05)
    im1 = axs[1].imshow(mse_map, aspect='auto', cmap='viridis')
    axs[1].set_title('MSE')
    plt.colorbar(im1, ax=axs[1], fraction=0.05)
    im2 = axs[2].imshow(rmse_map, aspect='auto', cmap='viridis')
    axs[2].set_title('RMSE')
    plt.colorbar(im2, ax=axs[2], fraction=0.05)
    for ax in axs:
        ax.set_xlabel('Time')
        ax.set_ylabel('ROI')

    plt.tight_layout()
    if show:
        plt.show()


def inference_vae_basic(
    model,
    data_loader,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    loss_per_feature=True,
    num_examples=3,
    plot_dir=None,
    sample_dir=None
):
    model.eval()
    all_losses = []
    all_recons = []
    all_klds = []
    errors = None
    top_n_examples = []

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(data_loader):
            x = data.to(device)
            recon_x, mu, logvar, z = model(x)
            loss, recon, kld = loss_function(x, recon_x, mu, logvar, loss_per_feature)
            if len(top_n_examples) < num_examples:
                top_n_examples = [(x[i], recon_x[i]) for i in range(num_examples)]
            all_losses.append(loss.item())
            all_recons.append(recon.item())
            all_klds.append(kld.item())

            if data_loader.dataset.normaliser is not None:
                x_denorm = denormalise(x.cpu(), data_loader.dataset.normaliser)
                x_hat_denorm = denormalise(recon_x.cpu(), data_loader.dataset.normaliser)
                if errors is None:
                    errors = x_denorm - x_hat_denorm
                else:
                    errors = np.concatenate([errors, (x_denorm - x_hat_denorm)])

    avg_loss = sum(all_losses) / len(all_losses)
    avg_recon = sum(all_recons) / len(all_recons)
    avg_kld = sum(all_klds) / len(all_klds)

    print(f"Average loss: {avg_loss}")
    print(f"Average reconstruction: {avg_recon}")
    print(f"Average KLD: {avg_kld}")

    # breakpoint()

    top_n_denormed = [
        (denormalise(x.cpu(), data_loader.dataset.normaliser), denormalise(recon_x.cpu(), data_loader.dataset.normaliser))
        for x, recon_x in top_n_examples
    ] if data_loader.dataset.normaliser is not None else top_n_examples

    visualise_error(errors, data_loader, show=plot_dir is None)
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, 'basicVAE_error.png'), dpi=150, bbox_inches='tight')
        plt.close()

    visualise_examples(top_n_denormed, data_loader, show=plot_dir is None)
    if plot_dir:
        plt.savefig(os.path.join(plot_dir, 'basicVAE_examples.png'), dpi=150, bbox_inches='tight')
        plt.close()

    if sample_dir:
        export_examples(top_n_denormed, data_loader, sample_dir)
    
    return avg_loss, avg_recon, avg_kld