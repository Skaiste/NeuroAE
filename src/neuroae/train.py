import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def loss_function(x, x_hat, mu, log_var, error_per_feature=True):
    # if selected error per feature, we are averaging everything
    if error_per_feature:
        # recon: mean mse loss
        recon = F.mse_loss(x_hat, x, reduction="mean")

        # KL: mean over batch, then mean over latent dims
        kld = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
        kld = kld.sum(dim=1).mean() / log_var.size(1)

    # if selected error per sample, we are summing everything
    else:
        # recon: sum over features per sample, then mean over batch
        recon = F.mse_loss(x_hat, x, reduction="none")  # [B, D]
        recon = recon.sum(dim=1).mean()

        # kld: sum over latent dims per sample, then mean over batch
        kld = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
        kld = kld.sum(dim=1).mean() / log_var.size(1)

    return recon + kld, recon, kld

import math

def denormalise_losses(recon, kld, min_val, max_val, input_dim, latent_dim, error_per_feature=True):

    R = max_val - min_val  # global range

    # Convert recon to per-feature MSE (normalized space)
    if not error_per_feature:
        mse_norm_per_feature = recon / input_dim
    else:
        mse_norm_per_feature = recon

    rmse_norm_per_feature = math.sqrt(mse_norm_per_feature)

    # Convert to original units
    mse_orig_per_feature = mse_norm_per_feature * (R ** 2)
    rmse_orig_per_feature = rmse_norm_per_feature * R

    # KL is unitless — just make it interpretable
    kld_per_dim = kld / latent_dim

    return {
        "mse_norm_per_feature": mse_norm_per_feature,
        "rmse_norm_per_feature": rmse_norm_per_feature,
        "mse_orig_per_feature": mse_orig_per_feature,
        "rmse_orig_per_feature": rmse_orig_per_feature,
        "kld_per_dim": kld_per_dim,
    }


def train_vae_basic(
    model,
    train_loader,
    val_loader,
    num_epochs=100,
    learning_rate=1e-4,
    loss_per_feature=True,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    save_dir='./checkpoints',
):
    history = {
        'train_loss': [],
        'train_reproduction_loss': [],
        'train_KLD': [],
        'val_loss': [],
        'val_reproduction_loss': [],
        'val_KLD': [],
    }
    best_model_losses = {
        'train_loss': float('inf'),
        'train_reproduction_loss': float('inf'),
        'train_KLD': float('inf'),
        'val_loss': float('inf'),
        'val_reproduction_loss': float('inf'),
        'val_KLD': float('inf'),
    }
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_reproduction_loss = 0.0
        train_KLD = 0.0

        model.train()
        for batch_idx, (data, _) in enumerate(train_loader):
            x = data.to(device)

            optimizer.zero_grad()
            
            recon_x, mu, logvar, z = model(x)
            loss, recon, kld = loss_function(x, recon_x, mu, logvar, loss_per_feature)
            train_loss += loss.item()
            train_reproduction_loss += recon.item()
            train_KLD += kld.item()

            loss.backward()
            optimizer.step()

        num_batches = batch_idx + 1
        history['train_loss'].append(train_loss / num_batches)
        history['train_reproduction_loss'].append(train_reproduction_loss / num_batches)
        history['train_KLD'].append(train_KLD / num_batches)

        # validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_reproduction_loss = 0.0
            val_KLD = 0.0
            for batch_idx, (data, _) in enumerate(val_loader):
                x = data.to(device)
                recon_x, mu, logvar, z = model(x)
                loss, recon, kld = loss_function(x, recon_x, mu, logvar, loss_per_feature)
                val_loss += loss.item()
                val_reproduction_loss += recon.item()
                val_KLD += kld.item()

        num_val_batches = batch_idx + 1
        history['val_loss'].append(val_loss / num_val_batches)
        history['val_reproduction_loss'].append(val_reproduction_loss / num_val_batches)
        history['val_KLD'].append(val_KLD / num_val_batches)

        print(f"Epoch {epoch}/{num_epochs} | Train Loss: {train_loss / num_batches:.4f} | Train Recon: {train_reproduction_loss / num_batches:.4f} | Train KLD: {train_KLD / num_batches:.4f} | Val Loss: {val_loss / num_val_batches:.4f} | Val Recon: {val_reproduction_loss / num_val_batches:.4f} | Val KLD: {val_KLD / num_val_batches:.4f}")

        # select best model based on validation loss
        avg_val_loss = val_loss / num_val_batches
        if avg_val_loss < best_model_losses['val_loss']:
            best_model_losses['val_loss'] = avg_val_loss
            best_model_losses['val_reproduction_loss'] = val_reproduction_loss / num_val_batches
            best_model_losses['val_KLD'] = val_KLD / num_val_batches
            best_model_losses['train_loss'] = train_loss / num_batches
            best_model_losses['train_reproduction_loss'] = train_reproduction_loss / num_batches
            best_model_losses['train_KLD'] = train_KLD / num_batches
            torch.save(model.state_dict(), f'{save_dir}/best_model.pt')
            
    print("Training complete!")

    # print("Calculated denormalised losses for best model:")
    # train_unnorm_losses = denormalise_losses(train_reproduction_loss, train_KLD, train_loader.dataset.data_min, train_loader.dataset.data_max, 78800, 64, loss_per_feature)
    # val_unnorm_losses = denormalise_losses(val_reproduction_loss, val_KLD, train_loader.dataset.data_min, train_loader.dataset.data_max, 78800, 64, loss_per_feature)
    # print(f"Train:")
    # print(f"\tRMSE per feature: {train_unnorm_losses['rmse_orig_per_feature']}")
    # print(f"\tKLD per dim: {train_unnorm_losses['kld_per_dim']}")
    # print(f"Validation:")
    # print(f"\tRMSE per feature: {val_unnorm_losses['rmse_orig_per_feature']}")
    # print(f"\tKLD per dim: {val_unnorm_losses['kld_per_dim']}")


    return history


def plot_training_history(
    history,
    figsize=(12, 4),
    save_path=None,
    show=True,
):
    """Plot training history from train_vae_basic.

    Args:
        history: Dict with keys train_loss, train_reproduction_loss, train_KLD,
            val_loss, val_reproduction_loss, val_KLD (each a list of per-epoch values).
        figsize: Figure size (width, height).
        save_path: Optional path to save the figure.
        show: Whether to display the plot (default True).
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    metrics = [
        ('loss', 'Total Loss'),
        ('reproduction_loss', 'Reconstruction Loss'),
        ('KLD', 'KL Divergence'),
    ]
    epochs = range(1, len(history['train_loss']) + 1)

    for ax, (suffix, title) in zip(axes, metrics):
        train_key = f'train_{suffix}'
        val_key = f'val_{suffix}'
        ax.plot(epochs, history[train_key], label='Train', color='tab:blue')
        ax.plot(epochs, history[val_key], label='Validation', color='tab:orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig
