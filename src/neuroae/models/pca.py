"""A PCA-backed autoencoder compatible with the NeuroAE training interface."""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA as SklearnPCA

from . import ModelBase


class PCAAE(ModelBase):
    """Non-learnable autoencoder which applies PCA independently per timepoint.

    Inputs have shape ``(B, T, R)``.  PCA is fit on the pooled timepoint
    samples ``(B * T, R)`` and produces a latent representation of shape
    ``(B, T, L)``.  The model deliberately runs for one epoch: PCA fitting is
    a closed-form operation, so further optimiser epochs cannot improve it.
    """

    requires_optimizer = False
    max_training_epochs = 1

    def __init__(self, region_dim, timepoint_dim, latent_dim, *, whiten=False, random_state=None):
        super().__init__()
        self.region_dim = int(region_dim)
        self.timepoint_dim = int(timepoint_dim)
        self.latent_dim = int(latent_dim)
        if self.region_dim <= 0 or self.timepoint_dim <= 0 or self.latent_dim <= 0:
            raise ValueError("region_dim, timepoint_dim, and latent_dim must be > 0.")
        if self.latent_dim > self.region_dim:
            raise ValueError("latent_dim cannot exceed region_dim for PCA.")

        # Compatibility with code that consumes LAE-style latent dimensions.
        self.feature_dim = self.region_dim
        self.input_dim = self.region_dim * self.timepoint_dim
        self.latent_per_timepoint = self.latent_dim
        self.latent_flat_dim = self.timepoint_dim * self.latent_dim
        self.whiten = bool(whiten)
        self.random_state = random_state
        self.pca = None
        self.fitted = False

        # Retain fitted state in model checkpoints.  Forward uses these buffers
        # if a checkpoint has been loaded in a fresh Python process.
        self.register_buffer("components", torch.empty(self.latent_dim, self.region_dim))
        self.register_buffer("mean", torch.empty(self.region_dim))
        self.register_buffer("explained_variance", torch.empty(self.latent_dim))
        self.register_buffer("is_fitted", torch.tensor(False))

    def _reshape_input(self, x):
        expected_shape = (self.timepoint_dim, self.region_dim)
        if x.ndim != 3 or tuple(x.shape[1:]) != expected_shape:
            raise ValueError(
                f"Expected x shape (B, {self.timepoint_dim}, {self.region_dim}), got {tuple(x.shape)}"
            )
        return x

    def fit(self, x):
        """Fit PCA from an input tensor/array shaped ``(B, T, R)``."""
        x = self._reshape_input(torch.as_tensor(x))
        flat_x = x.detach().cpu().reshape(-1, self.region_dim).numpy()
        if flat_x.shape[0] < self.latent_dim:
            raise ValueError(
                "PCA requires at least latent_dim pooled samples; got "
                f"{flat_x.shape[0]} samples for latent_dim={self.latent_dim}."
            )
        self.pca = SklearnPCA(
            n_components=self.latent_dim,
            whiten=self.whiten,
            random_state=self.random_state,
        ).fit(flat_x)
        self.components = torch.as_tensor(self.pca.components_, dtype=torch.float32, device=self.components.device)
        self.mean = torch.as_tensor(self.pca.mean_, dtype=torch.float32, device=self.mean.device)
        self.explained_variance = torch.as_tensor(
            self.pca.explained_variance_, dtype=torch.float32, device=self.explained_variance.device
        )
        self.fitted = True
        self.is_fitted.fill_(True)
        return self

    def fit_train_loader(self, train_loader, device=None):
        """Training-framework hook: collect all first-epoch training samples."""
        batches = []
        for data, _ in train_loader:
            batches.append(data.detach().cpu())
        if not batches:
            raise ValueError("Cannot fit PCAAE from an empty training loader.")
        return self.fit(torch.cat(batches, dim=0))

    def _transform_from_buffers(self, flat_x):
        centered = flat_x - self.mean.to(device=flat_x.device, dtype=flat_x.dtype)
        z = centered @ self.components.to(device=flat_x.device, dtype=flat_x.dtype).T
        if self.whiten:
            z = z / torch.sqrt(self.explained_variance.to(device=flat_x.device, dtype=flat_x.dtype))
        return z

    def _inverse_transform_from_buffers(self, flat_z):
        z = flat_z
        if self.whiten:
            z = z * torch.sqrt(self.explained_variance.to(device=z.device, dtype=z.dtype))
        return z @ self.components.to(device=z.device, dtype=z.dtype) + self.mean.to(device=z.device, dtype=z.dtype)

    def transform(self, x):
        """Project ``(B, T, R)`` inputs into PCA latents ``(B, T, L)``."""
        x = self._reshape_input(x)
        if not bool(self.is_fitted.item()):
            raise RuntimeError("PCAAE must be fitted before calling transform.")
        batch_size = x.shape[0]
        flat_x = x.reshape(-1, self.region_dim)
        flat_z = self._transform_from_buffers(flat_x)
        return flat_z.reshape(batch_size, self.timepoint_dim, self.latent_dim)

    def inverse_transform(self, z):
        """Reconstruct ``(B, T, L)`` PCA latents into ``(B, T, R)`` inputs."""
        expected_shape = (self.timepoint_dim, self.latent_dim)
        if z.ndim != 3 or tuple(z.shape[1:]) != expected_shape:
            raise ValueError(
                f"Expected z shape (B, {self.timepoint_dim}, {self.latent_dim}), got {tuple(z.shape)}"
            )
        if not bool(self.is_fitted.item()):
            raise RuntimeError("PCAAE must be fitted before calling inverse_transform.")
        flat_z = z.reshape(-1, self.latent_dim)
        flat_recon = self._inverse_transform_from_buffers(flat_z)
        return flat_recon.reshape(z.shape[0], self.timepoint_dim, self.region_dim)

    def forward(self, x):
        # transform followed by inverse_transform is the PCA autoencoder.
        z = self.transform(x)
        x_hat = self.inverse_transform(z)
        batch_size = x.shape[0]
        return x_hat, z.reshape(batch_size, self.latent_flat_dim)

    def loss(self, x, model_output):
        """Loss placeholder matching differentiable autoencoders in the framework."""
        x_hat, _ = model_output
        recon = self.reconstruction_loss(x_hat, x)
        loss = {"loss": recon, "recon": recon}
        return self.add_weighted_reconstruction_losses(loss, x, x_hat)
