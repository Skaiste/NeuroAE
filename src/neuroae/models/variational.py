"""Backward-compatible imports for the renamed deep autoencoder module."""

from .deep import DAE, DAEPredHeads, VAE, VDAE, VAEPredHeads

__all__ = ["DAE", "DAEPredHeads", "VAE", "VDAE", "VAEPredHeads"]
