"""Backward-compatible imports for the renamed deep autoencoder module."""

from .deep import DAE, DAEPredHeads, VAE, VAEPredHeads

__all__ = ["DAE", "DAEPredHeads", "VAE", "VAEPredHeads"]
