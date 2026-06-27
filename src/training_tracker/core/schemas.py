"""Type aliases for training tracker payloads."""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict


JSONValue = Any
JSONDict = Dict[str, JSONValue]


class IndexEntry(TypedDict, total=False):
    schema_version: int
    experiment_id: str
    created_at: str
    status: str
    model_type: str
    framework: str
    signature: str
    has_evaluation: bool
    has_model_evaluation: bool
    best_val_loss: float
    num_epochs: int
    learning_rate: float
    latent_dim: int
    tags: List[str]
    metadata_path: str
