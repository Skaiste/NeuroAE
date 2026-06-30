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


class ParameterIndexEntry(TypedDict, total=False):
    schema_version: int
    experiment_id: str
    model_type: str
    model_params: JSONDict
    training_params: JSONDict
    data_params: JSONDict
    experiment_params: JSONDict
    evaluation: JSONDict
    evaluation_scope: str
    evaluation_group_names: List[str]
    evaluation_groups: JSONDict
