"""Index helpers for experiments."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

from .schemas import IndexEntry
from .storage import append_jsonl_locked, file_lock, read_jsonl, write_jsonl_atomic


def read_index(index_path: Path) -> list[IndexEntry]:
    return read_jsonl(index_path)


def append_index_entry(index_path: Path, entry: IndexEntry) -> None:
    append_jsonl_locked(index_path, entry)


def write_index(index_path: Path, entries: list[IndexEntry]) -> None:
    with file_lock(index_path):
        write_jsonl_atomic(index_path, entries)


def replace_index_entry(index_path: Path, entry: IndexEntry) -> None:
    with file_lock(index_path):
        rows = read_index(index_path)
        experiment_id = entry.get("experiment_id")
        replaced = False
        for idx, row in enumerate(rows):
            if row.get("experiment_id") == experiment_id:
                rows[idx] = entry
                replaced = True
                break
        if not replaced:
            rows.append(entry)
        write_jsonl_atomic(index_path, rows)


def find_index_entry(index_path: Path, experiment_id: str) -> Optional[IndexEntry]:
    for entry in read_index(index_path):
        if entry.get("experiment_id") == experiment_id:
            return entry
    return None


def build_index_entry(metadata: dict, metadata_path: Path, base_dir: Path) -> IndexEntry:
    summary = metadata.get("summary", {})
    training_params = metadata.get("training_params", {})
    model_params = metadata.get("model_params", {})
    evaluation = metadata.get("evaluation")
    has_evaluation = isinstance(evaluation, dict) and len(evaluation) > 0
    model_metrics = evaluation.get("model") if isinstance(evaluation, dict) else None
    has_model_evaluation = isinstance(model_metrics, dict) and len(model_metrics) > 0
    signature = get_or_build_signature(metadata)

    return {
        "schema_version": int(metadata.get("schema_version", 1)),
        "experiment_id": metadata["experiment_id"],
        "created_at": metadata["created_at"],
        "status": metadata.get("status", "unknown"),
        "model_type": metadata.get("model_type", "unknown"),
        "framework": metadata.get("framework", "unknown"),
        "signature": signature,
        "has_evaluation": has_evaluation,
        "has_model_evaluation": has_model_evaluation,
        "best_val_loss": _to_float(summary.get("best_val_loss")),
        "num_epochs": _to_int(summary.get("num_epochs")),
        "learning_rate": _to_float(training_params.get("learning_rate")),
        "latent_dim": _to_int(model_params.get("latent_dim")),
        "tags": list(metadata.get("tags", [])),
        "metadata_path": os.path.relpath(metadata_path, base_dir),
    }


def _to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_or_build_signature(metadata: dict) -> Optional[str]:
    model_type = metadata.get("model_type")
    model_params = metadata.get("model_params")
    training_params = metadata.get("training_params")
    data_params = metadata.get("data_params")
    if not isinstance(model_type, str):
        return None
    if not isinstance(model_params, dict) or not isinstance(training_params, dict) or not isinstance(data_params, dict):
        return None
    canonical = {
        "model_type": model_type,
        "model_params": model_params,
        "training_params": training_params,
        "data_params": data_params,
    }
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]
