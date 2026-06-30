"""Helpers for the parameter-focused experiment index."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Optional

from .schemas import ParameterIndexEntry
from .storage import append_jsonl_locked, file_lock, read_jsonl, write_jsonl_atomic


def read_parameter_index(index_path: Path) -> list[ParameterIndexEntry]:
    return read_jsonl(index_path)


def append_parameter_index_entry(index_path: Path, entry: ParameterIndexEntry) -> None:
    append_jsonl_locked(index_path, entry)


def write_parameter_index(index_path: Path, entries: list[ParameterIndexEntry]) -> None:
    with file_lock(index_path):
        write_jsonl_atomic(index_path, entries)


def replace_parameter_index_entry(index_path: Path, entry: ParameterIndexEntry) -> None:
    with file_lock(index_path):
        rows = read_parameter_index(index_path)
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


def find_parameter_index_entry(index_path: Path, experiment_id: str) -> Optional[ParameterIndexEntry]:
    for entry in read_parameter_index(index_path):
        if entry.get("experiment_id") == experiment_id:
            return entry
    return None


def _encode_group_value(value: object) -> object:
    if isinstance(value, list):
        return tuple(_encode_group_value(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((str(k), _encode_group_value(v)) for k, v in value.items()))
    return value


def _collect_leaf_paths_from_dict(node: object, prefix: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
    if not isinstance(node, dict):
        return {prefix} if prefix else set()
    paths: set[tuple[str, ...]] = set()
    for key, value in node.items():
        key_path = prefix + (str(key),)
        if isinstance(value, dict):
            paths |= _collect_leaf_paths_from_dict(value, key_path)
        else:
            paths.add(key_path)
    return paths


def _get_nested_value(node: object, path: tuple[str, ...]) -> object:
    current = node
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _assign_nested_value(target: dict[str, Any], path: tuple[str, ...], value: object) -> None:
    current: dict[str, Any] = target
    for key in path[:-1]:
        next_value = current.get(key)
        if not isinstance(next_value, dict):
            next_value = {}
            current[key] = next_value
        current = next_value
    current[path[-1]] = value


def _extract_varying_subset(node: object, varying_paths: set[tuple[str, ...]]) -> dict[str, Any]:
    if not isinstance(node, dict):
        return {}
    output: dict[str, Any] = {}
    for path in varying_paths:
        value = _get_nested_value(node, path)
        if value is not None:
            _assign_nested_value(output, path, value)
    return output


def compute_varying_parameter_paths(metadata_rows: list[dict]) -> dict[str, set[tuple[str, ...]]]:
    sections = {
        "model_params": "model_params",
        "training_params": "training_params",
        "data_params": "data_params",
        "experiment_params": "experiment_params",
    }
    varying_paths: dict[str, set[tuple[str, ...]]] = {key: set() for key in sections}

    for output_key, metadata_key in sections.items():
        candidate_paths: set[tuple[str, ...]] = set()
        for metadata in metadata_rows:
            if metadata_key == "experiment_params":
                experiment_params = {}
                if metadata.get("target_group") is not None:
                    experiment_params["target_group"] = metadata.get("target_group")
                candidate_paths |= _collect_leaf_paths_from_dict(experiment_params)
                continue
            candidate_paths |= _collect_leaf_paths_from_dict(metadata.get(metadata_key))

        for path in candidate_paths:
            values = set()
            for metadata in metadata_rows:
                if metadata_key == "experiment_params":
                    experiment_params = {}
                    if metadata.get("target_group") is not None:
                        experiment_params["target_group"] = metadata.get("target_group")
                    value = _get_nested_value(experiment_params, path)
                else:
                    value = _get_nested_value(metadata.get(metadata_key), path)
                if value is not None:
                    values.add(_encode_group_value(value))
                if len(values) >= 2:
                    varying_paths[output_key].add(path)
                    break
    return varying_paths


def build_parameter_index_entry(
    metadata: dict,
    varying_paths: dict[str, set[tuple[str, ...]]] | None = None,
) -> ParameterIndexEntry:
    evaluation = metadata.get("evaluation")
    if not isinstance(evaluation, dict):
        evaluation = {}
    groups_payload = evaluation.get("groups")
    if not isinstance(groups_payload, dict):
        groups_payload = {}

    experiment_params = {}
    if metadata.get("target_group") is not None:
        experiment_params["target_group"] = metadata.get("target_group")

    model_params = metadata.get("model_params")
    training_params = metadata.get("training_params")
    data_params = metadata.get("data_params")
    varying_paths = varying_paths or {
        "model_params": _collect_leaf_paths_from_dict(model_params),
        "training_params": _collect_leaf_paths_from_dict(training_params),
        "data_params": _collect_leaf_paths_from_dict(data_params),
        "experiment_params": _collect_leaf_paths_from_dict(experiment_params),
    }

    return {
        "schema_version": int(metadata.get("schema_version", 1)),
        "experiment_id": str(metadata["experiment_id"]),
        "model_type": str(metadata.get("model_type", "unknown")),
        "model_params": _extract_varying_subset(model_params, varying_paths.get("model_params", set())),
        "training_params": _extract_varying_subset(training_params, varying_paths.get("training_params", set())),
        "data_params": _extract_varying_subset(data_params, varying_paths.get("data_params", set())),
        "experiment_params": _extract_varying_subset(experiment_params, varying_paths.get("experiment_params", set())),
        "evaluation": evaluation,
        "evaluation_scope": str(evaluation.get("scope", "combined")),
        "evaluation_group_names": sorted(groups_payload.keys()),
        "evaluation_groups": groups_payload,
    }
