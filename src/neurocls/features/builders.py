from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


def _as_timeseries(sample):
    array = np.asarray(sample, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected subject timeseries with ndim=2, got shape {array.shape}.")
    return array


def _subject_raw_flat(sample):
    return _as_timeseries(sample).reshape(-1).astype(np.float32, copy=False)


def _subject_fc(sample):
    ts = _as_timeseries(sample)
    fc = np.corrcoef(ts, rowvar=False)
    fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return fc


def _subject_fc_flat(sample):
    fc = _subject_fc(sample)
    tri = np.triu_indices(fc.shape[0], k=1)
    return fc[tri].astype(np.float32, copy=False)


def _bandpower(x, freqs, low, high):
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return np.zeros(x.shape[1], dtype=np.float32)
    return np.sum(np.abs(x[mask]) ** 2, axis=0).astype(np.float32)


def _subject_node_features(sample):
    ts = _as_timeseries(sample)
    n_timepoints = ts.shape[0]
    centered = ts - np.mean(ts, axis=0, keepdims=True)
    std = np.std(centered, axis=0)
    mean = np.mean(ts, axis=0)
    minimum = np.min(ts, axis=0)
    maximum = np.max(ts, axis=0)
    energy = np.mean(centered**2, axis=0)

    fft = np.fft.rfft(centered, axis=0)
    freqs = np.fft.rfftfreq(n_timepoints, d=1.0)
    low_band = _bandpower(fft, freqs, 0.01, 0.08)
    full_band = _bandpower(fft, freqs, 0.0, 0.25)
    alff = np.sqrt(low_band + 1e-8).astype(np.float32)
    falff = (low_band / np.maximum(full_band, 1e-8)).astype(np.float32)

    fc = _subject_fc(sample)
    stats = np.stack([mean, std, minimum, maximum, energy, alff, falff], axis=1)
    return np.concatenate([stats, fc], axis=1).astype(np.float32, copy=False), fc


def _build_graph_split(dataset):
    node_features = []
    adjacency = []
    labels = list(dataset.labels)
    subject_ids = list(getattr(dataset, "subject_ids", [None] * len(dataset)))
    for sample in np.asarray(dataset.data):
        nodes, fc = _subject_node_features(sample)
        node_features.append(nodes)
        adjacency.append(fc.astype(np.float32, copy=False))
    return {
        "node_features": np.asarray(node_features, dtype=np.float32),
        "adjacency": np.asarray(adjacency, dtype=np.float32),
        "labels": labels,
        "subject_ids": subject_ids,
    }


def _build_vector_split(dataset, mode):
    rows = []
    labels = list(dataset.labels)
    subject_ids = list(getattr(dataset, "subject_ids", [None] * len(dataset)))
    for sample in np.asarray(dataset.data):
        if mode == "raw_flat":
            rows.append(_subject_raw_flat(sample))
        elif mode == "fc_flat":
            rows.append(_subject_fc_flat(sample))
        else:
            raise ValueError(f"Unsupported vector feature mode: {mode}")
    return {
        "X": np.asarray(rows, dtype=np.float32),
        "labels": labels,
        "subject_ids": subject_ids,
    }


def _fit_vector_scaler(train_X, val_X=None, test_X=None, scaler_type="standard"):
    if scaler_type in (None, "none"):
        return train_X, val_X, test_X, None
    if scaler_type != "standard":
        raise ValueError(f"Unsupported scaler_type {scaler_type!r}.")
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X).astype(np.float32)
    if val_X is not None:
        val_X = scaler.transform(val_X).astype(np.float32)
    if test_X is not None:
        test_X = scaler.transform(test_X).astype(np.float32)
    return train_X, val_X, test_X, scaler


def _fit_graph_scaler(train_nodes, val_nodes=None, test_nodes=None, scaler_type="standard"):
    if scaler_type in (None, "none"):
        return train_nodes, val_nodes, test_nodes, None
    if scaler_type != "standard":
        raise ValueError(f"Unsupported scaler_type {scaler_type!r}.")
    scaler = StandardScaler()
    train_shape = train_nodes.shape
    train_nodes = scaler.fit_transform(train_nodes.reshape(-1, train_shape[-1])).reshape(train_shape).astype(np.float32)
    if val_nodes is not None:
        val_shape = val_nodes.shape
        val_nodes = scaler.transform(val_nodes.reshape(-1, val_shape[-1])).reshape(val_shape).astype(np.float32)
    if test_nodes is not None:
        test_shape = test_nodes.shape
        test_nodes = scaler.transform(test_nodes.reshape(-1, test_shape[-1])).reshape(test_shape).astype(np.float32)
    return train_nodes, val_nodes, test_nodes, scaler


def _feature_cache_path(data_config, input_mode, dataset_splits):
    feature_cache_dir = data_config.get("data", {}).get("feature_cache_dir")
    if not feature_cache_dir:
        return None
    cache_root = Path(feature_cache_dir)
    if not cache_root.is_absolute():
        cache_root = Path(__file__).resolve().parents[3] / cache_root
    subjects = {
        split_name: list(getattr(dataset_splits[split_name], "subject_ids", []))
        for split_name in dataset_splits
    }
    key_payload = {
        "input_mode": input_mode,
        "data": deepcopy(data_config),
        "subjects": subjects,
    }
    digest = hashlib.sha1(json.dumps(key_payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root / f"{input_mode}_{digest}.pt"


def build_feature_splits(loaders, model_config, training_config, data_config):
    input_mode = str(model_config["model"].get("input_mode", "raw_flat"))
    scaler_type = training_config.get("training", {}).get("feature_scaler", "standard")
    dataset_splits = {
        "train": loaders["train_loader"].dataset,
        "val": loaders.get("val_loader").dataset if loaders.get("val_loader") is not None else None,
        "test": loaders.get("test_loader").dataset if loaders.get("test_loader") is not None else None,
    }

    cache_path = _feature_cache_path(data_config, input_mode, {k: v for k, v in dataset_splits.items() if v is not None})
    if cache_path is not None and cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        return payload

    if input_mode == "graph":
        train_split = _build_graph_split(dataset_splits["train"])
        val_split = _build_graph_split(dataset_splits["val"]) if dataset_splits["val"] is not None else None
        test_split = _build_graph_split(dataset_splits["test"]) if dataset_splits["test"] is not None else None
        train_nodes, val_nodes, test_nodes, scaler = _fit_graph_scaler(
            train_split["node_features"],
            val_split["node_features"] if val_split else None,
            test_split["node_features"] if test_split else None,
            scaler_type=scaler_type if data_config.get("data", {}).get("scale_features", True) else "none",
        )
        train_split["node_features"] = train_nodes
        if val_split:
            val_split["node_features"] = val_nodes
        if test_split:
            test_split["node_features"] = test_nodes
    else:
        train_split = _build_vector_split(dataset_splits["train"], input_mode)
        val_split = _build_vector_split(dataset_splits["val"], input_mode) if dataset_splits["val"] is not None else None
        test_split = _build_vector_split(dataset_splits["test"], input_mode) if dataset_splits["test"] is not None else None
        train_X, val_X, test_X, scaler = _fit_vector_scaler(
            train_split["X"],
            val_split["X"] if val_split else None,
            test_split["X"] if test_split else None,
            scaler_type=scaler_type if data_config.get("data", {}).get("scale_features", True) else "none",
        )
        train_split["X"] = train_X
        if val_split:
            val_split["X"] = val_X
        if test_split:
            test_split["X"] = test_X

    payload = {
        "input_mode": input_mode,
        "train": train_split,
        "val": val_split,
        "test": test_split,
        "scaler": scaler,
    }
    if cache_path is not None:
        torch.save(payload, cache_path)
    return payload

