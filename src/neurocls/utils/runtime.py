from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from neuroae.__main__ import (
    build_experiment_signature,
    configure_reproducibility,
    load_config,
    load_data_from_config as neuroae_load_data_from_config,
    resolve_results_dir,
)
from training_tracker import TrainingResultsManager


def load_adni3_loaders(data_dir, data_config, num_workers=0):
    data_cfg = deepcopy(data_config)
    data_cfg.setdefault("data", {})
    data_cfg["data"]["type"] = "ADNI3"
    data_cfg["data"]["timepoints_as_samples"] = False
    data_cfg["data"]["flatten"] = False
    data_cfg["data"]["fc_input"] = False
    return neuroae_load_data_from_config(data_dir=data_dir, data_config=data_cfg, num_workers=num_workers)


def encode_labels(train_labels, val_labels=None, test_labels=None):
    classes = sorted({str(label) for label in train_labels})
    class_to_index = {label: idx for idx, label in enumerate(classes)}

    def _encode(labels):
        if labels is None:
            return None
        return np.asarray([class_to_index[str(label)] for label in labels], dtype=np.int64)

    return {
        "classes": classes,
        "class_to_index": class_to_index,
        "train": _encode(train_labels),
        "val": _encode(val_labels),
        "test": _encode(test_labels),
    }


def build_feature_metadata(feature_payload, label_payload):
    train = feature_payload["train"]
    input_mode = feature_payload["input_mode"]
    metadata = {
        "input_mode": input_mode,
        "label_classes": list(label_payload["classes"]),
    }
    if input_mode == "graph":
        metadata["input_shape"] = tuple(train["node_features"].shape[1:])
        metadata["num_nodes"] = int(train["node_features"].shape[1])
        metadata["node_feature_dim"] = int(train["node_features"].shape[2])
    else:
        metadata["input_shape"] = tuple(train["X"].shape[1:])
        metadata["feature_dim"] = int(train["X"].shape[1])
    return metadata


def save_artifact(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_artifact(path):
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def get_experiment_id(results_dir, experiment_id=None):
    tracker = TrainingResultsManager(results_dir=results_dir)
    if experiment_id:
        return experiment_id
    rows = tracker.list_experiments(limit=1)
    if not rows:
        raise ValueError(f"No experiments found in {results_dir}.")
    return rows[0]["experiment_id"]


def load_experiment_context(results_dir, experiment_id):
    tracker = TrainingResultsManager(results_dir=results_dir)
    metadata = tracker.get_experiment(experiment_id)
    return tracker, metadata


def build_experiment_summary(train_metrics, val_metrics=None, classifier_metric="macro_f1"):
    summary = {
        "selected_metric": classifier_metric,
        "train_accuracy": train_metrics.get("accuracy"),
        "train_macro_f1": train_metrics.get("macro_f1"),
    }
    if isinstance(val_metrics, dict):
        summary["best_val_accuracy"] = val_metrics.get("accuracy")
        summary["best_val_macro_f1"] = val_metrics.get("macro_f1")
    return summary


def build_signature(model_name, model_config, training_config, data_config):
    return build_experiment_signature(
        model_type=model_name,
        model_params=deepcopy(model_config.get("model", {})),
        training_params=deepcopy(training_config.get("training", {})),
        data_params=deepcopy(data_config),
    )


def load_cli_config_bundle(args):
    data_config = load_config(args.data_config)
    model_config = load_config(args.model_config)
    training_config = load_config(args.training_config)
    return data_config, model_config, training_config


def to_jsonable_metrics(metrics):
    return json.loads(json.dumps(metrics, default=str))

