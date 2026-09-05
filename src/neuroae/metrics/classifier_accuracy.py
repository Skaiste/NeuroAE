from __future__ import annotations

import random
from copy import deepcopy

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from neurocls.eval import compute_classification_metrics
from neurocls.models import create_model
from neurocls.train import run_torch_cross_validation_epoch_search, train_sklearn_model, train_torch_model


LATENT_BRAINGNN_MODEL_CONFIG = {
    "model": {
        "name": "braingnn",
        "hidden_dims": [256,128],
        "pool_ratios": [0.5,0.5],
        "dropout": 0.2,
        "aux_loss_weight": 0.5,
    }
}
LATENT_BRAINGNN_TRAINING_CONFIG = {
    "training": {
        "learning_rate": 0.005,
        "cross_validation_folds": 5,
        "classifier_metric": "loss",
        "batch_size": 16,
        "weight_decay": 0,
        "convergence_patience": 10,
        "convergence_min_delta": 0,
        "convergence_warmup_epochs": 0,
        "num_epochs": 100,
        "reproducibility": {"seed": 1717},
    }
}

# Classifier used by NeuroAE's standard reconstruction/latent evaluation.
# Keep this configuration here rather than in neurocls: it evaluates an AE's
# latent representation, whereas neurocls is a separate classifier workflow.
LATENT_SVM_MODEL_CONFIG = {
    "model": {
        "name": "svm",
        "input_mode": "latent_flat",
        "kernel": "sigmoid",
        "C": 1.0,
        "class_weight": "balanced",
        "probability": True,
    }
}


def _emit_classifier_progress(message):
    print(f"[latent-clf] {message}", flush=True)


def _set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_matrix(latents):
    array = np.asarray(latents, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, None]
    elif array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    return array.astype(np.float32, copy=False)


def _as_timeseries(sample):
    array = np.asarray(sample, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected latent timeseries with ndim=2, got shape {array.shape}.")
    return array


def _bandpower(x, freqs, low, high):
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return np.zeros(x.shape[1], dtype=np.float32)
    return np.sum(np.abs(x[mask]) ** 2, axis=0).astype(np.float32)


def _subject_latent_node_features(sample):
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

    fc = np.corrcoef(ts, rowvar=False)
    fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    stats = np.stack([mean, std, minimum, maximum, energy, alff, falff], axis=1)
    return np.concatenate([stats, fc], axis=1).astype(np.float32, copy=False), fc


def _latent_graph_split_from_timeseries(latents, labels):
    latent_ts = np.asarray(latents, dtype=np.float32)
    labels = list(labels)
    if latent_ts.ndim != 3:
        raise ValueError(f"Expected latent timeseries with shape (samples, timepoints, latent_dim), got {latent_ts.shape}.")
    if latent_ts.shape[0] != len(labels):
        raise ValueError(
            f"Latent sample count {latent_ts.shape[0]} does not match label count {len(labels)}."
        )
    _emit_classifier_progress(
        f"Building neurocls-style graph split: samples={latent_ts.shape[0]} "
        f"timepoints={latent_ts.shape[1]} latent_dim={latent_ts.shape[2]}"
    )
    node_features = []
    adjacency = []
    for sample in latent_ts:
        nodes, fc = _subject_latent_node_features(sample)
        node_features.append(nodes)
        adjacency.append(fc)
    node_features = np.asarray(node_features, dtype=np.float32)
    adjacency = np.asarray(adjacency, dtype=np.float32)
    adjacency_mb = adjacency.nbytes / (1024 * 1024)
    _emit_classifier_progress(
        f"Built neurocls-style adjacency tensor: shape={adjacency.shape} estimated_memory={adjacency_mb:.1f}MB"
    )
    return {
        "node_features": node_features,
        "adjacency": adjacency,
        "labels": labels,
    }


def _latent_graph_split(latents, labels):
    latent_array = np.asarray(latents, dtype=np.float32)
    if latent_array.ndim == 3:
        return _latent_graph_split_from_timeseries(latent_array, labels)

    latent_matrix = _to_matrix(latents)
    labels = list(labels)
    if latent_matrix.shape[0] != len(labels):
        raise ValueError(
            f"Latent sample count {latent_matrix.shape[0]} does not match label count {len(labels)}."
        )
    _emit_classifier_progress(
        f"Building graph split: samples={latent_matrix.shape[0]} latent_dim={latent_matrix.shape[1]}"
    )
    node_features = latent_matrix[:, :, None]
    norms = np.linalg.norm(latent_matrix, axis=1, keepdims=True)
    normalized = latent_matrix / np.maximum(norms, 1e-8)
    adjacency = np.abs(normalized[:, :, None] * normalized[:, None, :]).astype(np.float32, copy=False)
    diag = np.arange(adjacency.shape[1])
    adjacency[:, diag, diag] = 1.0
    adjacency_mb = adjacency.nbytes / (1024 * 1024)
    _emit_classifier_progress(
        f"Built adjacency tensor: shape={adjacency.shape} estimated_memory={adjacency_mb:.1f}MB"
    )
    return {
        "node_features": node_features.astype(np.float32, copy=False),
        "adjacency": adjacency,
        "labels": labels,
    }


def _latent_vector_split(latents, labels):
    latent_matrix = _to_matrix(latents)
    labels = list(labels)
    if latent_matrix.shape[0] != len(labels):
        raise ValueError(
            f"Latent sample count {latent_matrix.shape[0]} does not match label count {len(labels)}."
        )
    return {"X": latent_matrix, "labels": labels}


def _fit_graph_scaler(train_nodes, val_nodes=None, test_nodes=None):
    scaler = StandardScaler()
    train_shape = train_nodes.shape
    train_nodes = scaler.fit_transform(train_nodes.reshape(-1, train_shape[-1])).reshape(train_shape).astype(np.float32)
    if val_nodes is not None:
        val_shape = val_nodes.shape
        val_nodes = scaler.transform(val_nodes.reshape(-1, val_shape[-1])).reshape(val_shape).astype(np.float32)
    if test_nodes is not None:
        test_shape = test_nodes.shape
        test_nodes = scaler.transform(test_nodes.reshape(-1, test_shape[-1])).reshape(test_shape).astype(np.float32)
    return train_nodes, val_nodes, test_nodes


def _encode_labels(train_labels, val_labels=None, test_labels=None):
    classes = sorted({str(label) for label in train_labels})
    class_to_index = {label: idx for idx, label in enumerate(classes)}

    def _encode(values):
        if values is None:
            return None
        return np.asarray([class_to_index[str(label)] for label in values], dtype=np.int64)

    return {
        "classes": classes,
        "class_to_index": class_to_index,
        "train": _encode(train_labels),
        "val": _encode(val_labels),
        "test": _encode(test_labels),
    }


def _predict_graph_model(model, split_payload, device):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        node_features = torch.as_tensor(split_payload["node_features"], dtype=torch.float32, device=device)
        adjacency = torch.as_tensor(split_payload["adjacency"], dtype=torch.float32, device=device)
        logits = model(node_features, adjacency)["logits"]
        y_proba = torch.softmax(logits, dim=1).cpu().numpy()
        y_pred = np.argmax(y_proba, axis=1)
    return y_pred, y_proba


def _nan_metrics(label_classes):
    return {
        "accuracy": float("nan"),
        "balanced_accuracy": float("nan"),
        "macro_f1": float("nan"),
        "confusion_matrix": [],
        "per_class": {
            label: {"precision": float("nan"), "recall": float("nan"), "f1": float("nan"), "support": 0}
            for label in label_classes
        },
    }


def run_latent_svm_classifier(
    train_latents,
    train_labels,
    test_latents,
    test_labels,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """Fit the configured SVM on AE training latents and evaluate AE test latents."""
    _emit_classifier_progress(
        f"Starting latent SVM classifier: train={len(train_labels)} "
        f"test={len(test_labels)} device={device}"
    )
    label_payload = _encode_labels(train_labels, test_labels=test_labels)
    _emit_classifier_progress(f"Encoded labels: classes={label_payload['classes']}")
    if len(label_payload["classes"]) < 2 or len(train_labels) == 0:
        nan_metrics = _nan_metrics(label_payload["classes"])
        _emit_classifier_progress("Skipping classifier: insufficient classes or empty training labels")
        return {
            "model": None,
            "history": {"train": {}, "val": {}},
            "train_metrics": nan_metrics,
            "val_metrics": None,
            "test_metrics": nan_metrics,
            "test_predictions": None,
            "test_probabilities": None,
            "label_payload": label_payload,
        }

    _emit_classifier_progress("Preparing flattened latent feature vectors")
    train_split = _latent_vector_split(train_latents, train_labels)
    test_split = _latent_vector_split(test_latents, test_labels)
    scaler = StandardScaler()
    train_split["X"] = scaler.fit_transform(train_split["X"]).astype(np.float32)
    test_split["X"] = scaler.transform(test_split["X"]).astype(np.float32)

    feature_payload = {
        "input_mode": "latent_flat",
        "train": train_split,
        "val": None,
        "test": test_split,
        "scaler": scaler,
    }
    feature_metadata = {"input_shape": tuple(train_split["X"].shape[1:])}
    _emit_classifier_progress(f"Feature metadata prepared: input_shape={feature_metadata['input_shape']}")

    model_config = deepcopy(LATENT_SVM_MODEL_CONFIG)
    _emit_classifier_progress("Creating configured SVM runtime")
    runtime = create_model(model_config, feature_metadata["input_shape"], len(label_payload["classes"]))
    _emit_classifier_progress(
        f"Created runtime: family={runtime['family']} classes={len(label_payload['classes'])}"
    )
    model, history, train_metrics, _ = train_sklearn_model(runtime["model"], feature_payload, label_payload)

    _emit_classifier_progress("Running classifier predictions on test split")
    test_predictions = model.predict(test_split["X"])
    test_probabilities = model.predict_proba(test_split["X"])
    test_metrics = compute_classification_metrics(
        label_payload["test"],
        test_predictions,
        label_payload["classes"],
        y_proba=test_probabilities,
    )
    _emit_classifier_progress("Latent SVM classifier finished")

    return {
        "model": model,
        "history": history,
        "train_metrics": train_metrics,
        "val_metrics": None,
        "test_metrics": test_metrics,
        "test_predictions": test_predictions,
        "test_probabilities": test_probabilities,
        "label_payload": label_payload,
        "cross_validation": None,
    }
