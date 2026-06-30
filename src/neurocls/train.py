from __future__ import annotations

import logging
from copy import deepcopy

import numpy as np
import torch
from sklearn.base import clone
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .eval import compute_classification_metrics

LOGGER = logging.getLogger("neurocls")


def _emit_progress(message):
    LOGGER.info(message)
    print(message, flush=True)


class VectorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class GraphDataset(Dataset):
    def __init__(self, node_features, adjacency, y):
        self.node_features = torch.as_tensor(node_features, dtype=torch.float32)
        self.adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.node_features[idx], self.adjacency[idx], self.y[idx]


def _probabilities(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    if hasattr(model, "decision_function"):
        decision = model.decision_function(X)
        decision = np.asarray(decision)
        if decision.ndim == 1:
            probs_pos = 1.0 / (1.0 + np.exp(-decision))
            return np.stack([1.0 - probs_pos, probs_pos], axis=1)
        exp = np.exp(decision - np.max(decision, axis=1, keepdims=True))
        return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-8)
    return None


def train_sklearn_model(model, feature_payload, label_payload):
    train_X = feature_payload["train"]["X"]
    train_y = label_payload["train"]
    val_X = feature_payload["val"]["X"] if feature_payload["val"] is not None else None
    val_y = label_payload["val"] if label_payload["val"] is not None else None

    LOGGER.info(
        "Starting sklearn training: samples=%d features=%d classes=%d",
        train_X.shape[0],
        train_X.shape[1],
        len(label_payload["classes"]),
    )
    _emit_progress(
        f"[neurocls] Starting sklearn training: samples={train_X.shape[0]} "
        f"features={train_X.shape[1]} classes={len(label_payload['classes'])}"
    )
    estimator = clone(model)
    estimator.fit(train_X, train_y)
    train_pred = estimator.predict(train_X)
    train_metrics = compute_classification_metrics(
        train_y,
        train_pred,
        label_payload["classes"],
        y_proba=_probabilities(estimator, train_X),
    )
    val_metrics = None
    if val_X is not None and val_y is not None and len(val_y) > 0:
        val_pred = estimator.predict(val_X)
        val_metrics = compute_classification_metrics(
            val_y,
            val_pred,
            label_payload["classes"],
            y_proba=_probabilities(estimator, val_X),
        )
    LOGGER.info(
        "Finished sklearn training: train_accuracy=%.4f train_macro_f1=%.4f%s",
        train_metrics["accuracy"],
        train_metrics["macro_f1"],
        (
            f" val_accuracy={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}"
            if val_metrics
            else ""
        ),
    )
    _emit_progress(
        "[neurocls] Finished sklearn training: "
        f"train_accuracy={train_metrics['accuracy']:.4f} "
        f"train_macro_f1={train_metrics['macro_f1']:.4f}"
        + (
            f" val_accuracy={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}"
            if val_metrics
            else ""
        )
    )
    history = {
        "train": {"accuracy": [train_metrics["accuracy"]], "macro_f1": [train_metrics["macro_f1"]]},
        "val": {
            "accuracy": [val_metrics["accuracy"]] if val_metrics else [],
            "macro_f1": [val_metrics["macro_f1"]] if val_metrics else [],
        },
    }
    return estimator, history, train_metrics, val_metrics


def _run_epoch(model, loader, criterion, device, optimizer=None, graph_mode=False):
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    y_true = []
    y_pred = []
    y_proba = []

    for batch in loader:
        if graph_mode:
            node_features, adjacency, y = batch
            node_features = node_features.to(device)
            adjacency = adjacency.to(device)
            y = y.to(device)
            outputs = model(node_features, adjacency)
            logits = outputs["logits"]
            aux_loss = outputs.get("aux_loss", 0.0)
        else:
            X, y = batch
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            aux_loss = 0.0

        loss = criterion(logits, y) + (aux_loss if torch.is_tensor(aux_loss) else torch.tensor(aux_loss, device=device))
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * y.shape[0]
        probs = torch.softmax(logits.detach(), dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        y_true.extend(y.detach().cpu().numpy().tolist())
        y_pred.extend(preds.tolist())
        y_proba.append(probs)

    avg_loss = total_loss / max(len(loader.dataset), 1)
    y_proba = np.concatenate(y_proba, axis=0) if y_proba else None
    return avg_loss, np.asarray(y_true), np.asarray(y_pred), y_proba


def train_torch_model(model, family, feature_payload, label_payload, training_config, device):
    batch_size = int(training_config["training"].get("batch_size", 16))
    num_epochs = int(training_config["training"].get("num_epochs", 100))
    patience = training_config["training"].get("convergence_patience")
    min_delta = float(training_config["training"].get("convergence_min_delta", 0.0))
    warmup = int(training_config["training"].get("convergence_warmup_epochs", 0))
    metric_name = str(training_config["training"].get("classifier_metric", "macro_f1"))
    log_every_epochs = max(1, int(training_config["training"].get("log_every_epochs", 1)))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["training"].get("learning_rate", 1e-3)),
        weight_decay=float(training_config["training"].get("weight_decay", 1e-4)),
    )

    if family == "torch_graph":
        train_dataset = GraphDataset(
            feature_payload["train"]["node_features"],
            feature_payload["train"]["adjacency"],
            label_payload["train"],
        )
        val_dataset = GraphDataset(
            feature_payload["val"]["node_features"],
            feature_payload["val"]["adjacency"],
            label_payload["val"],
        ) if feature_payload["val"] is not None and label_payload["val"] is not None else None
        graph_mode = True
    else:
        train_dataset = VectorDataset(feature_payload["train"]["X"], label_payload["train"])
        val_dataset = VectorDataset(feature_payload["val"]["X"], label_payload["val"]) if feature_payload["val"] is not None and label_payload["val"] is not None else None
        graph_mode = False

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset is not None else None

    model = model.to(device)
    LOGGER.info(
        "Starting torch training: family=%s samples=%d batch_size=%d epochs=%d metric=%s device=%s",
        family,
        len(train_dataset),
        batch_size,
        num_epochs,
        metric_name,
        device,
    )
    _emit_progress(
        f"[neurocls] Starting torch training: family={family} samples={len(train_dataset)} "
        f"batch_size={batch_size} epochs={num_epochs} metric={metric_name} device={device}"
    )
    history = {"train": {"loss": [], "accuracy": [], "macro_f1": []}, "val": {"loss": [], "accuracy": [], "macro_f1": []}}
    best_state = deepcopy(model.state_dict())
    best_score = float("-inf")
    best_val_metrics = None
    epochs_without_improvement = 0

    for epoch_idx in range(num_epochs):
        train_loss, train_true, train_pred, train_proba = _run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer, graph_mode=graph_mode
        )
        train_metrics = compute_classification_metrics(
            train_true, train_pred, label_payload["classes"], y_proba=train_proba
        )
        history["train"]["loss"].append(train_loss)
        history["train"]["accuracy"].append(train_metrics["accuracy"])
        history["train"]["macro_f1"].append(train_metrics["macro_f1"])

        if val_loader is not None:
            val_loss, val_true, val_pred, val_proba = _run_epoch(
                model, val_loader, criterion, device, optimizer=None, graph_mode=graph_mode
            )
            val_metrics = compute_classification_metrics(
                val_true, val_pred, label_payload["classes"], y_proba=val_proba
            )
            history["val"]["loss"].append(val_loss)
            history["val"]["accuracy"].append(val_metrics["accuracy"])
            history["val"]["macro_f1"].append(val_metrics["macro_f1"])
            score = val_metrics.get(metric_name, val_metrics["macro_f1"])
            if (epoch_idx + 1) % log_every_epochs == 0:
                LOGGER.info(
                    "Epoch %d/%d: train_loss=%.4f train_accuracy=%.4f train_macro_f1=%.4f val_loss=%.4f val_accuracy=%.4f val_macro_f1=%.4f %s=%.4f",
                    epoch_idx + 1,
                    num_epochs,
                    train_loss,
                    train_metrics["accuracy"],
                    train_metrics["macro_f1"],
                    val_loss,
                    val_metrics["accuracy"],
                    val_metrics["macro_f1"],
                    metric_name,
                    score,
                )
                _emit_progress(
                    f"[neurocls] Epoch {epoch_idx + 1}/{num_epochs}: "
                    f"train_loss={train_loss:.4f} train_accuracy={train_metrics['accuracy']:.4f} "
                    f"train_macro_f1={train_metrics['macro_f1']:.4f} "
                    f"val_loss={val_loss:.4f} val_accuracy={val_metrics['accuracy']:.4f} "
                    f"val_macro_f1={val_metrics['macro_f1']:.4f} {metric_name}={score:.4f}"
                )
            if score > (best_score + min_delta):
                best_score = score
                best_state = deepcopy(model.state_dict())
                best_val_metrics = val_metrics
                epochs_without_improvement = 0
                LOGGER.info("Epoch %d/%d improved best %s to %.4f", epoch_idx + 1, num_epochs, metric_name, score)
            elif epoch_idx + 1 > warmup:
                epochs_without_improvement += 1
                LOGGER.info(
                    "Epoch %d/%d no improvement: patience=%s/%s",
                    epoch_idx + 1,
                    num_epochs,
                    epochs_without_improvement,
                    patience,
                )
                if patience is not None and epochs_without_improvement >= int(patience):
                    LOGGER.info(
                        "Early stopping at epoch %d/%d after %d epochs without improvement.",
                        epoch_idx + 1,
                        num_epochs,
                        epochs_without_improvement,
                    )
                    break
            else:
                LOGGER.info("Epoch %d/%d warmup active; patience not counted yet.", epoch_idx + 1, num_epochs)
        else:
            best_state = deepcopy(model.state_dict())
            if (epoch_idx + 1) % log_every_epochs == 0:
                LOGGER.info(
                    "Epoch %d/%d: train_loss=%.4f train_accuracy=%.4f train_macro_f1=%.4f",
                    epoch_idx + 1,
                    num_epochs,
                    train_loss,
                    train_metrics["accuracy"],
                    train_metrics["macro_f1"],
                )
                _emit_progress(
                    f"[neurocls] Epoch {epoch_idx + 1}/{num_epochs}: "
                    f"train_loss={train_loss:.4f} train_accuracy={train_metrics['accuracy']:.4f} "
                    f"train_macro_f1={train_metrics['macro_f1']:.4f}"
                )

    model.load_state_dict(best_state)
    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    _, train_true, train_pred, train_proba = _run_epoch(
        model, train_eval_loader, criterion, device, optimizer=None, graph_mode=graph_mode
    )
    train_metrics = compute_classification_metrics(train_true, train_pred, label_payload["classes"], y_proba=train_proba)
    LOGGER.info(
        "Finished torch training: train_accuracy=%.4f train_macro_f1=%.4f%s",
        train_metrics["accuracy"],
        train_metrics["macro_f1"],
        (
            f" best_val_accuracy={best_val_metrics['accuracy']:.4f} best_val_macro_f1={best_val_metrics['macro_f1']:.4f}"
            if best_val_metrics
            else ""
        ),
    )
    _emit_progress(
        "[neurocls] Finished torch training: "
        f"train_accuracy={train_metrics['accuracy']:.4f} "
        f"train_macro_f1={train_metrics['macro_f1']:.4f}"
        + (
            f" best_val_accuracy={best_val_metrics['accuracy']:.4f} "
            f"best_val_macro_f1={best_val_metrics['macro_f1']:.4f}"
            if best_val_metrics
            else ""
        )
    )
    return model, history, train_metrics, best_val_metrics
