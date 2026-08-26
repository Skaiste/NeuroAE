"""Fold-local AE-to-classifier experiments.

This module deliberately uses only ``train_loader`` from the AE data bundle.
The configured AE test split is never encoded, fitted, or evaluated here.
"""
from __future__ import annotations

import itertools
import json
import logging
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from neuroae.__main__ import (
    _build_fold_loaders,
    _classification_labels,
    _make_cross_validation_splits,
    _seed_everything,
    configure_reproducibility,
    load_data_from_config,
    load_model_from_config,
)
from neuroae.train import select_best_checkpoint, train_vae
from training_tracker import TrainingResultsManager

from .eval import compute_classification_metrics
from .models import create_model
from .train import train_torch_model
from .utils.runtime import build_signature


LOGGER = logging.getLogger("neurocls.ae_exp")


def _collect_vars(node, prefix=""):
    if isinstance(node, dict):
        result = {}
        for key, value in node.items():
            result.update(_collect_vars(value, f"{prefix}.{key}" if prefix else key))
        return result
    return {prefix: node}


def _set_value(node, dotted_name, value):
    keys = dotted_name.split(".")
    target = node
    for key in keys[:-1]:
        target = target.setdefault(key, {})
    target[keys[-1]] = value


def _extract_latents(model, loader, device):
    """Return the time-preserving LAE embedding and labels from one loader."""
    latents, labels = [], []
    model.eval()
    with torch.no_grad():
        for data, batch_labels in loader:
            output = model(data.to(device))
            # LAE returns (reconstruction, z); for a variational LAE the
            # deterministic mean is the penultimate value.
            latent = output[-2] if len(output) == 4 else output[-1]
            latents.append(latent.detach().cpu().numpy())
            labels.extend(list(batch_labels))
    if not latents:
        raise ValueError("A CV fold produced no latent samples.")
    return np.concatenate(latents, axis=0), labels


def _make_latent_feature_payload(train_latents, train_labels, val_latents, val_labels, latent_dim, classes):
    from neuroae.metrics.classifier_accuracy import _latent_graph_split_from_timeseries, _fit_graph_scaler

    def reshape(values):
        values = np.asarray(values, dtype=np.float32)
        if values.shape[1] % latent_dim:
            raise ValueError(f"Latent width {values.shape[1]} is not divisible by latent_dim={latent_dim}.")
        return values.reshape(values.shape[0], values.shape[1] // latent_dim, latent_dim)

    train_split = _latent_graph_split_from_timeseries(reshape(train_latents), train_labels)
    val_split = _latent_graph_split_from_timeseries(reshape(val_latents), val_labels)
    train_nodes, val_nodes, _ = _fit_graph_scaler(train_split["node_features"], val_split["node_features"])
    train_split["node_features"] = train_nodes
    val_split["node_features"] = val_nodes
    classes = list(classes)
    class_to_index = {label: index for index, label in enumerate(classes)}
    labels = {
        "classes": classes,
        "class_to_index": class_to_index,
        "train": np.asarray([class_to_index[str(label)] for label in train_labels], dtype=np.int64),
        "val": np.asarray([class_to_index[str(label)] for label in val_labels], dtype=np.int64),
        "test": None,
    }
    return {
        "input_mode": "graph", "train": train_split, "val": val_split, "test": None, "scaler": None
    }, labels


def _predict_graph(model, split, device):
    model.eval()
    with torch.no_grad():
        nodes = torch.as_tensor(split["node_features"], dtype=torch.float32, device=device)
        adjacency = torch.as_tensor(split["adjacency"], dtype=torch.float32, device=device)
        probabilities = torch.softmax(model(nodes, adjacency)["logits"], dim=1).cpu().numpy()
    return np.argmax(probabilities, axis=1), probabilities


def _mean_or_none(values):
    values = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(values)) if values else None


def _average_metrics(fold_metrics, classes):
    aggregate = {
        key: _mean_or_none([metrics.get(key) for metrics in fold_metrics])
        for key in ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc", "roc_auc_ovr_macro")
    }
    aggregate = {key: value for key, value in aggregate.items() if value is not None}
    aggregate["per_class"] = {
        label: {
            metric: _mean_or_none([fold.get("per_class", {}).get(label, {}).get(metric) for fold in fold_metrics])
            for metric in ("precision", "recall", "f1")
        }
        for label in classes
    }
    return aggregate


def _validate_configs(ae_model_config, classifier_model_config):
    ae_model = ae_model_config.get("model", {})
    if ae_model.get("name") != "LAE" or int(ae_model.get("latent_dim", 0)) != 32:
        raise ValueError("ae_exp requires an AE configuration with model.name=LAE and model.latent_dim=32.")
    if str(classifier_model_config.get("model", {}).get("name", "")).lower() != "braingnn":
        raise ValueError("ae_exp currently supports model.name=braingnn only.")


def run_ae_experiment_sweep(data_dir, device, ae_data_config, ae_model_config, ae_training_config, experiment_config, results_dir, num_workers=0, dry_run=False):
    """Run one 5-fold AE fit followed by one BrainGNN fit per fold per candidate."""
    folds = int(experiment_config.get("default", {}).get("cross_validation_folds", 5))
    if folds != 5:
        raise ValueError("ae_exp is defined as a 5-fold pipeline; cross_validation_folds must be 5.")
    classifier_base = deepcopy(experiment_config.get("default", {}).get("classifier", {}))
    classifier_base.setdefault("model", {"name": "braingnn"})
    classifier_base.setdefault("training", {})
    classifier_base["training"].setdefault("classifier_metric", "macro_f1")
    _validate_configs(ae_model_config, classifier_base)

    # The AE test loader may exist in the source configuration, but it is
    # intentionally discarded immediately and cannot enter this pipeline.
    configure_reproducibility(ae_data_config, ae_training_config)
    loaders = load_data_from_config(data_dir, ae_data_config, num_workers=num_workers)
    train_dataset = loaders["train_loader"].dataset
    loaders.pop("test_loader", None)
    loaders.pop("val_loader", None)
    labels = list(train_dataset.labels)
    global_classes = sorted(set(map(str, labels)))
    if len(global_classes) < 2:
        raise ValueError("ae_exp needs at least two classes in the AE training split.")
    split_iter, stratified = _make_cross_validation_splits(train_dataset, folds, int(ae_training_config["training"].get("reproducibility", {}).get("seed", 42)))
    split_indices = list(split_iter)
    LOGGER.info("ae_exp: %d fold %s CV using only AE training samples", folds, "stratified" if stratified else "KFold")

    candidates = []
    for set_name, config_set in experiment_config.items():
        if set_name in {"default", "autoencoder"}:
            continue
        base = deepcopy(classifier_base)
        for section in ("model", "training"):
            base.setdefault(section, {}).update(deepcopy(config_set.get("static_params", {}).get(section, {})))
        variables = _collect_vars(config_set.get("exp_params", {}))
        keys = list(variables)
        for values in itertools.product(*[list(variables[key]) for key in keys]) if keys else [()]:
            candidate = deepcopy(base)
            for key, value in zip(keys, values):
                _set_value(candidate, key, value)
            _validate_configs(ae_model_config, candidate)
            candidates.append((set_name, candidate))

    # Fit the unsupervised AE once per fold.  Every BrainGNN candidate below
    # therefore sees exactly the same fold-local latent representation.
    ae_fold_data = []
    for fold, (train_idx, val_idx) in enumerate(split_indices, start=1):
        seed = int(ae_training_config["training"].get("reproducibility", {}).get("seed", 42)) + fold
        _seed_everything(seed)
        fold_loaders = _build_fold_loaders(loaders, train_idx, val_idx, ae_training_config)
        LOGGER.info("ae_exp: fitting LAE for fold %d/%d", fold, folds)
        ae_model, _, latent_dim = load_model_from_config(
            ae_model_config, ae_data_config, loaders["input_dim"], loaders["timepoint_dim"], device,
            preserve_timepoints=fold_loaders.get("preserve_timepoints", False), class_labels=_classification_labels(loaders),
        )
        ae_model.set_loss_fn_params(ae_training_config["training"].get("loss_params"))
        with tempfile.TemporaryDirectory(prefix="neurocls_ae_exp_") as checkpoint_dir:
            history, _ = train_vae(
                ae_model, fold_loaders["train_loader"], fold_loaders["val_loader"],
                num_epochs=1 if dry_run else int(ae_training_config["training"].get("num_epochs", 50)),
                learning_rate=ae_training_config["training"].get("learning_rate", 1e-3),
                weight_decay=ae_training_config["training"].get("weight_decay", 1e-4), device=device,
                save_dir=checkpoint_dir, name="best", convergence_patience=ae_training_config["training"].get("convergence_patience"),
                convergence_min_delta=ae_training_config["training"].get("convergence_min_delta", 0.0),
                convergence_warmup_epochs=ae_training_config["training"].get("convergence_warmup_epochs", 0),
                checkpoint_selection_metric=ae_training_config["training"].get("checkpoint_selection_metric", "swfcd_loss_joint"),
                save_checkpoint=True, vectorize_val_reference=ae_training_config["training"].get("vectorize_val_reference", False),
                compute_swfcd_during_training=ae_training_config["training"].get("compute_swfcd_during_training"),
            )
            checkpoint = Path(checkpoint_dir) / "best_model.pt"
            if checkpoint.exists():
                ae_model.load_state_dict(torch.load(checkpoint, map_location=torch.device(device)))
            selection = select_best_checkpoint(history, ae_training_config["training"].get("checkpoint_selection_metric", "swfcd_loss_joint"), ae_training_config["training"].get("convergence_min_delta", 0.0))
            train_latent, train_labels = _extract_latents(ae_model, fold_loaders["train_loader"], device)
            val_latent, val_labels = _extract_latents(ae_model, fold_loaders["val_loader"], device)
        features, encoded_labels = _make_latent_feature_payload(
            train_latent, train_labels, val_latent, val_labels, latent_dim, global_classes
        )
        ae_fold_data.append({
            "fold": fold, "seed": seed, "features": features, "labels": encoded_labels,
            "train_samples": int(len(train_idx)), "validation_samples": int(len(val_idx)), "selection": selection,
        })

    tracker = None if dry_run else TrainingResultsManager(results_dir=results_dir)
    for candidate_index, (set_name, classifier_config) in enumerate(candidates, start=1):
        signature = build_signature(
            "ae_exp_lae32_braingnn",
            {"model": classifier_config["model"]},
            {"training": classifier_config["training"]},
            ae_data_config,
        )
        LOGGER.info("ae_exp candidate %d/%d: %s (%s)", candidate_index, len(candidates), set_name, signature)
        fold_records, histories = [], {}
        for fold_data in ae_fold_data:
            fold = fold_data["fold"]
            features = fold_data["features"]
            encoded_labels = fold_data["labels"]
            _seed_everything(fold_data["seed"])
            classifier_training = deepcopy(classifier_config["training"])
            # The classifier is intentionally seeded from the AE fold rather
            # than carrying an independent sweep seed.  This keeps each
            # fold's AE representation and classifier initialization paired.
            classifier_training.setdefault("reproducibility", {})["seed"] = fold_data["seed"]
            if dry_run:
                classifier_training["num_epochs"] = 1
            runtime = create_model({"model": classifier_config["model"]}, tuple(features["train"]["node_features"].shape[1:]), len(encoded_labels["classes"]))
            classifier, classifier_history, _, _ = train_torch_model(runtime["model"], runtime["family"], features, encoded_labels, {"training": classifier_training}, torch.device(device))
            predictions, probabilities = _predict_graph(classifier, features["val"], torch.device(device))
            metrics = compute_classification_metrics(encoded_labels["val"], predictions, encoded_labels["classes"], probabilities)
            fold_records.append({"fold": fold, "seed": fold_data["seed"], "ae_seed": fold_data["seed"], "classifier_seed": fold_data["seed"], "train_samples": fold_data["train_samples"], "validation_samples": fold_data["validation_samples"], "ae_best_epoch": int(fold_data["selection"]["best_epoch"]) if fold_data["selection"] else None, "ae_selection": fold_data["selection"], "classifier_metrics": metrics})
            # AE epoch histories are shared across candidates; the selected
            # epoch is retained in metadata while each experiment history
            # stores only the candidate-specific classifier trace.
            histories[f"fold_{fold}"] = {"classifier": classifier_history}

        metrics_by_fold = [record["classifier_metrics"] for record in fold_records]
        aggregate = _average_metrics(metrics_by_fold, global_classes)
        metadata = {
            "status": "completed", "framework": "neurocls", "pipeline": "ae_exp", "model_type": "ae_exp_lae32_braingnn",
            "signature": signature, "experiment_set": set_name, "model_params": deepcopy(classifier_config["model"]),
            "training_params": deepcopy(classifier_config["training"]), "data_params": deepcopy(ae_data_config),
            "ae_model_params": deepcopy(ae_model_config["model"]), "ae_training_params": deepcopy(ae_training_config["training"]),
            "cross_validation": {"num_folds": folds, "stratified": stratified, "shared_ae_classifier_seed": True, "test_set_used": False, "folds": fold_records, "average_classifier_metrics": aggregate},
            "summary": {"selected_metric": "macro_f1", "mean_validation_macro_f1": aggregate.get("macro_f1")},
            "evaluation": {"split": "AE training split 5-fold validation", "model": aggregate, "folds": fold_records, "test_set_used": False},
        }
        if dry_run:
            print(json.dumps(metadata["summary"], sort_keys=True))
        else:
            experiment_id = tracker.build_experiment_id("ae_exp_lae32_braingnn", metadata["model_params"], metadata["training_params"], metadata["data_params"], identity={"ae_model": metadata["ae_model_params"], "ae_training": metadata["ae_training_params"]})
            metadata["experiment_id"] = experiment_id
            tracker.register_experiment(metadata, histories)
            LOGGER.info("Registered ae_exp %s: mean macro-F1=%.4f", experiment_id, aggregate.get("macro_f1", float("nan")))
