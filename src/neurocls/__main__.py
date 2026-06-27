from __future__ import annotations

import argparse
import itertools
import json
import logging
import pathlib
import sys
from copy import deepcopy

import numpy as np
import torch

from neuroae.__main__ import (
    _dedupe_param_choices,
    configure_reproducibility,
    load_config,
    resolve_results_dir,
)
from neuroae.utils.dict_utils import deepupdate
from training_tracker import TrainingResultsManager

from .eval import compute_classification_metrics
from .features import build_feature_splits
from .models import create_model
from .train import train_sklearn_model, train_torch_model
from .utils.runtime import (
    build_experiment_summary,
    build_feature_metadata,
    build_signature,
    encode_labels,
    load_adni3_loaders,
    load_artifact,
    load_experiment_context,
    save_artifact,
)


project_path = pathlib.Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger("neurocls")


def _configure_logging(training_config=None):
    training_section = training_config.get("training", {}) if isinstance(training_config, dict) else {}
    level_name = str(training_section.get("log_level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    root_logger = logging.getLogger("neurocls")
    if not root_logger.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            stream=sys.stdout,
        )
    root_logger.setLevel(level)


def _latest_experiment_id(results_dir):
    tracker = TrainingResultsManager(results_dir=results_dir)
    rows = tracker.list_experiments(limit=1)
    if not rows:
        raise ValueError(f"No experiments found in {results_dir}.")
    return rows[0]["experiment_id"]


def load_completed_experiment_signatures(results_dir):
    LOGGER.info("Loading already completed neurocls experiments from %s", results_dir)
    tracker = TrainingResultsManager(results_dir=results_dir)
    entries = tracker.list_experiments()
    needs_index_refresh = any(
        "framework" not in entry or "signature" not in entry or "has_model_evaluation" not in entry
        for entry in entries
    )
    if needs_index_refresh:
        LOGGER.info("Rebuilding legacy results index to cache neurocls lookup fields")
        tracker.rebuild_index()
        entries = tracker.list_experiments()

    completed_signatures = set()
    for entry in entries:
        if entry.get("framework") != "neurocls" or not entry.get("has_model_evaluation"):
            continue
        signature = entry.get("signature")
        if signature:
            completed_signatures.add(signature)
    LOGGER.info("Loaded %d completed neurocls experiment signatures", len(completed_signatures))
    return completed_signatures


def _prepare_runtime(data_dir, data_config, model_config, training_config, num_workers):
    configure_reproducibility(data_config=data_config, training_config=training_config)
    LOGGER.info(
        "Preparing runtime: dataset=%s parcellation=%s parcellation_type=%s merge_groups=%s input_mode=%s",
        data_config.get("data", {}).get("type", "ADNI3"),
        data_config.get("data", {}).get("parcelations"),
        data_config.get("data", {}).get("parcellation_type"),
        data_config.get("data", {}).get("merge_groups"),
        model_config.get("model", {}).get("input_mode"),
    )
    loaders = load_adni3_loaders(data_dir=data_dir, data_config=data_config, num_workers=num_workers)
    feature_payload = build_feature_splits(loaders, model_config, training_config, data_config)
    label_payload = encode_labels(
        feature_payload["train"]["labels"],
        feature_payload["val"]["labels"] if feature_payload["val"] is not None else None,
        feature_payload["test"]["labels"] if feature_payload["test"] is not None else None,
    )
    feature_metadata = build_feature_metadata(feature_payload, label_payload)
    train_split = feature_payload["train"]
    if feature_payload["input_mode"] == "graph":
        LOGGER.info(
            "Prepared graph features: train_samples=%d num_nodes=%d node_feature_dim=%d classes=%s",
            train_split["node_features"].shape[0],
            train_split["node_features"].shape[1],
            train_split["node_features"].shape[2],
            label_payload["classes"],
        )
    else:
        LOGGER.info(
            "Prepared vector features: train_samples=%d feature_dim=%d classes=%s",
            train_split["X"].shape[0],
            train_split["X"].shape[1],
            label_payload["classes"],
        )
    return loaders, feature_payload, label_payload, feature_metadata


def _train_and_register(data_dir, data_config, model_config, training_config, device, results_dir, num_workers=0):
    loaders, feature_payload, label_payload, feature_metadata = _prepare_runtime(
        data_dir, data_config, model_config, training_config, num_workers
    )
    runtime = create_model(model_config, feature_metadata["input_shape"], len(label_payload["classes"]))
    model_name = runtime["name"]
    dry_run = bool(training_config.get("training", {}).get("dry_run", False))
    LOGGER.info(
        "Training model: name=%s family=%s dry_run=%s",
        model_name,
        runtime["family"],
        dry_run,
    )

    if runtime["family"] == "sklearn":
        trained_model, history, train_metrics, val_metrics = train_sklearn_model(
            runtime["model"], feature_payload, label_payload
        )
    else:
        trained_model, history, train_metrics, val_metrics = train_torch_model(
            runtime["model"], runtime["family"], feature_payload, label_payload, training_config, device
        )

    tracker = None if dry_run else TrainingResultsManager(results_dir=results_dir)
    experiment_id = None
    artifact_path = None
    if not dry_run:
        experiment_id = tracker.build_experiment_id(
            model_type=model_name,
            model_params=deepcopy(model_config.get("model", {})),
            training_params=deepcopy(training_config.get("training", {})),
            data_params=deepcopy(data_config),
        )
        artifact_path = pathlib.Path(training_config["training"]["save_dir"]) / f"{experiment_id}_model.pt"
        artifact_payload = {
            "family": runtime["family"],
            "model_name": model_name,
            "model_config": deepcopy(model_config),
            "training_config": deepcopy(training_config),
            "data_config": deepcopy(data_config),
            "feature_metadata": feature_metadata,
            "label_classes": list(label_payload["classes"]),
            "scaler": feature_payload["scaler"],
        }
        if runtime["family"] == "sklearn":
            artifact_payload["model_object"] = trained_model
        else:
            artifact_payload["state_dict"] = trained_model.state_dict()
        save_artifact(artifact_path, artifact_payload)
        LOGGER.info("Saved model artifact: %s", artifact_path)

        metadata = {
            "experiment_id": experiment_id,
            "status": "completed",
            "framework": "neurocls",
            "signature": build_signature(model_name, model_config, training_config, data_config),
            "model_type": model_name,
            "summary": build_experiment_summary(
                train_metrics,
                val_metrics,
                classifier_metric=training_config["training"].get("classifier_metric", "macro_f1"),
            ),
            "model_params": deepcopy(model_config.get("model", {})),
            "training_params": deepcopy(training_config.get("training", {})),
            "data_params": deepcopy(data_config),
            "feature_metadata": deepcopy(feature_metadata),
            "label_classes": list(label_payload["classes"]),
            "artifacts": {"model_path": str(artifact_path)},
        }
        tracker.register_experiment(metadata=metadata, history=history)
        LOGGER.info(
            "Registered experiment: id=%s train_accuracy=%.4f train_macro_f1=%.4f%s",
            experiment_id,
            train_metrics["accuracy"],
            train_metrics["macro_f1"],
            (
                f" val_accuracy={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}"
                if val_metrics
                else ""
            ),
        )
    return {
        "experiment_id": experiment_id,
        "artifact_path": artifact_path,
        "trained_model": trained_model,
        "runtime_family": runtime["family"],
        "model_name": model_name,
        "history": history,
        "loaders": loaders,
        "feature_payload": feature_payload,
        "label_payload": label_payload,
        "feature_metadata": feature_metadata,
    }


def _predict_for_split(runtime_family, model, split_payload, device):
    if runtime_family == "sklearn":
        X = split_payload["X"]
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
        return y_pred, y_proba

    model.eval()
    with torch.no_grad():
        if runtime_family == "torch_graph":
            node_features = torch.as_tensor(split_payload["node_features"], dtype=torch.float32, device=device)
            adjacency = torch.as_tensor(split_payload["adjacency"], dtype=torch.float32, device=device)
            logits = model(node_features, adjacency)["logits"]
        else:
            X = torch.as_tensor(split_payload["X"], dtype=torch.float32, device=device)
            logits = model(X)
        y_proba = torch.softmax(logits, dim=1).cpu().numpy()
        y_pred = np.argmax(y_proba, axis=1)
    return y_pred, y_proba


def _evaluate_experiment(results_dir, experiment_id, data_dir, device, num_workers=0, delete_model_after_eval=True):
    tracker, metadata = load_experiment_context(results_dir, experiment_id)
    LOGGER.info("Evaluating experiment: %s", experiment_id)
    data_config = deepcopy(metadata["data_params"])
    model_config = {"model": deepcopy(metadata["model_params"])}
    training_config = {"training": deepcopy(metadata["training_params"])}
    loaders, feature_payload, label_payload, feature_metadata = _prepare_runtime(
        data_dir, data_config, model_config, training_config, num_workers
    )
    artifact = load_artifact(metadata["artifacts"]["model_path"])
    runtime = create_model(model_config, feature_metadata["input_shape"], len(label_payload["classes"]))
    if artifact["family"] == "sklearn":
        model = artifact["model_object"]
    else:
        model = runtime["model"].to(device)
        model.load_state_dict(artifact["state_dict"])
    y_pred, y_proba = _predict_for_split(artifact["family"], model, feature_payload["test"], device)
    metrics = compute_classification_metrics(
        label_payload["test"],
        y_pred,
        label_payload["classes"],
        y_proba=y_proba,
    )
    evaluation = {
        "model": metrics,
        "split": "test",
        "updated_from": "neurocls",
    }
    tracker.set_evaluation_metrics(experiment_id=experiment_id, evaluation=evaluation)
    LOGGER.info(
        "Stored evaluation: experiment_id=%s accuracy=%.4f balanced_accuracy=%.4f macro_f1=%.4f",
        experiment_id,
        metrics["accuracy"],
        metrics["balanced_accuracy"],
        metrics["macro_f1"],
    )
    model_path = pathlib.Path(metadata["artifacts"]["model_path"])
    if delete_model_after_eval:
        if model_path.exists():
            model_path.unlink()
            LOGGER.info("Deleted model artifact after evaluation: %s", model_path)
        else:
            LOGGER.info("Model artifact already missing, nothing to delete: %s", model_path)
    return evaluation


def _run_single(data_dir, data_config, model_config, training_config, device, results_dir, num_workers=0, evaluate=False):
    result = _train_and_register(
        data_dir=data_dir,
        data_config=data_config,
        model_config=model_config,
        training_config=training_config,
        device=device,
        results_dir=results_dir,
        num_workers=num_workers,
    )
    if training_config.get("training", {}).get("dry_run", False):
        return result
    if evaluate:
        _evaluate_experiment(
            results_dir=results_dir,
            experiment_id=result["experiment_id"],
            data_dir=data_dir,
            device=device,
            num_workers=num_workers,
            delete_model_after_eval=bool(training_config.get("training", {}).get("delete_model_after_eval", True)),
        )
    return result


def _set_var_value(node, name, value):
    if "." in name and isinstance(node, dict) and name.split(".")[0] in node:
        node_name = name.split(".")[0]
        child_name = ".".join(name.split(".")[1:])
        _set_var_value(node[node_name], child_name, value)
    elif isinstance(node, dict) and "." not in name:
        if isinstance(value, dict) and name in node and isinstance(node[name], dict):
            deepupdate(node[name], value)
        else:
            node[name] = value


def _collect_vars(node, name=""):
    if isinstance(node, dict):
        collected = {}
        for key, value in node.items():
            deepupdate(collected, _collect_vars(value, f"{name}.{key}" if name else key))
        return collected
    return {name: node}


def _validate_shard_args(shard_index, num_shards):
    if (shard_index is None) != (num_shards is None):
        raise ValueError("--shard-index and --num-shards must be provided together.")
    if shard_index is None:
        return
    if num_shards < 1:
        raise ValueError("--num-shards must be at least 1.")
    if shard_index < 1 or shard_index > num_shards:
        raise ValueError("--shard-index must be between 1 and --num-shards inclusive.")


def _is_signature_in_shard(signature, shard_index, num_shards):
    if shard_index is None or num_shards is None:
        return True
    return (int(signature, 16) % num_shards) + 1 == shard_index


def main():
    parser = argparse.ArgumentParser(description="Train classification models on ADNI3 data.")
    parser.add_argument("-m", "--mode", type=str, default="train", choices=["train", "eval", "load", "exp"])
    parser.add_argument("-d", "--data-dir", type=pathlib.Path, default=project_path / "data")
    parser.add_argument("--data-config", type=pathlib.Path, default=project_path / "cls_config" / "data.yml")
    parser.add_argument("--model-config", type=pathlib.Path, default=project_path / "cls_config" / "model.yml")
    parser.add_argument("--training-config", type=pathlib.Path, default=project_path / "cls_config" / "training.yml")
    parser.add_argument("--experiment-config", type=pathlib.Path, default=project_path / "cls_config" / "experiments.yml")
    parser.add_argument("--device", type=str, default="mps", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--exp-name", type=str)
    parser.add_argument("--num-parallel-experiments", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--results-dir-name", type=str, default="results")
    parser.add_argument("--max-experiment-combinations", type=int, default=100000)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    _validate_shard_args(args.shard_index, args.num_shards)

    results_dir = resolve_results_dir(project_path, args.results_dir_name)

    if args.mode == "load":
        data_config = load_config(args.data_config)
        training_config = load_config(args.training_config)
        _configure_logging(training_config)
        configure_reproducibility(data_config=data_config, training_config=training_config)
        loaders = load_adni3_loaders(args.data_dir, data_config, args.num_workers)
        sample_batch = next(iter(loaders["train_loader"]))
        batch_data, batch_labels = sample_batch
        print(f"Batch shape: {batch_data.shape}")
        print(f"Batch labels preview: {batch_labels[:5]}")
        return

    if args.mode == "train":
        data_config = load_config(args.data_config)
        model_config = load_config(args.model_config)
        training_config = load_config(args.training_config)
        if args.dry_run:
            training_config.setdefault("training", {})
            training_config["training"]["dry_run"] = True
        _configure_logging(training_config)
        _run_single(
            data_dir=args.data_dir,
            data_config=data_config,
            model_config=model_config,
            training_config=training_config,
            device=args.device,
            results_dir=results_dir,
            num_workers=args.num_workers,
            evaluate=not args.dry_run,
        )
        return

    if args.mode == "eval":
        training_config = load_config(args.training_config)
        _configure_logging(training_config)
        experiment_id = args.exp_name or _latest_experiment_id(results_dir)
        evaluation = _evaluate_experiment(
            results_dir=results_dir,
            experiment_id=experiment_id,
            data_dir=args.data_dir,
            device=args.device,
            num_workers=args.num_workers,
            delete_model_after_eval=bool(training_config.get("training", {}).get("delete_model_after_eval", True)),
        )
        print(json.dumps(evaluation, sort_keys=True, default=str))
        return

    if args.mode == "exp":
        exp_config = load_config(args.experiment_config)
        _configure_logging(exp_config.get("default", {}))
        completed_signatures = set()
        if not args.dry_run:
            completed_signatures = load_completed_experiment_signatures(results_dir)
        seen_signatures = set()
        run_queue = []
        skipped_duplicates = 0
        skipped_completed = 0
        skipped_shard = 0

        for set_name, config_set in exp_config.items():
            if set_name == "default":
                continue
            data_config = deepcopy(exp_config["default"]["data"])
            model_config = {"model": deepcopy(exp_config["default"]["model"])}
            training_config = {"training": deepcopy(exp_config["default"]["training"])}
            if args.dry_run:
                training_config["training"]["dry_run"] = True

            if "data" in config_set["static_params"]:
                deepupdate(data_config, config_set["static_params"]["data"])
            if "model" in config_set["static_params"]:
                deepupdate(model_config["model"], config_set["static_params"]["model"])
            if "training" in config_set["static_params"]:
                deepupdate(training_config["training"], config_set["static_params"]["training"])

            vars_map = _collect_vars(config_set["exp_params"], "")
            keys = list(vars_map.keys())
            values = [_dedupe_param_choices(list(vars_map[key])) for key in keys]
            combinations = list(itertools.product(*values)) if values else [()]
            if len(combinations) > args.max_experiment_combinations:
                raise ValueError(
                    f"{set_name} expands to {len(combinations)} combinations, "
                    f"which exceeds --max-experiment-combinations={args.max_experiment_combinations}."
                )

            for combo in combinations:
                collection = dict(zip(keys, combo)) if keys else {}
                dc = deepcopy(data_config)
                mc = deepcopy(model_config)
                tc = deepcopy(training_config)
                for var_name, value in collection.items():
                    _set_var_value({"data": dc, "model": mc["model"], "training": tc["training"]}, var_name, value)
                signature = build_signature(mc["model"]["name"], mc, tc, dc)
                if signature in seen_signatures:
                    LOGGER.info("Skipping duplicate candidate in current sweep: set=%s signature=%s", set_name, signature)
                    skipped_duplicates += 1
                    continue
                seen_signatures.add(signature)
                if not _is_signature_in_shard(signature, args.shard_index, args.num_shards):
                    skipped_shard += 1
                    continue
                if not args.dry_run and signature in completed_signatures:
                    skipped_completed += 1
                    continue
                run_queue.append(
                    {
                        "set_name": set_name,
                        "signature": signature,
                        "data_config": dc,
                        "model_config": mc,
                        "training_config": tc,
                    }
                )

        total_candidates = len(run_queue) + skipped_duplicates + skipped_completed
        total_candidates += skipped_shard
        LOGGER.info(
            "Prepared experiment queue: total=%d runnable=%d skipped=%d skipped_duplicates=%d skipped_completed=%d skipped_shard=%d%s",
            total_candidates,
            len(run_queue),
            skipped_duplicates + skipped_completed + skipped_shard,
            skipped_duplicates,
            skipped_completed,
            skipped_shard,
            (
                f" shard={args.shard_index}/{args.num_shards}"
                if args.shard_index is not None and args.num_shards is not None
                else ""
            ),
        )

        for run_idx, queued in enumerate(run_queue, start=1):
            LOGGER.info(
                "Running experiment %d/%d: set=%s model=%s input_mode=%s signature=%s",
                run_idx,
                len(run_queue),
                queued["set_name"],
                queued["model_config"]["model"]["name"],
                queued["model_config"]["model"].get("input_mode"),
                queued["signature"],
            )
            _run_single(
                    data_dir=args.data_dir,
                    data_config=queued["data_config"],
                    model_config=queued["model_config"],
                    training_config=queued["training_config"],
                    device=args.device,
                    results_dir=results_dir,
                    num_workers=args.num_workers,
                    evaluate=not args.dry_run,
                )
        return


if __name__ == "__main__":
    main()
