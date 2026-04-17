"""Training results manager for experiment metadata/history."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from .filters import apply_filters
from .index import append_index_entry, build_index_entry, find_index_entry, read_index, write_index
from .storage import read_json, write_json_atomic


class TrainingResultsManager:
    """Stores experiment records and provides query primitives."""

    def __init__(self, results_dir: str | Path = "results") -> None:
        self.results_dir = Path(results_dir)
        self.base_dir = self.results_dir.parent
        self.experiments_dir = self.results_dir / "experiments"
        self.index_path = self.results_dir / "index.jsonl"

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

    def register_experiment(self, metadata: dict, history: dict) -> str:
        payload = deepcopy(metadata)

        experiment_id = payload.get("experiment_id") or self._build_experiment_id()
        created_at = payload.get("created_at") or self._now_iso()

        payload["schema_version"] = int(payload.get("schema_version", 1))
        payload["experiment_id"] = experiment_id
        payload["created_at"] = created_at
        payload["status"] = payload.get("status", "completed")
        payload["model_type"] = payload.get("model_type", "unknown")

        summary = payload.get("summary", {})
        derived_summary = self._derive_summary_from_history(history)
        for key, value in derived_summary.items():
            if summary.get(key) is None:
                summary[key] = value
        payload["summary"] = summary

        experiment_dir = self.experiments_dir / experiment_id
        if experiment_dir.exists():
            raise ValueError(f"Experiment already exists: {experiment_id}")
        experiment_dir.mkdir(parents=True, exist_ok=False)

        metadata_path = experiment_dir / "metadata.json"
        history_path = experiment_dir / "history.json"

        payload["history_path"] = self._relative_to_base(history_path)

        write_json_atomic(history_path, history)
        write_json_atomic(metadata_path, payload)

        index_entry = build_index_entry(payload, metadata_path, self.base_dir)
        append_index_entry(self.index_path, index_entry)

        return experiment_id

    def list_experiments(
        self,
        filters: Optional[dict] = None,
        sort_by: str = "created_at",
        ascending: bool = False,
        limit: Optional[int] = None,
    ) -> list[dict]:
        rows = read_index(self.index_path)
        rows = apply_filters(rows, filters)

        rows.sort(
            key=lambda row: (row.get(sort_by) is None, row.get(sort_by)),
            reverse=not ascending,
        )

        if limit is not None:
            rows = rows[:limit]
        return rows

    def get_experiment(self, experiment_id: str) -> dict:
        index_entry = find_index_entry(self.index_path, experiment_id)
        if index_entry is None:
            raise FileNotFoundError(f"Experiment not found in index: {experiment_id}")

        metadata_path = self._resolve_from_base(index_entry["metadata_path"])
        return read_json(metadata_path)

    def get_history(self, experiment_id: str) -> dict:
        metadata = self.get_experiment(experiment_id)
        history_path = metadata.get("history_path")
        if not history_path:
            raise FileNotFoundError(f"Missing history path for experiment: {experiment_id}")

        return read_json(self._resolve_from_base(history_path))

    def set_evaluation_metrics(
        self,
        experiment_id: str,
        model_metrics: Optional[dict] = None,
        pca_metrics: Optional[dict] = None,
        evaluation: Optional[dict] = None,
    ) -> None:
        """Store evaluation metrics on an existing experiment metadata record."""
        index_entry = find_index_entry(self.index_path, experiment_id)
        if index_entry is None:
            raise FileNotFoundError(f"Experiment not found in index: {experiment_id}")

        metadata_path = self._resolve_from_base(index_entry["metadata_path"])
        metadata = read_json(metadata_path)

        if isinstance(evaluation, dict):
            evaluation_payload = deepcopy(evaluation)
        else:
            evaluation_payload = {
                "model": deepcopy(model_metrics) if isinstance(model_metrics, dict) else {},
                "pca": deepcopy(pca_metrics) if isinstance(pca_metrics, dict) else None,
            }
        evaluation_payload["updated_at"] = self._now_iso()
        metadata["evaluation"] = evaluation_payload

        write_json_atomic(metadata_path, metadata)

    def get_evaluation_metrics(self, experiment_id: str) -> dict:
        """Return stored evaluation metrics for an experiment, if present."""
        metadata = self.get_experiment(experiment_id)
        evaluation = metadata.get("evaluation")
        if not isinstance(evaluation, dict):
            return {}
        return evaluation

    def rebuild_index(self) -> None:
        entries: list[dict[str, Any]] = []
        for metadata_path in sorted(self.experiments_dir.glob("*/metadata.json")):
            metadata = read_json(metadata_path)
            entries.append(build_index_entry(metadata, metadata_path, self.base_dir))
        write_index(self.index_path, entries)

    def build_experiment_id(
        self,
        model_type: str,
        model_params: dict,
        training_params: dict,
        data_params: dict,
    ) -> str:
        """Build an experiment id from params plus timestamp."""
        canonical = {
            "model_type": model_type,
            "model_params": model_params,
            "training_params": training_params,
            "data_params": data_params,
        }
        payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]
        ts = datetime.now(timezone.utc).strftime("exp_%Y%m%d_%H%M%S")
        model_slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in model_type).strip("-") or "model"
        return f"{ts}_{model_slug}_{digest}"

    def _resolve_from_base(self, path_str: str) -> Path:
        path = Path(path_str)
        if path.is_absolute():
            return path
        return self.base_dir / path

    def _relative_to_base(self, path: Path) -> str:
        return str(path.relative_to(self.base_dir))

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _build_experiment_id() -> str:
        ts = datetime.now(timezone.utc).strftime("exp_%Y%m%d_%H%M%S")
        return f"{ts}_adhoc"

    @staticmethod
    def _metric_values(history: dict, split: str, metric: str) -> list[float]:
        split_metrics = history.get(split)
        if isinstance(split_metrics, dict):
            values = split_metrics.get(metric, [])
            if isinstance(values, list):
                return [float(v) for v in values]

        values = history.get(f"{split}_{metric}", [])
        if isinstance(values, list):
            return [float(v) for v in values]

        metrics = history.get("metrics")
        if isinstance(metrics, dict):
            values = metrics.get(f"{split}_{metric}", [])
            if isinstance(values, list):
                return [float(v) for v in values]

        return []

    @classmethod
    def _derive_summary_from_history(cls, history: dict) -> dict[str, Any]:
        train_losses = cls._metric_values(history, "train", "loss")
        val_losses = cls._metric_values(history, "val", "loss")

        best_epoch = None
        best_val = None
        if val_losses:
            best_idx = min(range(len(val_losses)), key=val_losses.__getitem__)
            best_epoch = best_idx + 1
            best_val = float(val_losses[best_idx])

        return {
            "num_epochs": max(len(train_losses), len(val_losses)),
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "final_train_loss": float(train_losses[-1]) if train_losses else None,
            "final_val_loss": float(val_losses[-1]) if val_losses else None,
        }
