"""Streamlit UI for browsing tracked training experiments."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yaml

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None

from training_tracker import TrainingResultsManager


def _default_results_dir() -> Path:
    env_value = os.environ.get("TRAINING_TRACKER_RESULTS_DIR")
    if env_value:
        return Path(env_value)
    return Path("results")


def _default_config_dir() -> Path:
    env_value = os.environ.get("TRAINING_TRACKER_CONFIG_DIR")
    if env_value:
        return Path(env_value)
    return Path("config")


def _default_experiments_config_path() -> Path:
    return Path("config") / "experiments.yml"


def _default_index_path(results_dir: Path) -> Path:
    env_value = os.environ.get("TRAINING_TRACKER_INDEX_FILE")
    if env_value:
        return Path(env_value)
    return results_dir / "index.jsonl"


def _results_fingerprint(index_path: Path, results_dir: Path) -> int:
    """Return a lightweight cache key that changes when tracker files change."""
    candidates = [index_path]
    candidates.extend(results_dir.glob("experiments/*/metadata.json"))
    candidates.extend(results_dir.glob("experiments/*/history.json"))

    latest_mtime_ns = 0
    for file_path in candidates:
        if file_path.exists():
            latest_mtime_ns = max(latest_mtime_ns, file_path.stat().st_mtime_ns)
    return latest_mtime_ns


@st.cache_data(show_spinner=False)
def _load_rows_cached(
    results_dir_str: str,
    index_path_str: str,
    fingerprint: int,
    score_schema_version: int,
) -> list[dict]:
    # `fingerprint` is only used for cache invalidation.
    _ = fingerprint
    _ = score_schema_version
    manager = TrainingResultsManager(results_dir=Path(results_dir_str))
    manager.index_path = Path(index_path_str)
    rows = manager.list_experiments(
        sort_by="created_at",
        ascending=False,
        limit=None,
    )
    rows = _attach_significance(rows, manager, metadata_cache=None)
    for row in rows:
        category = _classify_vs_pca(row)
        row["pca_class"] = _classification_label(category)
    _add_scores(rows)
    return rows


def _history_to_frame(history: dict) -> pd.DataFrame:
    if "metrics" in history:
        metrics = history["metrics"]
    elif isinstance(history.get("train"), dict) or isinstance(history.get("val"), dict):
        metrics = {}
        for split in ("train", "val"):
            split_metrics = history.get(split)
            if not isinstance(split_metrics, dict):
                continue
            for metric_name, values in split_metrics.items():
                metrics[f"{split}_{metric_name}"] = values
    else:
        metrics = history

    if not metrics:
        return pd.DataFrame()

    rows = []
    num_epochs = max((len(values) for values in metrics.values() if isinstance(values, list)), default=0)
    for epoch in range(1, num_epochs + 1):
        row = {"epoch": epoch}
        for metric_name, values in metrics.items():
            if isinstance(values, list):
                row[metric_name] = values[epoch - 1] if epoch - 1 < len(values) else None
        rows.append(row)
    return pd.DataFrame(rows)


def _first_existing(columns: list[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _render_pair_plot(
    container,
    history_df: pd.DataFrame,
    title: str,
    train_candidates: list[str],
    val_candidates: list[str],
    pca_reference: float | None = None,
) -> None:
    cols = list(history_df.columns)
    train_col = _first_existing(cols, train_candidates)
    val_col = _first_existing(cols, val_candidates)

    with container:
        st.markdown(f"**{title}**")
        if train_col is None and val_col is None:
            st.info("Metric not available.")
            return

        plot_cols = ["epoch"]
        rename_map: dict[str, str] = {}
        colors: list[str] = []
        if train_col is not None:
            plot_cols.append(train_col)
            rename_map[train_col] = "train"
            colors.append("#1f77b4")
        if val_col is not None:
            plot_cols.append(val_col)
            rename_map[val_col] = "val"
            colors.append("#ff7f0e")

        chart_df = history_df[plot_cols].rename(columns=rename_map).set_index("epoch")
        if pca_reference is not None and val_col is not None:
            chart_df["PCA MSE"] = float(pca_reference)
            colors.append("#d62728")
        st.line_chart(chart_df, color=colors)


def _available_metric_suffixes(columns: list[str]) -> list[str]:
    suffixes = set()
    for col in columns:
        if col.startswith("train_"):
            suffixes.add(col[len("train_"):])
        elif col.startswith("val_"):
            suffixes.add(col[len("val_"):])
    return sorted(suffixes, key=lambda metric: (metric != "loss", metric))


def _metric_title(metric_suffix: str) -> str:
    if metric_suffix == "loss":
        return "Total Loss"
    return metric_suffix.replace("_", " ").title()


def _is_recon_metric(metric_suffix: str) -> bool:
    lowered = metric_suffix.lower()
    return "recon" in lowered or "repro" in lowered


def _to_float(value) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _lighten_hex(hex_color: str, factor: float = 0.45) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def _short_experiment_id(experiment_id: str, keep: int = 8) -> str:
    tail = experiment_id.split("_")[-1]
    return tail if len(tail) <= keep else tail[:keep]


def _build_compare_labels(selected_rows: list[dict]) -> list[str]:
    """
    Build labels from parameters that differ across selected experiments.
    Falls back to model + short experiment id if needed.
    """
    fields = [
        ("model_type", "model"),
        ("latent_dim", "latent"),
        ("model_hidden_dim", "hidden"),
        ("training_beta", "beta"),
        ("data_flatten", "flatten"),
        ("data_transpose", "transpose"),
        ("data_timepoints_as_samples", "timepts"),
        ("data_fc_input", "fc_input"),
    ]

    varying_fields: list[tuple[str, str]] = []
    for key, alias in fields:
        values = {_encode_group_value(row.get(key)) for row in selected_rows}
        if len(values) > 1:
            varying_fields.append((key, alias))

    labels: list[str] = []
    for row in selected_rows:
        parts: list[str] = []
        for key, alias in varying_fields:
            value = row.get(key)
            if isinstance(value, bool):
                value_text = "yes" if value else "no"
            elif value is None:
                value_text = "NA"
            else:
                value_text = str(value)
            parts.append(f"{alias}={value_text}")

        if parts:
            label = " | ".join(parts)
        else:
            model_name = str(row.get("model_type", "model"))
            exp_id = str(row.get("experiment_id", "exp"))
            label = f"model={model_name} | id={_short_experiment_id(exp_id)}"
        labels.append(label)

    # Ensure uniqueness in case two experiments share identical varying params.
    counts: dict[str, int] = {}
    unique_labels: list[str] = []
    for label, row in zip(labels, selected_rows):
        counts[label] = counts.get(label, 0) + 1
        if counts[label] == 1 and labels.count(label) == 1:
            unique_labels.append(label)
            continue
        exp_id = str(row.get("experiment_id", "exp"))
        unique_labels.append(f"{label} | id={_short_experiment_id(exp_id)}")
    return unique_labels


def _build_compare_param_details(selected_rows: list[dict]) -> list[str]:
    """Build hover details from varying parameters across selected experiments."""
    fields = [
        ("model_type", "model"),
        ("latent_dim", "latent"),
        ("model_hidden_dim", "hidden"),
        ("training_beta", "beta"),
        ("data_flatten", "flatten"),
        ("data_transpose", "transpose"),
        ("data_timepoints_as_samples", "timepts"),
        ("data_fc_input", "fc_input"),
    ]

    varying_fields: list[tuple[str, str]] = []
    for key, alias in fields:
        values = {_encode_group_value(row.get(key)) for row in selected_rows}
        if len(values) > 1:
            varying_fields.append((key, alias))

    details: list[str] = []
    for row in selected_rows:
        parts = []
        for key, alias in varying_fields:
            value = row.get(key)
            if isinstance(value, bool):
                value_text = "true" if value else "false"
            elif value is None:
                value_text = "NA"
            else:
                value_text = str(value)
            parts.append(f"{alias}={value_text}")
        details.append(" | ".join(parts) if parts else "No varying params")
    return details


def _to_display_value(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ": "))
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def _render_kv_section(title: str, data: dict | None) -> None:
    st.markdown(f"**{title}**")
    if not data:
        st.caption("No data")
        return
    rows = [{"Field": key, "Value": _to_display_value(value)} for key, value in data.items()]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, default_flow_style=False)


def _extract_group_metric_rows(groups_payload: dict | None) -> pd.DataFrame:
    if not isinstance(groups_payload, dict):
        return pd.DataFrame()

    metric_specs = [
        ("mse", "MSE"),
        ("fc_preservation", "FC Preservation"),
        ("silhouette", "Silhouette"),
        ("logreg_accuracy", "LogReg Accuracy"),
        ("swfcd_pearson", "SWFCD Pearson"),
        ("swfcd_mad", "SWFCD MAD"),
        ("swfcd_rmse", "SWFCD RMSE"),
    ]
    rows: list[dict[str, object]] = []
    for group_name in sorted(groups_payload):
        group_payload = groups_payload.get(group_name)
        if not isinstance(group_payload, dict):
            continue
        model_metrics = group_payload.get("model")
        if not isinstance(model_metrics, dict):
            continue
        row = {"group": group_name}
        for metric_key, metric_title in metric_specs:
            row[metric_title] = _to_float(model_metrics.get(metric_key))
        rows.append(row)
    return pd.DataFrame(rows)


def _apply_metric_bundle_to_row(
    row: dict,
    model_eval: dict | None,
    pca_eval: dict | None,
    comparison_eval: dict | None,
) -> None:
    if isinstance(model_eval, dict):
        row["test_mse"] = _to_float(model_eval.get("mse"))
        row["test_fc_preservation"] = _to_float(model_eval.get("fc_preservation"))
        row["test_silhouette"] = _to_float(model_eval.get("silhouette"))
        row["test_logreg_accuracy"] = _to_float(model_eval.get("logreg_accuracy"))
        row["test_swfcd_pearson"] = _to_float(model_eval.get("swfcd_pearson"))
        row["test_swfcd_mad"] = _to_float(model_eval.get("swfcd_mad"))
        row["test_swfcd_rmse"] = _to_float(model_eval.get("swfcd_rmse"))
    else:
        row["test_mse"] = None
        row["test_fc_preservation"] = None
        row["test_silhouette"] = None
        row["test_logreg_accuracy"] = None
        row["test_swfcd_pearson"] = None
        row["test_swfcd_mad"] = None
        row["test_swfcd_rmse"] = None

    if isinstance(pca_eval, dict):
        row["pca_mse"] = _to_float(pca_eval.get("mse"))
        row["pca_fc_preservation"] = _to_float(pca_eval.get("fc_preservation"))
        row["pca_silhouette"] = _to_float(pca_eval.get("silhouette"))
        row["pca_logreg_accuracy"] = _to_float(pca_eval.get("logreg_accuracy"))
        row["pca_swfcd_pearson"] = _to_float(pca_eval.get("swfcd_pearson"))
        row["pca_swfcd_rmse"] = _to_float(pca_eval.get("swfcd_rmse"))
    else:
        row["pca_mse"] = None
        row["pca_fc_preservation"] = None
        row["pca_silhouette"] = None
        row["pca_logreg_accuracy"] = None
        row["pca_swfcd_pearson"] = None
        row["pca_swfcd_rmse"] = None

    if isinstance(comparison_eval, dict):
        row["delta_mse"] = _to_float(comparison_eval.get("mse_delta_model_minus_pca"))
        row["delta_fc_preservation"] = _to_float(comparison_eval.get("fc_delta_model_minus_pca"))
        row["delta_silhouette"] = _to_float(comparison_eval.get("silhouette_delta_model_minus_pca"))
        row["delta_logreg_accuracy"] = _to_float(comparison_eval.get("logreg_delta_model_minus_pca"))
    else:
        row["delta_mse"] = None
        row["delta_fc_preservation"] = None
        row["delta_silhouette"] = None
        row["delta_logreg_accuracy"] = None

    if row["delta_mse"] is None and row["test_mse"] is not None and row["pca_mse"] is not None:
        row["delta_mse"] = row["test_mse"] - row["pca_mse"]
    if row["delta_fc_preservation"] is None and row["test_fc_preservation"] is not None and row["pca_fc_preservation"] is not None:
        row["delta_fc_preservation"] = row["test_fc_preservation"] - row["pca_fc_preservation"]
    if row["delta_silhouette"] is None and row["test_silhouette"] is not None and row["pca_silhouette"] is not None:
        row["delta_silhouette"] = row["test_silhouette"] - row["pca_silhouette"]
    if row["delta_logreg_accuracy"] is None and row["test_logreg_accuracy"] is not None and row["pca_logreg_accuracy"] is not None:
        row["delta_logreg_accuracy"] = row["test_logreg_accuracy"] - row["pca_logreg_accuracy"]


def _overwrite_configs_from_metadata(metadata: dict, config_dir: Path) -> None:
    model_params = metadata.get("model_params")
    training_params = metadata.get("training_params")
    data_params = metadata.get("data_params")

    if not isinstance(model_params, dict):
        raise ValueError("model_params missing or invalid in metadata")
    if not isinstance(training_params, dict):
        raise ValueError("training_params missing or invalid in metadata")
    if not isinstance(data_params, dict):
        raise ValueError("data_params missing or invalid in metadata")

    _write_yaml(config_dir / "model.yml", {"model": model_params})
    _write_yaml(config_dir / "training.yml", {"training": training_params})
    _write_yaml(config_dir / "data.yml", data_params)


def _render_evaluation_tab(
    metadata: dict,
    best_fc_logreg_row: dict | None,
) -> None:
    evaluation = metadata.get("evaluation")
    if not isinstance(evaluation, dict):
        st.info("No evaluation metrics available for this experiment.")
        return

    selected_model_metrics = evaluation.get("model")
    if not isinstance(selected_model_metrics, dict):
        st.info("Evaluation metrics are incomplete (model missing).")
        return

    updated_at = evaluation.get("updated_at")
    if updated_at:
        st.caption(f"Evaluation updated: {updated_at}")
    st.caption(f"Evaluation scope: {evaluation.get('scope', 'combined')}")

    metric_row_key_map = {
        "mse": "test_mse",
        "fc_preservation": "test_fc_preservation",
        "silhouette": "test_silhouette",
        "logreg_accuracy": "test_logreg_accuracy",
        "swfcd_pearson": "test_swfcd_pearson",
        "swfcd_mad": "test_swfcd_mad",
        "swfcd_rmse": "test_swfcd_rmse",
    }

    selected_id = str(metadata.get("experiment_id", "selected"))
    selected_label = f"selected_{_short_experiment_id(selected_id)}"
    best_fc_logreg_label = (
        f"best_fc_logreg_{_short_experiment_id(str(best_fc_logreg_row.get('experiment_id', 'na')))}"
        if best_fc_logreg_row is not None
        else "best_fc_logreg"
    )
    numeric_metric_keys = []
    for metric_key in sorted(metric_row_key_map.keys()):
        selected_value = _to_float(selected_model_metrics.get(metric_key))
        best_fc_logreg_value = (
            _to_float(best_fc_logreg_row.get(metric_row_key_map[metric_key]))
            if best_fc_logreg_row is not None
            else None
        )
        if selected_value is None and best_fc_logreg_value is None:
            continue
        numeric_metric_keys.append(metric_key)

    if not numeric_metric_keys:
        st.info("No comparable numeric metrics found.")
        return

    for idx in range(0, len(numeric_metric_keys), 4):
        row_metrics = numeric_metric_keys[idx:idx + 4]
        row_cols = st.columns(4)
        for col, metric_name in zip(row_cols, row_metrics):
            selected_value = _to_float(selected_model_metrics.get(metric_name))
            best_fc_logreg_value = (
                _to_float(best_fc_logreg_row.get(metric_row_key_map[metric_name]))
                if best_fc_logreg_row is not None
                else None
            )
            metric_label = metric_name.replace('_', ' ').title()
            labels = []
            values = []
            if selected_value is not None:
                labels.append(selected_label)
                values.append(selected_value)
            if best_fc_logreg_value is not None:
                labels.append(best_fc_logreg_label)
                values.append(best_fc_logreg_value)
            if not values:
                continue
            metric_df = pd.DataFrame(
                {
                    "parameter_value": labels,
                    "avg_metric": values,
                }
            )
            with col:
                st.bar_chart(
                    metric_df,
                    x="parameter_value",
                    y="avg_metric",
                    y_label=metric_label,
                    color="parameter_value"
                )

    group_df = _extract_group_metric_rows(evaluation.get("groups"))
    if not group_df.empty:
        st.markdown("**Per-group metrics**")
        st.dataframe(group_df, width="stretch", hide_index=True)


def _attach_significance(
    rows: list[dict],
    manager: TrainingResultsManager,
    metadata_cache: dict[str, dict] | None = None,
) -> list[dict]:
    output: list[dict] = []
    for row in rows:
        row_copy = dict(row)
        experiment_id = str(row_copy.get("experiment_id", ""))
        metadata = (metadata_cache or {}).get(experiment_id)
        if not isinstance(metadata, dict):
            try:
                loaded = manager.get_experiment(experiment_id)
                metadata = loaded if isinstance(loaded, dict) else {}
            except Exception:
                metadata = {}
        model_params = metadata.get("model_params", {}) if isinstance(metadata, dict) else {}
        training_params = metadata.get("training_params", {}) if isinstance(metadata, dict) else {}
        data_params = metadata.get("data_params", {}) if isinstance(metadata, dict) else {}
        row_copy["_model_params"] = model_params if isinstance(model_params, dict) else {}
        row_copy["_training_params"] = training_params if isinstance(training_params, dict) else {}
        row_copy["_data_params"] = data_params if isinstance(data_params, dict) else {}
        significance = _to_float(metadata.get("summary", {}).get("significance"))
        row_copy["significance"] = significance
        evaluation = metadata.get("evaluation", {}) if isinstance(metadata, dict) else {}
        if not isinstance(evaluation, dict):
            evaluation = {}
        model_eval = evaluation.get("model", {})
        pca_eval = evaluation.get("pca", {})
        comparison_eval = evaluation.get("comparison", {})
        row_copy["evaluation_scope"] = str(evaluation.get("scope", "combined"))
        groups_payload = evaluation.get("groups", {})
        row_copy["_evaluation_groups"] = groups_payload if isinstance(groups_payload, dict) else {}
        row_copy["evaluation_group_names"] = sorted(row_copy["_evaluation_groups"].keys())
        _apply_metric_bundle_to_row(row_copy, model_eval, pca_eval, comparison_eval)
        if isinstance(model_params, dict):
            # Standard models expose latent_dim/hidden_dim.
            if row_copy.get("latent_dim") is None:
                row_copy["latent_dim"] = model_params.get("latent_dim")
            if row_copy.get("latent_dim") is None:
                # AutoencoderKL-style naming.
                row_copy["latent_dim"] = model_params.get("latent_channels")

            row_copy["model_hidden_dim"] = model_params.get("hidden_dim")
            if row_copy["model_hidden_dim"] is None:
                # AutoencoderKL-style naming.
                row_copy["model_hidden_dim"] = model_params.get("channels")
        else:
            row_copy["model_hidden_dim"] = None

        data_cfg = data_params.get("data", {}) if isinstance(data_params, dict) else {}
        if not isinstance(data_cfg, dict):
            data_cfg = {}
        row_copy["data_flatten"] = bool(data_cfg.get("flatten", False))
        row_copy["data_transpose"] = bool(data_cfg.get("transpose", False))
        row_copy["data_timepoints_as_samples"] = bool(data_cfg.get("timepoints_as_samples", False))
        row_copy["data_fc_input"] = bool(data_cfg.get("fc_input", False))

        loss_params = training_params.get("loss_params", {}) if isinstance(training_params, dict) else {}
        if not isinstance(loss_params, dict):
            loss_params = {}
        row_copy["training_beta"] = _to_float(loss_params.get("beta"))
        output.append(row_copy)
    return output


def _metric_outcome(model_value: float | None, pca_value: float | None, lower_is_better: bool) -> int | None:
    """
    Compare model vs PCA for a metric.
    Returns:
      1 -> model better
      0 -> around same
     -1 -> model worse
    """
    if model_value is None or pca_value is None:
        return None
    tolerance = max(1e-6, abs(pca_value) * 0.05)
    delta = model_value - pca_value
    if abs(delta) <= tolerance:
        return 0
    if lower_is_better:
        return 1 if model_value < pca_value else -1
    return 1 if model_value > pca_value else -1


def _classify_vs_pca(row: dict) -> str | None:
    outcomes = [
        _metric_outcome(row.get("test_mse"), row.get("pca_mse"), lower_is_better=True),
        _metric_outcome(row.get("test_fc_preservation"), row.get("pca_fc_preservation"), lower_is_better=False),
        _metric_outcome(row.get("test_silhouette"), row.get("pca_silhouette"), lower_is_better=False),
        _metric_outcome(row.get("test_logreg_accuracy"), row.get("pca_logreg_accuracy"), lower_is_better=False),
    ]
    if any(outcome is None for outcome in outcomes):
        return None
    if all(outcome == 1 for outcome in outcomes):
        return "better"
    if all(outcome == -1 for outcome in outcomes):
        return "worse"
    return "same"


def _classification_label(category: str | None) -> str:
    if category == "better":
        return ">PCA"
    if category == "worse":
        return "<PCA"
    if category == "same":
        return "~PCA"
    return "N/A"


def _minmax_normalize(values: list[float | None]) -> list[float | None]:
    valid = [v for v in values if v is not None]
    if not valid:
        return [None for _ in values]
    vmin = min(valid)
    vmax = max(valid)
    if vmax == vmin:
        return [0.0 if v is not None else None for v in values]
    return [((v - vmin) / (vmax - vmin)) if v is not None else None for v in values]


def _add_scores(rows: list[dict]) -> None:
    """Compute aggregate score columns used by the tracker UI."""
    mse_deltas = [_to_float(row.get("delta_mse")) for row in rows]
    fc_deltas = [_to_float(row.get("delta_fc_preservation")) for row in rows]
    sil_deltas = [_to_float(row.get("delta_silhouette")) for row in rows]
    logreg_deltas = [_to_float(row.get("delta_logreg_accuracy")) for row in rows]

    mse_norm = _minmax_normalize(mse_deltas)
    fc_norm = _minmax_normalize(fc_deltas)
    sil_norm = _minmax_normalize(sil_deltas)
    logreg_norm = _minmax_normalize(logreg_deltas)

    for idx, row in enumerate(rows):
        fc_input_enabled = row.get("data_fc_input")
        if isinstance(fc_input_enabled, str):
            fc_input_enabled = fc_input_enabled.strip().lower() == "true"
        fc_component = 0.5 if fc_input_enabled else fc_norm[idx]
        silhouette_component = 0.0
        if (
            mse_norm[idx] is None
            or fc_component is None
            or logreg_norm[idx] is None
        ):
            row["pca_score"] = None
            continue
        row["pca_score"] = (1.0 - mse_norm[idx]) + fc_component + silhouette_component + logreg_norm[idx]

    test_mse = [_to_float(row.get("test_mse")) for row in rows]
    test_fc = [_to_float(row.get("test_fc_preservation")) for row in rows]
    test_sil = [_to_float(row.get("test_silhouette")) for row in rows]
    test_logreg = [_to_float(row.get("test_logreg_accuracy")) for row in rows]
    test_swfcd_pearson = [_to_float(row.get("test_swfcd_pearson")) for row in rows]

    test_mse_norm = _minmax_normalize(test_mse)
    test_fc_norm = _minmax_normalize(test_fc)
    test_sil_norm = _minmax_normalize(test_sil)
    test_logreg_norm = _minmax_normalize(test_logreg)
    test_swfcd_pearson_norm = _minmax_normalize(test_swfcd_pearson)

    for idx, row in enumerate(rows):
        fc_input_enabled = row.get("data_fc_input")
        if isinstance(fc_input_enabled, str):
            fc_input_enabled = fc_input_enabled.strip().lower() == "true"
        mse_component = 0.5 if test_mse_norm[idx] is None else (1.0 - test_mse_norm[idx])
        fc_component = 0.5 if fc_input_enabled else test_fc_norm[idx]
        if fc_component is None:
            fc_component = 0.5
        silhouette_component = 0.0
        logreg_component = 0.5 if test_logreg_norm[idx] is None else test_logreg_norm[idx]
        swfcd_component = 0.5
        if test_swfcd_pearson_norm[idx] is not None:
            swfcd_component = test_swfcd_pearson_norm[idx]
        row["score"] = (
            mse_component
            + fc_component
            + silhouette_component
            + logreg_component
            + swfcd_component
        )
        if test_fc_norm[idx] is None or test_swfcd_pearson_norm[idx] is None:
            row["fc_score"] = None
        else:
            row["fc_score"] = test_fc_norm[idx] + test_swfcd_pearson_norm[idx]
        if test_swfcd_pearson_norm[idx] is None or test_logreg_norm[idx] is None:
            row["swfcd_logreg_score"] = None
        else:
            row["swfcd_logreg_score"] = (
                test_swfcd_pearson_norm[idx]
                + test_logreg_norm[idx]
            )
        if (
            test_fc_norm[idx] is None
            or test_swfcd_pearson_norm[idx] is None
            or test_logreg_norm[idx] is None
        ):
            row["fc_logreg_score"] = None
        else:
            row["fc_logreg_score"] = (
                test_fc_norm[idx]
                + test_swfcd_pearson_norm[idx]
                + test_logreg_norm[idx]
            )


def _value_sort_key(value):
    if isinstance(value, bool):
        return (0, int(value))
    if isinstance(value, (int, float)):
        return (1, float(value))
    return (2, str(value))


def _value_label(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _load_experiments_spec(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _collect_param_paths(node, prefix: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
    paths: set[tuple[str, ...]] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            paths |= _collect_param_paths(value, prefix + (str(key),))
        return paths
    if isinstance(node, list):
        if all(isinstance(item, dict) for item in node):
            for item in node:
                paths |= _collect_param_paths(item, prefix)
            return paths
        paths.add(prefix)
        return paths
    paths.add(prefix)
    return paths


def _collect_leaf_paths_from_dict(node, prefix: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
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


def _derived_parameter_options(rows: list[dict]) -> dict[tuple[str, ...], str]:
    options: dict[tuple[str, ...], str] = {}
    scope_values = {
        _encode_group_value(value)
        for row in rows
        for value in [row.get("evaluation_scope")]
        if value is not None
    }
    if len(scope_values) >= 2:
        options[("evaluation", "scope")] = "evaluation / scope"

    group_values = {
        _encode_group_value(value)
        for row in rows
        for value in _get_param_values_from_row(row, ("evaluation", "group"))
    }
    if len(group_values) >= 2:
        options[("evaluation", "group")] = "evaluation / group"
    return options


def _get_param_values_from_row(row: dict, path: tuple[str, ...]) -> list[object]:
    if not path:
        return []

    if path == ("evaluation", "scope"):
        value = row.get("evaluation_scope")
        return [] if value is None else [value]

    if path == ("evaluation", "group"):
        groups = row.get("evaluation_group_names")
        if isinstance(groups, list) and groups:
            return list(groups)
        return ["all"]

    value = _get_param_value_from_row(row, path)
    return [] if value is None else [value]


def _parameter_options_for_model_type(model_type: str, rows: list[dict]) -> dict[str, str]:
    model_rows = [row for row in rows if str(row.get("model_type", "unknown")) == model_type]
    if not model_rows:
        return {}

    paths: set[tuple[str, ...]] = set()
    for row in model_rows:
        model_params = row.get("_model_params")
        training_params = row.get("_training_params")
        data_params = row.get("_data_params")
        if isinstance(model_params, dict):
            paths |= {("model",) + path for path in _collect_leaf_paths_from_dict(model_params)}
        if isinstance(training_params, dict):
            paths |= {("training",) + path for path in _collect_leaf_paths_from_dict(training_params)}
        if isinstance(data_params, dict):
            paths |= {("data",) + path for path in _collect_leaf_paths_from_dict(data_params)}

    options: dict[str, str] = {}
    for path in sorted(paths):
        values = {
            _encode_group_value(value)
            for row in model_rows
            for value in _get_param_values_from_row(row, path)
        }
        if len(values) < 2:
            continue
        path_key = ".".join(path)
        options[path_key] = " / ".join(path)
    for path, label in _derived_parameter_options(model_rows).items():
        options[".".join(path)] = label
    return options


def _parameter_options_for_rows(
    rows: list[dict],
    exclude_keys: set[str] | None = None,
) -> dict[str, str]:
    exclude_keys = exclude_keys or set()
    paths: set[tuple[str, ...]] = set()
    for row in rows:
        model_params = row.get("_model_params")
        training_params = row.get("_training_params")
        data_params = row.get("_data_params")
        if isinstance(model_params, dict):
            paths |= {("model",) + path for path in _collect_leaf_paths_from_dict(model_params)}
        if isinstance(training_params, dict):
            paths |= {("training",) + path for path in _collect_leaf_paths_from_dict(training_params)}
        if isinstance(data_params, dict):
            paths |= {("data",) + path for path in _collect_leaf_paths_from_dict(data_params)}

    options: dict[str, str] = {}
    for path in sorted(paths):
        path_key = ".".join(path)
        if path_key in exclude_keys:
            continue
        values = {
            _encode_group_value(value)
            for row in rows
            for value in _get_param_values_from_row(row, path)
        }
        if len(values) < 2:
            continue
        options[path_key] = " / ".join(path)
    for path, label in _derived_parameter_options(rows).items():
        path_key = ".".join(path)
        if path_key not in exclude_keys:
            options[path_key] = label
    return options


def _get_param_value_from_row(row: dict, path: tuple[str, ...]):
    if not path:
        return None
    if path == ("evaluation", "scope"):
        return row.get("evaluation_scope")
    if path == ("evaluation", "group"):
        group_names = row.get("evaluation_group_names")
        return group_names if isinstance(group_names, list) and group_names else "all"
    root = path[0]
    if root == "model":
        current = row.get("_model_params")
    elif root == "training":
        current = row.get("_training_params")
    elif root == "data":
        current = row.get("_data_params")
    else:
        return None

    for key in path[1:]:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _encode_group_value(value):
    if isinstance(value, list):
        return tuple(_encode_group_value(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((str(k), _encode_group_value(v)) for k, v in value.items()))
    return value


def _display_group_value(value) -> str:
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, sort_keys=True)
    return _value_label(value)


def _parameter_value_options(rows: list[dict], path: tuple[str, ...]) -> list[object]:
    values = {
        _encode_group_value(value)
        for row in rows
        for value in _get_param_values_from_row(row, path)
    }
    return sorted(values, key=_value_sort_key)


def _rows_matching_param_filters(
    rows: list[dict],
    filters: list[tuple[str, object]],
) -> list[dict]:
    filtered_rows = rows
    for param_key, expected_value in filters:
        param_path = tuple(param_key.split("."))
        filtered_rows = [
            row
            for row in filtered_rows
            if any(
                _encode_group_value(value) == expected_value
                for value in _get_param_values_from_row(row, param_path)
            )
        ]
    return filtered_rows


def _expand_rows_for_parameter(rows: list[dict], path: tuple[str, ...]) -> list[dict]:
    if path != ("evaluation", "group"):
        return list(rows)

    expanded_rows: list[dict] = []
    for row in rows:
        groups_payload = row.get("_evaluation_groups")
        if isinstance(groups_payload, dict) and groups_payload:
            for group_name in sorted(groups_payload):
                group_payload = groups_payload.get(group_name)
                if not isinstance(group_payload, dict):
                    continue
                group_row = dict(row)
                group_row["_comparison_parameter_value"] = group_name
                _apply_metric_bundle_to_row(
                    group_row,
                    group_payload.get("model"),
                    group_payload.get("pca"),
                    group_payload.get("comparison"),
                )
                expanded_rows.append(group_row)
            continue

        combined_row = dict(row)
        combined_row["_comparison_parameter_value"] = "all"
        expanded_rows.append(combined_row)

    return expanded_rows


def _data_combination_label(row: dict) -> str:
    parts: list[str] = []
    if bool(row.get("data_fc_input", False)):
        parts.append("fc_input")
    if bool(row.get("data_flatten", False)):
        parts.append("flattened")
    if bool(row.get("data_transpose", False)):
        parts.append("transposed")
    if bool(row.get("data_timepoints_as_samples", False)):
        parts.append("timepoints_as_samples")
    if not parts:
        return "default"
    return " & ".join(parts)


def _build_compare_markdown_table(
    selected_rows: list[dict],
    include_fc: bool = True,
) -> str:
    headers = ["Experiment"]
    if include_fc:
        headers.append("FC Preservation")
    headers.extend(["LogReg Accuracy", "SWFCD Pearson"])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in selected_rows:
        experiment_id = str(row.get("experiment_id", ""))
        fc_value = _to_float(row.get("test_fc_preservation"))
        logreg_value = _to_float(row.get("test_logreg_accuracy"))
        swfcd_value = _to_float(row.get("test_swfcd_pearson"))
        lines.append(
            "| "
            + " | ".join(
                [value for value in [
                    experiment_id,
                    ("N/A" if fc_value is None else f"{fc_value:.6f}") if include_fc else None,
                    "N/A" if logreg_value is None else f"{logreg_value:.6f}",
                    "N/A" if swfcd_value is None else f"{swfcd_value:.6f}",
                ] if value is not None]
            )
            + " |"
        )
    return "\n".join(lines)


def _build_parameter_compare_markdown_table(
    grouped: dict[object, dict[str, list[float]]],
    include_fc: bool = True,
) -> str:
    headers = ["Parameter value"]
    if include_fc:
        headers.append("FC Preservation")
    headers.extend(["LogReg Accuracy", "SWFCD Pearson"])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    sorted_param_values = sorted(grouped.keys(), key=_value_sort_key)
    for param_value in sorted_param_values:
        fc_values = grouped[param_value].get("test_fc_preservation", [])
        logreg_values = grouped[param_value].get("test_logreg_accuracy", [])
        swfcd_values = grouped[param_value].get("test_swfcd_pearson", [])

        fc_avg = (sum(fc_values) / len(fc_values)) if fc_values else None
        logreg_avg = (sum(logreg_values) / len(logreg_values)) if logreg_values else None
        swfcd_avg = (sum(swfcd_values) / len(swfcd_values)) if swfcd_values else None

        lines.append(
            "| "
            + " | ".join(
                [value for value in [
                    _display_group_value(param_value),
                    ("N/A" if fc_avg is None else f"{fc_avg:.6f}") if include_fc else None,
                    "N/A" if logreg_avg is None else f"{logreg_avg:.6f}",
                    "N/A" if swfcd_avg is None else f"{swfcd_avg:.6f}",
                ] if value is not None]
            )
            + " |"
        )
    return "\n".join(lines)


def _boxplot_summary(values: list[float]) -> dict[str, object] | None:
    cleaned_values = sorted(_to_float(value) for value in values)
    cleaned_values = [value for value in cleaned_values if value is not None]
    if not cleaned_values:
        return None

    series = pd.Series(cleaned_values, dtype=float)
    q1 = float(series.quantile(0.25))
    median = float(series.quantile(0.5))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    non_outliers = [value for value in cleaned_values if lower_fence <= value <= upper_fence]
    whisker_min = min(non_outliers) if non_outliers else cleaned_values[0]
    whisker_max = max(non_outliers) if non_outliers else cleaned_values[-1]
    outliers = [value for value in cleaned_values if value < whisker_min or value > whisker_max]

    summary: dict[str, object] = {
        "min": float(cleaned_values[0]),
        "q1": q1,
        "median": median,
        "q3": q3,
        "max": float(cleaned_values[-1]),
        "whiskerMin": float(whisker_min),
        "whiskerMax": float(whisker_max),
    }
    if outliers:
        summary["outliers"] = [float(value) for value in outliers]
    return summary


def _build_parameter_compare_boxplot_spec(
    grouped: dict[object, dict[str, list[float]]],
    include_fc: bool = True,
) -> str:
    metric_specs = []
    if include_fc:
        metric_specs.append(("test_fc_preservation", "FC Preservation"))
    metric_specs.extend([
        ("test_logreg_accuracy", "LogReg Accuracy"),
        ("test_swfcd_pearson", "SWFCD Pearson"),
    ])
    sorted_param_values = sorted(grouped.keys(), key=_value_sort_key)

    lines = [
        "```chart",
        "type: boxplot",
        "facetByLabel: true",
        "labels: [" + ", ".join(metric_title for _, metric_title in metric_specs) + "]",
        "series:",
    ]

    for param_value in sorted_param_values:
        lines.append(f"    - title: \"{_display_group_value(param_value)}\"")
        lines.append("      data:")
        for metric_key, _metric_title in metric_specs:
            summary = _boxplot_summary(grouped[param_value].get(metric_key, []))
            if summary is None:
                summary = {
                    "min": None,
                    "q1": None,
                    "median": None,
                    "q3": None,
                    "max": None,
                    "whiskerMin": None,
                    "whiskerMax": None,
                }
            lines.append("          - min: " + ("null" if summary["min"] is None else f"{summary['min']:.6f}"))
            lines.append("            q1: " + ("null" if summary["q1"] is None else f"{summary['q1']:.6f}"))
            lines.append("            median: " + ("null" if summary["median"] is None else f"{summary['median']:.6f}"))
            lines.append("            q3: " + ("null" if summary["q3"] is None else f"{summary['q3']:.6f}"))
            lines.append("            max: " + ("null" if summary["max"] is None else f"{summary['max']:.6f}"))
            lines.append("            whiskerMin: " + ("null" if summary["whiskerMin"] is None else f"{summary['whiskerMin']:.6f}"))
            lines.append("            whiskerMax: " + ("null" if summary["whiskerMax"] is None else f"{summary['whiskerMax']:.6f}"))
            if summary.get("outliers"):
                outlier_values = ", ".join(f"{float(value):.6f}" for value in summary["outliers"])
                lines.append(f"            outliers: [{outlier_values}]")

    lines.append("```")
    return "\n".join(lines)


def _build_model_compare_boxplot_spec(grouped: dict[str, dict[str, list[float]]]) -> str:
    metric_specs = [
        ("test_fc_preservation", "FC Preservation"),
        ("test_logreg_accuracy", "LogReg Accuracy"),
        ("test_swfcd_pearson", "SWFCD Pearson"),
    ]
    model_names = list(grouped.keys())

    lines = [
        "```chart",
        "type: boxplot",
        "facetByLabel: true",
        "labels: [" + ", ".join(metric_title for _, metric_title in metric_specs) + "]",
        "series:",
    ]

    for model_name in model_names:
        lines.append(f"    - title: {model_name}")
        lines.append("      data:")
        for metric_key, _metric_title in metric_specs:
            summary = _boxplot_summary(grouped[model_name].get(metric_key, []))
            if summary is None:
                summary = {
                    "min": None,
                    "q1": None,
                    "median": None,
                    "q3": None,
                    "max": None,
                    "whiskerMin": None,
                    "whiskerMax": None,
                }
            lines.append("          - min: " + ("null" if summary["min"] is None else f"{summary['min']:.6f}"))
            lines.append("            q1: " + ("null" if summary["q1"] is None else f"{summary['q1']:.6f}"))
            lines.append("            median: " + ("null" if summary["median"] is None else f"{summary['median']:.6f}"))
            lines.append("            q3: " + ("null" if summary["q3"] is None else f"{summary['q3']:.6f}"))
            lines.append("            max: " + ("null" if summary["max"] is None else f"{summary['max']:.6f}"))
            lines.append("            whiskerMin: " + ("null" if summary["whiskerMin"] is None else f"{summary['whiskerMin']:.6f}"))
            lines.append("            whiskerMax: " + ("null" if summary["whiskerMax"] is None else f"{summary['whiskerMax']:.6f}"))
            if summary.get("outliers"):
                outlier_values = ", ".join(f"{float(value):.6f}" for value in summary["outliers"])
                lines.append(f"            outliers: [{outlier_values}]")

    lines.append("```")
    return "\n".join(lines)


def _build_raincloud_spec(
    grouped: dict[str, dict[str, list[float]]],
    include_fc: bool = True,
) -> dict[str, object]:
    metric_specs: list[tuple[str, str]] = []
    if include_fc:
        metric_specs.append(("test_fc_preservation", "FC Preservation"))
    metric_specs.extend([
        ("test_logreg_accuracy", "LogReg Accuracy"),
        ("test_swfcd_pearson", "SWFCD Pearson"),
    ])

    series = []
    for title, metrics_by_key in grouped.items():
        series.append(
            {
                "title": str(title),
                "itemRadius": 2,
                "data": [
                    [float(value) for value in metrics_by_key.get(metric_key, []) if _to_float(value) is not None]
                    for metric_key, _ in metric_specs
                ],
            }
        )

    return {
        "type": "raincloud",
        "labels": [metric_title for _, metric_title in metric_specs],
        "series": series,
    }


def _save_raincloud_spec(
    grouped: dict[str, dict[str, list[float]]],
    results_dir: Path,
    tab_name: str,
    include_fc: bool = True,
) -> Path:
    plot_data_dir = Path("plot_data")
    plot_data_dir.mkdir(parents=True, exist_ok=True)
    tab_slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in tab_name).strip("_")
    if results_dir.name == "results":
        run_name = results_dir.parent.name or "current"
    else:
        run_name = results_dir.name or "current"
    output_path = plot_data_dir / f"{run_name}_{tab_slug}.yml"
    payload = _build_raincloud_spec(grouped, include_fc=include_fc)
    _write_yaml(output_path, payload)
    return output_path


def _pairwise_pvalue_matrix(
    grouped: dict[str, dict[str, list[float]]],
    metric_key: str,
) -> pd.DataFrame:
    labels = list(grouped.keys())
    matrix_rows: list[list[str]] = []
    for left_label in labels:
        left_values = [_to_float(value) for value in grouped[left_label].get(metric_key, [])]
        left_values = [value for value in left_values if value is not None]
        row_values: list[str] = []
        for right_label in labels:
            right_values = [_to_float(value) for value in grouped[right_label].get(metric_key, [])]
            right_values = [value for value in right_values if value is not None]
            if left_label == right_label:
                row_values.append("1.0000")
            elif len(left_values) < 2 or len(right_values) < 2 or scipy_stats is None:
                row_values.append("N/A")
            else:
                p_value = scipy_stats.ttest_ind(left_values, right_values, equal_var=False, nan_policy="omit").pvalue
                p_value = _to_float(p_value)
                if p_value is None:
                    left_unique = {value for value in left_values}
                    right_unique = {value for value in right_values}
                    if len(left_unique) == 1 and len(right_unique) == 1 and left_unique == right_unique:
                        row_values.append("1.0000")
                    else:
                        row_values.append("N/A")
                else:
                    row_values.append(f"{p_value:.4f}")
        matrix_rows.append(row_values)
    return pd.DataFrame(matrix_rows, index=labels, columns=labels)


def _style_pvalue_matrix(matrix_df: pd.DataFrame):
    def _cell_style(value: object) -> str:
        numeric = _to_float(value)
        if numeric is None:
            return "background-color: #f5f5f5; color: #666666;"
        if numeric < 0.01:
            bg = "#b7e4c7"
        elif numeric < 0.05:
            bg = "#d8f3dc"
        elif numeric < 0.1:
            bg = "#f3f4b1"
        elif 0.1 < numeric < 0.9:
            bg = "#f8edc7"
        else:
            bg = "#f4c7c3"
        return f"background-color: {bg}; color: #303030;"

    return matrix_df.style.applymap(_cell_style)


def _render_pvalue_matrices(grouped: dict[str, dict[str, list[float]]]) -> None:
    st.markdown("**P-value Matrices**")
    metric_specs = [
        ("test_fc_preservation", "FC Preservation"),
        ("test_logreg_accuracy", "LogReg Accuracy"),
        ("test_swfcd_pearson", "SWFCD Pearson"),
    ]
    row_cols = st.columns(3)
    for col, (metric_key, metric_title) in zip(row_cols, metric_specs):
        with col:
            st.markdown(f"**{metric_title}**")
            matrix_df = _pairwise_pvalue_matrix(grouped, metric_key)
            st.dataframe(_style_pvalue_matrix(matrix_df), width="stretch")


def _extract_selected_row(selection) -> int | None:
    """Handle Streamlit dataframe selection payload variants."""
    if selection is None:
        return None

    rows = None
    cells = None
    if isinstance(selection, dict):
        rows = selection.get("rows")
        cells = selection.get("cells")
    else:
        rows = getattr(selection, "rows", None)
        cells = getattr(selection, "cells", None)

    if isinstance(cells, list) and cells:
        cell = cells[0]
        if isinstance(cell, dict):
            row = cell.get("row")
            if isinstance(row, int):
                return row

    if isinstance(rows, list) and rows:
        row = rows[0]
        if isinstance(row, int):
            return row

    return None


def _parse_optional_float(text: str) -> float | None:
    stripped = text.strip()
    if not stripped:
        return None
    return float(stripped)


def _in_range(value, min_value: float | None, max_value: float | None) -> bool:
    numeric = _to_float(value)
    if numeric is None:
        return False
    if min_value is not None and numeric < min_value:
        return False
    if max_value is not None and numeric > max_value:
        return False
    return True


def _bool_match(value: bool, want_true: bool, want_false: bool) -> bool:
    if want_true and want_false:
        return True
    if want_true:
        return value is True
    if want_false:
        return value is False
    return True


def main() -> None:
    st.set_page_config(page_title="Training Tracker", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Training Tracker")
    results_dir = _default_results_dir()
    manager = TrainingResultsManager(results_dir=results_dir)
    index_path = _default_index_path(results_dir)
    manager.index_path = index_path

    rows = _load_rows_cached(
        str(results_dir),
        str(index_path),
        _results_fingerprint(index_path, results_dir),
        7,
    )

    st.caption(f"All experiments: {len(rows)}")
    if not rows:
        st.info("No experiments available.")
        return

    table_df = pd.DataFrame(rows)
    visible_cols = [
        "experiment_id",
        "created_at",
        "model_type",
        "latent_dim",
        "score",
        "swfcd_logreg_score",
        "fc_logreg_score",
        "best_val_loss",
        "test_mse",
        "test_fc_preservation",
        "test_silhouette",
        "test_logreg_accuracy",
        "test_swfcd_pearson",
        "test_swfcd_mad",
        "test_swfcd_rmse",
    ]
    present_cols = [col for col in visible_cols if col in table_df.columns]
    view_options = [
        "All Experiments",
        "Experiment Details",
        "Compare Experiments",
        "Model Parameters",
        "Parameter Comparison",
        "Model Comparison",
        "Data Comparison",
    ]

    if "pending_main_view" in st.session_state:
        st.session_state["main_view"] = st.session_state.pop("pending_main_view")
    if "main_view" not in st.session_state:
        st.session_state["main_view"] = "All Experiments"
    active_view = st.segmented_control(
        "View",
        options=view_options,
        default=st.session_state["main_view"],
        key="main_view",
        label_visibility="collapsed",
    )

    experiment_options = [row["experiment_id"] for row in rows]
    row_by_id = {str(row.get("experiment_id")): row for row in rows}
    if "details_experiment_id" not in st.session_state or st.session_state["details_experiment_id"] not in experiment_options:
        st.session_state["details_experiment_id"] = experiment_options[0]

    if active_view == "All Experiments":
        st.subheader("Experiments")
        st.caption("Data filters")
        filter_cols = st.columns(4)
        with filter_cols[0]:
            st.markdown("**Flatten**")
            flatten_true = st.checkbox("True", value=False, key="allf_flatten_true")
            flatten_false = st.checkbox("False", value=False, key="allf_flatten_false")
        with filter_cols[1]:
            st.markdown("**Transpose**")
            transpose_true = st.checkbox("True", value=False, key="allf_transpose_true")
            transpose_false = st.checkbox("False", value=False, key="allf_transpose_false")
        with filter_cols[2]:
            st.markdown("**Timepoints as samples**")
            tas_true = st.checkbox("True", value=False, key="allf_tas_true")
            tas_false = st.checkbox("False", value=False, key="allf_tas_false")
        with filter_cols[3]:
            st.markdown("**FC input**")
            fc_input_true = st.checkbox("True", value=False, key="allf_fc_input_true")
            fc_input_false = st.checkbox("False", value=False, key="allf_fc_input_false")

        filtered_table_df = table_df[
            table_df.apply(
                lambda row: _bool_match(bool(row.get("data_flatten", False)), flatten_true, flatten_false)
                and _bool_match(bool(row.get("data_transpose", False)), transpose_true, transpose_false)
                and _bool_match(bool(row.get("data_timepoints_as_samples", False)), tas_true, tas_false)
                and _bool_match(bool(row.get("data_fc_input", False)), fc_input_true, fc_input_false),
                axis=1,
            )
        ]
        if filtered_table_df.empty:
            st.info("No experiments match the selected data filters.")
            return

        table_state = st.dataframe(
            filtered_table_df[present_cols],
            width="stretch",
            hide_index=True,
            column_config={
                "score": "score",
                "swfcd_logreg_score": "SWFCD+LogReg score",
                "fc_logreg_score": "FC+LogReg score",
                "test_swfcd_pearson": "test_swfcd_pearson",
                "test_swfcd_mad": "test_swfcd_mad",
                "test_swfcd_rmse": "test_swfcd_rmse",
            },
            on_select="rerun",
            selection_mode="single-cell",
            key="all_experiments_table",
        )
        selected_row = _extract_selected_row(getattr(table_state, "selection", None))
        if selected_row is not None:
            selected_experiment_id = filtered_table_df.iloc[selected_row]["experiment_id"]
            if (
                st.session_state.get("details_experiment_id") != selected_experiment_id
                or st.session_state.get("main_view") != "Experiment Details"
            ):
                st.session_state["details_experiment_id"] = selected_experiment_id
                st.session_state["pending_main_view"] = "Experiment Details"

    elif active_view == "Experiment Details":
        experiment_id = st.selectbox(
            "Experiment",
            options=experiment_options,
            index=experiment_options.index(st.session_state["details_experiment_id"]),
            key="details_experiment_id",
        )

        metadata = manager.get_experiment(experiment_id)
        history = manager.get_history(experiment_id)
        best_fc_logreg_row = max(
            (row for row in rows if _to_float(row.get("fc_logreg_score")) is not None),
            key=lambda row: float(row["fc_logreg_score"]),
            default=None,
        )
        tabs = st.tabs(["Evaluation", "History", "Overview", "Raw JSON"])

        with tabs[0]:
            _render_evaluation_tab(
                metadata=metadata,
                best_fc_logreg_row=best_fc_logreg_row,
            )

        with tabs[1]:
            history_df = _history_to_frame(history)
            if history_df.empty:
                st.warning("No history metrics available.")
            else:
                metric_suffixes = _available_metric_suffixes(list(history_df.columns))
                visible_metrics = metric_suffixes if metric_suffixes else []
                if not visible_metrics:
                    st.warning("No train/val metrics available.")
                else:
                    summary = metadata.get("summary", {})
                    val_pca_mse = _to_float(summary.get("val_pca_mse"))
                    recon_metrics = [metric for metric in visible_metrics if _is_recon_metric(metric)]
                    if recon_metrics:
                        pca_target_metrics = set(recon_metrics)
                    elif "loss" in visible_metrics:
                        pca_target_metrics = {"loss"}
                    else:
                        pca_target_metrics = set()

                    for idx in range(0, len(visible_metrics), 4):
                        row_metrics = visible_metrics[idx:idx + 4]
                        plot_columns = st.columns(4)
                        for container, metric_suffix in zip(plot_columns, row_metrics):
                            _render_pair_plot(
                                container,
                                history_df,
                                _metric_title(metric_suffix),
                                train_candidates=[f"train_{metric_suffix}"],
                                val_candidates=[f"val_{metric_suffix}"],
                                pca_reference=val_pca_mse if metric_suffix in pca_target_metrics else None,
                            )

        with tabs[2]:
            left_meta, right_meta = st.columns(2)
            with left_meta:
                st.markdown("**Run Info**")
                st.write(f"Experiment ID: `{metadata.get('experiment_id', 'N/A')}`")
                st.write(f"Created: `{metadata.get('created_at', 'N/A')}`")
                st.write(f"Model type: `{metadata.get('model_type', 'N/A')}`")
                st.write(f"Schema version: `{metadata.get('schema_version', 'N/A')}`")
            with right_meta:
                summary = metadata.get("summary", {})
                st.markdown("**Summary**")
                c1, c2, c3 = st.columns(3)
                c1.metric("Best val loss", f"{summary.get('best_val_loss', 'N/A')}")
                c2.metric("Best epoch", f"{summary.get('best_epoch', 'N/A')}")
                c3.metric("Epochs", f"{summary.get('num_epochs', 'N/A')}")

            _render_kv_section("Model Parameters", metadata.get("model_params"))
            _render_kv_section("Training Parameters", metadata.get("training_params"))
            data_params = metadata.get("data_params")
            if isinstance(data_params, dict) and ("data" in data_params or "filter" in data_params):
                if "data" in data_params:
                    _render_kv_section("Data Parameters: data", data_params.get("data"))
                if "filter" in data_params:
                    _render_kv_section("Data Parameters: filter", data_params.get("filter"))
                remaining = {k: v for k, v in data_params.items() if k not in {"data", "filter"}}
                if remaining:
                    _render_kv_section("Data Parameters: other", remaining)
            else:
                _render_kv_section("Data Parameters", data_params)
            _render_kv_section("Artifacts", metadata.get("artifacts"))
            st.markdown("**Tags**")
            st.write(", ".join(metadata.get("tags", [])) or "None")

            st.markdown("---")
            if st.button("Overwrite current config with this experiment", key=f"overwrite_{experiment_id}"):
                try:
                    _overwrite_configs_from_metadata(metadata, _default_config_dir())
                except Exception as exc:
                    st.error(f"Failed to overwrite configs: {exc}")
                else:
                    st.success("Updated config/model.yml, config/training.yml, and config/data.yml")

        with tabs[3]:
            st.subheader("Metadata JSON")
            st.code(json.dumps(metadata, indent=2), language="json")
            st.subheader("History JSON")
            st.code(json.dumps(history, indent=2), language="json")

    elif active_view == "Compare Experiments":
        st.subheader("Compare Experiments")
        default_compare = experiment_options[:2] if len(experiment_options) >= 2 else experiment_options[:1]
        selected_compare_ids = st.multiselect(
            "Experiments to compare",
            options=experiment_options,
            default=st.session_state.get("compare_experiment_ids", default_compare),
            key="compare_experiment_ids",
        )

        selected_compare_ids = [exp_id for exp_id in selected_compare_ids if exp_id in row_by_id]
        if not selected_compare_ids:
            st.info("Select at least one experiment.")
            return

        metric_specs = [
            ("test_fc_preservation", "FC Preservation"),
            ("test_logreg_accuracy", "LogReg Accuracy"),
            ("test_swfcd_pearson", "SWFCD Pearson"),
            ("test_mse", "MSE"),
            ("test_silhouette", "Silhouette"),
            ("test_swfcd_rmse", "SWFCD RMSE"),
        ]
        selected_rows = [row_by_id[exp_id] for exp_id in selected_compare_ids]
        display_labels = [
            _short_experiment_id(str(row.get("experiment_id", "exp")))
            for row in selected_rows
        ]
        export_grouped: dict[str, dict[str, list[float]]] = {
            label: {metric_key: [] for metric_key, _ in metric_specs}
            for label in display_labels
        }
        for label, exp_row in zip(display_labels, selected_rows):
            for metric_key, _ in metric_specs:
                metric_value = _to_float(exp_row.get(metric_key))
                if metric_value is not None:
                    export_grouped[label][metric_key].append(metric_value)
        param_details = _build_compare_param_details(selected_rows)

        for idx in range(0, len(metric_specs), 3):
            row_metrics = metric_specs[idx:idx + 3]
            row_cols = st.columns(3)
            for col, (metric_key, metric_title) in zip(row_cols, row_metrics):
                plot_rows = []
                for label, detail, exp_row in zip(display_labels, param_details, selected_rows):
                    metric_value = _to_float(exp_row.get(metric_key))
                    if metric_value is None:
                        continue
                    plot_rows.append(
                        {
                            "model": label,
                            "model_value": metric_value,
                            "parameters": detail,
                        }
                    )

                with col:
                    st.markdown(f"**{metric_title}**")
                    if not plot_rows:
                        st.info("No data")
                    else:
                        metric_df = pd.DataFrame(plot_rows)
                        bars = alt.Chart(metric_df).mark_bar().encode(
                            x=alt.X("model:N", sort=None, title=None),
                            y=alt.Y("model_value:Q", title=metric_title),
                            color=alt.Color("model:N", legend=None),
                            tooltip=[
                                alt.Tooltip("model:N", title="Experiment"),
                                alt.Tooltip("model_value:Q", title=f"Model {metric_title}", format=".6f"),
                                alt.Tooltip("parameters:N", title="Unique params"),
                            ],
                        )
                        st.altair_chart(
                            bars.properties(height=320),
                            use_container_width=True,
                        )

        _render_pvalue_matrices(export_grouped)

        st.markdown("---")
        export_cols = st.columns([1, 1.2])
        with export_cols[1]:
            exclude_fc = st.checkbox(
                "Exclude FC preservation",
                value=False,
                key="compare_exclude_fc",
            )
        with export_cols[0]:
            if st.button("Copy as markdown table", key="compare_copy_markdown"):
                markdown_table = _build_compare_markdown_table(
                    selected_rows,
                    include_fc=not exclude_fc,
                )
                components.html(
                    f"""
                    <script>
                    navigator.clipboard.writeText({json.dumps(markdown_table)});
                    </script>
                    """,
                    height=0,
                )
                st.success("Markdown table copied to clipboard.")
                st.code(markdown_table, language="markdown")

    elif active_view == "Model Parameters":
        st.subheader("Model Parameters")
        filter_col, charts_col = st.columns([1, 3], gap="large")

        with filter_col:
            st.markdown("**Filters**")
            model_type_options = sorted({str(row.get("model_type", "unknown")) for row in rows})
            selected_model_type = st.selectbox(
                "Model type",
                options=model_type_options,
                key="param_filter_model_type",
            )
            model_rows = [
                row for row in rows
                if str(row.get("model_type", "unknown")) == selected_model_type
            ]
            parameter_options = _parameter_options_for_model_type(selected_model_type, rows)
            if not parameter_options:
                st.info("No varying parameters found for this model type.")
                return
            parameter_keys = list(parameter_options.keys())
            if (
                "param_filter_param_key" in st.session_state
                and st.session_state["param_filter_param_key"] not in parameter_keys
            ):
                st.session_state["param_filter_param_key"] = parameter_keys[0]
            selected_param_key = st.selectbox(
                "Parameter",
                options=parameter_keys,
                format_func=lambda key: parameter_options.get(key, key),
                key="param_filter_param_key",
            )
            selected_param_path = tuple(selected_param_key.split("."))

            base_rows = [
                row for row in model_rows
                if _get_param_values_from_row(row, selected_param_path)
            ]

            extra_filters: list[tuple[str, object]] = []
            current_rows = base_rows
            used_param_keys = {selected_param_key}
            for filter_idx in range(2, 6):
                filter_options = _parameter_options_for_rows(current_rows, exclude_keys=used_param_keys)
                if not filter_options:
                    break

                filter_key_key = f"param_filter_extra_key_{filter_idx}"
                filter_value_key = f"param_filter_extra_value_{filter_idx}"

                if (
                    filter_key_key in st.session_state
                    and st.session_state[filter_key_key] not in filter_options
                ):
                    del st.session_state[filter_key_key]
                if (
                    filter_value_key in st.session_state
                    and filter_key_key not in st.session_state
                ):
                    del st.session_state[filter_value_key]

                line_cols = st.columns([1, 1])
                with line_cols[0]:
                    selected_extra_key = st.selectbox(
                        f"Parameter {filter_idx}",
                        options=[""] + list(filter_options.keys()),
                        format_func=lambda key: "Select parameter" if key == "" else filter_options.get(key, key),
                        key=filter_key_key,
                    )
                with line_cols[1]:
                    if selected_extra_key:
                        value_options = _parameter_value_options(current_rows, tuple(selected_extra_key.split(".")))
                        if (
                            filter_value_key in st.session_state
                            and st.session_state[filter_value_key] not in value_options
                        ):
                            del st.session_state[filter_value_key]
                        selected_extra_value = st.selectbox(
                            f"Value {filter_idx}",
                            options=value_options,
                            format_func=_display_group_value,
                            key=filter_value_key,
                        )
                    else:
                        selected_extra_value = None
                        st.empty()

                if not selected_extra_key or selected_extra_value is None:
                    break

                extra_filters.append((selected_extra_key, selected_extra_value))
                used_param_keys.add(selected_extra_key)
                current_rows = _rows_matching_param_filters(current_rows, [(selected_extra_key, selected_extra_value)])

            candidate_rows = _rows_matching_param_filters(base_rows, extra_filters)
            st.caption(f"Selected for comparison: {len(candidate_rows)}")
            if st.button("Load for comparison", key="param_filter_load"):
                st.session_state["param_compare_model_type"] = selected_model_type
                st.session_state["param_compare_key"] = selected_param_key
                st.session_state["param_compare_extra_filters"] = extra_filters

        with charts_col:
            loaded_model_type = st.session_state.get("param_compare_model_type")
            loaded_param_key = st.session_state.get("param_compare_key")
            loaded_extra_filters = st.session_state.get("param_compare_extra_filters", [])
            if not loaded_model_type or not loaded_param_key:
                st.info("Select model + parameter and click Load for comparison.")
                return

            selected_rows = [
                row for row in rows
                if str(row.get("model_type", "unknown")) == loaded_model_type
                and _get_param_values_from_row(row, tuple(loaded_param_key.split(".")))
            ]
            selected_rows = _rows_matching_param_filters(selected_rows, loaded_extra_filters)
            if not selected_rows:
                st.info("No runs available for loaded model/parameter selection.")
                return

            loaded_parameter_options = _parameter_options_for_model_type(loaded_model_type, rows)
            loaded_filter_text = ", ".join(
                f"{loaded_parameter_options.get(param_key, param_key)}={_display_group_value(param_value)}"
                for param_key, param_value in loaded_extra_filters
            )
            st.caption(
                f"Loaded: model={loaded_model_type}, "
                f"parameter={loaded_parameter_options.get(loaded_param_key, loaded_param_key)}"
                + (f" | filters: {loaded_filter_text}" if loaded_filter_text else "")
            )

            metric_specs = [
                ("test_mse", "MSE"),
                ("test_fc_preservation", "FC Preservation"),
                ("test_swfcd_pearson", "SWFCD Pearson"),
                ("test_logreg_accuracy", "LogReg Accuracy"),
            ]

            # Aggregate each metric by parameter value.
            grouped: dict[object, dict[str, list[float]]] = {}
            loaded_param_path = tuple(loaded_param_key.split("."))
            comparison_rows = _expand_rows_for_parameter(selected_rows, loaded_param_path)
            for row in comparison_rows:
                param_value = row.get("_comparison_parameter_value")
                if param_value is None:
                    param_value = _get_param_value_from_row(row, loaded_param_path)
                encoded_param_value = _encode_group_value(param_value)
                if encoded_param_value not in grouped:
                    grouped[encoded_param_value] = {metric_key: [] for metric_key, _ in metric_specs}
                for metric_key, _ in metric_specs:
                    metric_value = _to_float(row.get(metric_key))
                    if metric_value is not None:
                        grouped[encoded_param_value][metric_key].append(metric_value)

            sorted_param_values = sorted(grouped.keys(), key=_value_sort_key)
            display_grouped = {
                _display_group_value(param_value): grouped[param_value]
                for param_value in sorted_param_values
            }

            for idx in range(0, len(metric_specs), 2):
                row_metrics = metric_specs[idx:idx + 2]
                row_cols = st.columns(2)
                for col, (metric_key, metric_title) in zip(row_cols, row_metrics):
                    plot_rows = []
                    for param_value in sorted_param_values:
                        values = grouped[param_value].get(metric_key, [])
                        if not values:
                            continue
                        param_label = _display_group_value(param_value)
                        for value in values:
                            plot_rows.append(
                                {
                                    "parameter_value": param_label,
                                    "metric_value": value,
                                }
                            )

                    with col:
                        st.markdown(f"**{metric_title}**")
                        if not plot_rows:
                            st.info("No data")
                        else:
                            metric_df = pd.DataFrame(plot_rows)
                            metric_min = float(metric_df["metric_value"].min())
                            metric_max = float(metric_df["metric_value"].max())
                            value_span = metric_max - metric_min
                            padding = (value_span * 0.08) if value_span > 0 else max(abs(metric_min) * 0.08, 1e-6)
                            y_domain = [metric_min - padding, metric_max + padding]
                            param_order = [
                                _display_group_value(param_value)
                                for param_value in sorted_param_values
                                if grouped[param_value].get(metric_key, [])
                            ]
                            chart = alt.Chart(metric_df).mark_boxplot(
                                size=30,
                                ticks={"size": 30},
                            ).encode(
                                x=alt.X("parameter_value:N", title=None, sort=param_order),
                                y=alt.Y("metric_value:Q", title=metric_title, scale=alt.Scale(domain=y_domain, zero=False)),
                                color=alt.Color("parameter_value:N", legend=None),
                                tooltip=[
                                    alt.Tooltip("parameter_value:N", title="Parameter value"),
                                    alt.Tooltip("metric_value:Q", title=metric_title, format=".6f"),
                                ],
                            )
                            st.altair_chart(
                                chart.properties(height=320),
                                use_container_width=True,
                            )

            _render_pvalue_matrices(display_grouped)
            st.markdown("---")
            export_cols = st.columns([1, 1, 1, 1.2])
            with export_cols[3]:
                exclude_fc = st.checkbox(
                    "Exclude FC preservation",
                    value=False,
                    key="param_compare_exclude_fc",
                )
            with export_cols[2]:
                if st.button("Save Raincloud plot spec", key="param_compare_save_raincloud"):
                    output_path = _save_raincloud_spec(
                        display_grouped,
                        results_dir=results_dir,
                        tab_name="model_parameters",
                        include_fc=not exclude_fc,
                    )
                    st.success(f"Raincloud plot spec saved to {output_path}")
            with export_cols[0]:
                if st.button("Copy as markdown table", key="param_compare_copy_markdown"):
                    markdown_table = _build_parameter_compare_markdown_table(
                        grouped,
                        include_fc=not exclude_fc,
                    )
                    components.html(
                        f"""
                        <script>
                        navigator.clipboard.writeText({json.dumps(markdown_table)});
                        </script>
                        """,
                        height=0,
                    )
                    st.success("Markdown table copied to clipboard.")
                    st.code(markdown_table, language="markdown")
            with export_cols[1]:
                if st.button("Copy boxplot spec", key="param_compare_copy_boxplot_spec"):
                    boxplot_spec = _build_parameter_compare_boxplot_spec(
                        grouped,
                        include_fc=not exclude_fc,
                    )
                    components.html(
                        f"""
                        <script>
                        navigator.clipboard.writeText({json.dumps(boxplot_spec)});
                        </script>
                        """,
                        height=0,
                    )
                    st.success("Boxplot spec copied to clipboard.")
                    st.code(boxplot_spec, language="yaml")

    elif active_view == "Parameter Comparison":
        st.subheader("Parameter Comparison")
        filter_col, charts_col = st.columns([1, 3], gap="large")

        with filter_col:
            st.markdown("**Filters**")
            parameter_options = _parameter_options_for_rows(rows)
            if not parameter_options:
                st.info("No varying parameters found across experiments.")
                return
            parameter_keys = list(parameter_options.keys())
            if (
                "allparam_filter_param_key" in st.session_state
                and st.session_state["allparam_filter_param_key"] not in parameter_keys
            ):
                st.session_state["allparam_filter_param_key"] = parameter_keys[0]
            selected_param_key = st.selectbox(
                "Parameter",
                options=parameter_keys,
                format_func=lambda key: parameter_options.get(key, key),
                key="allparam_filter_param_key",
            )
            selected_param_path = tuple(selected_param_key.split("."))

            base_rows = [
                row for row in rows
                if _get_param_values_from_row(row, selected_param_path)
            ]

            extra_filters: list[tuple[str, object]] = []
            current_rows = base_rows
            used_param_keys = {selected_param_key}
            for filter_idx in range(2, 6):
                filter_options = _parameter_options_for_rows(current_rows, exclude_keys=used_param_keys)
                if not filter_options:
                    break

                filter_key_key = f"allparam_filter_extra_key_{filter_idx}"
                filter_value_key = f"allparam_filter_extra_value_{filter_idx}"

                if (
                    filter_key_key in st.session_state
                    and st.session_state[filter_key_key] not in filter_options
                ):
                    del st.session_state[filter_key_key]
                if (
                    filter_value_key in st.session_state
                    and filter_key_key not in st.session_state
                ):
                    del st.session_state[filter_value_key]

                line_cols = st.columns([1, 1])
                with line_cols[0]:
                    selected_extra_key = st.selectbox(
                        f"Parameter {filter_idx}",
                        options=[""] + list(filter_options.keys()),
                        format_func=lambda key: "Select parameter" if key == "" else filter_options.get(key, key),
                        key=filter_key_key,
                    )
                with line_cols[1]:
                    if selected_extra_key:
                        value_options = _parameter_value_options(current_rows, tuple(selected_extra_key.split(".")))
                        if (
                            filter_value_key in st.session_state
                            and st.session_state[filter_value_key] not in value_options
                        ):
                            del st.session_state[filter_value_key]
                        selected_extra_value = st.selectbox(
                            f"Value {filter_idx}",
                            options=value_options,
                            format_func=_display_group_value,
                            key=filter_value_key,
                        )
                    else:
                        selected_extra_value = None
                        st.empty()

                if not selected_extra_key or selected_extra_value is None:
                    break

                extra_filters.append((selected_extra_key, selected_extra_value))
                used_param_keys.add(selected_extra_key)
                current_rows = _rows_matching_param_filters(current_rows, [(selected_extra_key, selected_extra_value)])

            candidate_rows = _rows_matching_param_filters(base_rows, extra_filters)
            st.caption(f"Selected for comparison: {len(candidate_rows)}")
            if st.button("Load for comparison", key="allparam_filter_load"):
                st.session_state["allparam_compare_key"] = selected_param_key
                st.session_state["allparam_compare_extra_filters"] = extra_filters

        with charts_col:
            loaded_param_key = st.session_state.get("allparam_compare_key")
            loaded_extra_filters = st.session_state.get("allparam_compare_extra_filters", [])
            if not loaded_param_key:
                st.info("Select parameter(s) and click Load for comparison.")
                return

            selected_rows = [
                row for row in rows
                if _get_param_values_from_row(row, tuple(loaded_param_key.split(".")))
            ]
            selected_rows = _rows_matching_param_filters(selected_rows, loaded_extra_filters)
            if not selected_rows:
                st.info("No runs available for loaded parameter selection.")
                return

            loaded_parameter_options = _parameter_options_for_rows(rows)
            loaded_filter_text = ", ".join(
                f"{loaded_parameter_options.get(param_key, param_key)}={_display_group_value(param_value)}"
                for param_key, param_value in loaded_extra_filters
            )
            st.caption(
                f"Loaded: parameter={loaded_parameter_options.get(loaded_param_key, loaded_param_key)}"
                + (f" | filters: {loaded_filter_text}" if loaded_filter_text else "")
            )

            metric_specs = [
                ("test_mse", "MSE"),
                ("test_fc_preservation", "FC Preservation"),
                ("test_swfcd_pearson", "SWFCD Pearson"),
                ("test_logreg_accuracy", "LogReg Accuracy"),
            ]

            grouped: dict[object, dict[str, list[float]]] = {}
            loaded_param_path = tuple(loaded_param_key.split("."))
            comparison_rows = _expand_rows_for_parameter(selected_rows, loaded_param_path)
            for row in comparison_rows:
                param_value = row.get("_comparison_parameter_value")
                if param_value is None:
                    param_value = _get_param_value_from_row(row, loaded_param_path)
                encoded_param_value = _encode_group_value(param_value)
                if encoded_param_value not in grouped:
                    grouped[encoded_param_value] = {metric_key: [] for metric_key, _ in metric_specs}
                for metric_key, _ in metric_specs:
                    metric_value = _to_float(row.get(metric_key))
                    if metric_value is not None:
                        grouped[encoded_param_value][metric_key].append(metric_value)

            sorted_param_values = sorted(grouped.keys(), key=_value_sort_key)
            display_grouped = {
                _display_group_value(param_value): grouped[param_value]
                for param_value in sorted_param_values
            }

            for idx in range(0, len(metric_specs), 2):
                row_metrics = metric_specs[idx:idx + 2]
                row_cols = st.columns(2)
                for col, (metric_key, metric_title) in zip(row_cols, row_metrics):
                    plot_rows = []
                    for param_value in sorted_param_values:
                        values = grouped[param_value].get(metric_key, [])
                        if not values:
                            continue
                        param_label = _display_group_value(param_value)
                        for value in values:
                            plot_rows.append(
                                {
                                    "parameter_value": param_label,
                                    "metric_value": value,
                                }
                            )

                    with col:
                        st.markdown(f"**{metric_title}**")
                        if not plot_rows:
                            st.info("No data")
                        else:
                            metric_df = pd.DataFrame(plot_rows)
                            metric_min = float(metric_df["metric_value"].min())
                            metric_max = float(metric_df["metric_value"].max())
                            value_span = metric_max - metric_min
                            padding = (value_span * 0.08) if value_span > 0 else max(abs(metric_min) * 0.08, 1e-6)
                            y_domain = [metric_min - padding, metric_max + padding]
                            param_order = [
                                _display_group_value(param_value)
                                for param_value in sorted_param_values
                                if grouped[param_value].get(metric_key, [])
                            ]
                            chart = alt.Chart(metric_df).mark_boxplot(
                                size=30,
                                ticks={"size": 30},
                            ).encode(
                                x=alt.X("parameter_value:N", title=None, sort=param_order),
                                y=alt.Y("metric_value:Q", title=metric_title, scale=alt.Scale(domain=y_domain, zero=False)),
                                color=alt.Color("parameter_value:N", legend=None),
                                tooltip=[
                                    alt.Tooltip("parameter_value:N", title="Parameter value"),
                                    alt.Tooltip("metric_value:Q", title=metric_title, format=".6f"),
                                ],
                            )
                            st.altair_chart(
                                chart.properties(height=320),
                                use_container_width=True,
                            )

            _render_pvalue_matrices(display_grouped)
            st.markdown("---")
            export_cols = st.columns([1, 1, 1, 1.2])
            with export_cols[3]:
                exclude_fc = st.checkbox(
                    "Exclude FC preservation",
                    value=False,
                    key="allparam_compare_exclude_fc",
                )
            with export_cols[2]:
                if st.button("Save Raincloud plot spec", key="allparam_compare_save_raincloud"):
                    output_path = _save_raincloud_spec(
                        display_grouped,
                        results_dir=results_dir,
                        tab_name="parameter_comparison",
                        include_fc=not exclude_fc,
                    )
                    st.success(f"Raincloud plot spec saved to {output_path}")
            with export_cols[0]:
                if st.button("Copy as markdown table", key="allparam_compare_copy_markdown"):
                    markdown_table = _build_parameter_compare_markdown_table(
                        grouped,
                        include_fc=not exclude_fc,
                    )
                    components.html(
                        f"""
                        <script>
                        navigator.clipboard.writeText({json.dumps(markdown_table)});
                        </script>
                        """,
                        height=0,
                    )
                    st.success("Markdown table copied to clipboard.")
                    st.code(markdown_table, language="markdown")
            with export_cols[1]:
                if st.button("Copy boxplot spec", key="allparam_compare_copy_boxplot_spec"):
                    boxplot_spec = _build_parameter_compare_boxplot_spec(
                        grouped,
                        include_fc=not exclude_fc,
                    )
                    components.html(
                        f"""
                        <script>
                        navigator.clipboard.writeText({json.dumps(boxplot_spec)});
                        </script>
                        """,
                        height=0,
                    )
                    st.success("Boxplot spec copied to clipboard.")
                    st.code(boxplot_spec, language="yaml")

    elif active_view == "Model Comparison":
        st.subheader("Model Comparison")
        st.caption("Data filters")
        filter_cols = st.columns(4)
        with filter_cols[0]:
            st.markdown("**Flatten**")
            flatten_true = st.checkbox("True", value=False, key="modelf_flatten_true")
            flatten_false = st.checkbox("False", value=False, key="modelf_flatten_false")
        with filter_cols[1]:
            st.markdown("**Transpose**")
            transpose_true = st.checkbox("True", value=False, key="modelf_transpose_true")
            transpose_false = st.checkbox("False", value=False, key="modelf_transpose_false")
        with filter_cols[2]:
            st.markdown("**Timepoints as samples**")
            tas_true = st.checkbox("True", value=False, key="modelf_tas_true")
            tas_false = st.checkbox("False", value=False, key="modelf_tas_false")
        with filter_cols[3]:
            st.markdown("**FC input**")
            fc_input_true = st.checkbox("True", value=False, key="modelf_fc_input_true")
            fc_input_false = st.checkbox("False", value=False, key="modelf_fc_input_false")

        filtered_model_rows = [
            row for row in rows
            if _bool_match(bool(row.get("data_flatten", False)), flatten_true, flatten_false)
            and _bool_match(bool(row.get("data_transpose", False)), transpose_true, transpose_false)
            and _bool_match(bool(row.get("data_timepoints_as_samples", False)), tas_true, tas_false)
            and _bool_match(bool(row.get("data_fc_input", False)), fc_input_true, fc_input_false)
        ]
        if not filtered_model_rows:
            st.info("No experiments match the selected data filters.")
            return

        metric_specs = [
            ("test_fc_preservation", "FC Preservation"),
            ("test_logreg_accuracy", "LogReg Accuracy"),
            ("test_swfcd_pearson", "SWFCD Pearson"),
            ("test_mse", "MSE"),
            ("test_silhouette", "Silhouette"),
            ("test_swfcd_rmse", "SWFCD RMSE"),
        ]
        model_order = list(
            dict.fromkeys(str(row.get("model_type", "unknown")) for row in filtered_model_rows)
        )
        export_grouped: dict[str, dict[str, list[float]]] = {
            model_name: {metric_key: [] for metric_key, _ in metric_specs}
            for model_name in model_order
        }
        for row in filtered_model_rows:
            model_name = str(row.get("model_type", "unknown"))
            for metric_key, _ in metric_specs:
                metric_value = _to_float(row.get(metric_key))
                if metric_value is not None:
                    export_grouped[model_name][metric_key].append(metric_value)
        st.caption("Box plots show metric distributions across all runs for each model type.")

        for idx in range(0, len(metric_specs), 3):
            row_metrics = metric_specs[idx:idx + 3]
            row_cols = st.columns(3)
            for col, (metric_key, metric_title) in zip(row_cols, row_metrics):
                plot_rows = []
                for row in filtered_model_rows:
                    metric_value = _to_float(row.get(metric_key))
                    model_type = str(row.get("model_type", "unknown"))
                    if metric_value is None:
                        continue
                    plot_rows.append(
                        {
                            "model_type": model_type,
                            "metric_value": metric_value,
                        }
                    )

                with col:
                    st.markdown(f"**{metric_title}**")
                    if not plot_rows:
                        st.info("No data")
                    else:
                        metric_df = pd.DataFrame(plot_rows)
                        metric_min = float(metric_df["metric_value"].min())
                        metric_max = float(metric_df["metric_value"].max())
                        value_span = metric_max - metric_min
                        padding = (value_span * 0.08) if value_span > 0 else max(abs(metric_min) * 0.08, 1e-6)
                        y_domain = [metric_min - padding, metric_max + padding]
                        chart = alt.Chart(metric_df).mark_boxplot(
                            size=30,
                            ticks={"size": 30},
                        ).encode(
                            x=alt.X("model_type:N", title=None, sort=model_order),
                            y=alt.Y("metric_value:Q", title=metric_title, scale=alt.Scale(domain=y_domain, zero=False)),
                            color=alt.Color("model_type:N", legend=None),
                            tooltip=[
                                alt.Tooltip("model_type:N", title="Model"),
                                alt.Tooltip("metric_value:Q", title=metric_title, format=".6f"),
                            ],
                        )
                        st.altair_chart(
                            chart.properties(height=320),
                            use_container_width=True,
                        )

        _render_pvalue_matrices(export_grouped)
        st.markdown("---")
        export_cols = st.columns(2)
        with export_cols[0]:
            if st.button("Copy boxplot spec", key="model_compare_copy_boxplot_spec"):
                boxplot_spec = _build_model_compare_boxplot_spec(export_grouped)
                components.html(
                    f"""
                    <script>
                    navigator.clipboard.writeText({json.dumps(boxplot_spec)});
                    </script>
                    """,
                    height=0,
                )
                st.success("Boxplot spec copied to clipboard.")
                st.code(boxplot_spec, language="yaml")
        with export_cols[1]:
            if st.button("Save Raincloud plot spec", key="model_compare_save_raincloud"):
                output_path = _save_raincloud_spec(
                    export_grouped,
                    results_dir=results_dir,
                    tab_name="model_comparison",
                    include_fc=True,
                )
                st.success(f"Raincloud plot spec saved to {output_path}")

    elif active_view == "Data Comparison":
        st.subheader("Data Comparison")
        metric_specs = [
            ("test_fc_preservation", "FC Preservation"),
            ("test_logreg_accuracy", "LogReg Accuracy"),
            ("test_swfcd_pearson", "SWFCD Pearson"),
            ("test_mse", "MSE"),
            ("test_silhouette", "Silhouette"),
            ("test_swfcd_rmse", "SWFCD RMSE"),
        ]
        data_combo_order = list(
            dict.fromkeys(_data_combination_label(row) for row in rows)
        )
        export_grouped: dict[str, dict[str, list[float]]] = {
            data_combo: {metric_key: [] for metric_key, _ in metric_specs}
            for data_combo in data_combo_order
        }
        for row in rows:
            data_combo = _data_combination_label(row)
            for metric_key, _ in metric_specs:
                metric_value = _to_float(row.get(metric_key))
                if metric_value is not None:
                    export_grouped[data_combo][metric_key].append(metric_value)
        st.caption("Box plots show metric distributions across all runs for each observed data-parameter combination.")

        for idx in range(0, len(metric_specs), 3):
            row_metrics = metric_specs[idx:idx + 3]
            row_cols = st.columns(3)
            for col, (metric_key, metric_title) in zip(row_cols, row_metrics):
                plot_rows = []
                for row in rows:
                    metric_value = _to_float(row.get(metric_key))
                    if metric_value is None:
                        continue
                    plot_rows.append(
                        {
                            "data_combo": _data_combination_label(row),
                            "metric_value": metric_value,
                        }
                    )

                with col:
                    st.markdown(f"**{metric_title}**")
                    if not plot_rows:
                        st.info("No data")
                    else:
                        metric_df = pd.DataFrame(plot_rows)
                        metric_min = float(metric_df["metric_value"].min())
                        metric_max = float(metric_df["metric_value"].max())
                        value_span = metric_max - metric_min
                        padding = (value_span * 0.08) if value_span > 0 else max(abs(metric_min) * 0.08, 1e-6)
                        y_domain = [metric_min - padding, metric_max + padding]
                        chart = alt.Chart(metric_df).mark_boxplot(
                            size=30,
                            ticks={"size": 30},
                        ).encode(
                            x=alt.X("data_combo:N", title=None, sort=data_combo_order),
                            y=alt.Y("metric_value:Q", title=metric_title, scale=alt.Scale(domain=y_domain, zero=False)),
                            color=alt.Color("data_combo:N", legend=None),
                            tooltip=[
                                alt.Tooltip("data_combo:N", title="Data parameters"),
                                alt.Tooltip("metric_value:Q", title=metric_title, format=".6f"),
                            ],
                        )
                        st.altair_chart(
                            chart.properties(height=320),
                            use_container_width=True,
                        )

        _render_pvalue_matrices(export_grouped)
        st.markdown("---")
        export_cols = st.columns(2)
        with export_cols[0]:
            if st.button("Copy boxplot spec", key="data_compare_copy_boxplot_spec"):
                boxplot_spec = _build_model_compare_boxplot_spec(export_grouped)
                components.html(
                    f"""
                    <script>
                    navigator.clipboard.writeText({json.dumps(boxplot_spec)});
                    </script>
                    """,
                    height=0,
                )
                st.success("Boxplot spec copied to clipboard.")
                st.code(boxplot_spec, language="yaml")
        with export_cols[1]:
            if st.button("Save Raincloud plot spec", key="data_compare_save_raincloud"):
                output_path = _save_raincloud_spec(
                    export_grouped,
                    results_dir=results_dir,
                    tab_name="data_comparison",
                    include_fc=True,
                )
                st.success(f"Raincloud plot spec saved to {output_path}")


if __name__ == "__main__":
    main()
