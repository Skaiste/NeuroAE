"""Streamlit UI for browsing tracked training experiments."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from training_tracker import TrainingResultsManager


def _default_results_dir() -> Path:
    env_value = os.environ.get("TRAINING_TRACKER_RESULTS_DIR")
    if env_value:
        return Path(env_value)
    return Path("results")


def _parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _history_to_frame(history: dict) -> pd.DataFrame:
    metrics = history if "metrics" not in history else history["metrics"]
    if not metrics:
        return pd.DataFrame()

    rows = []
    num_epochs = max((len(values) for values in metrics.values()), default=0)
    for epoch in range(1, num_epochs + 1):
        row = {"epoch": epoch}
        for metric_name, values in metrics.items():
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
        if train_col is not None:
            plot_cols.append(train_col)
            rename_map[train_col] = "train"
        if val_col is not None:
            plot_cols.append(val_col)
            rename_map[val_col] = "val"

        chart_df = history_df[plot_cols].rename(columns=rename_map).set_index("epoch")
        st.line_chart(chart_df)


def _build_sidebar(manager: TrainingResultsManager) -> dict:
    st.sidebar.header("Filters")
    rows = manager.list_experiments(limit=None)

    model_types = sorted({row.get("model_type") for row in rows if row.get("model_type")})

    selected_model_types = st.sidebar.multiselect("Model type", options=model_types, default=model_types)

    max_best_val_loss = st.sidebar.number_input("Max best val loss", min_value=0.0, value=10.0, step=0.1)
    min_epochs = st.sidebar.number_input("Min epochs", min_value=0, value=0, step=1)
    max_days_old = st.sidebar.number_input("Max age (days)", min_value=1, value=3650, step=1)

    if st.sidebar.button("Rebuild index"):
        manager.rebuild_index()
        st.sidebar.success("Index rebuilt")

    filters: dict = {}
    if selected_model_types and len(selected_model_types) != len(model_types):
        filters["model_type"] = selected_model_types[0] if len(selected_model_types) == 1 else None

    filters["best_val_loss_max"] = max_best_val_loss
    filters["num_epochs_min"] = int(min_epochs)

    created_after = datetime.now(timezone.utc) - timedelta(days=int(max_days_old))
    filters["created_after"] = created_after.isoformat().replace("+00:00", "Z")

    # Additional in-memory filters for multi-select model/status.
    filters["_selected_model_types"] = set(selected_model_types)

    return filters


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

    manager = TrainingResultsManager(results_dir=_default_results_dir())
    filters = _build_sidebar(manager)

    rows = manager.list_experiments(
        filters={
            "model_type": filters.get("model_type"),
            "best_val_loss_max": filters.get("best_val_loss_max"),
            "num_epochs_min": filters.get("num_epochs_min"),
            "created_after": filters.get("created_after"),
        },
        sort_by="created_at",
        ascending=False,
        limit=None,
    )

    selected_model_types = filters.get("_selected_model_types", set())
    rows = [
        row for row in rows
        if (not selected_model_types or row.get("model_type") in selected_model_types)
    ]

    total_rows = manager.list_experiments(limit=None)
    st.title("Training Tracker")
    best_loss = min((row.get("best_val_loss") for row in rows if row.get("best_val_loss") is not None), default=None)
    st.caption(
        f"Experiments: {len(total_rows)}   |   "
        f"Filtered: {len(rows)}   |   "
        f"Best val loss: {best_loss:.4f}" if best_loss is not None else
        f"Experiments: {len(total_rows)}   |   Filtered: {len(rows)}   |   Best val loss: N/A"
    )

    if not rows:
        st.info("No experiments match the selected filters.")
        return

    table_df = pd.DataFrame(rows)
    visible_cols = [
        "experiment_id",
        "created_at",
        "model_type",
        "best_val_loss",
        "num_epochs",
        "learning_rate",
        "latent_dim",
    ]
    present_cols = [col for col in visible_cols if col in table_df.columns]
    top_tabs = st.tabs(["Selected Experiments", "Experiment Details"])

    with top_tabs[0]:
        st.subheader("Experiments")
        st.dataframe(table_df[present_cols], width="stretch", hide_index=True)

    with top_tabs[1]:
        experiment_id = st.selectbox(
            "Experiment",
            options=[row["experiment_id"] for row in rows],
            index=0,
            key="details_experiment_id",
        )

    metadata = manager.get_experiment(experiment_id)
    history = manager.get_history(experiment_id)
    tabs = top_tabs[1].tabs(["History", "Overview", "Raw JSON"])

    with tabs[0]:
        history_df = _history_to_frame(history)
        if history_df.empty:
            st.warning("No history metrics available.")
        else:
            col_total, col_recon, col_kld = st.columns(3)
            _render_pair_plot(
                col_total,
                history_df,
                "Total Loss",
                train_candidates=["train_loss"],
                val_candidates=["val_loss"],
            )
            _render_pair_plot(
                col_recon,
                history_df,
                "Reconstruction Loss",
                train_candidates=["train_reproduction_loss", "train_reconstruction_loss"],
                val_candidates=["val_reproduction_loss", "val_reconstruction_loss"],
            )
            _render_pair_plot(
                col_kld,
                history_df,
                "KLD",
                train_candidates=["train_KLD", "train_kld"],
                val_candidates=["val_KLD", "val_kld"],
            )

    with tabs[1]:
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

    with tabs[2]:
        st.subheader("Metadata JSON")
        st.code(json.dumps(metadata, indent=2), language="json")
        st.subheader("History JSON")
        st.code(json.dumps(history, indent=2), language="json")


if __name__ == "__main__":
    main()
