"""Experiments page: launch a fixed Experiment Config, watch Training Jobs, compare Runs."""

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from mlflow.tracking import MlflowClient

from app.db.repositories import Job, JobRepository
from app.db.session import get_engine
from app.jobs import service
from app.jobs.config import ConfigError
from app.ml.algorithms import ALGORITHMS
from app.ml.data import load_dataset
from app.settings import get_settings
from app.tracking.runs import LeaderboardRow, leaderboard, run_url

REFRESH_SECONDS = 2

ALGORITHM_LABELS = {
    "linear_regression": "Linear regression",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
}


@st.cache_resource
def _jobs() -> JobRepository:
    return JobRepository(get_engine())


@st.cache_resource
def _mlflow() -> MlflowClient:
    uri = get_settings().mlflow_tracking_uri
    return MlflowClient(tracking_uri=uri, registry_uri=uri)


@st.cache_data
def _dataset_sha256(path: str, modified_ns: int) -> str:
    # Keyed on the modification time too, so replacing the file changes the current Dataset.
    return load_dataset(path).sha256


def _current_dataset_sha256() -> str:
    path = Path(get_settings().data_path)
    return _dataset_sha256(str(path), path.stat().st_mtime_ns)


def render() -> None:
    st.title("Experiments")
    _config_form()
    _jobs_section()
    _leaderboard_section()


# Launch -----------------------------------------------------------------------------------


def _param_input(algorithm: str, name: str, default: Any) -> Any:
    key = f"param.{algorithm}.{name}"
    if isinstance(default, bool):
        return st.checkbox(name, value=default, key=key)
    if isinstance(default, int):
        return int(st.number_input(name, value=default, step=1, key=key))
    if isinstance(default, float):
        return float(st.number_input(name, value=default, format="%g", key=key))
    return st.text_input(name, value=str(default), key=key)


def _config_form() -> None:
    st.subheader("Launch a Training Job")
    # Outside the form, so changing it reruns the page and refills the defaults below.
    algorithm = st.selectbox(
        "Algorithm", list(ALGORITHMS), format_func=lambda a: ALGORITHM_LABELS.get(a, a)
    )
    defaults = ALGORITHMS[algorithm].default_params

    with st.form("experiment-config"):
        name = st.text_input("Name", value=f"{algorithm}-fixed", key=f"name.{algorithm}")
        left, right = st.columns(2)
        bedrooms_per_room = left.toggle("Add bedrooms-per-room feature", value=True)
        target_transform = right.radio(
            "Target transform", ["none", "log1p"], horizontal=True, index=1
        )

        st.markdown("**Hyperparameters** (fixed)")
        params = {}
        if not defaults:
            st.caption("This algorithm has no hyperparameters to set.")
        columns = st.columns(len(defaults)) if defaults else []
        for column, (param, default) in zip(columns, defaults.items(), strict=True):
            with column:
                params[param] = _param_input(algorithm, param, default)

        if st.form_submit_button("Launch", type="primary"):
            raw = {
                "name": name,
                "algorithm": algorithm,
                "features": {"bedrooms_per_room": bedrooms_per_room},
                "target_transform": target_transform,
                "tuning": {"mode": "fixed", "params": params},
            }
            try:
                job = service.submit(_jobs(), raw)
            except ConfigError as exc:
                st.error(str(exc))
            else:
                st.success(f"Queued {job.name} ({job.id}).")


# Watch ------------------------------------------------------------------------------------


def _jobs_frame(jobs: list[Job]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Name": job.name,
                "Algorithm": job.config.get("algorithm"),
                "Status": job.status,
                "Phase": job.phase or "",
                "Progress": job.progress_done / job.progress_total if job.progress_total else 0.0,
                "Candidates": f"{job.progress_done}/{job.progress_total}",
                "Submitted": job.created_at,
                "Error": job.error or "",
            }
            for job in jobs
        ]
    )


def _jobs_section() -> None:
    st.subheader("Training Jobs")
    active = any(not job.finished for job in service.recent(_jobs()))

    # Refreshes itself only while a job is queued or running.
    @st.fragment(run_every=REFRESH_SECONDS if active else None)
    def jobs_table() -> None:
        jobs = service.recent(_jobs())
        if not jobs:
            st.info("No Training Jobs yet. Launch one above or with `make train`.")
        else:
            st.dataframe(
                _jobs_frame(jobs),
                hide_index=True,
                column_config={
                    "Progress": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0),
                    "Submitted": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm:ss"),
                },
            )
        if active and all(job.finished for job in jobs):
            # Rerun the whole page: stop refreshing and show the new Runs on the Leaderboard.
            st.rerun()

    jobs_table()
    if not active:
        st.button("Refresh", key="refresh-jobs", help="Pick up jobs submitted from the CLI.")


# Compare ----------------------------------------------------------------------------------


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def _leaderboard_frame(rows: list[LeaderboardRow], mlflow_ui_url: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Best": "★ Best Run" if row.is_best else "",
                "Config": row.config_name,
                "Algorithm": ALGORITHM_LABELS.get(row.algorithm, row.algorithm),
                "CV RMSE": f"{row.cv_rmse_mean:.4f} ± {row.cv_rmse_std:.4f}",
                "CV MAE": _fmt(row.cv_mae_mean),
                "CV R²": _fmt(row.cv_r2_mean),
                "Test RMSE": _fmt(row.test_rmse),
                "Test MAE": _fmt(row.test_mae),
                "Test R²": _fmt(row.test_r2),
                "Key params": ", ".join(f"{k}={v}" for k, v in row.params.items()),
                "MLflow": run_url(row, mlflow_ui_url),
            }
            for row in rows
        ]
    )


def _highlight_best(row: pd.Series) -> list[str]:
    style = "background-color: rgba(255, 196, 0, 0.25); font-weight: 600" if row["Best"] else ""
    return [style] * len(row)


def _leaderboard_section() -> None:
    settings = get_settings()
    st.subheader("Leaderboard")
    try:
        dataset_sha256 = _current_dataset_sha256()
        rows = leaderboard(_mlflow(), dataset_sha256)
    except Exception as exc:  # MLflow down or Dataset unreadable: keep the rest of the page.
        st.error(f"Cannot load the Leaderboard: {exc}")
        return
    st.caption(
        f"Completed Runs on the current Dataset (`{dataset_sha256[:12]}`), ranked by mean CV "
        "RMSE. Test-Set scores are shown for information and never affect the ranking. "
        f"[Open the MLflow UI]({settings.mlflow_ui_url})"
    )
    if not rows:
        st.info("No completed Runs on this Dataset yet.")
        return

    frame = _leaderboard_frame(rows, settings.mlflow_ui_url)
    selection = st.dataframe(
        frame.style.apply(_highlight_best, axis=1),
        hide_index=True,
        key="leaderboard",
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "MLflow": st.column_config.LinkColumn("MLflow", display_text="Open run"),
        },
    )
    selected = selection.selection.rows
    if selected:
        _run_details(rows[selected[0]], settings.mlflow_ui_url)
    else:
        st.caption("Select a row to see the Run's details.")


def _run_details(row: LeaderboardRow, mlflow_ui_url: str) -> None:
    run = _mlflow().get_run(row.run_id)
    st.markdown(
        f"#### {row.config_name} · {ALGORITHM_LABELS.get(row.algorithm, row.algorithm)}"
        f"{' · ★ Best Run' if row.is_best else ''}"
    )
    st.markdown(f"Run `{row.run_id}` · [Open in the MLflow UI]({run_url(row, mlflow_ui_url)})")
    metrics, params, tags = st.columns(3)
    metrics.markdown("**Metrics**")
    metrics.dataframe(
        pd.DataFrame(sorted(run.data.metrics.items()), columns=["Metric", "Value"]),
        hide_index=True,
    )
    params.markdown("**Params**")
    params.dataframe(
        pd.DataFrame(sorted(run.data.params.items()), columns=["Param", "Value"]),
        hide_index=True,
    )
    tags.markdown("**Tags**")
    visible_tags = {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")}
    tags.dataframe(
        pd.DataFrame(sorted(visible_tags.items()), columns=["Tag", "Value"]), hide_index=True
    )
