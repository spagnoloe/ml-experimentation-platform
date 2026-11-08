"""Experiments page: launch an Experiment Config, watch Training Jobs, compare Runs."""

import hashlib
from typing import Any

import pandas as pd
import streamlit as st
import yaml

from app.db.repositories import Job, JobRepository
from app.db.session import get_engine
from app.jobs import service
from app.jobs.config import ConfigError
from app.ml.algorithms import ALGORITHMS
from app.registry.client import RegistryError
from app.settings import get_settings
from app.tracking.runs import LeaderboardRow, baseline, leaderboard, run_details, run_url
from app.ui import shared

REFRESH_SECONDS = 2
# Set by a successful register, shown once after the rerun that disables the button.
_REGISTERED = "experiments.registered"
IDLE_REFRESH_SECONDS = 10


@st.cache_resource
def _jobs() -> JobRepository:
    return JobRepository(get_engine())


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


def _search_space_yaml(algorithm: str) -> str:
    return yaml.safe_dump(
        ALGORITHMS[algorithm].search_space, sort_keys=False, default_flow_style=None
    )


def _config_form() -> None:
    st.subheader("Launch a Training Job")
    # Outside the form, so changing them reruns the page: the defaults below are refilled
    # and the Candidate count is up to date before launch.
    algorithm = st.selectbox("Algorithm", list(ALGORITHMS), format_func=shared.algorithm_label)
    defaults = ALGORITHMS[algorithm].default_params
    searchable = bool(ALGORITHMS[algorithm].search_space)
    left, right = st.columns(2)
    mode = left.radio(
        "Hyperparameters",
        ["fixed", "random"] if searchable else ["fixed"],
        format_func={"fixed": "Fixed", "random": "Random search"}.get,
        horizontal=True,
        key=f"mode.{algorithm}",
    )
    n_iter = 1
    if mode == "random":
        n_iter = int(right.number_input("Candidates (n_iter)", min_value=1, max_value=50, value=10))
    folds = get_settings().cv_folds
    st.caption(
        f"This job evaluates **{n_iter} Candidate{'s' if n_iter != 1 else ''}** with "
        f"{folds}-fold CV ({n_iter * folds} fits), then refits the winner once."
        + ("" if searchable else " This algorithm has no search space; it runs fixed only.")
    )

    with st.form("experiment-config"):
        name = st.text_input("Name", value=f"{algorithm}-{mode}", key=f"name.{algorithm}.{mode}")
        left, right = st.columns(2)
        bedrooms_per_room = left.toggle("Add bedrooms-per-room feature", value=True)
        target_transform = right.radio(
            "Target transform", ["none", "log1p"], horizontal=True, index=1
        )

        if mode == "random":
            st.markdown("**Search space**")
            space_text = st.text_area(
                "Lists are sampled uniformly; ranges are `{type: int|float, low, high}`, "
                "with `log: true` for a log-scaled float range.",
                value=_search_space_yaml(algorithm),
                height=150,
                key=f"space.{algorithm}",
            )
        else:
            st.markdown("**Hyperparameters** (fixed)")
            params = {}
            if not defaults:
                st.caption("This algorithm has no hyperparameters to set.")
            columns = st.columns(len(defaults)) if defaults else []
            for column, (param, default) in zip(columns, defaults.items(), strict=True):
                with column:
                    params[param] = _param_input(algorithm, param, default)

        if st.form_submit_button("Launch", type="primary"):
            if mode == "random":
                try:
                    space = yaml.safe_load(space_text)
                except yaml.YAMLError as exc:
                    st.error(f"The search space is not valid YAML:\n```\n{exc}\n```")
                    return
                tuning = {"mode": "random", "n_iter": n_iter, "params": space}
            else:
                tuning = {"mode": "fixed", "params": params}
            raw = {
                "name": name,
                "algorithm": algorithm,
                "features": {"bedrooms_per_room": bedrooms_per_room},
                "target_transform": target_transform,
                "tuning": tuning,
            }
            try:
                job = service.submit(_jobs(), raw)
            except ConfigError as exc:
                # A code block keeps one problem per line; st.error alone renders Markdown.
                st.error(f"```\n{exc}\n```")
            else:
                st.success(f"Queued {job.name} ({job.id}): {job.progress_total} Candidate(s).")


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

    # Every 2 seconds while a job is queued or running; otherwise a slow poll, so a job
    # submitted from the CLI or another tab still switches the fast refresh on.
    @st.fragment(run_every=REFRESH_SECONDS if active else IDLE_REFRESH_SECONDS)
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
        if active != any(not job.finished for job in jobs):
            # Rerun the whole page to change the refresh rate, and when jobs have finished,
            # to show their Runs on the Leaderboard.
            st.rerun()

    jobs_table()


# Compare ----------------------------------------------------------------------------------


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def _marker(row: LeaderboardRow, baseline_row: LeaderboardRow | None) -> str:
    if row.is_best:
        return "★ Best Run"
    return "Baseline" if row is baseline_row else ""


def _leaderboard_frame(rows: list[LeaderboardRow], mlflow_ui_url: str) -> pd.DataFrame:
    baseline_row = baseline(rows)
    return pd.DataFrame(
        [
            {
                "Best": _marker(row, baseline_row),
                "Config": row.config_name,
                "Algorithm": shared.algorithm_label(row.algorithm),
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
    best = row["Best"].startswith("★")
    style = "background-color: rgba(255, 196, 0, 0.25); font-weight: 600" if best else ""
    return [style] * len(row)


def _leaderboard_section() -> None:
    settings = get_settings()
    st.subheader("Leaderboard")
    try:
        dataset_sha256 = shared.current_dataset_sha256()
        rows = leaderboard(shared.mlflow_client(), dataset_sha256)
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
    if baseline_row := baseline(rows):
        st.markdown(
            f"**Baseline** (mean predictor, `{baseline_row.config_name}`): CV RMSE "
            f"{baseline_row.cv_rmse_mean:.4f} ± {baseline_row.cv_rmse_std:.4f}"
        )
    else:
        st.caption("No baseline yet: launch a Dummy Run to compare every model with a mean.")

    frame = _leaderboard_frame(rows, settings.mlflow_ui_url)
    # Selections are row positions, so a new key whenever the Runs change clears the
    # selection instead of silently pointing it at another Run.
    run_ids = ",".join(row.run_id for row in rows)
    selection = st.dataframe(
        frame.style.apply(_highlight_best, axis=1),
        hide_index=True,
        key=f"leaderboard-{hashlib.sha1(run_ids.encode()).hexdigest()[:12]}",
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
    details = run_details(shared.mlflow_client(), row.run_id)
    best = " · ★ Best Run" if row.is_best else ""
    st.markdown(f"#### {row.config_name} · {shared.algorithm_label(row.algorithm)}{best}")
    st.markdown(f"Run `{row.run_id}` · [Open in the MLflow UI]({run_url(row, mlflow_ui_url)})")
    metrics, params, tags = st.columns(3)
    metrics.markdown("**Metrics**")
    metrics.dataframe(
        pd.DataFrame(sorted(details.metrics.items()), columns=["Metric", "Value"]),
        hide_index=True,
    )
    params.markdown("**Params**")
    params.dataframe(
        pd.DataFrame(sorted(details.params.items()), columns=["Param", "Value"]),
        hide_index=True,
    )
    tags.markdown("**Tags**")
    tags.dataframe(
        pd.DataFrame(sorted(details.tags.items()), columns=["Tag", "Value"]), hide_index=True
    )
    _register(row)


def _register(row: LeaderboardRow) -> None:
    st.markdown("**Register as a Model Version**")
    st.caption(
        "Registering makes the Run deployable; it never changes the Active Model. "
        "Deploy it from the Deployments page."
    )
    name_column, button_column = st.columns([3, 1], vertical_alignment="bottom")
    name = name_column.text_input(
        "Registered Model", value=get_settings().registered_model, key=f"register.{row.run_id}"
    ).strip()
    registry = shared.registry()
    try:
        existing = registry.version_of_run(row.run_id, name) if name else None
    except RegistryError as exc:
        st.error(str(exc))
        return
    clicked = button_column.button(
        "Register",
        type="primary",
        disabled=not name or existing is not None,
        key=f"register-button.{row.run_id}",
    )
    if registered := st.session_state.pop(_REGISTERED, None):
        st.success(f"Registered as Model Version {registered}.")
    elif existing is not None:
        st.info(f"This Run is registered under {name} as Model Version {existing}.")
    if clicked:
        try:
            ref = registry.register_run(row.run_id, name)
        except RegistryError as exc:
            st.error(str(exc))
        else:
            # Rerun, so the button is drawn disabled for the Run now registered.
            st.session_state[_REGISTERED] = str(ref)
            st.rerun()
