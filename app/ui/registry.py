"""Registry page: Registered Models and their Model Versions, read-only."""

import pandas as pd
import streamlit as st

from app.registry.client import RegisteredModelSummary
from app.tracking.runs import best_run
from app.ui import shared


def render() -> None:
    st.title("Registry")
    st.caption(
        "Registered Models and their Model Versions, newest first. Register a Run from the "
        "Experiments page; deploy a Model Version from the Deployments page."
    )
    try:
        best = best_run(shared.mlflow_client(), shared.current_dataset_sha256())
        models = shared.registry().registered_models(best_run_id=best.run_id if best else None)
    except Exception as exc:  # MLflow down or Dataset unreadable.
        st.error(f"Cannot load the registry: {exc}")
        return
    if not models:
        st.info("No Registered Models yet. Register a Run from the Experiments page.")
        return
    for model in models:
        st.subheader(model.name)
        if not model.versions:
            st.caption("No Model Versions.")
            continue
        st.dataframe(_versions_frame(model), hide_index=True)


def _metric(tags: dict[str, str], name: str) -> str:
    try:
        return f"{float(tags[name]):.4f}"
    except (KeyError, ValueError):
        return ""


def _versions_frame(model: RegisteredModelSummary) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Version": version.ref.model_version,
                "Best": "★ best" if version.is_best else "",
                "Algorithm": shared.algorithm_label(version.tags.get("algorithm")),
                "CV RMSE": f"{_metric(version.tags, 'cv_rmse_mean')} ± "
                f"{_metric(version.tags, 'cv_rmse_std')}",
                "Test RMSE": _metric(version.tags, "test_rmse"),
                "Dataset": version.tags.get("dataset_sha256", "")[:12],
                "Job": version.tags.get("job_id", ""),
                "Run": version.run_id,
                "Registered": version.created_at,
            }
            for version in model.versions
        ]
    )
