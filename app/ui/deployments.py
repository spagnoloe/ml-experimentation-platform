"""Deployments page: the Active Model, deploying, the history and a test prediction.

Everything goes through the serving API (ADR-0001); this page never reads the registry.
"""

import pandas as pd
import streamlit as st

from app.api.client import ApiError
from app.ml.schema import FEATURES, SAMPLE_ROWS, TARGET_UNIT
from app.registry.model_version import MAX_MODEL_VERSION, ModelVersionRef
from app.settings import get_settings
from app.ui import shared

# Set by a successful deploy, shown once after the rerun that refreshes the page.
_DEPLOYED = "deployments.deployed"


def render() -> None:
    st.title("Deployments")
    _active_model()
    _deploy_form()
    _history()
    _test_prediction()


def _active_model() -> None:
    st.subheader("Active Model")
    try:
        status = shared.api().model_status()
    except ApiError as exc:
        st.error(f"Cannot read the Active Model: {exc}")
        return
    loaded, desired = status.loaded, status.desired
    if loaded is None:
        st.warning("No Model Version is loaded: `/v1/predict` answers 503.")
    else:
        ref = ModelVersionRef(loaded.model_name, loaded.model_version)
        st.markdown(
            f"Serving **{ref}** · {shared.algorithm_label(loaded.algorithm)} · "
            f"Run `{loaded.run_id}` · loaded {loaded.loaded_at:%Y-%m-%d %H:%M:%S} UTC"
        )
    named = "none" if desired is None else f"{desired.model_name}/{desired.model_version}"
    if status.in_sync:
        st.success(f"`in_sync: true` · the latest Deployment names {named}.")
    else:
        st.error(f"`in_sync: false` · the latest Deployment names {named}.")


def _deploy_form() -> None:
    st.subheader("Deploy a Model Version")
    if deployed := st.session_state.pop(_DEPLOYED, None):
        st.success(f"Deployed {deployed}; it is the Active Model.")
    with st.form("deploy"):
        name_column, version_column = st.columns([3, 1])
        name = name_column.text_input("Registered Model", value=get_settings().registered_model)
        version = version_column.number_input(
            "Version", min_value=1, max_value=MAX_MODEL_VERSION, value=1, step=1
        )
        submitted = st.form_submit_button("Deploy", type="primary")
    if not submitted:
        return
    ref = ModelVersionRef(name.strip(), int(version))
    try:
        shared.api().deploy(ref)
    except ApiError as exc:
        # The API's reason as-is; the Active Model above is unchanged.
        st.error(f"Deploying {ref} failed. {exc}")
        return
    st.session_state[_DEPLOYED] = str(ref)
    st.rerun()


def _history() -> None:
    st.subheader("Deployment history")
    try:
        deployments = shared.api().deployments()
    except ApiError as exc:
        st.error(f"Cannot read the Deployment history: {exc}")
        return
    if not deployments:
        st.info("No Deployments yet.")
        return
    st.caption("Every Deployment in this environment, newest first. Deployments never change.")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Deployed": d.created_at,
                    "Model Version": f"{d.model_name}/{d.model_version}",
                    "Environment": d.environment,
                    "Id": d.id,
                }
                for d in deployments
            ]
        ),
        hide_index=True,
        column_config={
            "Deployed": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm:ss"),
        },
    )


def _test_prediction() -> None:
    st.subheader("Test prediction")
    with st.form("predict"):
        columns = st.columns(4)
        row = {
            name: float(columns[i % 4].number_input(name, value=value, format="%g"))
            for i, (name, value) in enumerate(SAMPLE_ROWS[0].items())
        }
        submitted = st.form_submit_button("Predict")
    if not submitted:
        return
    try:
        answer = shared.api().predict([{name: row[name] for name in FEATURES}])
    except ApiError as exc:
        st.error(f"Prediction failed. {exc}")
        return
    answered_by = ModelVersionRef(answer.model_name, answer.model_version)
    st.metric(f"Predicted house value ({TARGET_UNIT})", f"{answer.predictions[0]:.3f}")
    st.markdown(f"Answered by Model Version **{answered_by}**.")
