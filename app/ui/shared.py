"""Clients and lookups the UI pages share, created once per Streamlit server."""

from pathlib import Path

import streamlit as st
from mlflow.tracking import MlflowClient

from app.api.client import ApiClient
from app.ml.data import load_dataset
from app.registry.client import MlflowRegistryClient
from app.settings import get_settings

ALGORITHM_LABELS = {
    "linear_regression": "Linear regression",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
    "dummy": "Dummy (mean baseline)",
}


def algorithm_label(algorithm: str | None) -> str:
    return "" if algorithm is None else ALGORITHM_LABELS.get(algorithm, algorithm)


@st.cache_resource
def mlflow_client() -> MlflowClient:
    uri = get_settings().mlflow_tracking_uri
    return MlflowClient(tracking_uri=uri, registry_uri=uri)


@st.cache_resource
def registry() -> MlflowRegistryClient:
    return MlflowRegistryClient(get_settings().mlflow_tracking_uri)


@st.cache_resource
def api() -> ApiClient:
    return ApiClient.at(get_settings().api_url)


@st.cache_data
def _dataset_sha256(path: str, modified_ns: int) -> str:
    # Keyed on the modification time too, so replacing the file changes the current Dataset.
    return load_dataset(path).sha256


def current_dataset_sha256() -> str:
    path = Path(get_settings().data_path)
    return _dataset_sha256(str(path), path.stat().st_mtime_ns)
