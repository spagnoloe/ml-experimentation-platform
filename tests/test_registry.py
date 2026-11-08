"""The MLflow registry client against a real, temporary MLflow store with tiny logged models."""

import uuid

import mlflow
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from app.ml.schema import FEATURES, SAMPLE_ROWS
from app.registry.client import MlflowRegistryClient, RegistryError
from app.registry.model_version import ModelVersionRef
from app.settings import get_settings


@pytest.fixture
def registry(mlflow_store) -> MlflowRegistryClient:
    mlflow.set_experiment(get_settings().mlflow_experiment)
    return MlflowRegistryClient(mlflow_store)


@pytest.fixture
def model_name() -> str:
    """A Registered Model name of its own per test, so versions never collide across tests."""
    return f"test-{uuid.uuid4().hex[:8]}"


def _completed_run(*, cv_rmse: float = 0.5, algorithm: str = "random_forest") -> str:
    """A Run shaped like the runner's: tags, metrics and one logged pipeline."""
    X = pd.DataFrame(SAMPLE_ROWS, columns=list(FEATURES))
    tags = {
        "job_id": str(uuid.uuid4()),
        "algorithm": algorithm,
        "dataset_sha256": "abc123",
        "config_name": "cfg",
    }
    with mlflow.start_run(tags=tags) as run:
        mlflow.log_metrics({"cv_rmse_mean": cv_rmse, "cv_rmse_std": 0.02, "test_rmse": 0.61})
        mlflow.sklearn.log_model(LinearRegression().fit(X, X["MedInc"]), name="model")
    return run.info.run_id


def test_registering_a_run_creates_the_next_model_version_tagged_from_the_run(registry, model_name):
    first = registry.register_run(_completed_run(), model_name)
    run_id = _completed_run(cv_rmse=0.4321, algorithm="gradient_boosting")

    second = registry.register_run(run_id, model_name)

    assert (first, second) == (ModelVersionRef(model_name, 1), ModelVersionRef(model_name, 2))
    version = mlflow.MlflowClient().get_model_version(model_name, "2")
    assert version.run_id == run_id
    tags = version.tags
    assert tags["algorithm"] == "gradient_boosting"
    assert tags["dataset_sha256"] == "abc123"
    assert tags["job_id"]
    assert float(tags["cv_rmse_mean"]) == pytest.approx(0.4321)
    assert float(tags["cv_rmse_std"]) == pytest.approx(0.02)
    assert float(tags["test_rmse"]) == pytest.approx(0.61)


def test_a_run_is_registered_at_most_once_per_registered_model(registry, model_name):
    run_id = _completed_run()
    registry.register_run(run_id, model_name)

    with pytest.raises(RegistryError, match="already registered"):
        registry.register_run(run_id, model_name)

    assert registry.version_of_run(run_id, model_name) == ModelVersionRef(model_name, 1)
    # Another name is another line of Model Versions, so the same Run may go there too.
    assert registry.register_run(run_id, f"{model_name}-b") == ModelVersionRef(f"{model_name}-b", 1)


def test_an_unregistered_run_has_no_model_version(registry, model_name):
    assert registry.version_of_run(_completed_run(), model_name) is None


def test_registered_models_list_their_versions_newest_first_with_tags_and_the_best_badge(
    registry, model_name
):
    best_run = _completed_run(cv_rmse=0.41)
    registry.register_run(_completed_run(cv_rmse=0.55), model_name)
    registry.register_run(best_run, model_name)
    registry.register_run(best_run, f"{model_name}-b")

    models = {m.name: m for m in registry.registered_models(best_run_id=best_run)}

    versions = models[model_name].versions
    assert [v.ref.model_version for v in versions] == [2, 1]
    assert [v.is_best for v in versions] == [True, False]
    assert versions[0].run_id == best_run
    assert float(versions[1].tags["cv_rmse_mean"]) == pytest.approx(0.55)
    assert versions[0].created_at is not None
    assert [v.is_best for v in models[f"{model_name}-b"].versions] == [True]


def test_without_a_best_run_no_version_has_the_badge(registry, model_name):
    registry.register_run(_completed_run(), model_name)

    models = {m.name: m for m in registry.registered_models(best_run_id=None)}

    assert [v.is_best for v in models[model_name].versions] == [False]
