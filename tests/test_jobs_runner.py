import platform

import mlflow
import numpy as np
import pandas as pd
import pytest
import sklearn
from mlflow.tracking import MlflowClient

from app.jobs import service
from app.jobs.runner import run_job
from app.ml.data import load_dataset
from app.ml.schema import FEATURES, SAMPLE_ROWS
from app.settings import get_settings

PROVIDED_CSV_SHA256 = "ac6910c7db55ba6d0eb63e9fd0e18e69ad471ddf3928d25473a74808132d964f"

RF_LOG1P = {
    "name": "rf-log1p",
    "algorithm": "random_forest",
    "features": {"bedrooms_per_room": True},
    "target_transform": "log1p",
    "tuning": {"mode": "fixed", "params": {"n_estimators": 10, "max_depth": 6}},
}


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(get_settings().data_path)


def _run(run_id: str):
    return MlflowClient().get_run(run_id)


def test_a_fixed_job_completes_with_a_run_holding_one_candidate_metrics_and_a_model(
    jobs, mlflow_store, dataset
):
    job = service.submit(jobs, RF_LOG1P)
    jobs.claim(job.id)

    run_job(jobs, job.id, dataset=dataset)

    done = jobs.get(job.id)
    assert done.status == "completed", done.error
    assert (done.progress_done, done.progress_total) == (1, 1)
    assert done.finished_at is not None and done.error is None

    run = _run(done.mlflow_run_id)
    experiment = MlflowClient().get_experiment(run.info.experiment_id)
    assert experiment.name == "california-housing"
    assert run.info.status == "FINISHED"
    assert run.data.tags["job_id"] == str(job.id)
    assert run.data.tags["config_name"] == "rf-log1p"
    assert run.data.tags["algorithm"] == "random_forest"
    assert run.data.tags["tuning_mode"] == "fixed"
    assert run.data.tags["dataset_sha256"] == PROVIDED_CSV_SHA256
    assert run.data.tags["python_version"] == platform.python_version()
    assert run.data.tags["sklearn_version"] == sklearn.__version__
    assert run.data.tags["git_sha"]
    assert run.data.params["split_seed"] == "42"
    assert run.data.params["test_size"] == "0.2"
    assert run.data.params["cv_folds"] == "5"
    assert run.data.params["n_estimators"] == "10"
    assert run.data.params["max_depth"] == "6"
    for metric in ("cv_rmse_mean", "cv_rmse_std", "cv_mae_mean", "cv_r2_mean"):
        assert metric in run.data.metrics
    for metric in ("test_rmse", "test_mae", "test_r2"):
        assert metric in run.data.metrics
    config_text = mlflow.artifacts.load_text(f"runs:/{run.info.run_id}/config.yaml")
    assert "rf-log1p" in config_text

    children = MlflowClient().search_runs(
        [run.info.experiment_id], filter_string=f"tags.mlflow.parentRunId = '{run.info.run_id}'"
    )
    assert len(children) == 1
    assert children[0].data.params["n_estimators"] == "10"
    assert children[0].data.metrics["cv_rmse_mean"] == run.data.metrics["cv_rmse_mean"]


def test_with_log1p_the_logged_model_and_every_metric_are_in_original_units(
    jobs, mlflow_store, dataset
):
    job = service.submit(jobs, RF_LOG1P)
    jobs.claim(job.id)

    run_job(jobs, job.id, dataset=dataset)

    run_id = jobs.get(job.id).mlflow_run_id
    metrics = _run(run_id).data.metrics
    # In original units (100k USD) the RMSE of a weak forest is about 0.6; in log1p space ~0.2.
    assert 0.4 < metrics["test_rmse"] < 1.0
    assert 0.4 < metrics["cv_rmse_mean"] < 1.0
    assert 0.5 < metrics["test_r2"] < 1.0

    model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    assert model.metadata.get_input_schema().input_names() == list(FEATURES)
    predictions = np.asarray(model.predict(pd.DataFrame(SAMPLE_ROWS)))
    assert predictions.shape == (len(SAMPLE_ROWS),)
    assert np.isfinite(predictions).all()
    assert 0.3 < predictions.mean() < 5.5


def test_an_exception_during_a_job_marks_the_job_and_the_run_failed(jobs, mlflow_store, dataset):
    # Valid at submit (known name), but scikit-learn rejects the value when fitting.
    broken = {**RF_LOG1P, "tuning": {"mode": "fixed", "params": {"n_estimators": -1}}}
    job = service.submit(jobs, broken)
    jobs.claim(job.id)

    run_job(jobs, job.id, dataset=dataset)

    failed = jobs.get(job.id)
    assert failed.status == "failed"
    assert "n_estimators" in failed.error
    assert failed.finished_at is not None
    assert _run(failed.mlflow_run_id).info.status == "FAILED"
