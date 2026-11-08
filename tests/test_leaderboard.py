import uuid

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from app.settings import get_settings
from app.tracking.runs import leaderboard, run_url


@pytest.fixture
def dataset_sha256(mlflow_store) -> str:
    """A hash of its own per test, so Runs logged by other tests never appear."""
    mlflow.set_experiment(get_settings().mlflow_experiment)
    return uuid.uuid4().hex


def _log_run(
    dataset_sha256: str,
    *,
    name: str,
    cv_rmse: float,
    test_rmse: float = 0.5,
    algorithm: str = "random_forest",
    status: str = "FINISHED",
    params: dict | None = None,
) -> str:
    tags = {"config_name": name, "algorithm": algorithm, "dataset_sha256": dataset_sha256}
    with mlflow.start_run(run_name=name, tags=tags) as run:
        mlflow.log_params(params or {})
        mlflow.log_metrics(
            {
                "cv_rmse_mean": cv_rmse,
                "cv_rmse_std": 0.01,
                "cv_mae_mean": 0.3,
                "cv_r2_mean": 0.8,
                "test_rmse": test_rmse,
                "test_mae": 0.31,
                "test_r2": 0.79,
            }
        )
    if status != "FINISHED":
        MlflowClient().set_terminated(run.info.run_id, status=status)
    return run.info.run_id


def test_completed_runs_are_ranked_by_mean_cv_rmse_with_only_the_first_marked_best(
    dataset_sha256,
):
    _log_run(dataset_sha256, name="middle", cv_rmse=0.55)
    _log_run(dataset_sha256, name="best", cv_rmse=0.48)
    _log_run(dataset_sha256, name="worst", cv_rmse=0.72)

    rows = leaderboard(MlflowClient(), dataset_sha256)

    assert [row.config_name for row in rows] == ["best", "middle", "worst"]
    assert [row.is_best for row in rows] == [True, False, False]
    assert rows[0].cv_rmse_mean == pytest.approx(0.48)
    assert rows[0].cv_rmse_std == pytest.approx(0.01)
    assert rows[0].algorithm == "random_forest"


def test_only_completed_top_level_runs_on_the_current_dataset_are_listed(dataset_sha256):
    _log_run(dataset_sha256, name="done", cv_rmse=0.6)
    _log_run(dataset_sha256, name="failed", cv_rmse=0.4, status="FAILED")
    _log_run(dataset_sha256, name="running", cv_rmse=0.4, status="RUNNING")
    _log_run(uuid.uuid4().hex, name="other-dataset", cv_rmse=0.4)
    with (
        mlflow.start_run(
            run_name="with-candidate", tags={"dataset_sha256": dataset_sha256}
        ) as parent,
        mlflow.start_run(
            run_name="candidate", nested=True, tags={"dataset_sha256": dataset_sha256}
        ),
    ):
        mlflow.log_metrics({"cv_rmse_mean": 0.3, "cv_rmse_std": 0.01})
    MlflowClient().set_terminated(parent.info.run_id, status="FAILED")

    rows = leaderboard(MlflowClient(), dataset_sha256)

    assert [row.config_name for row in rows] == ["done"]


def test_test_rmse_is_shown_but_never_changes_the_ranking(dataset_sha256):
    _log_run(dataset_sha256, name="better-cv", cv_rmse=0.50, test_rmse=0.70)
    _log_run(dataset_sha256, name="better-test", cv_rmse=0.60, test_rmse=0.30)

    rows = leaderboard(MlflowClient(), dataset_sha256)

    assert [row.config_name for row in rows] == ["better-cv", "better-test"]
    assert rows[0].is_best
    assert rows[0].test_rmse == pytest.approx(0.70)
    assert (rows[0].test_mae, rows[0].test_r2) == (pytest.approx(0.31), pytest.approx(0.79))
    assert (rows[0].cv_mae_mean, rows[0].cv_r2_mean) == (pytest.approx(0.3), pytest.approx(0.8))


def test_key_params_are_the_estimators_hyperparameters_and_the_feature_options(dataset_sha256):
    _log_run(
        dataset_sha256,
        name="rf",
        cv_rmse=0.5,
        params={
            "n_estimators": 100,
            "max_depth": 12,
            "bedrooms_per_room": True,
            "target_transform": "log1p",
            "split_seed": 42,
            "cv_folds": 5,
            "random_state": 42,
        },
    )

    (row,) = leaderboard(MlflowClient(), dataset_sha256)

    assert row.params == {
        "n_estimators": "100",
        "max_depth": "12",
        "bedrooms_per_room": "True",
        "target_transform": "log1p",
    }


def test_each_row_links_to_its_run_in_the_mlflow_ui(dataset_sha256):
    run_id = _log_run(dataset_sha256, name="linked", cv_rmse=0.5)
    (row,) = leaderboard(MlflowClient(), dataset_sha256)

    url = run_url(row, "http://localhost:5001/")

    assert url == f"http://localhost:5001/#/experiments/{row.experiment_id}/runs/{run_id}"
