"""Read completed Runs back from MLflow: the Leaderboard and links to the MLflow UI."""

from dataclasses import dataclass, field

from mlflow.entities import Run
from mlflow.tracking import MlflowClient

from app.ml.algorithms import ALGORITHMS
from app.settings import get_settings

# Run params that describe the Experiment Config rather than the shared evaluation protocol.
_FEATURE_PARAMS = ("bedrooms_per_room", "target_transform")


@dataclass(frozen=True)
class LeaderboardRow:
    run_id: str
    experiment_id: str
    config_name: str
    algorithm: str
    cv_rmse_mean: float
    cv_rmse_std: float
    cv_mae_mean: float | None
    cv_r2_mean: float | None
    # Shown for information only: the Test Set never decides the ranking.
    test_rmse: float | None
    test_mae: float | None
    test_r2: float | None
    is_best: bool
    # The winning Candidate's hyperparameters plus the feature options, as MLflow strings.
    params: dict[str, str] = field(default_factory=dict)


def leaderboard(client: MlflowClient, dataset_sha256: str) -> list[LeaderboardRow]:
    """Completed Runs on the Dataset with this hash, best (lowest mean CV RMSE) first."""
    experiment = client.get_experiment_by_name(get_settings().mlflow_experiment)
    if experiment is None:
        return []
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=(
            f"tags.dataset_sha256 = '{dataset_sha256}' and attributes.status = 'FINISHED'"
        ),
        order_by=["metrics.cv_rmse_mean ASC"],
    )
    # Candidates are child runs of their Run; only Runs belong on the Leaderboard.
    runs = [
        run
        for run in runs
        if "mlflow.parentRunId" not in run.data.tags and "cv_rmse_mean" in run.data.metrics
    ]
    return [_row(run, is_best=rank == 0) for rank, run in enumerate(runs)]


def _row(run: Run, *, is_best: bool) -> LeaderboardRow:
    metrics = run.data.metrics
    algorithm = run.data.tags.get("algorithm", "")
    return LeaderboardRow(
        run_id=run.info.run_id,
        experiment_id=run.info.experiment_id,
        config_name=run.data.tags.get("config_name", run.info.run_name),
        algorithm=algorithm,
        cv_rmse_mean=metrics["cv_rmse_mean"],
        cv_rmse_std=metrics.get("cv_rmse_std", 0.0),
        cv_mae_mean=metrics.get("cv_mae_mean"),
        cv_r2_mean=metrics.get("cv_r2_mean"),
        test_rmse=metrics.get("test_rmse"),
        test_mae=metrics.get("test_mae"),
        test_r2=metrics.get("test_r2"),
        is_best=is_best,
        params=_key_params(algorithm, run.data.params),
    )


def _key_params(algorithm: str, params: dict[str, str]) -> dict[str, str]:
    known = ALGORITHMS.get(algorithm)
    # random_state is set once per config and logged with the protocol, not as a hyperparameter.
    hyperparameters = (known.param_names() if known else set()) - {"random_state"}
    return {
        name: value
        for name, value in params.items()
        if name in hyperparameters or name in _FEATURE_PARAMS
    }


def run_url(row: LeaderboardRow, mlflow_ui_url: str) -> str:
    """The Run's page in the MLflow UI, at the address a browser reaches it on."""
    return f"{mlflow_ui_url.rstrip('/')}/#/experiments/{row.experiment_id}/runs/{row.run_id}"
