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
    settings = get_settings()
    experiment = client.get_experiment_by_name(settings.mlflow_experiment)
    if experiment is None:
        return []
    # Runs scored with another split or other folds are not comparable (ADR-0004).
    protocol = {
        "split_seed": settings.split_seed,
        "test_size": settings.test_size,
        "cv_folds": settings.cv_folds,
        "cv_seed": settings.cv_seed,
    }
    filters = [
        f"tags.dataset_sha256 = '{dataset_sha256}'",
        "attributes.status = 'FINISHED'",
        *(f"params.{name} = '{value}'" for name, value in protocol.items()),
    ]
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=" and ".join(filters),
        # On a tie (e.g. a re-seed of the same config) the earliest Run stays the Best Run.
        order_by=["metrics.cv_rmse_mean ASC", "attributes.start_time ASC"],
    )
    # Candidates are child runs of their Run; only Runs belong on the Leaderboard.
    runs = [
        run
        for run in runs
        if "mlflow.parentRunId" not in run.data.tags and "cv_rmse_mean" in run.data.metrics
    ]
    return [_row(run, is_best=rank == 0) for rank, run in enumerate(runs)]


def best_run(client: MlflowClient, dataset_sha256: str) -> LeaderboardRow | None:
    """The Best Run on the Dataset with this hash, or None before any Run has completed."""
    rows = leaderboard(client, dataset_sha256)
    return rows[0] if rows else None


BASELINE_ALGORITHM = "dummy"


def baseline(rows: list[LeaderboardRow]) -> LeaderboardRow | None:
    """The best dummy Run among these ranked Leaderboard rows, or None without one.

    It is the reference every model should beat; it is ranked like any other Run.
    """
    return next((row for row in rows if row.algorithm == BASELINE_ALGORITHM), None)


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


@dataclass(frozen=True)
class RunDetails:
    metrics: dict[str, float]
    params: dict[str, str]
    # The tags this platform sets; MLflow's own `mlflow.*` tags are left out.
    tags: dict[str, str]


def run_details(client: MlflowClient, run_id: str) -> RunDetails:
    data = client.get_run(run_id).data
    return RunDetails(
        metrics=dict(data.metrics),
        params=dict(data.params),
        tags={k: v for k, v in data.tags.items() if not k.startswith("mlflow.")},
    )


def run_url(row: LeaderboardRow, mlflow_ui_url: str) -> str:
    """The Run's page in the MLflow UI, at the address a browser reaches it on."""
    return f"{mlflow_ui_url.rstrip('/')}/#/experiments/{row.experiment_id}/runs/{row.run_id}"
