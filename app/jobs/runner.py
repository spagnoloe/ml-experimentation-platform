"""Run one claimed Training Job: train it and record the Run in MLflow."""

import logging
import platform
import subprocess
import uuid
from functools import lru_cache

import mlflow
import pandas as pd
import sklearn
import yaml
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema

from app.db.repositories import JobRepository
from app.jobs.config import ExperimentConfig
from app.ml.data import Dataset, load_dataset, split_dataset
from app.ml.pipeline import SKOPS_TRUSTED_TYPES
from app.ml.schema import FEATURES, SAMPLE_ROWS, TARGET
from app.ml.training import best_candidate, evaluate_candidate, refit_and_score
from app.settings import get_settings

log = logging.getLogger(__name__)

SIGNATURE = ModelSignature(
    inputs=Schema([ColSpec("double", name) for name in FEATURES]),
    outputs=Schema([ColSpec("double", TARGET)]),
)


def run_job(jobs: JobRepository, job_id: uuid.UUID, *, dataset: Dataset | None = None) -> None:
    """Train a claimed (running) job. Never raises: a failure is recorded on the job and Run."""
    try:
        _train_and_log(jobs, job_id, dataset)
    except Exception as exc:
        log.exception("training job %s failed", job_id)
        jobs.fail(job_id, f"{type(exc).__name__}: {exc}")
    else:
        jobs.complete(job_id)


def _train_and_log(jobs: JobRepository, job_id: uuid.UUID, dataset: Dataset | None) -> None:
    settings = get_settings()
    config = ExperimentConfig.model_validate(jobs.get(job_id).config)
    candidates = config.candidates()

    mlflow.set_experiment(settings.mlflow_experiment)
    tags = {
        "job_id": str(job_id),
        "config_name": config.name,
        "algorithm": config.algorithm,
        "tuning_mode": config.tuning.mode,
        "git_sha": _git_sha(),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
    }
    # Leaving the `with` block on an exception marks the Run FAILED.
    with mlflow.start_run(run_name=config.name, tags=tags) as run:
        jobs.record_progress(job_id, mlflow_run_id=run.info.run_id, phase="cv")
        config_yaml = yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False)
        mlflow.log_text(config_yaml, "config.yaml")
        mlflow.log_params(
            {
                "bedrooms_per_room": config.features.bedrooms_per_room,
                "target_transform": config.target_transform,
                "random_state": config.random_state,
                "split_seed": settings.split_seed,
                "test_size": settings.test_size,
                "cv_folds": settings.cv_folds,
                "cv_seed": settings.cv_seed,
            }
        )

        dataset = dataset or load_dataset(settings.data_path)
        mlflow.set_tag("dataset_sha256", dataset.sha256)
        split = split_dataset(dataset)

        results = []
        for index, params in enumerate(candidates):
            with mlflow.start_run(run_name=f"{config.name}-candidate-{index}", nested=True):
                result = evaluate_candidate(config, params, split)
                mlflow.log_params(params)
                mlflow.log_metrics(result.cv_metrics())
            results.append(result)
            jobs.record_progress(job_id, progress_done=index + 1)

        winner = best_candidate(results)
        mlflow.log_params(winner.params)
        mlflow.log_metrics(winner.cv_metrics())

        jobs.record_progress(job_id, phase="refit")
        pipeline, test_metrics = refit_and_score(config, winner.params, split)
        mlflow.log_metrics(test_metrics)

        jobs.record_progress(job_id, phase="logging")
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=SIGNATURE,
            input_example=pd.DataFrame(SAMPLE_ROWS, columns=list(FEATURES)),
            skops_trusted_types=list(SKOPS_TRUSTED_TYPES),
        )


@lru_cache
def _git_sha() -> str:
    if sha := get_settings().git_sha:
        return sha
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
