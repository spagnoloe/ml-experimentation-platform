"""Populate a fresh system: train, register and deploy through the real code path.

Submits every `configs/seed/*.yaml` like any other Experiment Config, waits for the worker,
registers each Run as a Model Version, and deploys the Best Run through `POST /deploy`.
Running it again adds new Runs and Model Versions; it never resets anything.
"""

from collections.abc import Callable
from pathlib import Path

import httpx
import yaml
from mlflow.tracking import MlflowClient
from sqlalchemy import make_url
from sqlalchemy.exc import SQLAlchemyError

from app.db.repositories import Job, JobRepository
from app.db.session import get_engine
from app.jobs import service
from app.jobs.config import ConfigError, ExperimentConfig, parse_config
from app.ml.schema import SAMPLE_ROWS
from app.registry.client import MlflowRegistryClient
from app.registry.model_version import ModelVersionRef
from app.settings import get_settings
from app.tracking.runs import leaderboard

SEED_CONFIGS = Path(__file__).resolve().parent.parent / "configs" / "seed"
# Per job; seed jobs take seconds, so hitting this means the worker is not picking them up.
JOB_TIMEOUT_SECONDS = 300
HTTP_TIMEOUT_SECONDS = 60


class SeedError(Exception):
    """Seed cannot finish; the message says why and what to do."""


def _print(message: str) -> None:
    # Flushed, so progress interleaves correctly with MLflow's own messages on stderr.
    print(message, flush=True)


def seed(log: Callable[[str], None] = _print) -> ModelVersionRef:
    """Run the whole seed and return the Model Version it left as the Active Model."""
    settings = get_settings()
    configs = _load_configs(SEED_CONFIGS)
    jobs = _check_stack()

    queued = [service.submit(jobs, config) for config in configs]
    for job in queued:
        log(f"queued job {job.id} ({job.name})")
    finished = [_wait(jobs, job, log) for job in queued]
    if failed := [job for job in finished if job.status != "completed"]:
        details = "\n".join(f"  {job.name} (job {job.id}): {job.error}" for job in failed)
        raise SeedError(f"{len(failed)} seed job(s) failed; nothing was registered:\n{details}")

    registry = MlflowRegistryClient(settings.mlflow_tracking_uri)
    for job in finished:
        ref = registry.register_run(job.mlflow_run_id, settings.registered_model)
        log(f"registered Run {job.mlflow_run_id} ({job.name}) as Model Version {ref}")

    ref = _best_run_version(registry, finished[0].mlflow_run_id, log)
    _deploy_and_check(ref, log)
    return ref


def _load_configs(directory: Path) -> list[ExperimentConfig]:
    """Every seed config, all validated before any is submitted."""
    paths = sorted(directory.glob("*.yaml"))
    if not paths:
        raise SeedError(f"no seed configs in {directory}")
    configs = []
    for path in paths:
        raw = yaml.safe_load(path.read_text())
        if not isinstance(raw, dict):
            raise SeedError(f"{path}: expected a YAML mapping")
        try:
            configs.append(parse_config(raw))
        except ConfigError as exc:
            raise SeedError(f"{path}: {exc}") from None
    return configs


def _check_stack() -> JobRepository:
    """Fail fast, before submitting anything, if any service seed needs is unreachable."""
    settings = get_settings()
    hint = "is the stack running? Start it with `make up`"
    try:
        jobs = JobRepository(get_engine())
    except SQLAlchemyError as exc:
        where = make_url(settings.app_db_url).render_as_string(hide_password=True)
        raise SeedError(f"cannot reach Postgres at {where}: {hint}") from exc
    for service_name, url in (
        ("MLflow", f"{settings.mlflow_tracking_uri.rstrip('/')}/health"),
        ("the API", f"{settings.api_url.rstrip('/')}/health"),
    ):
        try:
            httpx.get(url, timeout=5).raise_for_status()
        except httpx.HTTPError as exc:
            raise SeedError(f"cannot reach {service_name} at {url}: {hint}") from exc
    return jobs


def _wait(jobs: JobRepository, job: Job, log: Callable[[str], None]) -> Job:
    try:
        done = service.wait(jobs, job.id, timeout_seconds=JOB_TIMEOUT_SECONDS)
    except TimeoutError as exc:
        raise SeedError(
            f"{exc}; is the worker running? Check `docker compose ps` and `make logs`"
        ) from exc
    log(f"job {done.id} ({done.name}): {done.status}")
    return done


def _best_run_version(
    registry: MlflowRegistryClient, seeded_run_id: str, log: Callable[[str], None]
) -> ModelVersionRef:
    """A Model Version of the Best Run, registering the Run first if it has none.

    The Best Run is taken over every completed Run on the Dataset the seed jobs trained on,
    so it can be an earlier Run, from an earlier seed or from the data scientist.
    """
    settings = get_settings()
    client = MlflowClient(settings.mlflow_tracking_uri)
    dataset_sha256 = client.get_run(seeded_run_id).data.tags["dataset_sha256"]
    rows = leaderboard(client, dataset_sha256)
    if not rows:
        raise SeedError("no completed Run on the Leaderboard after seeding")
    best = rows[0]
    log(f"Best Run: {best.run_id} ({best.config_name}), mean CV RMSE {best.cv_rmse_mean:.4f}")
    versions = registry.versions_of_run(best.run_id, settings.registered_model)
    if versions:
        return versions[-1]
    ref = registry.register_run(best.run_id, settings.registered_model)
    log(f"registered the Best Run as Model Version {ref}")
    return ref


def _deploy_and_check(ref: ModelVersionRef, log: Callable[[str], None]) -> None:
    api = get_settings().api_url.rstrip("/")
    body = {"model_name": ref.model_name, "version": ref.model_version}
    try:
        response = httpx.post(f"{api}/deploy", json=body, timeout=HTTP_TIMEOUT_SECONDS)
    except httpx.HTTPError as exc:
        raise SeedError(f"cannot reach the API at {api}/deploy: {exc}") from exc
    if response.is_error:
        raise SeedError(f"deploying {ref} failed ({response.status_code}): {_detail(response)}")
    log(f"deployed Model Version {ref}; it is the Active Model")

    try:
        response = httpx.post(
            f"{api}/v1/predict", json={"rows": [SAMPLE_ROWS[0]]}, timeout=HTTP_TIMEOUT_SECONDS
        )
    except httpx.HTTPError as exc:
        raise SeedError(f"cannot reach the API at {api}/v1/predict: {exc}") from exc
    if response.is_error:
        raise SeedError(f"test prediction failed ({response.status_code}): {_detail(response)}")
    answer = response.json()
    answered_by = ModelVersionRef(answer["model_name"], answer["model_version"])
    if answered_by != ref:
        raise SeedError(f"/v1/predict answered with {answered_by}, expected {ref}")
    log(f"/v1/predict answered {answer['predictions'][0]:.3f} {answer['unit']} with {ref}")


def _detail(response: httpx.Response) -> str:
    try:
        return str(response.json().get("detail", response.text))
    except ValueError:
        return response.text
