"""Populate a fresh system: train, register and deploy through the real code path.

Submits every `configs/seed/*.yaml` like any other Experiment Config, waits for the worker,
registers each Run as a Model Version, and deploys the Best Run through `POST /deploy`.
Running it again adds new Runs and Model Versions; it never resets anything.
"""

from pathlib import Path
from typing import Any

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
from app.registry.client import MlflowRegistryClient, RegistryError
from app.registry.model_version import ModelVersionRef
from app.settings import Settings, get_settings
from app.tracking.runs import leaderboard

SEED_CONFIGS = Path(__file__).resolve().parent.parent / "configs" / "seed"
# Per job; seed jobs take seconds, so hitting this means the worker is not picking them up.
JOB_TIMEOUT_SECONDS = 300
HTTP_TIMEOUT_SECONDS = 60
_NOT_RUNNING_HINT = "is the stack running? Start it with `make up`"


class SeedError(Exception):
    """Seed cannot finish; the message says why and what to do."""


def seed() -> ModelVersionRef:
    """Run the whole seed and return the Model Version it left as the Active Model."""
    settings = get_settings()
    configs = _load_configs(SEED_CONFIGS)
    _check_services(settings)
    jobs = _connect_jobs(settings)

    queued = [service.submit(jobs, config) for config in configs]
    for job in queued:
        _log(f"queued job {job.id} ({job.name})")
    finished = [_wait(jobs, job) for job in queued]
    if failed := [job for job in finished if job.status != "completed"]:
        details = "\n".join(f"  {job.name} (job {job.id}): {job.error}" for job in failed)
        raise SeedError(f"{len(failed)} seed job(s) failed; nothing was registered:\n{details}")

    registry = MlflowRegistryClient(settings.mlflow_tracking_uri)
    try:
        for job in finished:
            ref = registry.register_run(job.mlflow_run_id, settings.registered_model)
            _log(f"registered Run {job.mlflow_run_id} ({job.name}) as Model Version {ref}")
        ref = _best_run_version(settings, registry, finished[0].mlflow_run_id)
    except RegistryError as exc:
        raise SeedError(str(exc)) from exc
    _deploy_and_check(settings, ref)
    return ref


def _log(message: str) -> None:
    # Flushed, so progress interleaves correctly with MLflow's own messages on stderr.
    print(message, flush=True)


def _load_configs(directory: Path) -> list[ExperimentConfig]:
    """Every seed config, all validated before any is submitted."""
    paths = sorted(directory.glob("*.yaml"))
    if not paths:
        raise SeedError(f"no seed configs in {directory}")
    configs = []
    for path in paths:
        try:
            raw = yaml.safe_load(path.read_text())
        except (OSError, yaml.YAMLError) as exc:
            raise SeedError(f"cannot read {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise SeedError(f"{path}: expected a YAML mapping")
        try:
            configs.append(parse_config(raw))
        except ConfigError as exc:
            raise SeedError(f"{path}: {exc}") from exc
    return configs


def _check_services(settings: Settings) -> None:
    """Fail before submitting anything if MLflow or the API is unreachable."""
    for name, base in (("MLflow", settings.mlflow_tracking_uri), ("the API", settings.api_url)):
        url = f"{base.rstrip('/')}/health"
        try:
            httpx.get(url, timeout=5).raise_for_status()
        except httpx.HTTPError as exc:
            raise SeedError(f"cannot reach {name} at {url}: {_NOT_RUNNING_HINT}") from exc


def _connect_jobs(settings: Settings) -> JobRepository:
    try:
        return JobRepository(get_engine())
    except SQLAlchemyError as exc:
        where = make_url(settings.app_db_url).render_as_string(hide_password=True)
        raise SeedError(f"cannot reach Postgres at {where}: {_NOT_RUNNING_HINT}") from exc


def _wait(jobs: JobRepository, job: Job) -> Job:
    try:
        done = service.wait(jobs, job.id, timeout_seconds=JOB_TIMEOUT_SECONDS)
    except TimeoutError as exc:
        raise SeedError(
            f"{exc}; is the worker running? Check `docker compose ps` and `make logs`"
        ) from exc
    _log(f"job {done.id} ({done.name}): {done.status}")
    return done


def _best_run_version(
    settings: Settings, registry: MlflowRegistryClient, seeded_run_id: str
) -> ModelVersionRef:
    """The Model Version of the Best Run, registering the Run first if it has none.

    The Best Run is taken over every completed Run on the Dataset the seed jobs trained on,
    so it can be an earlier Run, from an earlier seed or from the data scientist.
    """
    client = MlflowClient(settings.mlflow_tracking_uri)
    dataset_sha256 = client.get_run(seeded_run_id).data.tags["dataset_sha256"]
    rows = leaderboard(client, dataset_sha256)
    if not rows:
        raise SeedError("no completed Run on the Leaderboard after seeding")
    best = rows[0]
    _log(f"Best Run: {best.run_id} ({best.config_name}), mean CV RMSE {best.cv_rmse_mean:.4f}")
    if ref := registry.version_of_run(best.run_id, settings.registered_model):
        return ref
    ref = registry.register_run(best.run_id, settings.registered_model)
    _log(f"registered the Best Run as Model Version {ref}")
    return ref


def _deploy_and_check(settings: Settings, ref: ModelVersionRef) -> None:
    _post(settings, "/deploy", {"model_name": ref.model_name, "version": ref.model_version})
    _log(f"deployed Model Version {ref}; it is the Active Model")

    answer = _post(settings, "/v1/predict", {"rows": [SAMPLE_ROWS[0]]})
    answered_by = ModelVersionRef(answer["model_name"], answer["model_version"])
    if answered_by != ref:
        raise SeedError(f"/v1/predict answered with {answered_by}, expected {ref}")
    _log(f"/v1/predict answered {answer['predictions'][0]:.3f} {answer['unit']} with {ref}")


def _post(settings: Settings, path: str, body: dict[str, Any]) -> dict[str, Any]:
    url = f"{settings.api_url.rstrip('/')}{path}"
    try:
        response = httpx.post(url, json=body, timeout=HTTP_TIMEOUT_SECONDS)
    except httpx.HTTPError as exc:
        raise SeedError(f"cannot reach the API at {url}: {exc}") from exc
    if response.is_error:
        raise SeedError(f"POST {path} failed ({response.status_code}): {_detail(response)}")
    return response.json()


def _detail(response: httpx.Response) -> str:
    try:
        return str(response.json().get("detail", response.text))
    except ValueError:
        return response.text
