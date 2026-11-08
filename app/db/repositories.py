"""The only code that reads or writes our tables."""

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import Engine, func, insert, select, update

from app.db.tables import deployments, training_jobs


@dataclass(frozen=True)
class Job:
    id: uuid.UUID
    name: str
    config: dict[str, Any]
    status: str
    phase: str | None
    progress_done: int
    progress_total: int
    mlflow_run_id: str | None
    error: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    @property
    def finished(self) -> bool:
        return self.status in ("completed", "failed")


class JobRepository:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def insert(self, *, name: str, config: dict[str, Any], progress_total: int) -> Job:
        with self._engine.begin() as conn:
            row = conn.execute(
                insert(training_jobs)
                .values(name=name, config=config, progress_total=progress_total)
                .returning(training_jobs)
            ).one()
        return Job(**row._mapping)

    def get(self, job_id: uuid.UUID) -> Job:
        with self._engine.connect() as conn:
            row = conn.execute(select(training_jobs).where(training_jobs.c.id == job_id)).one()
        return Job(**row._mapping)

    def recent(self, limit: int = 50) -> list[Job]:
        """Most recently created first."""
        query = select(training_jobs).order_by(training_jobs.c.created_at.desc()).limit(limit)
        with self._engine.connect() as conn:
            return [Job(**row._mapping) for row in conn.execute(query)]

    def next_queued(self) -> uuid.UUID | None:
        """The oldest queued job, so jobs run in submission order."""
        query = (
            select(training_jobs.c.id)
            .where(training_jobs.c.status == "queued")
            .order_by(training_jobs.c.created_at, training_jobs.c.id)
            .limit(1)
        )
        with self._engine.connect() as conn:
            return conn.execute(query).scalar_one_or_none()

    def claim(self, job_id: uuid.UUID) -> bool:
        """Move a queued job to running. True only for the one caller whose update won."""
        with self._engine.begin() as conn:
            result = conn.execute(
                update(training_jobs)
                .where(training_jobs.c.id == job_id, training_jobs.c.status == "queued")
                .values(status="running", started_at=func.now())
            )
        return result.rowcount == 1

    def record_progress(self, job_id: uuid.UUID, **fields: Any) -> None:
        """Update `phase`, `progress_done` or `mlflow_run_id` of a running job."""
        allowed = {"phase", "progress_done", "mlflow_run_id"}
        if unknown := set(fields) - allowed:
            raise ValueError(f"not progress fields: {', '.join(sorted(unknown))}")
        self._update_running(job_id, **fields)

    def complete(self, job_id: uuid.UUID) -> None:
        self._update_running(job_id, status="completed", phase=None, finished_at=func.now())

    def fail(self, job_id: uuid.UUID, error: str) -> None:
        self._update_running(job_id, status="failed", error=error, finished_at=func.now())

    def fail_orphans(self, error: str = "worker restarted") -> list[Job]:
        """Mark every running job failed and return them. Only safe with a single worker."""
        with self._engine.begin() as conn:
            rows = conn.execute(
                update(training_jobs)
                .where(training_jobs.c.status == "running")
                .values(status="failed", error=error, finished_at=func.now())
                .returning(training_jobs)
            )
            return [Job(**row._mapping) for row in rows]

    def _update_running(self, job_id: uuid.UUID, **values: Any) -> None:
        # Guarded on status, so a job already marked failed (e.g. orphaned) stays failed.
        with self._engine.begin() as conn:
            conn.execute(
                update(training_jobs)
                .where(training_jobs.c.id == job_id, training_jobs.c.status == "running")
                .values(**values)
            )


@dataclass(frozen=True)
class Deployment:
    id: int
    environment: str
    model_name: str
    model_version: int
    created_at: datetime


class DeploymentRepository:
    """Append-only: there is deliberately no update or delete."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def append(self, environment: str, model_name: str, model_version: int) -> Deployment:
        with self._engine.begin() as conn:
            row = conn.execute(
                insert(deployments)
                .values(environment=environment, model_name=model_name, model_version=model_version)
                .returning(deployments)
            ).one()
        return Deployment(**row._mapping)

    def latest(self, environment: str) -> Deployment | None:
        history = self.history(environment, limit=1)
        return history[0] if history else None

    def history(self, environment: str, limit: int = 50) -> list[Deployment]:
        """Newest first."""
        query = (
            select(deployments)
            .where(deployments.c.environment == environment)
            .order_by(deployments.c.id.desc())
            .limit(limit)
        )
        with self._engine.connect() as conn:
            return [Deployment(**row._mapping) for row in conn.execute(query)]
