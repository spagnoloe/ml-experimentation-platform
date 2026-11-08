"""Submit and follow Training Jobs. The CLI and the UI both go through here."""

import time
import uuid
from typing import Any

from app.db.repositories import Job, JobRepository
from app.jobs.config import ExperimentConfig, parse_config


def submit(jobs: JobRepository, raw_config: dict[str, Any] | ExperimentConfig) -> Job:
    """Validate the Experiment Config and queue a Training Job. Raises ConfigError if invalid."""
    config = parse_config(raw_config)
    return jobs.insert(
        name=config.name,
        config=config.model_dump(mode="json"),
        progress_total=config.candidate_count,
    )


def recent(jobs: JobRepository, limit: int = 50) -> list[Job]:
    """The most recently submitted Training Jobs, newest first."""
    return jobs.recent(limit)


def wait(
    jobs: JobRepository,
    job_id: uuid.UUID,
    *,
    poll_seconds: float = 1.0,
    timeout_seconds: float | None = None,
) -> Job:
    """Block until the job is completed or failed, and return it."""
    deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
    while not (job := jobs.get(job_id)).finished:
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError(f"job {job_id} still {job.status} after {timeout_seconds}s")
        time.sleep(poll_seconds)
    return job
