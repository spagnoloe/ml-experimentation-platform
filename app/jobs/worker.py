"""The training worker: claims queued Training Jobs one at a time and runs them.

Run with `python -m app.jobs.worker`. Exactly one worker is expected (see ADR-0003).
"""

import logging
import time

import mlflow
from mlflow.tracking import MlflowClient

from app.db.repositories import JobRepository
from app.db.session import get_engine
from app.jobs.runner import run_job
from app.settings import get_settings

log = logging.getLogger(__name__)


def recover_orphans(jobs: JobRepository) -> int:
    """Fail jobs left running by a previous worker process, and their Runs.

    With a single worker, any job still running at startup is an orphan.
    """
    orphans = jobs.fail_orphans("worker restarted")
    for job in orphans:
        log.warning("marked orphaned training job %s failed", job.id)
        if job.mlflow_run_id:
            try:
                MlflowClient().set_terminated(job.mlflow_run_id, status="FAILED")
            except Exception:
                # The job row is authoritative; a stale Run status must not stop the worker.
                log.exception("could not mark Run %s failed", job.mlflow_run_id)
    return len(orphans)


def work_once(jobs: JobRepository) -> bool:
    """Claim and run the oldest queued job. False when there was nothing to do."""
    job_id = jobs.next_queued()
    if job_id is None:
        return False
    if jobs.claim(job_id):
        log.info("running training job %s", job_id)
        run_job(jobs, job_id)
        log.info("training job %s finished: %s", job_id, jobs.get(job_id).status)
    return True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    jobs = JobRepository(get_engine())
    recover_orphans(jobs)
    log.info("worker polling every %ss", settings.worker_poll_seconds)
    while True:
        if not work_once(jobs):
            time.sleep(settings.worker_poll_seconds)


if __name__ == "__main__":
    main()
