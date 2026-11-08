from app.jobs import service
from app.jobs.worker import recover_orphans

CONFIG = {"name": "lr", "algorithm": "linear_regression"}


def test_the_first_claim_of_a_queued_job_wins_and_a_second_changes_nothing(jobs):
    job = service.submit(jobs, CONFIG)

    assert jobs.claim(job.id) is True
    after_first = jobs.get(job.id)
    assert jobs.claim(job.id) is False

    assert after_first.status == "running"
    assert after_first.started_at is not None
    assert jobs.get(job.id) == after_first


def test_jobs_are_picked_in_submission_order(jobs):
    first = service.submit(jobs, CONFIG)
    second = service.submit(jobs, CONFIG)

    assert jobs.next_queued() == first.id
    jobs.claim(first.id)
    assert jobs.next_queued() == second.id


def test_on_startup_orphaned_running_jobs_are_marked_failed_and_others_left_alone(jobs):
    orphan = service.submit(jobs, CONFIG)
    jobs.claim(orphan.id)
    waiting = service.submit(jobs, CONFIG)

    recover_orphans(jobs)

    failed = jobs.get(orphan.id)
    assert failed.status == "failed"
    assert failed.error == "worker restarted"
    assert failed.finished_at is not None
    assert jobs.get(waiting.id).status == "queued"


def test_a_job_failed_as_an_orphan_is_not_revived_by_late_progress(jobs):
    job = service.submit(jobs, CONFIG)
    jobs.claim(job.id)
    recover_orphans(jobs)

    jobs.record_progress(job.id, progress_done=1)
    jobs.complete(job.id)

    assert jobs.get(job.id).status == "failed"
    assert jobs.get(job.id).progress_done == 0


def test_on_startup_the_run_of_an_orphaned_job_is_marked_failed_too(jobs, mlflow_store):
    import mlflow
    from mlflow.tracking import MlflowClient

    job = service.submit(jobs, CONFIG)
    jobs.claim(job.id)
    experiment = mlflow.get_experiment_by_name("california-housing")
    run = MlflowClient().create_run(experiment.experiment_id)
    jobs.record_progress(job.id, mlflow_run_id=run.info.run_id)

    recover_orphans(jobs)

    assert MlflowClient().get_run(run.info.run_id).info.status == "FAILED"
