import pytest

from app.jobs import service
from app.jobs.config import ConfigError

FIXED_CONFIG = {
    "name": "rf-fixed",
    "algorithm": "random_forest",
    "features": {"bedrooms_per_room": True},
    "target_transform": "log1p",
    "tuning": {"mode": "fixed", "params": {"n_estimators": 20, "max_depth": 6}},
}


def test_submitting_a_valid_config_queues_one_job_with_one_candidate(jobs):
    job = service.submit(jobs, FIXED_CONFIG)

    stored = jobs.get(job.id)
    assert stored.status == "queued"
    assert stored.name == "rf-fixed"
    assert stored.config["tuning"]["params"] == {"n_estimators": 20, "max_depth": 6}
    assert (stored.progress_done, stored.progress_total) == (0, 1)
    assert stored.mlflow_run_id is None
    assert [j.id for j in jobs.recent()] == [job.id]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"algorithm": "svm"}, "algorithm"),
        (
            {"tuning": {"mode": "fixed", "params": {"n_trees": 10}}},
            "unknown hyperparameter.*n_trees",
        ),
        ({"tuning": {"mode": "fixed", "params": {"random_state": 1}}}, "random_state"),
        ({"target_transform": "sqrt"}, "target_transform"),
        ({"name": "  "}, "name"),
        ({"surprise": True}, "surprise"),
    ],
)
def test_an_invalid_config_is_rejected_at_submit_and_nothing_is_queued(jobs, change, message):
    with pytest.raises(ConfigError, match=message):
        service.submit(jobs, {**FIXED_CONFIG, **change})

    assert jobs.recent() == []
