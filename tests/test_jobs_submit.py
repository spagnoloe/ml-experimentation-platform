import pytest

from app.jobs import service
from app.jobs.config import ConfigError
from app.ml.algorithms import ALGORITHMS

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


RANDOM_CONFIG = {
    "name": "rf-random",
    "algorithm": "random_forest",
    "tuning": {
        "mode": "random",
        "n_iter": 7,
        "params": {
            "n_estimators": {"type": "int", "low": 10, "high": 30},
            "max_features": {"type": "float", "low": 0.3, "high": 1.0},
            "min_impurity_decrease": {"type": "float", "low": 1e-6, "high": 1e-2, "log": True},
            "max_depth": [4, 6, 8],
        },
    },
}


def test_a_random_search_queues_one_job_with_n_iter_candidates(jobs):
    job = service.submit(jobs, RANDOM_CONFIG)

    stored = jobs.get(job.id)
    assert stored.config["tuning"]["mode"] == "random"
    assert stored.config["tuning"]["n_iter"] == 7
    assert (stored.progress_done, stored.progress_total) == (0, 7)


def _random(n_iter: int = 7, **params) -> dict:
    tuning = {**RANDOM_CONFIG["tuning"], "n_iter": n_iter}
    if params:
        tuning["params"] = params
    return {**RANDOM_CONFIG, "tuning": tuning}


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (_random(n_iter=0), "n_iter"),
        (_random(n_iter=51), "n_iter"),
        (_random(n_iter=2, n_trees=[10, 20]), "unknown hyperparameter.*n_trees"),
        (_random(n_iter=2, random_state=[1, 2]), "random_state"),
        (
            {**FIXED_CONFIG, "tuning": {"mode": "fixed", "params": {"max_depth": [4, 6]}}},
            "fixed mode takes single values",
        ),
        (
            {
                **FIXED_CONFIG,
                "tuning": {
                    "mode": "fixed",
                    "params": {"n_estimators": {"type": "int", "low": 10, "high": 30}},
                },
            },
            "fixed mode takes single values",
        ),
        (_random(max_depth={"type": "int", "low": 8, "high": 4}), "high must be at least low"),
        (_random(max_features={"type": "float", "low": 0.0, "high": 1.0, "log": True}), "low > 0"),
        (_random(max_depth={"type": "int", "low": 2, "high": 8, "log": True}), "log"),
        (_random(max_depth=[]), "max_depth"),
        (_random(n_iter=7, max_depth=[4, 6], min_samples_leaf=[1, 2, 3]), "only 6 distinct"),
        ({**RANDOM_CONFIG, "tuning": {"mode": "random", "params": {}}}, "at least one param"),
    ],
)
def test_an_invalid_random_search_is_rejected_at_submit_and_nothing_is_queued(
    jobs, config, message
):
    with pytest.raises(ConfigError, match=message):
        service.submit(jobs, config)

    assert jobs.recent() == []


@pytest.mark.parametrize("algorithm", ["random_forest", "gradient_boosting"])
def test_the_default_search_space_submits_as_a_random_search_of_ten_candidates(jobs, algorithm):
    config = {
        "name": f"{algorithm}-random",
        "algorithm": algorithm,
        "tuning": {"mode": "random", "params": ALGORITHMS[algorithm].search_space},
    }

    job = service.submit(jobs, config)

    assert jobs.get(job.id).progress_total == 10


def test_the_random_forest_default_search_space_is_capped_for_runtime():
    space = ALGORITHMS["random_forest"].search_space

    assert space["max_depth"]["high"] <= 20
    assert space["n_estimators"]["high"] <= 200


def test_tuning_without_a_mode_is_fixed(jobs):
    config = {**FIXED_CONFIG, "tuning": {"params": {"max_depth": 5}}}

    job = service.submit(jobs, config)

    assert jobs.get(job.id).config["tuning"] == {"mode": "fixed", "params": {"max_depth": 5}}
