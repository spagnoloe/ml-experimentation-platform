"""The serving API, through the HTTP test client, against `app_test` and a fake registry."""

import json
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sqlalchemy import text

from app.api.main import create_app
from app.ml.schema import FEATURES, SAMPLE_ROWS
from app.registry.client import RegistryError, VersionMetadata

ROW = SAMPLE_ROWS[0]


@dataclass
class FakeModel:
    pipeline: object
    input_names: list[str] | None

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(frame)


class FakeRegistry:
    """Serves small, real fitted pipelines by `(model_name, version)`."""

    def __init__(self) -> None:
        self.models: dict[tuple[str, int], tuple[FakeModel, VersionMetadata]] = {}

    def add(self, name: str, version: int, pipeline, *, input_names=FEATURES, algorithm="lr"):
        names = None if input_names is None else list(input_names)
        metadata = VersionMetadata(run_id=f"run-{name}-{version}", algorithm=algorithm)
        self.models[(name, version)] = (FakeModel(pipeline, names), metadata)

    def _get(self, name: str, version: int):
        try:
            return self.models[(name, version)]
        except KeyError:
            raise RegistryError(f"no Model Version {name}/{version}") from None

    def load_model(self, name: str, version: int) -> FakeModel:
        return self._get(name, version)[0]

    def get_version_metadata(self, name: str, version: int) -> VersionMetadata:
        return self._get(name, version)[1]


def _fitted(scale: float) -> LinearRegression:
    """A linear model that predicts exactly `scale * MedInc`, so expected values are known."""
    X = pd.DataFrame(SAMPLE_ROWS, columns=list(FEATURES))
    return LinearRegression().fit(X, scale * X["MedInc"])


def _predicts_infinity() -> TransformedTargetRegressor:
    X = pd.DataFrame(SAMPLE_ROWS, columns=list(FEATURES))
    return TransformedTargetRegressor(
        regressor=LinearRegression(),
        func=lambda y: y,
        inverse_func=lambda y: np.full_like(y, np.inf),
        check_inverse=False,
    ).fit(X, X["MedInc"])


@pytest.fixture
def registry() -> FakeRegistry:
    registry = FakeRegistry()
    registry.add("california-housing", 1, _fitted(1.0))
    registry.add("california-housing", 2, _fitted(2.0), algorithm="rf")
    registry.add("california-housing", 3, _fitted(1.0), input_names=FEATURES[:-1])
    registry.add("california-housing", 4, _predicts_infinity())
    return registry


@pytest.fixture
def engine(db_engine):
    """`app_test` with our tables and no Deployments."""
    from app.db.session import create_tables

    create_tables(db_engine)
    with db_engine.begin() as conn:
        conn.execute(text("TRUNCATE deployments"))
    return db_engine


@pytest.fixture
def start(registry, engine):
    """Start the API (running its startup) and return the test client; stopped afterwards."""
    clients = []

    def _start() -> TestClient:
        client = TestClient(create_app(registry=registry, engine=engine))
        client.__enter__()
        clients.append(client)
        return client

    yield _start
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def api(start) -> TestClient:
    return start()


def _deploy(api: TestClient, version: int, name: str = "california-housing"):
    return api.post("/deploy", json={"model_name": name, "version": version})


def test_a_deployed_model_version_answers_a_batch_in_input_order(api):
    assert _deploy(api, 1).status_code == 200

    rows = [SAMPLE_ROWS[2], SAMPLE_ROWS[0], SAMPLE_ROWS[1]]
    response = api.post("/v1/predict", json={"rows": rows})

    assert response.status_code == 200
    body = response.json()
    assert body["predictions"] == pytest.approx([1.7159, 8.3252, 3.3841])
    assert body["unit"] == "100k USD"
    assert body["model_name"] == "california-housing"
    assert body["model_version"] == 1


def test_deploying_another_version_switches_the_active_model_and_both_deployments_are_listed(api):
    _deploy(api, 1)
    response = _deploy(api, 2)

    assert response.status_code == 200
    assert response.json()["model_version"] == 2
    predicted = api.post("/v1/predict", json={"rows": [ROW]}).json()
    assert predicted["model_version"] == 2
    assert predicted["predictions"] == pytest.approx([2 * ROW["MedInc"]])

    history = api.get("/deployments").json()
    assert [(d["model_name"], d["model_version"]) for d in history] == [
        ("california-housing", 2),
        ("california-housing", 1),
    ]
    assert all(d["environment"] == "local" and d["created_at"] for d in history)
    assert [d["model_version"] for d in api.get("/deployments?limit=1").json()] == [2]


def test_redeploying_the_loaded_model_version_returns_200_and_writes_no_deployment(api):
    _deploy(api, 1)

    response = _deploy(api, 1)

    assert response.status_code == 200
    assert response.json()["model_version"] == 1
    assert len(api.get("/deployments").json()) == 1


@pytest.mark.parametrize(
    ("version", "reason"),
    [
        (3, "signature"),  # signature lacks Longitude
        (4, "finite"),  # predicts inf
        (99, "99"),  # not in the registry
    ],
)
def test_a_model_version_that_fails_validation_is_rejected_and_the_previous_keeps_serving(
    api, version, reason
):
    _deploy(api, 1)

    response = _deploy(api, version)

    assert response.status_code == 422
    assert reason in response.json()["detail"]
    assert [d["model_version"] for d in api.get("/deployments").json()] == [1]
    assert api.post("/v1/predict", json={"rows": [ROW]}).json()["model_version"] == 1


def test_a_model_version_without_a_signature_is_rejected(registry, api):
    registry.add("no-signature", 1, _fitted(1.0), input_names=None)

    response = _deploy(api, 1, name="no-signature")

    assert response.status_code == 422
    assert "signature" in response.json()["detail"]


def test_if_the_deployment_cannot_be_written_deploy_returns_500_and_does_not_swap(registry, api):
    # A valid model whose version overflows the integer column, so the insert itself fails.
    too_big = 2**31
    registry.add("california-housing", too_big, _fitted(2.0))
    _deploy(api, 1)

    response = _deploy(api, too_big)

    assert response.status_code == 500
    assert [d["model_version"] for d in api.get("/deployments").json()] == [1]
    assert api.post("/v1/predict", json={"rows": [ROW]}).json()["model_version"] == 1


_WITHOUT_LATITUDE = {k: v for k, v in ROW.items() if k != "Latitude"}


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"rows": [_WITHOUT_LATITUDE]}, id="missing field"),
        pytest.param({"rows": [{**ROW, "Rooms": 3.0}]}, id="extra field"),
        pytest.param({"rows": [{**ROW, "MedInc": "8.3"}]}, id="string for a float"),
        pytest.param({"rows": [{**ROW, "MedInc": True}]}, id="bool for a float"),
        pytest.param({"rows": [{**ROW, "MedInc": None}]}, id="null for a float"),
        pytest.param({"rows": []}, id="empty batch"),
        pytest.param({"rows": [ROW] * 1001}, id="over 1000 rows"),
        pytest.param({}, id="no rows key"),
        pytest.param({"rows": [ROW], "model_version": 1}, id="extra top-level key"),
    ],
)
def test_a_malformed_predict_request_returns_422(api, body):
    _deploy(api, 1)

    assert api.post("/v1/predict", json=body).status_code == 422


def test_a_batch_of_1000_rows_with_integer_values_is_accepted(api):
    _deploy(api, 1)
    row = {**ROW, "HouseAge": 41, "Population": 322}

    response = api.post("/v1/predict", json={"rows": [row] * 1000})

    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 1000


def test_with_no_model_loaded_predict_returns_503(api):
    response = api.post("/v1/predict", json={"rows": [ROW]})

    assert response.status_code == 503


def test_health_and_model_report_the_loaded_and_desired_model_version(api):
    assert api.get("/health").json() == {"status": "degraded", "model_loaded": False}
    assert api.get("/v1/model").json() == {"loaded": None, "desired": None, "in_sync": False}

    _deploy(api, 2)

    assert api.get("/health").json() == {"status": "ok", "model_loaded": True}
    model = api.get("/v1/model").json()
    assert model["in_sync"] is True
    assert model["desired"] == {"model_name": "california-housing", "model_version": 2}
    loaded = model["loaded"]
    assert (loaded["model_name"], loaded["model_version"]) == ("california-housing", 2)
    assert loaded["run_id"] == "run-california-housing-2"
    assert loaded["algorithm"] == "rf"
    assert loaded["loaded_at"]


def test_at_startup_the_api_loads_the_latest_deployment_without_writing_one(start):
    first = start()
    _deploy(first, 1)
    _deploy(first, 2)

    restarted = start()

    assert restarted.get("/health").json()["model_loaded"] is True
    assert restarted.get("/v1/model").json()["in_sync"] is True
    assert restarted.post("/v1/predict", json={"rows": [ROW]}).json()["model_version"] == 2
    assert len(restarted.get("/deployments").json()) == 2


def test_if_the_latest_deployment_cannot_be_loaded_at_startup_the_api_starts_degraded(
    registry, start
):
    _deploy(start(), 2)
    del registry.models[("california-housing", 2)]

    restarted = start()

    assert restarted.get("/health").json() == {"status": "degraded", "model_loaded": False}
    model = restarted.get("/v1/model").json()
    assert model["loaded"] is None
    assert model["desired"] == {"model_name": "california-housing", "model_version": 2}
    assert model["in_sync"] is False
    assert restarted.post("/v1/predict", json={"rows": [ROW]}).status_code == 503


def test_each_predict_request_logs_one_structured_line(api, caplog):
    _deploy(api, 1)
    caplog.set_level(logging.INFO, logger="app.api")
    caplog.clear()

    api.post("/v1/predict", json={"rows": [ROW, ROW, ROW]})

    lines = [json.loads(r.getMessage()) for r in caplog.records if r.name == "app.api.predict"]
    assert len(lines) == 1
    assert lines[0]["model_name"] == "california-housing"
    assert lines[0]["model_version"] == 1
    assert lines[0]["rows"] == 3
    assert lines[0]["latency_ms"] >= 0
