import numpy as np
import pandas as pd
import pytest

from app.jobs.config import ExperimentConfig
from app.ml.data import load_dataset, split_dataset
from app.ml.pipeline import build_pipeline
from app.ml.schema import FEATURES
from app.settings import get_settings


@pytest.fixture(scope="module")
def split():
    return split_dataset(load_dataset(get_settings().data_path))


def _config(**overrides) -> ExperimentConfig:
    raw = {"name": "test", "algorithm": "linear_regression", **overrides}
    return ExperimentConfig.model_validate(raw)


def test_the_scaler_is_fitted_only_on_the_rows_it_is_trained_on(split):
    subset_X, subset_y = split.X_train.iloc[:500], split.y_train.iloc[:500]
    pipeline = build_pipeline(_config(), {})

    pipeline.fit(subset_X, subset_y)

    scaler = pipeline.named_steps["scaler"]
    np.testing.assert_allclose(scaler.mean_, subset_X.mean().to_numpy())
    assert not np.allclose(scaler.mean_, split.X_train.mean().to_numpy())


def test_bedrooms_per_room_is_finite_when_a_district_has_zero_rooms(split):
    pipeline = build_pipeline(_config(features={"bedrooms_per_room": True}), {})
    pipeline.fit(split.X_train.iloc[:500], split.y_train.iloc[:500])
    zero_rooms = split.X_test.iloc[:3].assign(AveRooms=0.0)

    features = pipeline.named_steps["bedrooms_per_room"].transform(zero_rooms)
    predictions = pipeline.predict(zero_rooms)

    assert "BedroomsPerRoom" in features.columns
    assert np.isfinite(features["BedroomsPerRoom"]).all()
    assert np.isfinite(predictions).all()


def test_fitting_and_predicting_does_not_mutate_the_input(split):
    X, y = split.X_train.iloc[:500].copy(), split.y_train.iloc[:500].copy()
    X_before, y_before = X.copy(), y.copy()
    pipeline = build_pipeline(
        _config(features={"bedrooms_per_room": True}, target_transform="log1p"), {}
    )

    pipeline.fit(X, y)
    pipeline.predict(X)

    pd.testing.assert_frame_equal(X, X_before)
    pd.testing.assert_series_equal(y, y_before)
    assert list(X.columns) == list(FEATURES)


def test_with_log1p_predictions_are_in_original_units(split):
    pipeline = build_pipeline(
        _config(algorithm="gradient_boosting", target_transform="log1p"),
        {"n_estimators": 50},
    )

    pipeline.fit(split.X_train, split.y_train)
    predictions = pipeline.predict(split.X_test)

    # MedHouseVal in 100k USD averages about 2; in log1p space it would average about 1.
    assert predictions.mean() == pytest.approx(split.y_test.mean(), rel=0.1)


def test_fixed_hyperparameters_use_estimator_level_names():
    pipeline = build_pipeline(
        _config(algorithm="random_forest", target_transform="log1p"),
        {"n_estimators": 7, "max_depth": 3},
    )

    estimator = pipeline.regressor.named_steps["estimator"]
    assert estimator.n_estimators == 7
    assert estimator.max_depth == 3
    assert estimator.random_state == 42
