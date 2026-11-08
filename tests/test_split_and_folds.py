import numpy as np
import pytest

from app.ml.data import cv_folds, load_dataset, split_dataset
from app.ml.schema import FEATURES, TARGET
from app.settings import get_settings


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(get_settings().data_path)


def test_the_fixed_split_holds_out_a_test_set_of_3303_rows(dataset):
    split = split_dataset(dataset)

    assert len(split.X_test) == len(split.y_test) == 3_303
    assert len(split.X_train) == len(split.y_train) == 16_512 - 3_303
    assert list(split.X_train.columns) == list(FEATURES)
    assert split.y_train.name == TARGET
    assert split.X_train.index.intersection(split.X_test.index).empty


def test_the_split_is_identical_across_calls(dataset):
    first, second = split_dataset(dataset), split_dataset(dataset)

    assert first.X_train.index.equals(second.X_train.index)
    assert first.X_test.index.equals(second.X_test.index)


def test_the_cv_folds_are_five_shuffled_folds_identical_across_calls(dataset):
    X_train = split_dataset(dataset).X_train

    first = list(cv_folds().split(X_train))
    second = list(cv_folds().split(X_train))

    assert len(first) == 5
    for (train_a, val_a), (train_b, val_b) in zip(first, second, strict=True):
        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(val_a, val_b)
    # Every training row is validated exactly once, and the folds are shuffled.
    validated = np.concatenate([val for _, val in first])
    np.testing.assert_array_equal(np.sort(validated), np.arange(len(X_train)))
    assert not np.array_equal(first[0][1], np.arange(len(first[0][1])))
