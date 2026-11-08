import pandas as pd
import pytest

from app.ml.data import DatasetError, load_dataset
from app.settings import get_settings

EXPECTED_COLUMNS = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
    "MedHouseVal",
]
# `shasum -a 256 data/training_data.csv`
PROVIDED_CSV_SHA256 = "ac6910c7db55ba6d0eb63e9fd0e18e69ad471ddf3928d25473a74808132d964f"


def test_loading_the_provided_csv_gives_all_rows_with_expected_columns():
    dataset = load_dataset(get_settings().data_path)

    assert len(dataset.frame) == 16_512
    assert list(dataset.frame.columns) == EXPECTED_COLUMNS


def _write_sample_csv(tmp_path, transform=lambda frame: frame):
    """Write the first 50 rows of the provided CSV, after `transform`, and return the path."""
    frame = transform(pd.read_csv(get_settings().data_path, nrows=50))
    path = tmp_path / "sample.csv"
    frame.to_csv(path, index=False)
    return path


def test_a_csv_with_a_missing_column_is_rejected_naming_the_column(tmp_path):
    path = _write_sample_csv(tmp_path, lambda frame: frame.drop(columns=["AveOccup"]))

    with pytest.raises(DatasetError, match="missing column.*AveOccup"):
        load_dataset(path)


def test_a_csv_with_a_non_numeric_column_is_rejected_naming_the_column(tmp_path):
    def corrupt(frame):
        frame["HouseAge"] = frame["HouseAge"].astype(object)
        frame.loc[3, "HouseAge"] = "forty-one"
        return frame

    path = _write_sample_csv(tmp_path, corrupt)

    with pytest.raises(DatasetError, match="HouseAge.*float"):
        load_dataset(path)


def test_the_dataset_is_identified_by_the_sha256_of_its_file_stably_across_loads():
    path = get_settings().data_path

    first, second = load_dataset(path), load_dataset(path)

    assert first.sha256 == PROVIDED_CSV_SHA256
    assert second.sha256 == first.sha256


def test_a_csv_with_a_true_false_column_is_rejected_naming_the_column(tmp_path):
    path = _write_sample_csv(tmp_path, lambda frame: frame.assign(Latitude=frame.index % 2 == 0))

    with pytest.raises(DatasetError, match="Latitude.*bool"):
        load_dataset(path)


def test_a_csv_with_empty_cells_is_rejected_naming_the_column(tmp_path):
    def blank(frame):
        frame.loc[[2, 7], "AveRooms"] = None
        return frame

    path = _write_sample_csv(tmp_path, blank)

    with pytest.raises(DatasetError, match="missing values.*AveRooms"):
        load_dataset(path)
