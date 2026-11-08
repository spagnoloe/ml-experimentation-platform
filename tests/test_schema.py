from app.ml import schema
from app.ml.data import load_dataset
from app.settings import get_settings


def test_the_schema_names_eight_features_and_the_target_in_100k_usd():
    assert len(schema.FEATURES) == 8
    assert schema.TARGET == "MedHouseVal"
    assert schema.TARGET_UNIT == "100k USD"


def test_sample_rows_have_every_feature_as_a_float_and_come_from_the_dataset():
    features = load_dataset(get_settings().data_path).frame[list(schema.FEATURES)]
    dataset_rows = set(features.itertuples(index=False, name=None))

    assert len(schema.SAMPLE_ROWS) >= 3
    for row in schema.SAMPLE_ROWS:
        assert list(row) == list(schema.FEATURES)
        assert all(isinstance(value, float) for value in row.values())
        assert tuple(row.values()) in dataset_rows
