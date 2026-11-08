"""The one definition of the Dataset's shape. Training, the API and the UI all import it."""

FEATURES: tuple[str, ...] = (
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
)

TARGET = "MedHouseVal"
TARGET_UNIT = "100k USD"

COLUMNS: tuple[str, ...] = (*FEATURES, TARGET)

# Real rows from the Dataset (CSV rows 1, 1001, 5001, 10001, 16001), features only.
# Used to check that a model accepts schema input and returns finite predictions.
_SAMPLE_VALUES: tuple[tuple[float, ...], ...] = (
    (
        8.3252,
        41.0,
        6.984126984126984,
        1.0238095238095237,
        322.0,
        2.5555555555555554,
        37.88,
        -122.23,
    ),
    (3.3841, 29.0, 4.842031029619182, 1.002820874471086, 1919.0, 2.706629055007052, 37.69, -121.76),
    (
        1.7159,
        49.0,
        4.491735537190083,
        0.993801652892562,
        1861.0,
        3.8450413223140494,
        33.99,
        -118.28,
    ),
    (4.2031, 13.0, 6.811111111111111, 1.0185185185185186, 752.0, 2.785185185185185, 39.02, -121.08),
    (
        4.0846,
        52.0,
        4.821316614420063,
        0.9561128526645768,
        819.0,
        2.5673981191222572,
        37.74,
        -122.47,
    ),
)

SAMPLE_ROWS: tuple[dict[str, float], ...] = tuple(
    dict(zip(FEATURES, values, strict=True)) for values in _SAMPLE_VALUES
)
