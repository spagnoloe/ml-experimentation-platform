"""Load the Dataset: validate it, identify it by hash, and give the fixed split and folds."""

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from app.ml.schema import COLUMNS, FEATURES, TARGET
from app.settings import get_settings


class DatasetError(ValueError):
    """The file does not match the schema."""


@dataclass(frozen=True)
class Dataset:
    frame: pd.DataFrame
    sha256: str


def load_dataset(path: str | Path) -> Dataset:
    # Hash and parse the same bytes, so the hash always names what was loaded.
    content = Path(path).read_bytes()
    frame = pd.read_csv(io.BytesIO(content))
    missing = [c for c in COLUMNS if c not in frame.columns]
    if missing:
        raise DatasetError(f"{path}: missing column(s) {', '.join(missing)}")
    not_numeric = [
        c
        for c in COLUMNS
        if not pd.api.types.is_numeric_dtype(frame[c]) or pd.api.types.is_bool_dtype(frame[c])
    ]
    if not_numeric:
        details = ", ".join(f"{c} is {frame[c].dtype}" for c in not_numeric)
        raise DatasetError(f"{path}: non-numeric column(s): {details}; expected float")
    with_nulls = [c for c in COLUMNS if frame[c].isna().any()]
    if with_nulls:
        raise DatasetError(f"{path}: missing values in column(s) {', '.join(with_nulls)}")
    return Dataset(
        frame=frame[list(COLUMNS)].astype("float64"),
        sha256=hashlib.sha256(content).hexdigest(),
    )


@dataclass(frozen=True)
class Split:
    """The training portion and the Test Set. The Test Set is never used for selection."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def split_dataset(dataset: Dataset) -> Split:
    """The fixed train/test split from settings, the same for every Training Job."""
    settings = get_settings()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.frame[list(FEATURES)],
        dataset.frame[TARGET],
        test_size=settings.test_size,
        random_state=settings.split_seed,
    )
    return Split(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def cv_folds() -> KFold:
    """The fixed shuffled k-fold CV on the training portion; every Candidate sees these folds."""
    settings = get_settings()
    return KFold(n_splits=settings.cv_folds, shuffle=True, random_state=settings.cv_seed)
