"""Data loading and preprocessing for supervised router training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from app.ml.model_utils import (
        DATASET_PATH,
        NON_PRODUCTION_COLUMNS,
        POST_INFERENCE_COLUMNS,
        TARGET_COLUMN,
    )
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution
    from model_utils import DATASET_PATH, NON_PRODUCTION_COLUMNS, POST_INFERENCE_COLUMNS, TARGET_COLUMN


@dataclass(frozen=True)
class PreprocessResult:
    """Container for prepared training inputs and preprocessing metadata."""

    dataframe: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    preprocessor: ColumnTransformer
    feature_columns: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    removed_columns: dict[str, list[str]]


def load_training_data(dataset_path: Path | str = DATASET_PATH) -> pd.DataFrame:
    """Load the Phase 2 training dataset."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Training dataset not found: {path}")
    return pd.read_csv(path)


def get_removed_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Identify columns removed before supervised training."""
    leakage = sorted([col for col in POST_INFERENCE_COLUMNS if col in df.columns])
    non_production = sorted([col for col in NON_PRODUCTION_COLUMNS if col in df.columns])
    missing_target = [] if TARGET_COLUMN in df.columns else [TARGET_COLUMN]
    return {
        "target_leakage": leakage,
        "non_production_metadata": non_production,
        "missing_required": missing_target,
    }


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict[str, list[str]]]:
    """Remove leakage/metadata columns and split X/y."""
    removed = get_removed_columns(df)
    if removed["missing_required"]:
        raise ValueError(f"Dataset is missing required target column: {TARGET_COLUMN}")

    drop_columns = set(removed["target_leakage"]) | set(removed["non_production_metadata"]) | {TARGET_COLUMN}
    feature_columns = [col for col in df.columns if col not in drop_columns]
    if not feature_columns:
        raise ValueError("No trainable pre-routing feature columns remain after preprocessing.")

    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN].astype(str).str.strip().str.upper()
    return X, y, removed


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool = True) -> ColumnTransformer:
    """Build a sklearn preprocessing pipeline for numeric and categorical columns."""
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - older sklearn compatibility
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_features:
        transformers.append(("num", Pipeline(numeric_steps), numeric_features))
    if categorical_features:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", encoder),
            ]),
            categorical_features,
        ))

    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def prepare_training_data(
    dataset_path: Path | str = DATASET_PATH,
    scale_numeric: bool = True,
) -> PreprocessResult:
    """Load the dataset, remove leakage columns, split target, and build preprocessor."""
    df = load_training_data(dataset_path)
    df.attrs["path"] = str(Path(dataset_path).resolve())
    X, y, removed = split_features_target(df)
    preprocessor = build_preprocessor(X, scale_numeric=scale_numeric)
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]
    return PreprocessResult(
        dataframe=df,
        X=X,
        y=y,
        preprocessor=preprocessor,
        feature_columns=X.columns.tolist(),
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        removed_columns=removed,
    )


def align_features(features: dict[str, Any], feature_columns: list[str]) -> pd.DataFrame:
    """Convert a feature dictionary into a one-row DataFrame matching training columns."""
    row = {col: _to_model_value(features.get(col, 0)) for col in feature_columns}
    return pd.DataFrame([row], columns=feature_columns)


def _to_model_value(value: Any) -> Any:
    """Normalize live extractor outputs to the tabular training representation."""
    if isinstance(value, dict):
        return value.get("score", 0)
    if isinstance(value, bool):
        return int(value)
    return value
