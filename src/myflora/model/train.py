"""Gradient boosting yield model: training and evaluation.

Only a gradient boosting baseline (XGBoost) is implemented in this pass.
A simple RNN or transformer trained directly on the raw per-batch sensor
time series (rather than the aggregated Phase 3 features used here) is a
natural next comparison -- SPEC.md's Phase 4 calls it out explicitly --
but is out of scope for this pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from myflora.model.config import TrainConfig
from myflora.model.split import split_batches

#: Identifier/label columns that are never valid model inputs.
NON_FEATURE_COLUMNS = {"batch_id", "yield", "closeness_score"}


def select_feature_columns(dataset: pd.DataFrame) -> list[str]:
    """Default feature set: every numeric column except id/label columns."""
    numeric = dataset.select_dtypes(include="number").columns
    return [c for c in numeric if c not in NON_FEATURE_COLUMNS]


def compute_regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Standard regression metrics: RMSE, MAE, R^2."""
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_yield_model(
    dataset: pd.DataFrame,
    config: TrainConfig | None = None,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Train and evaluate a gradient boosting yield model.

    Args:
        dataset: Batch features joined with yield labels (one row per
            batch_id, must include a "yield" column).
        config: Training knobs; defaults to TrainConfig().
        feature_columns: Columns to use as model input; defaults to every
            numeric column except batch_id/yield/closeness_score (see
            select_feature_columns).

    Returns:
        Dict with keys: model, feature_columns, train_ids, test_ids,
        metrics_train, metrics_test.
    """
    config = config or TrainConfig()
    feature_columns = feature_columns or select_feature_columns(dataset)
    if not feature_columns:
        raise ValueError("No feature columns available to train on")

    train, test = split_batches(dataset, config.test_fraction, config.master_seed)

    model = xgb.XGBRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        reg_lambda=config.reg_lambda,
        random_state=config.xgb_random_state,
    )
    model.fit(train[feature_columns], train["yield"])

    metrics_train = compute_regression_metrics(train["yield"], model.predict(train[feature_columns]))
    metrics_test = compute_regression_metrics(test["yield"], model.predict(test[feature_columns]))

    return {
        "model": model,
        "feature_columns": feature_columns,
        "train_ids": train["batch_id"].tolist(),
        "test_ids": test["batch_id"].tolist(),
        "metrics_train": metrics_train,
        "metrics_test": metrics_test,
    }


def save_model(model: xgb.XGBRegressor, path: Path | str) -> Path:
    """Save a trained model to disk, in XGBoost's native JSON format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    return path


def load_model(path: Path | str) -> xgb.XGBRegressor:
    """Load a model previously saved with save_model."""
    model = xgb.XGBRegressor()
    model.load_model(str(path))
    return model
