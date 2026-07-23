import numpy as np
import pandas as pd
import pytest

from myflora.model.config import TrainConfig
from myflora.model.train import (
    NON_FEATURE_COLUMNS,
    compute_regression_metrics,
    load_model,
    save_model,
    select_feature_columns,
    train_yield_model,
)


def _linear_signal_dataset(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    feature_a = rng.uniform(0.0, 1.0, size=n)
    feature_b = rng.uniform(0.0, 1.0, size=n)
    noise = rng.normal(0.0, 2.0, size=n)  # small noise relative to signal scale
    yield_value = 100.0 + 300.0 * feature_a + noise
    return pd.DataFrame(
        {
            "batch_id": [f"batch_{i:04d}" for i in range(n)],
            "closeness_score": feature_a,  # excluded from features, must not leak
            "feature_a": feature_a,
            "feature_b": feature_b,
            "yield": yield_value,
        }
    )


def test_select_feature_columns_excludes_ids_and_labels():
    dataset = _linear_signal_dataset(n=10)
    columns = select_feature_columns(dataset)

    assert set(columns) == {"feature_a", "feature_b"}
    assert NON_FEATURE_COLUMNS.isdisjoint(columns)


def test_compute_regression_metrics_perfect_prediction():
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    metrics = compute_regression_metrics(y, y)

    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)


def test_train_yield_model_learns_a_strong_linear_signal():
    dataset = _linear_signal_dataset(n=200)
    config = TrainConfig(test_fraction=0.2, master_seed=3, n_estimators=100, max_depth=3)

    result = train_yield_model(dataset, config)

    assert set(result["feature_columns"]) == {"feature_a", "feature_b"}
    assert len(result["train_ids"]) + len(result["test_ids"]) == len(dataset)
    assert set(result["train_ids"]).isdisjoint(result["test_ids"])
    # A near-noiseless linear relationship should be learnable well above chance.
    assert result["metrics_test"]["r2"] > 0.7


def test_train_yield_model_rejects_dataset_with_no_feature_columns():
    dataset = pd.DataFrame({"batch_id": ["a", "b", "c"], "yield": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        train_yield_model(dataset)


def test_save_and_load_model_round_trips_predictions(tmp_path):
    dataset = _linear_signal_dataset(n=100)
    result = train_yield_model(dataset, TrainConfig(n_estimators=20, master_seed=1))

    model_path = save_model(result["model"], tmp_path / "model.json")
    loaded = load_model(model_path)

    original_preds = result["model"].predict(dataset[result["feature_columns"]])
    loaded_preds = loaded.predict(dataset[result["feature_columns"]])
    np.testing.assert_allclose(original_preds, loaded_preds)
