import math

from myflora.features.engineer import compute_feature_table
from myflora.generator.config import GeneratorConfig
from myflora.generator.simulate import generate_dataset
from myflora.model.config import TrainConfig, YieldLabelConfig
from myflora.model.labels import compute_yield_labels
from myflora.model.train import train_yield_model

FAST_CONFIG = GeneratorConfig(duration_days_range=(2.0, 3.0), reading_interval_minutes=60.0)


def test_full_pipeline_generate_features_labels_train():
    readings, metadata = generate_dataset(40, master_seed=51, config=FAST_CONFIG)
    features = compute_feature_table(readings, metadata)
    labels = compute_yield_labels(features, YieldLabelConfig(master_seed=51))

    dataset = features.merge(labels[["batch_id", "yield"]], on="batch_id")
    result = train_yield_model(dataset, TrainConfig(master_seed=51))

    assert len(result["train_ids"]) + len(result["test_ids"]) == len(dataset)
    for metrics in (result["metrics_train"], result["metrics_test"]):
        assert math.isfinite(metrics["rmse"])
        assert math.isfinite(metrics["mae"])
        assert math.isfinite(metrics["r2"])
        assert metrics["rmse"] >= 0.0
        assert metrics["mae"] >= 0.0


def test_pipeline_is_reproducible_given_same_seeds():
    readings, metadata = generate_dataset(20, master_seed=52, config=FAST_CONFIG)
    features = compute_feature_table(readings, metadata)

    def run():
        labels = compute_yield_labels(features, YieldLabelConfig(master_seed=52))
        dataset = features.merge(labels[["batch_id", "yield"]], on="batch_id")
        return train_yield_model(dataset, TrainConfig(master_seed=52))

    result_a = run()
    result_b = run()

    assert result_a["metrics_test"] == result_b["metrics_test"]
    assert result_a["train_ids"] == result_b["train_ids"]
