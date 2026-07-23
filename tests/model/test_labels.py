import pandas as pd
import pytest

from myflora.model.config import YieldLabelConfig
from myflora.model.labels import compute_yield_labels


def _features(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_compute_yield_labels_has_expected_shape_and_columns():
    features = _features(
        [
            {"batch_id": "b0", "temperature_stress_event_fraction": 0.1, "humidity_stress_event_fraction": 0.2},
            {"batch_id": "b1", "temperature_stress_event_fraction": 0.5, "humidity_stress_event_fraction": 0.4},
        ]
    )
    labels = compute_yield_labels(features)

    assert list(labels.columns) == ["batch_id", "closeness_score", "yield"]
    assert list(labels["batch_id"]) == ["b0", "b1"]
    assert len(labels) == 2


def test_perfect_closeness_yields_base_plus_gain_with_zero_noise():
    features = _features(
        [{"batch_id": "b0", "temperature_stress_event_fraction": 0.0, "humidity_stress_event_fraction": 0.0}]
    )
    config = YieldLabelConfig(base_yield=100.0, max_yield_gain=50.0, noise_sigma=0.0)

    labels = compute_yield_labels(features, config)

    assert labels.loc[0, "closeness_score"] == pytest.approx(1.0)
    assert labels.loc[0, "yield"] == pytest.approx(150.0)


def test_zero_closeness_yields_base_with_zero_noise():
    features = _features(
        [{"batch_id": "b0", "temperature_stress_event_fraction": 1.0, "humidity_stress_event_fraction": 1.0}]
    )
    config = YieldLabelConfig(base_yield=100.0, max_yield_gain=50.0, noise_sigma=0.0)

    labels = compute_yield_labels(features, config)

    assert labels.loc[0, "closeness_score"] == pytest.approx(0.0)
    assert labels.loc[0, "yield"] == pytest.approx(100.0)


def test_closeness_score_is_equal_weighted_average_by_default():
    features = _features(
        [{"batch_id": "b0", "temperature_stress_event_fraction": 0.0, "humidity_stress_event_fraction": 0.4}]
    )
    labels = compute_yield_labels(features, YieldLabelConfig(noise_sigma=0.0))
    # closeness = mean(1 - 0.0, 1 - 0.4) = mean(1.0, 0.6) = 0.8
    assert labels.loc[0, "closeness_score"] == pytest.approx(0.8)


def test_sensor_weights_change_the_closeness_score():
    features = _features(
        [{"batch_id": "b0", "temperature_stress_event_fraction": 0.0, "humidity_stress_event_fraction": 1.0}]
    )
    config = YieldLabelConfig(noise_sigma=0.0, sensor_weights={"temperature": 1.0, "humidity": 0.0})

    labels = compute_yield_labels(features, config)

    # Humidity fully weighted out -> closeness driven entirely by temperature.
    assert labels.loc[0, "closeness_score"] == pytest.approx(1.0)


def test_missing_stress_columns_raises():
    features = _features([{"batch_id": "b0", "some_other_column": 1.0}])
    with pytest.raises(ValueError):
        compute_yield_labels(features)


def test_labels_are_reproducible_given_same_config():
    features = _features(
        [
            {"batch_id": f"b{i}", "temperature_stress_event_fraction": 0.1 * i, "humidity_stress_event_fraction": 0.05 * i}
            for i in range(10)
        ]
    )
    config = YieldLabelConfig(master_seed=99)

    labels_a = compute_yield_labels(features, config)
    labels_b = compute_yield_labels(features, config)

    pd.testing.assert_frame_equal(labels_a, labels_b)


def test_yields_are_never_negative():
    # Worst-case closeness with large noise sigma should still clip at 0.
    features = _features(
        [
            {"batch_id": f"b{i}", "temperature_stress_event_fraction": 1.0, "humidity_stress_event_fraction": 1.0}
            for i in range(200)
        ]
    )
    config = YieldLabelConfig(base_yield=10.0, max_yield_gain=0.0, noise_sigma=50.0, master_seed=1)

    labels = compute_yield_labels(features, config)

    assert (labels["yield"] >= 0.0).all()
