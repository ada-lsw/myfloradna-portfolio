import pandas as pd
import pytest

from myflora.features.engineer import compute_batch_features
from myflora.generator.config import SensorSpec

TEMP_SPEC = SensorSpec(
    name="temperature",
    unit="degC",
    target_low=20.0,
    target_high=25.0,
    theta_per_hour=0.5,
    sigma=1.0,
    physical_low=0.0,
    physical_high=50.0,
)

PPFD_SPEC = SensorSpec(
    name="ppfd",
    unit="umol_m2_s",
    target_low=400.0,
    target_high=800.0,
    theta_per_hour=2.0,
    sigma=15.0,
    physical_low=0.0,
    physical_high=1200.0,
)


def _hourly_batch(column: str, values: list[float], batch_id: str = "batch_test") -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=len(values), freq="h")
    return pd.DataFrame({"batch_id": batch_id, "timestamp": timestamps, column: values})


def test_degree_days_above_and_below_optimal():
    # 2 hours at 27 (target_high=25 -> +2 degC exceedance each) and
    # 2 hours at 18 (target_low=20 -> +2 degC shortfall each), hourly samples.
    readings = _hourly_batch("temperature", [27.0, 27.0, 18.0, 18.0])
    meta = pd.Series({"batch_id": "batch_test", "target_temp": 22.0})

    features = compute_batch_features(readings, meta, sensors={"temperature": TEMP_SPEC})

    assert features["degree_days_above_optimal"] == pytest.approx(4.0 / 24.0)
    assert features["degree_days_below_optimal"] == pytest.approx(4.0 / 24.0)


def test_stress_event_count_and_fraction():
    readings = _hourly_batch("temperature", [22.0, 30.0, 10.0, 23.0])  # 2 of 4 outside [20, 25]
    meta = pd.Series({"batch_id": "batch_test", "target_temp": 22.0})

    features = compute_batch_features(readings, meta, sensors={"temperature": TEMP_SPEC})

    assert features["temperature_stress_event_count"] == 2
    assert features["temperature_stress_event_fraction"] == pytest.approx(0.5)


def test_frac_near_target_uses_sigma_tolerance():
    # target=22, sigma=1 -> default 2-sigma tolerance => within [20, 24].
    readings = _hourly_batch("temperature", [22.0, 23.9, 24.1, 30.0])
    meta = pd.Series({"batch_id": "batch_test", "target_temp": 22.0})

    features = compute_batch_features(readings, meta, sensors={"temperature": TEMP_SPEC})

    assert features["temperature_frac_near_target"] == pytest.approx(0.5)  # 22.0 and 23.9 are within


def test_rolling_mean_and_std_for_constant_series():
    readings = _hourly_batch("temperature", [22.0] * 48)
    meta = pd.Series({"batch_id": "batch_test", "target_temp": 22.0})

    features = compute_batch_features(readings, meta, sensors={"temperature": TEMP_SPEC})

    assert features["temperature_rolling_mean"] == pytest.approx(22.0)
    assert features["temperature_rolling_std"] == pytest.approx(0.0)


def test_nan_readings_are_excluded_from_fraction_features():
    readings = _hourly_batch("temperature", [22.0, float("nan"), 30.0, float("nan")])
    meta = pd.Series({"batch_id": "batch_test", "target_temp": 22.0})

    features = compute_batch_features(readings, meta, sensors={"temperature": TEMP_SPEC})

    # Only 2 valid readings: 22 (in range) and 30 (stress event) -> fraction 0.5.
    assert features["temperature_stress_event_count"] == 1
    assert features["temperature_stress_event_fraction"] == pytest.approx(0.5)


def test_light_integral_converts_umol_to_mol_per_m2():
    # 4 hourly readings of 500 umol/m2/s => integral = 500 * 3600s * 4h / 1e6.
    readings = _hourly_batch("ppfd", [500.0, 500.0, 500.0, 500.0])
    meta = pd.Series({"batch_id": "batch_test", "target_ppfd": 600.0})

    features = compute_batch_features(readings, meta, sensors={"ppfd": PPFD_SPEC})

    assert features["light_integral_mol_m2"] == pytest.approx(500.0 * 3600.0 * 4 / 1e6)


def test_only_requested_sensors_produce_derived_features():
    readings = _hourly_batch("ppfd", [500.0, 500.0])
    meta = pd.Series({"batch_id": "batch_test", "target_ppfd": 600.0})

    features = compute_batch_features(readings, meta, sensors={"ppfd": PPFD_SPEC})

    assert "light_integral_mol_m2" in features
    assert "degree_days_above_optimal" not in features
    assert "degree_days_below_optimal" not in features
