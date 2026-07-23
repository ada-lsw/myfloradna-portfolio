"""Batch-level feature engineering from raw per-batch sensor time series.

"Optimal"/"stress" ranges and the "near target" tolerance are derived from
existing myflora.generator.config.SensorSpec fields (target_low/target_high
and sigma) rather than new hardcoded thresholds, so agronomic assumptions
live in one place.
"""

from __future__ import annotations

import pandas as pd

from myflora.features.config import FeatureConfig
from myflora.generator.batches import TARGET_COLUMN
from myflora.generator.config import DEFAULT_SENSORS, SensorSpec

#: Temperature is the channel the spec's degree-days concept applies to;
#: the reference thresholds are the sensor's own optimal band edges
#: (SensorSpec.target_low / target_high), not a new constant.
DEGREE_DAY_SENSOR = "temperature"
#: Channel the cumulative light integral is computed over.
LIGHT_INTEGRAL_SENSOR = "ppfd"


def _median_interval_hours(timestamps: pd.Series) -> float:
    """Infer a batch's sampling interval directly from its timestamps."""
    diffs = timestamps.sort_values().diff().dropna()
    if diffs.empty:
        raise ValueError("Need at least two readings to infer a sampling interval")
    return diffs.median().total_seconds() / 3600.0


def _rolling_mean_std(values: pd.Series, window_hours: float, interval_hours: float) -> tuple[float, float]:
    """Mean of a rolling-window mean/std, as batch-level summary features.

    Using the mean of a rolling statistic rather than a single flat
    mean/std over the whole batch captures typical *local* behavior (e.g.
    short-term control noise) instead of conflating it with longer-term
    structure like the diurnal cycle.
    """
    clean = values.dropna()
    window = max(int(round(window_hours / interval_hours)), 1)
    rolling = clean.rolling(window=window, min_periods=1)
    return float(rolling.mean().mean()), float(rolling.std().mean())


def compute_batch_features(
    batch_readings: pd.DataFrame,
    batch_meta: pd.Series,
    sensors: dict[str, SensorSpec] | None = None,
    config: FeatureConfig | None = None,
) -> dict[str, float | str]:
    """Engineer one batch's feature row from its raw time series.

    Args:
        batch_readings: Long-format readings for a single batch_id.
        batch_meta: The matching row of batch metadata (targets, duration,
            fault_rate); must include "batch_id" and, for each sensor
            present, its target column (see
            myflora.generator.batches.TARGET_COLUMN).
        sensors: Sensor specs to engineer features for; defaults to
            DEFAULT_SENSORS. Their target_low/target_high define each
            channel's "optimal" range (used for stress-event counts and
            degree-day thresholds); their sigma defines the "near target"
            tolerance.
        config: Feature engineering knobs; defaults to FeatureConfig().

    Returns:
        A flat dict of feature_name -> value, including "batch_id".
    """
    sensors = sensors or DEFAULT_SENSORS
    config = config or FeatureConfig()

    batch_readings = batch_readings.sort_values("timestamp")
    interval_hours = _median_interval_hours(batch_readings["timestamp"])

    features: dict[str, float | str] = {"batch_id": batch_meta["batch_id"]}

    for sensor_name, spec in sensors.items():
        values = batch_readings[sensor_name]
        clean = values.dropna()
        n_valid = len(clean)

        rolling_mean, rolling_std = _rolling_mean_std(values, config.rolling_window_hours, interval_hours)
        features[f"{sensor_name}_rolling_mean"] = rolling_mean
        features[f"{sensor_name}_rolling_std"] = rolling_std

        out_of_optimal = (clean < spec.target_low) | (clean > spec.target_high)
        features[f"{sensor_name}_stress_event_count"] = int(out_of_optimal.sum())
        features[f"{sensor_name}_stress_event_fraction"] = float(out_of_optimal.mean()) if n_valid else 0.0

        target = batch_meta[TARGET_COLUMN[sensor_name]]
        tolerance = config.near_target_sigma_multiplier * spec.sigma
        near_target = (clean - target).abs() <= tolerance
        features[f"{sensor_name}_frac_near_target"] = float(near_target.mean()) if n_valid else 0.0

    if DEGREE_DAY_SENSOR in sensors:
        temp_spec = sensors[DEGREE_DAY_SENSOR]
        temperature = batch_readings[DEGREE_DAY_SENSOR].dropna()
        features["degree_days_above_optimal"] = float(
            (temperature - temp_spec.target_high).clip(lower=0.0).sum() * interval_hours / 24.0
        )
        features["degree_days_below_optimal"] = float(
            (temp_spec.target_low - temperature).clip(lower=0.0).sum() * interval_hours / 24.0
        )

    if LIGHT_INTEGRAL_SENSOR in sensors:
        ppfd = batch_readings[LIGHT_INTEGRAL_SENSOR].dropna()
        # umol/m2/s * s -> umol/m2, then /1e6 -> mol/m2.
        features["light_integral_mol_m2"] = float(ppfd.sum() * interval_hours * 3600.0 / 1e6)

    return features


def compute_feature_table(
    readings: pd.DataFrame,
    metadata: pd.DataFrame,
    sensors: dict[str, SensorSpec] | None = None,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Engineer one feature row per batch from full readings + metadata tables.

    Args:
        readings: Long-format sensor readings across all batches.
        metadata: Batch metadata across all batches.
        sensors: Sensor specs to engineer features for; defaults to
            DEFAULT_SENSORS.
        config: Feature engineering knobs; defaults to FeatureConfig().

    Returns:
        DataFrame with one row per batch_id, ready to join with a yield
        label on batch_id.
    """
    sensors = sensors or DEFAULT_SENSORS
    config = config or FeatureConfig()

    grouped_readings = dict(tuple(readings.groupby("batch_id")))
    rows = [
        compute_batch_features(grouped_readings[batch_meta["batch_id"]], batch_meta, sensors, config)
        for _, batch_meta in metadata.iterrows()
    ]
    return pd.DataFrame(rows)
