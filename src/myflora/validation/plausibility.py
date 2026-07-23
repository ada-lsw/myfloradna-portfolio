"""Physical plausibility checks for generated sensor data.

Injected faults are expected to occasionally push readings outside
physical bounds (that's the point of a spike) or produce NaNs (dropouts).
These checks therefore bound how *much* implausibility is acceptable using
each batch's own fault_rate, rather than requiring zero violations.
"""

from __future__ import annotations

import math

import pandas as pd

from myflora.generator.config import GeneratorConfig, SensorSpec


def _poisson_quantile(mean: float, confidence: float) -> int:
    """Smallest k such that P(X <= k) >= confidence for X ~ Poisson(mean).

    Used instead of a normal approximation because expected fault counts
    are often well under 1 for short batches / rare faults, a regime where
    the normal approximation badly underestimates the tail.
    """
    if mean <= 0.0:
        return 0
    cumulative = pmf = math.exp(-mean)
    k = 0
    while cumulative < confidence and k < 10_000:
        k += 1
        pmf *= mean / k
        cumulative += pmf
    return k


def summarize_bounds_violations(
    readings: pd.DataFrame,
    metadata: pd.DataFrame,
    sensors: dict[str, SensorSpec] | None = None,
) -> pd.DataFrame:
    """Per (batch, sensor) counts of NaNs and out-of-physical-bounds readings.

    Args:
        readings: Long-format sensor readings, as from generate_dataset.
        metadata: Batch metadata, as from generate_dataset; must include
            batch_id and fault_rate.
        sensors: Sensor specs to check; defaults to GeneratorConfig().sensors.

    Returns:
        DataFrame with one row per (batch_id, sensor): n, n_nan, nan_rate,
        n_out_of_bounds, out_of_bounds_rate, and the batch's fault_rate.
    """
    sensors = sensors or GeneratorConfig().sensors
    fault_rate_by_batch = metadata.set_index("batch_id")["fault_rate"]

    rows = []
    for batch_id, group in readings.groupby("batch_id"):
        for sensor_name, spec in sensors.items():
            values = group[sensor_name]
            n = len(values)
            n_nan = int(values.isna().sum())
            non_nan = values.dropna()
            n_out_of_bounds = int(
                ((non_nan < spec.physical_low) | (non_nan > spec.physical_high)).sum()
            )
            rows.append(
                {
                    "batch_id": batch_id,
                    "sensor": sensor_name,
                    "n": n,
                    "n_nan": n_nan,
                    "nan_rate": n_nan / n if n else 0.0,
                    "n_out_of_bounds": n_out_of_bounds,
                    "out_of_bounds_rate": n_out_of_bounds / n if n else 0.0,
                    "fault_rate": fault_rate_by_batch.loc[batch_id],
                }
            )
    return pd.DataFrame(rows)


def find_plausibility_violations(
    readings: pd.DataFrame,
    metadata: pd.DataFrame,
    sensors: dict[str, SensorSpec] | None = None,
    confidence: float = 0.9999,
) -> pd.DataFrame:
    """Flag (batch, sensor) combinations whose implausibility looks like a bug.

    By construction, fault injection assigns each fault event to dropout,
    spike, or stuck-value with equal probability, and only spikes can push
    an (already-clipped) clean reading out of physical bounds. So both the
    NaN count and the out-of-bounds count are, in expectation, bounded by
    n * fault_rate / 3 -- modeled as Poisson(n * fault_rate / 3), since
    expected counts are often well under 1 for short batches or low fault
    rates. A count above the `confidence` quantile of that distribution
    can't be explained by fault injection alone, and points at a generator
    bug (e.g. a broken clip, or faults firing more often than configured).

    Args:
        readings: Long-format sensor readings.
        metadata: Batch metadata, with fault_rate per batch.
        sensors: Sensor specs to check; defaults to GeneratorConfig().sensors.
        confidence: Poisson quantile used as the flagging threshold; higher
            means fewer false positives from pure sampling noise.

    Returns:
        Subset of summarize_bounds_violations' output whose n_nan or
        n_out_of_bounds exceeds its statistical threshold. Empty means clean.
    """
    summary = summarize_bounds_violations(readings, metadata, sensors)

    expected = summary["n"] * summary["fault_rate"] / 3.0  # dropout share == spike share
    threshold = expected.map(lambda mean: _poisson_quantile(mean, confidence))

    violates = (summary["n_nan"] > threshold) | (summary["n_out_of_bounds"] > threshold)
    return summary[violates].reset_index(drop=True)
