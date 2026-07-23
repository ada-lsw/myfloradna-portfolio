"""Synthetic ground-truth yield labels.

There is no real yield data for this project, so labels are generated
from a hand-specified function of "how close a batch's actual conditions
stayed to each sensor's optimal range" -- Phase 3's per-sensor
stress_event_fraction, already defined relative to
myflora.generator.config.SensorSpec.target_low/target_high -- plus noise.
This is a deliberately synthetic stand-in for a real yield outcome, not a
fitted or literature-derived model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from myflora.model.config import YieldLabelConfig

_STRESS_SUFFIX = "_stress_event_fraction"


def _closeness_score(features: pd.DataFrame, sensor_weights: dict[str, float] | None) -> pd.Series:
    """Weighted average, across sensors, of time spent inside the optimal range."""
    stress_columns = [c for c in features.columns if c.endswith(_STRESS_SUFFIX)]
    if not stress_columns:
        raise ValueError(f"No '{_STRESS_SUFFIX}' columns found in the feature table")

    in_optimal = 1.0 - features[stress_columns]

    if sensor_weights is None:
        return in_optimal.mean(axis=1)

    weights = (
        pd.Series({f"{sensor}{_STRESS_SUFFIX}": w for sensor, w in sensor_weights.items()})
        .reindex(stress_columns)
        .fillna(0.0)
    )
    if weights.sum() == 0:
        raise ValueError("sensor_weights assigns zero total weight to sensors present in the feature table")
    return in_optimal.mul(weights, axis=1).sum(axis=1) / weights.sum()


def compute_yield_labels(
    features: pd.DataFrame,
    config: YieldLabelConfig | None = None,
) -> pd.DataFrame:
    """Generate synthetic yield labels from batch-level features.

    yield = base_yield + max_yield_gain * closeness_score + noise, where
    closeness_score in [0, 1] is a (possibly weighted) average across
    sensors of the fraction of time each batch's readings stayed inside
    that sensor's optimal range (1 - stress_event_fraction).

    Args:
        features: Batch feature table, as from
            myflora.features.engineer.compute_feature_table. Must include
            "batch_id" and at least one "*_stress_event_fraction" column.
        config: Label generation knobs; defaults to YieldLabelConfig().

    Returns:
        DataFrame with columns "batch_id", "closeness_score", "yield".
    """
    config = config or YieldLabelConfig()
    rng = np.random.default_rng(config.master_seed)

    closeness = _closeness_score(features, config.sensor_weights)
    noise = rng.normal(loc=0.0, scale=config.noise_sigma, size=len(features))
    yield_value = config.base_yield + config.max_yield_gain * closeness.to_numpy() + noise
    yield_value = np.clip(yield_value, a_min=0.0, a_max=None)

    return pd.DataFrame(
        {
            "batch_id": features["batch_id"].to_numpy(),
            "closeness_score": closeness.to_numpy(),
            "yield": yield_value,
        }
    )
