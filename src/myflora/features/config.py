"""Configuration for batch-level feature engineering."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureConfig:
    """Knobs for turning raw per-batch time series into batch-level features.

    Attributes:
        rolling_window_hours: Window size for the rolling mean/std
            features, in hours. Samples-per-window is derived from each
            batch's own median sampling interval, so this works across
            datasets generated at different reading intervals.
        near_target_sigma_multiplier: A reading counts as "near target" if
            it falls within this many multiples of the sensor's OU sigma
            (myflora.generator.config.SensorSpec.sigma) of the batch's own
            target setpoint -- ties the tolerance to the sensor's actual
            process noise instead of an arbitrary absolute threshold.
    """

    rolling_window_hours: float = 24.0
    near_target_sigma_multiplier: float = 2.0
