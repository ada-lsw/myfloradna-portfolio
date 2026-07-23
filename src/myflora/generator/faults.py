"""Sensor fault injection: dropouts, spikes, and stuck-value runs."""

from __future__ import annotations

import numpy as np


def inject_faults(
    rng: np.random.Generator,
    values: np.ndarray,
    fault_rate: float,
    sigma: float,
    spike_magnitude_sigma: float = 8.0,
    stuck_run_range: tuple[int, int] = (3, 20),
) -> np.ndarray:
    """Inject dropout, spike, and stuck-value faults into a clean series.

    Faults are deliberately allowed to violate physical plausibility
    bounds -- that's what makes a spike a spike. Downstream validation of
    generated data should treat these as expected exceptions, not bugs.

    Args:
        rng: Source of randomness.
        values: Clean sensor readings, shape (n,). Not modified in place.
        fault_rate: Probability that any given reading starts a fault
            event. Each event becomes a dropout, spike, or stuck-run with
            equal probability.
        sigma: Sensor's OU volatility, used to scale spike magnitude.
        spike_magnitude_sigma: Spike size, in multiples of sigma.
        stuck_run_range: (min, max) run length, in readings, for
            stuck-value faults.

    Returns:
        A new array, same shape as values, with faults applied.
    """
    n = values.shape[0]
    out = values.copy()
    if fault_rate <= 0.0:
        return out

    event_indices = np.flatnonzero(rng.random(n) < fault_rate)
    if event_indices.size == 0:
        return out

    kinds = rng.integers(0, 3, size=event_indices.size)  # 0=dropout 1=spike 2=stuck

    dropout_idx = event_indices[kinds == 0]
    out[dropout_idx] = np.nan

    spike_idx = event_indices[kinds == 1]
    if spike_idx.size:
        signs = rng.choice(np.array([-1.0, 1.0]), size=spike_idx.size)
        magnitudes = rng.uniform(0.5, 1.5, size=spike_idx.size) * spike_magnitude_sigma * sigma
        out[spike_idx] = out[spike_idx] + signs * magnitudes

    stuck_idx = event_indices[kinds == 2]
    if stuck_idx.size:
        run_lengths = rng.integers(stuck_run_range[0], stuck_run_range[1] + 1, size=stuck_idx.size)
        for start, length in zip(stuck_idx, run_lengths):
            end = min(start + length, n)
            out[start:end] = values[start]  # hold at the pre-fault clean value

    return out
