"""Ornstein-Uhlenbeck process simulation and diurnal light envelope."""

from __future__ import annotations

import numpy as np


def ou_process(
    rng: np.random.Generator,
    mu: np.ndarray | float,
    theta: float,
    sigma: float,
    dt_hours: float,
    n_steps: int,
    x0: float | None = None,
) -> np.ndarray:
    """Simulate a mean-reverting Ornstein-Uhlenbeck process.

    Uses the exact OU transition between steps (treating mu as piecewise
    constant over each interval), so the discretization stays well-behaved
    for any dt_hours rather than only for very small steps.

    Args:
        rng: Source of randomness.
        mu: Reversion target. Either a scalar, or an array of length
            n_steps giving a time-varying target (e.g. a diurnal cycle).
        theta: Mean-reversion rate, in 1/hours. Larger snaps back faster.
        sigma: Process volatility, in signal units per sqrt(hour).
        dt_hours: Time step size, in hours.
        n_steps: Number of steps to simulate.
        x0: Starting value. Defaults to mu's first value.

    Returns:
        Array of shape (n_steps,) with the simulated path.
    """
    mu_arr = np.full(n_steps, mu, dtype=float) if np.isscalar(mu) else np.asarray(mu, dtype=float)
    if mu_arr.shape[0] != n_steps:
        raise ValueError("mu array must have length n_steps")
    if theta <= 0:
        raise ValueError("theta must be positive for a mean-reverting process")

    alpha = np.exp(-theta * dt_hours)
    step_std = sigma * np.sqrt((1.0 - alpha**2) / (2.0 * theta))
    noise = rng.normal(loc=0.0, scale=1.0, size=n_steps)

    x = np.empty(n_steps, dtype=float)
    x[0] = mu_arr[0] if x0 is None else x0
    for t in range(1, n_steps):
        x[t] = mu_arr[t - 1] + (x[t - 1] - mu_arr[t - 1]) * alpha + step_std * noise[t]
    return x


def light_envelope(
    minute_of_day: np.ndarray,
    lights_on_hours: float,
    ramp_minutes: float,
    lights_on_start_hour: float,
) -> np.ndarray:
    """Piecewise ramp-up / plateau / ramp-down light envelope, in [0, 1].

    Args:
        minute_of_day: Minutes since local midnight for each timestamp.
        lights_on_hours: Length of the "lights on" period, in hours.
        ramp_minutes: Duration of the ramp up/down transitions, in minutes.
        lights_on_start_hour: Hour of day (0-24) lights turn on.

    Returns:
        Array shaped like minute_of_day, with values in [0, 1]: 0 during
        "night", 1 at full plateau brightness, and a linear ramp between.
    """
    start = lights_on_start_hour * 60.0
    on_duration = lights_on_hours * 60.0
    relative = np.mod(minute_of_day - start, 1440.0)

    envelope = np.zeros_like(relative, dtype=float)
    ramp_up = relative < ramp_minutes
    plateau = (relative >= ramp_minutes) & (relative <= on_duration - ramp_minutes)
    ramp_down = (relative > on_duration - ramp_minutes) & (relative <= on_duration)

    envelope[ramp_up] = relative[ramp_up] / ramp_minutes
    envelope[plateau] = 1.0
    envelope[ramp_down] = (on_duration - relative[ramp_down]) / ramp_minutes
    return envelope
