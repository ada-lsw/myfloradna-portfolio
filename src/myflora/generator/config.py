"""Configuration objects and default sensor specifications for the
synthetic cultivation data generator."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SensorSpec:
    """Static physical properties of one simulated sensor channel.

    Attributes:
        name: Channel name, used as the DataFrame column name.
        unit: Physical unit, for documentation/plotting only.
        target_low: Lower bound of the range a batch's target setpoint is
            drawn from.
        target_high: Upper bound of the range a batch's target setpoint is
            drawn from.
        theta_per_hour: Ornstein-Uhlenbeck mean-reversion rate, in 1/hours.
            Larger values snap back to the target faster.
        sigma: Ornstein-Uhlenbeck volatility, in sensor units per sqrt(hour).
        physical_low: Hard physical floor; clean (pre-fault) readings are
            clipped here.
        physical_high: Hard physical ceiling; clean (pre-fault) readings are
            clipped here.
        diurnal: Whether this channel's reversion target follows the light
            cycle rather than staying constant across the batch.
    """

    name: str
    unit: str
    target_low: float
    target_high: float
    theta_per_hour: float
    sigma: float
    physical_low: float
    physical_high: float
    diurnal: bool = False


DEFAULT_SENSORS: dict[str, SensorSpec] = {
    "temperature": SensorSpec(
        name="temperature",
        unit="degC",
        target_low=22.0,
        target_high=26.0,
        theta_per_hour=0.5,
        sigma=0.15,
        physical_low=5.0,
        physical_high=45.0,
        diurnal=True,
    ),
    "humidity": SensorSpec(
        name="humidity",
        unit="pct_RH",
        target_low=40.0,
        target_high=70.0,
        theta_per_hour=0.3,
        sigma=1.0,
        physical_low=0.0,
        physical_high=100.0,
    ),
    "ppfd": SensorSpec(
        name="ppfd",
        unit="umol_m2_s",
        target_low=400.0,
        target_high=800.0,
        theta_per_hour=2.0,
        sigma=15.0,
        physical_low=0.0,
        physical_high=1200.0,
        diurnal=True,
    ),
    "co2": SensorSpec(
        name="co2",
        unit="ppm",
        target_low=800.0,
        target_high=1500.0,
        theta_per_hour=0.2,
        sigma=20.0,
        physical_low=300.0,
        physical_high=3000.0,
    ),
    "soil_moisture": SensorSpec(
        name="soil_moisture",
        unit="pct_VWC",
        target_low=55.0,
        target_high=75.0,
        theta_per_hour=0.1,
        sigma=0.5,
        physical_low=0.0,
        physical_high=100.0,
    ),
}


@dataclass(frozen=True)
class GeneratorConfig:
    """Top-level knobs for a synthetic dataset generation run.

    Attributes:
        duration_days_range: Range each batch's grow-cycle length (in
            days) is drawn from.
        fault_rate_range: Range each batch's per-reading sensor fault
            probability is drawn from.
        reading_interval_minutes: Spacing between sensor readings.
        lights_on_hours: Length of the "lights on" period per day.
        lights_on_start_hour: Hour of day (0-24) lights turn on.
        ramp_minutes: Duration of the light ramp-up/ramp-down transitions.
        temp_night_drop: How many degrees the temperature reversion target
            dips below its daytime target during "lights off".
        spike_magnitude_sigma: Spike fault size, in multiples of the
            sensor's OU sigma.
        stuck_run_range: (min, max) run length, in readings, for
            stuck-value faults.
        dataset_start: Earliest possible batch start date.
        dataset_span_days: Window (in days, from dataset_start) that batch
            start dates are drawn from, so batches overlap/stagger like
            independent rooms would.
        sensors: Sensor channels to simulate, keyed by column name.
    """

    duration_days_range: tuple[float, float] = (60.0, 90.0)
    fault_rate_range: tuple[float, float] = (0.0005, 0.02)
    reading_interval_minutes: float = 15.0
    lights_on_hours: float = 18.0
    lights_on_start_hour: float = 6.0
    ramp_minutes: float = 30.0
    temp_night_drop: float = 1.5
    spike_magnitude_sigma: float = 8.0
    stuck_run_range: tuple[int, int] = (3, 20)
    dataset_start: str = "2023-01-01"
    dataset_span_days: int = 730
    sensors: dict[str, SensorSpec] = field(default_factory=lambda: dict(DEFAULT_SENSORS))
