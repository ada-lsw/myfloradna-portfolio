"""Full synthetic dataset generation: batches -> per-sensor OU time series."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from myflora.generator.batches import TARGET_COLUMN, generate_batch_metadata
from myflora.generator.config import GeneratorConfig
from myflora.generator.faults import inject_faults
from myflora.generator.ou_process import light_envelope, ou_process


def generate_batch_timeseries(
    batch_row: pd.Series,
    rng: np.random.Generator,
    config: GeneratorConfig,
) -> pd.DataFrame:
    """Simulate one batch's multivariate sensor time series.

    Args:
        batch_row: One row of batch metadata (target setpoints, duration,
            fault rate, start time).
        rng: This batch's RNG stream, reused from metadata generation so
            the whole batch is one deterministic draw from its own seed.
        config: Generator configuration.

    Returns:
        Long-format DataFrame with columns: batch_id, timestamp, and one
        column per sensor channel.
    """
    dt_hours = config.reading_interval_minutes / 60.0
    n_steps = int(round(batch_row["duration_days"] * 24.0 / dt_hours))

    timestamps = batch_row["start_time"] + pd.to_timedelta(
        np.arange(n_steps) * config.reading_interval_minutes, unit="min"
    )
    minute_of_day = (
        timestamps.hour * 60 + timestamps.minute + timestamps.second / 60.0
    ).to_numpy(dtype=float)
    envelope = light_envelope(
        minute_of_day, config.lights_on_hours, config.ramp_minutes, config.lights_on_start_hour
    )

    data: dict[str, object] = {"batch_id": batch_row["batch_id"], "timestamp": timestamps}
    for sensor_name, spec in config.sensors.items():
        target = batch_row[TARGET_COLUMN[sensor_name]]

        if sensor_name == "ppfd":
            mu = target * envelope
        elif sensor_name == "temperature":
            mu = target - config.temp_night_drop * (1.0 - envelope)
        else:
            mu = target

        clean = ou_process(
            rng,
            mu=mu,
            theta=spec.theta_per_hour,
            sigma=spec.sigma,
            dt_hours=dt_hours,
            n_steps=n_steps,
        )
        clean = np.clip(clean, spec.physical_low, spec.physical_high)
        data[sensor_name] = inject_faults(
            rng,
            clean,
            fault_rate=batch_row["fault_rate"],
            sigma=spec.sigma,
            spike_magnitude_sigma=config.spike_magnitude_sigma,
            stuck_run_range=config.stuck_run_range,
        )

    return pd.DataFrame(data)


def generate_dataset(
    n_batches: int,
    master_seed: int,
    config: GeneratorConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a full synthetic dataset: batch metadata + sensor readings.

    Args:
        n_batches: Number of independent grow batches to simulate.
        master_seed: Single seed the entire dataset derives from
            bit-exactly.
        config: Generator configuration; defaults to GeneratorConfig().

    Returns:
        (readings, metadata) tuple of DataFrames, matching the long-format
        readings table and batch metadata table described in the spec.
    """
    config = config or GeneratorConfig()
    metadata, rngs = generate_batch_metadata(n_batches, master_seed, config)

    batch_frames = [
        generate_batch_timeseries(row, rng, config)
        for (_, row), rng in zip(metadata.iterrows(), rngs)
    ]
    readings = pd.concat(batch_frames, ignore_index=True)
    return readings, metadata


def write_dataset(
    readings: pd.DataFrame,
    metadata: pd.DataFrame,
    output_dir: Path | str,
) -> tuple[Path, Path]:
    """Write readings and metadata DataFrames to Parquet.

    Args:
        readings: Long-format sensor readings, as returned by
            generate_dataset.
        metadata: Batch metadata, as returned by generate_dataset.
        output_dir: Directory to write into; created if missing.

    Returns:
        (readings_path, metadata_path) tuple of the written file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    readings_path = output_dir / "readings.parquet"
    metadata_path = output_dir / "batch_metadata.parquet"
    readings.to_parquet(readings_path, index=False)
    metadata.to_parquet(metadata_path, index=False)
    return readings_path, metadata_path
