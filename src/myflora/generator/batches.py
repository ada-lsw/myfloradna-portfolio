"""Batch metadata generation: per-batch setpoints and deterministic RNG seeding."""

from __future__ import annotations

import numpy as np
import pandas as pd

from myflora.generator.config import GeneratorConfig

#: Maps sensor column name -> batch metadata target column name. Matches
#: the abbreviation used in the spec's metadata table ("target_temp"),
#: falling back to "target_<sensor>" for everything else.
TARGET_COLUMN = {
    "temperature": "target_temp",
    "humidity": "target_humidity",
    "ppfd": "target_ppfd",
    "co2": "target_co2",
    "soil_moisture": "target_soil_moisture",
}


def spawn_batch_rngs(n_batches: int, master_seed: int) -> list[np.random.Generator]:
    """Derive n_batches independent, reproducible RNG streams from one seed.

    Uses NumPy's SeedSequence.spawn, which deterministically derives child
    seeds from (master_seed, batch_index) so the full dataset is bit-exact
    reproducible from master_seed alone, while each batch is still an
    independent random draw.

    Args:
        n_batches: Number of RNG streams to spawn.
        master_seed: Single seed the entire dataset derives from.

    Returns:
        A list of n_batches numpy Generator instances, one per batch.
    """
    seed_seq = np.random.SeedSequence(master_seed)
    return [np.random.default_rng(s) for s in seed_seq.spawn(n_batches)]


def generate_batch_metadata(
    n_batches: int,
    master_seed: int,
    config: GeneratorConfig | None = None,
) -> tuple[pd.DataFrame, list[np.random.Generator]]:
    """Draw random per-batch setpoints, duration, fault rate, and start date.

    Args:
        n_batches: Number of independent grow batches to generate.
        master_seed: Single seed the entire dataset derives from.
        config: Generator configuration; defaults to GeneratorConfig().

    Returns:
        A (metadata, rngs) tuple. metadata has one row per batch. rngs are
        the same per-batch generators used to draw this metadata -- reuse
        them (in order) to generate each batch's time series so the whole
        run stays reproducible from master_seed alone.
    """
    config = config or GeneratorConfig()
    rngs = spawn_batch_rngs(n_batches, master_seed)

    dataset_start = pd.Timestamp(config.dataset_start)
    rows = []
    for i, rng in enumerate(rngs):
        duration_days = float(rng.uniform(*config.duration_days_range))
        fault_rate = float(rng.uniform(*config.fault_rate_range))
        max_offset = max(config.dataset_span_days - duration_days, 0.0)
        start_time = dataset_start + pd.Timedelta(days=float(rng.uniform(0.0, max_offset)))

        row = {
            "batch_id": f"batch_{i:04d}",
            "start_time": start_time,
            "duration_days": duration_days,
            "fault_rate": fault_rate,
        }
        for sensor_name, spec in config.sensors.items():
            row[TARGET_COLUMN[sensor_name]] = float(rng.uniform(spec.target_low, spec.target_high))
        rows.append(row)

    return pd.DataFrame(rows), rngs
