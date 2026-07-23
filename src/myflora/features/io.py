"""I/O helpers for the feature engineering pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_generator_output(
    readings_path: Path | str,
    metadata_path: Path | str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the readings + batch metadata Parquet files written by the generator.

    Args:
        readings_path: Path to a long-format readings Parquet file.
        metadata_path: Path to a batch metadata Parquet file.

    Returns:
        (readings, metadata) tuple of DataFrames.
    """
    return pd.read_parquet(readings_path), pd.read_parquet(metadata_path)


def write_feature_table(features: pd.DataFrame, output_path: Path | str) -> Path:
    """Write the batch feature table to Parquet, creating parent dirs as needed.

    Args:
        features: One row per batch_id, as from compute_feature_table.
        output_path: Destination Parquet file path.

    Returns:
        The output path written to.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    return output_path
