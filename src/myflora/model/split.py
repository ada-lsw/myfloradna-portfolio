"""Reproducible train/test split at the batch level.

Uses NumPy's Generator API directly (rng.permutation) rather than
scikit-learn's train_test_split, which is seeded through the legacy
RandomState API -- keeping this project's "Generator, never RandomState"
reproducibility convention (SPEC.md's Tech Stack & Conventions) consistent
across every pipeline stage, not just data generation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def split_batches(
    dataset: pd.DataFrame,
    test_fraction: float,
    master_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shuffle and split a batch-indexed DataFrame into train/test sets.

    Args:
        dataset: One row per batch_id (e.g. features joined with labels).
        test_fraction: Fraction of rows to hold out for the test set.
        master_seed: Seed for the shuffle.

    Returns:
        (train, test) DataFrames, row order shuffled, index reset.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")
    if len(dataset) < 2:
        raise ValueError("Need at least 2 rows to split into train/test")

    rng = np.random.default_rng(master_seed)
    shuffled_positions = rng.permutation(len(dataset))
    n_test = max(int(round(len(dataset) * test_fraction)), 1)

    test_positions = shuffled_positions[:n_test]
    train_positions = shuffled_positions[n_test:]

    train = dataset.iloc[train_positions].reset_index(drop=True)
    test = dataset.iloc[test_positions].reset_index(drop=True)
    return train, test
