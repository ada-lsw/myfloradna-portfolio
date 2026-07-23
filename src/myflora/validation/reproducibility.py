"""Bit-exact reproducibility checks: hash datasets to confirm two runs with
the same master seed produce identical output."""

from __future__ import annotations

import hashlib
from typing import Callable

import pandas as pd


def hash_dataframe(df: pd.DataFrame) -> str:
    """Return a stable SHA-256 hex digest of a DataFrame's contents.

    Captures both column names/dtypes and every value (including the
    index), so any bit-level difference between two runs changes the
    digest. Used to confirm bit-exact reproducibility across two dataset
    generation runs without diffing both full DataFrames in memory.
    """
    row_hashes = pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes()
    column_signature = "|".join(f"{col}:{dtype}" for col, dtype in df.dtypes.items()).encode()

    digest = hashlib.sha256()
    digest.update(column_signature)
    digest.update(row_hashes)
    return digest.hexdigest()


def is_bit_exact_reproducible(generate_fn: Callable[..., object], *args: object, **kwargs: object) -> bool:
    """Call generate_fn(*args, **kwargs) twice and confirm identical hashes.

    Args:
        generate_fn: A callable returning either a single DataFrame or a
            tuple of DataFrames (e.g. (readings, metadata)).
        *args: Positional arguments forwarded to generate_fn.
        **kwargs: Keyword arguments forwarded to generate_fn.

    Returns:
        True if every returned DataFrame hashes identically across the two
        calls, False otherwise.
    """
    result_a = generate_fn(*args, **kwargs)
    result_b = generate_fn(*args, **kwargs)

    frames_a = result_a if isinstance(result_a, tuple) else (result_a,)
    frames_b = result_b if isinstance(result_b, tuple) else (result_b,)

    if len(frames_a) != len(frames_b):
        return False
    return all(hash_dataframe(a) == hash_dataframe(b) for a, b in zip(frames_a, frames_b))
