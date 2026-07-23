import pandas as pd

from myflora.generator.config import GeneratorConfig
from myflora.generator.simulate import generate_dataset
from myflora.validation.reproducibility import hash_dataframe, is_bit_exact_reproducible

FAST_CONFIG = GeneratorConfig(duration_days_range=(1.0, 2.0), reading_interval_minutes=60.0)


def test_hash_dataframe_is_stable_across_equal_frames():
    df1 = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df2 = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert hash_dataframe(df1) == hash_dataframe(df2)


def test_hash_dataframe_differs_for_different_content():
    df1 = pd.DataFrame({"a": [1, 2, 3]})
    df2 = pd.DataFrame({"a": [1, 2, 4]})
    assert hash_dataframe(df1) != hash_dataframe(df2)


def test_hash_dataframe_differs_for_different_dtype():
    df1 = pd.DataFrame({"a": [1, 2, 3]})
    df2 = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    assert hash_dataframe(df1) != hash_dataframe(df2)


def test_full_dataset_generation_is_bit_exact_reproducible_via_hash():
    def generate():
        return generate_dataset(8, master_seed=21, config=FAST_CONFIG)

    assert is_bit_exact_reproducible(generate)


def test_different_master_seeds_hash_differently():
    readings_a, metadata_a = generate_dataset(8, master_seed=1, config=FAST_CONFIG)
    readings_b, metadata_b = generate_dataset(8, master_seed=2, config=FAST_CONFIG)
    assert hash_dataframe(readings_a) != hash_dataframe(readings_b)
    assert hash_dataframe(metadata_a) != hash_dataframe(metadata_b)
