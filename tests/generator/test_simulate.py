import pandas as pd
import pandas.testing as pdt

from myflora.generator.config import GeneratorConfig
from myflora.generator.simulate import generate_dataset

# Small/fast config for tests: short batches, coarse interval.
FAST_CONFIG = GeneratorConfig(
    duration_days_range=(1.0, 2.0),
    reading_interval_minutes=60.0,
)


def test_generate_dataset_has_expected_shape_and_columns():
    readings, metadata = generate_dataset(4, master_seed=1, config=FAST_CONFIG)

    assert set(metadata["batch_id"]) == set(readings["batch_id"].unique())
    expected_reading_columns = {
        "batch_id", "timestamp", "temperature", "humidity", "ppfd", "co2", "soil_moisture",
    }
    assert expected_reading_columns.issubset(readings.columns)

    # Each batch should have roughly duration_days * 24 hourly readings.
    for batch_id, group in readings.groupby("batch_id"):
        duration_days = metadata.loc[metadata["batch_id"] == batch_id, "duration_days"].iloc[0]
        expected_steps = round(duration_days * 24.0)
        assert len(group) == expected_steps


def test_generate_dataset_is_bit_exact_reproducible():
    readings_a, metadata_a = generate_dataset(6, master_seed=99, config=FAST_CONFIG)
    readings_b, metadata_b = generate_dataset(6, master_seed=99, config=FAST_CONFIG)

    pdt.assert_frame_equal(readings_a, readings_b)
    pdt.assert_frame_equal(metadata_a, metadata_b)


def test_different_master_seeds_produce_different_datasets():
    readings_a, _ = generate_dataset(4, master_seed=1, config=FAST_CONFIG)
    readings_b, _ = generate_dataset(4, master_seed=2, config=FAST_CONFIG)
    assert not readings_a["temperature"].equals(readings_b["temperature"])


def test_physical_bounds_hold_when_faults_are_disabled():
    # Faults are allowed to violate plausibility bounds by design (that's
    # what makes a spike a spike), so only assert bounds with faults off.
    no_fault_config = GeneratorConfig(
        duration_days_range=(1.0, 2.0),
        reading_interval_minutes=60.0,
        fault_rate_range=(0.0, 0.0),
    )
    readings, _ = generate_dataset(10, master_seed=3, config=no_fault_config)

    assert readings["humidity"].between(0.0, 100.0).all()
    assert readings["ppfd"].between(0.0, 1200.0).all()
    assert readings["soil_moisture"].between(0.0, 100.0).all()
    assert readings["temperature"].between(5.0, 45.0).all()
    assert readings["co2"].between(300.0, 3000.0).all()
