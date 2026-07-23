import numpy as np

from myflora.generator.batches import generate_batch_metadata, spawn_batch_rngs
from myflora.generator.config import GeneratorConfig


def test_spawn_batch_rngs_is_deterministic_given_same_master_seed():
    rngs_a = spawn_batch_rngs(5, master_seed=123)
    rngs_b = spawn_batch_rngs(5, master_seed=123)
    for a, b in zip(rngs_a, rngs_b):
        np.testing.assert_array_equal(a.random(10), b.random(10))


def test_spawn_batch_rngs_gives_independent_streams():
    rngs = spawn_batch_rngs(3, master_seed=123)
    draws = [r.random(20) for r in rngs]
    assert not np.allclose(draws[0], draws[1])
    assert not np.allclose(draws[1], draws[2])


def test_generate_batch_metadata_has_expected_columns_and_row_count():
    metadata, rngs = generate_batch_metadata(10, master_seed=1)
    assert len(metadata) == 10
    assert len(rngs) == 10
    expected_columns = {
        "batch_id", "start_time", "duration_days", "fault_rate",
        "target_temp", "target_humidity", "target_ppfd", "target_co2",
        "target_soil_moisture",
    }
    assert expected_columns.issubset(metadata.columns)
    assert metadata["batch_id"].is_unique


def test_generate_batch_metadata_targets_are_within_configured_ranges():
    config = GeneratorConfig()
    metadata, _ = generate_batch_metadata(30, master_seed=2, config=config)

    assert metadata["target_temp"].between(22.0, 26.0).all()
    assert metadata["target_humidity"].between(40.0, 70.0).all()
    assert metadata["target_ppfd"].between(400.0, 800.0).all()
    assert metadata["target_co2"].between(800.0, 1500.0).all()
    assert metadata["duration_days"].between(*config.duration_days_range).all()
    assert metadata["fault_rate"].between(*config.fault_rate_range).all()
