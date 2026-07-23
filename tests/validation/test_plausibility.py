from myflora.generator.config import GeneratorConfig
from myflora.generator.simulate import generate_dataset
from myflora.validation.plausibility import find_plausibility_violations, summarize_bounds_violations

FAST_CONFIG = GeneratorConfig(duration_days_range=(2.0, 3.0), reading_interval_minutes=60.0)


def test_generated_data_has_no_plausibility_violations():
    readings, metadata = generate_dataset(15, master_seed=11, config=FAST_CONFIG)
    violations = find_plausibility_violations(readings, metadata)
    assert violations.empty, violations.to_string()


def test_summarize_bounds_violations_has_expected_shape():
    readings, metadata = generate_dataset(5, master_seed=12, config=FAST_CONFIG)
    summary = summarize_bounds_violations(readings, metadata)
    assert set(summary["batch_id"]) == set(metadata["batch_id"])
    assert set(summary["sensor"]) == set(FAST_CONFIG.sensors)
    assert (summary["nan_rate"] >= 0).all()
    assert (summary["out_of_bounds_rate"] >= 0).all()


def test_find_plausibility_violations_catches_a_broken_generator():
    # Simulate a bug: humidity readings that blow way past physical bounds
    # far more often than the batch's fault_rate could explain.
    readings, metadata = generate_dataset(5, master_seed=13, config=FAST_CONFIG)
    broken = readings.copy()
    n = len(broken)
    broken.loc[broken.index[: n // 2], "humidity"] = 150.0  # way past [0, 100]

    violations = find_plausibility_violations(broken, metadata)
    assert not violations.empty
    assert (violations["sensor"] == "humidity").any()


def test_zero_fault_rate_yields_zero_implausible_readings():
    no_fault_config = GeneratorConfig(
        duration_days_range=(2.0, 3.0),
        reading_interval_minutes=60.0,
        fault_rate_range=(0.0, 0.0),
    )
    readings, metadata = generate_dataset(5, master_seed=14, config=no_fault_config)
    summary = summarize_bounds_violations(readings, metadata)
    assert (summary["n_out_of_bounds"] == 0).all()
    assert (summary["n_nan"] == 0).all()
