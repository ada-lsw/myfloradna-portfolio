from myflora.features.engineer import compute_feature_table
from myflora.generator.config import GeneratorConfig
from myflora.generator.simulate import generate_dataset

FAST_CONFIG = GeneratorConfig(duration_days_range=(2.0, 3.0), reading_interval_minutes=60.0)


def test_compute_feature_table_has_one_row_per_batch():
    readings, metadata = generate_dataset(6, master_seed=31, config=FAST_CONFIG)
    features = compute_feature_table(readings, metadata)

    assert len(features) == len(metadata)
    assert set(features["batch_id"]) == set(metadata["batch_id"])


def test_compute_feature_table_has_expected_columns_per_sensor():
    readings, metadata = generate_dataset(3, master_seed=32, config=FAST_CONFIG)
    features = compute_feature_table(readings, metadata)

    for sensor_name in FAST_CONFIG.sensors:
        for suffix in (
            "rolling_mean", "rolling_std", "stress_event_count",
            "stress_event_fraction", "frac_near_target",
        ):
            assert f"{sensor_name}_{suffix}" in features.columns

    assert {"degree_days_above_optimal", "degree_days_below_optimal", "light_integral_mol_m2"}.issubset(
        features.columns
    )


def test_compute_feature_table_values_are_finite_and_in_range():
    readings, metadata = generate_dataset(10, master_seed=33, config=FAST_CONFIG)
    features = compute_feature_table(readings, metadata)

    for sensor_name in FAST_CONFIG.sensors:
        assert features[f"{sensor_name}_stress_event_fraction"].between(0.0, 1.0).all()
        assert features[f"{sensor_name}_frac_near_target"].between(0.0, 1.0).all()

    assert (features["degree_days_above_optimal"] >= 0.0).all()
    assert (features["degree_days_below_optimal"] >= 0.0).all()
    assert (features["light_integral_mol_m2"] >= 0.0).all()


def test_compute_feature_table_is_deterministic_given_same_inputs():
    readings, metadata = generate_dataset(5, master_seed=34, config=FAST_CONFIG)
    features_a = compute_feature_table(readings, metadata)
    features_b = compute_feature_table(readings, metadata)
    assert features_a.equals(features_b)
