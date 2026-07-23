import pandas as pd

from myflora.features.io import read_generator_output, write_feature_table
from myflora.generator.config import GeneratorConfig
from myflora.generator.simulate import generate_dataset, write_dataset


def test_read_generator_output_round_trips_parquet(tmp_path):
    config = GeneratorConfig(duration_days_range=(1.0, 2.0), reading_interval_minutes=60.0)
    readings, metadata = generate_dataset(3, master_seed=41, config=config)
    write_dataset(readings, metadata, tmp_path)

    loaded_readings, loaded_metadata = read_generator_output(
        tmp_path / "readings.parquet", tmp_path / "batch_metadata.parquet"
    )

    pd.testing.assert_frame_equal(loaded_readings, readings)
    pd.testing.assert_frame_equal(loaded_metadata, metadata)


def test_write_feature_table_creates_parent_dirs_and_round_trips(tmp_path):
    features = pd.DataFrame({"batch_id": ["a", "b"], "x": [1.0, 2.0]})
    output_path = tmp_path / "nested" / "dir" / "batch_features.parquet"

    written_path = write_feature_table(features, output_path)

    assert written_path == output_path
    assert output_path.exists()
    pd.testing.assert_frame_equal(pd.read_parquet(output_path), features)
