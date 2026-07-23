"""Command-line entry point for engineering batch-level features."""

from __future__ import annotations

import argparse

from myflora.features.engineer import compute_feature_table
from myflora.features.io import read_generator_output, write_feature_table


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Engineer batch-level features from generated sensor data.")
    parser.add_argument(
        "--readings-path", type=str, default="data/raw/readings.parquet",
        help="Path to the readings Parquet file (default: data/raw/readings.parquet).",
    )
    parser.add_argument(
        "--metadata-path", type=str, default="data/raw/batch_metadata.parquet",
        help="Path to the batch metadata Parquet file (default: data/raw/batch_metadata.parquet).",
    )
    parser.add_argument(
        "--output-path", type=str, default="data/interim/batch_features.parquet",
        help="Path to write the batch feature table to (default: data/interim/batch_features.parquet).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse CLI args, engineer batch features, and write them to Parquet."""
    args = build_arg_parser().parse_args(argv)

    readings, metadata = read_generator_output(args.readings_path, args.metadata_path)
    features = compute_feature_table(readings, metadata)
    output_path = write_feature_table(features, args.output_path)

    print(f"Wrote {len(features)} batch feature rows to {output_path}")


if __name__ == "__main__":
    main()
