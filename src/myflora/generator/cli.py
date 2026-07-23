"""Command-line entry point for generating a synthetic dataset."""

from __future__ import annotations

import argparse

from myflora.generator.config import GeneratorConfig
from myflora.generator.simulate import generate_dataset, write_dataset


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Generate synthetic cultivation sensor data.")
    parser.add_argument(
        "--n-batches", type=int, default=60,
        help="Number of grow batches to simulate (default: 60).",
    )
    parser.add_argument(
        "--master-seed", type=int, default=42,
        help="Master RNG seed the whole dataset derives from (default: 42).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/raw",
        help="Directory to write readings.parquet and batch_metadata.parquet to (default: data/raw).",
    )
    parser.add_argument(
        "--interval-minutes", type=float, default=15.0,
        help="Sensor reading interval in minutes (default: 15).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse CLI args, generate a dataset, and write it to Parquet."""
    args = build_arg_parser().parse_args(argv)

    config = GeneratorConfig(reading_interval_minutes=args.interval_minutes)
    readings, metadata = generate_dataset(args.n_batches, args.master_seed, config)
    readings_path, metadata_path = write_dataset(readings, metadata, args.output_dir)

    print(f"Wrote {len(readings):,} readings across {len(metadata)} batches to {readings_path}")
    print(f"Wrote batch metadata to {metadata_path}")


if __name__ == "__main__":
    main()
