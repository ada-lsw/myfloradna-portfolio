"""Command-line entry point: label, split, train, evaluate, and log a
yield-prediction run in one shot."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from myflora.model.config import TrainConfig, YieldLabelConfig
from myflora.model.labels import compute_yield_labels
from myflora.model.train import save_model, train_yield_model
from myflora.model.tracking import build_run_record, log_run


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic yield labels and train a gradient boosting yield model."
    )
    parser.add_argument(
        "--features-path", type=str, default="data/interim/batch_features.parquet",
        help="Path to the Phase 3 batch feature table (default: data/interim/batch_features.parquet).",
    )
    parser.add_argument(
        "--labels-output-path", type=str, default="data/processed/labels.parquet",
        help="Path to write the generated yield labels to (default: data/processed/labels.parquet).",
    )
    parser.add_argument(
        "--model-output-path", type=str, default="models/yield_xgb.json",
        help="Path to save the trained model artifact to (default: models/yield_xgb.json).",
    )
    parser.add_argument(
        "--log-path", type=str, default="models/experiment_log.jsonl",
        help="Path to append this run's record to (default: models/experiment_log.jsonl).",
    )
    parser.add_argument(
        "--test-fraction", type=float, default=TrainConfig().test_fraction,
        help="Fraction of batches held out for the test set.",
    )
    parser.add_argument(
        "--master-seed", type=int, default=TrainConfig().master_seed,
        help="Seed for the train/test split.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse CLI args and run the full label -> train -> evaluate -> log pipeline."""
    args = build_arg_parser().parse_args(argv)

    features = pd.read_parquet(args.features_path)

    label_config = YieldLabelConfig()
    labels = compute_yield_labels(features, label_config)
    labels_output_path = Path(args.labels_output_path)
    labels_output_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(labels_output_path, index=False)

    dataset = features.merge(labels[["batch_id", "yield"]], on="batch_id")
    train_config = TrainConfig(test_fraction=args.test_fraction, master_seed=args.master_seed)

    result = train_yield_model(dataset, train_config)
    model_path = save_model(result["model"], args.model_output_path)

    record = build_run_record(
        train_config,
        label_config,
        result["feature_columns"],
        len(result["train_ids"]),
        len(result["test_ids"]),
        result["metrics_train"],
        result["metrics_test"],
        model_path=str(model_path),
    )
    log_run(args.log_path, record)

    print(f"Wrote {len(labels)} yield labels to {labels_output_path}")
    print(f"Trained on {len(result['train_ids'])} batches, tested on {len(result['test_ids'])}")
    print(f"Train metrics: {result['metrics_train']}")
    print(f"Test metrics:  {result['metrics_test']}")
    print(f"Model saved to {model_path}")
    print(f"Run logged to {args.log_path}")


if __name__ == "__main__":
    main()
