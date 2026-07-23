"""Lightweight experiment tracking: an append-only JSON-lines run log.

Not a full tracking framework (e.g. MLflow) -- just enough structure to
compare runs later. Each training run appends one JSON record (config,
feature set, and resulting metrics) to a log file; log_run/read_runs are
the two operations needed to write and later inspect that history.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from myflora.model.config import TrainConfig, YieldLabelConfig


def build_run_record(
    train_config: TrainConfig,
    label_config: YieldLabelConfig,
    feature_columns: list[str],
    n_train: int,
    n_test: int,
    metrics_train: dict[str, float],
    metrics_test: dict[str, float],
    model_path: str | None = None,
) -> dict[str, Any]:
    """Assemble a structured record describing one training run.

    Args:
        train_config: The TrainConfig used for this run.
        label_config: The YieldLabelConfig used to generate the labels
            trained against.
        feature_columns: Model input columns used.
        n_train: Number of training-set batches.
        n_test: Number of test-set batches.
        metrics_train: Training-set regression metrics.
        metrics_test: Test-set regression metrics.
        model_path: Where the trained model artifact was saved, if any --
            this is the "model registry" link back from a log entry to
            its artifact.

    Returns:
        A JSON-serializable dict, ready for tracking.log_run.
    """
    return {
        "run_id": uuid.uuid4().hex[:12],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "train_config": asdict(train_config),
        "label_config": asdict(label_config),
        "feature_columns": feature_columns,
        "n_train": n_train,
        "n_test": n_test,
        "metrics_train": metrics_train,
        "metrics_test": metrics_test,
        "model_path": model_path,
    }


def log_run(log_path: Path | str, run_record: dict[str, Any]) -> Path:
    """Append one run record as a JSON line to the experiment log.

    Args:
        log_path: Path to a .jsonl experiment log file; created (along
            with parent dirs) if it doesn't exist yet.
        run_record: JSON-serializable dict describing the run, e.g. from
            build_run_record.

    Returns:
        The log path written to.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(run_record, sort_keys=True, default=str) + "\n")
    return log_path


def read_runs(log_path: Path | str) -> pd.DataFrame:
    """Load an experiment log into a DataFrame, one row per run.

    Returns an empty DataFrame if the log doesn't exist yet.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        return pd.DataFrame()
    lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(json.loads(line) for line in lines)
