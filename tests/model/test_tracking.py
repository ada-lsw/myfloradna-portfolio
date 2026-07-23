import pandas as pd

from myflora.model.config import TrainConfig, YieldLabelConfig
from myflora.model.tracking import build_run_record, log_run, read_runs


def test_build_run_record_has_expected_shape():
    record = build_run_record(
        train_config=TrainConfig(),
        label_config=YieldLabelConfig(),
        feature_columns=["a", "b"],
        n_train=8,
        n_test=2,
        metrics_train={"rmse": 1.0, "mae": 0.5, "r2": 0.9},
        metrics_test={"rmse": 2.0, "mae": 1.5, "r2": 0.7},
        model_path="models/model.json",
    )

    assert "run_id" in record
    assert "timestamp" in record
    assert record["feature_columns"] == ["a", "b"]
    assert record["n_train"] == 8
    assert record["n_test"] == 2
    assert record["metrics_test"]["r2"] == 0.7
    assert record["train_config"]["test_fraction"] == TrainConfig().test_fraction
    assert record["model_path"] == "models/model.json"


def test_log_run_appends_jsonl_and_read_runs_round_trips(tmp_path):
    log_path = tmp_path / "runs.jsonl"

    record_a = build_run_record(TrainConfig(), YieldLabelConfig(), ["a"], 5, 1, {"r2": 0.1}, {"r2": 0.2})
    record_b = build_run_record(TrainConfig(), YieldLabelConfig(), ["a"], 5, 1, {"r2": 0.3}, {"r2": 0.4})

    log_run(log_path, record_a)
    log_run(log_path, record_b)

    runs = read_runs(log_path)

    assert len(runs) == 2
    assert set(runs["run_id"]) == {record_a["run_id"], record_b["run_id"]}
    assert list(runs["n_train"]) == [5, 5]


def test_read_runs_returns_empty_dataframe_when_log_missing(tmp_path):
    runs = read_runs(tmp_path / "does_not_exist.jsonl")
    assert isinstance(runs, pd.DataFrame)
    assert runs.empty


def test_log_run_creates_parent_directories(tmp_path):
    log_path = tmp_path / "nested" / "dir" / "runs.jsonl"
    record = build_run_record(TrainConfig(), YieldLabelConfig(), ["a"], 5, 1, {"r2": 0.1}, {"r2": 0.2})

    written_path = log_run(log_path, record)

    assert written_path == log_path
    assert log_path.exists()
