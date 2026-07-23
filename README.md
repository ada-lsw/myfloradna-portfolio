# myflora-dna-portfolio

Synthetic cultivation data generation and crop yield prediction pipeline.
See [SPEC.md](SPEC.md) for the full project plan. Phases 1-4 (synthetic
data generator, validation suite, feature engineering, gradient boosting
yield model) are implemented; Phase 5 (documentation polish) and the
RNN/transformer comparison called out in Phase 4 are not yet built.

## Setup

```
python -m pip install -e ".[dev]"
```

## Generate a dataset

```
python -m myflora.generator.cli --n-batches 60 --master-seed 42 --output-dir data/raw
```

or, equivalently, the installed console script:

```
myflora-generate --n-batches 60 --master-seed 42 --output-dir data/raw
```

Writes `readings.parquet` (long-format sensor time series) and
`batch_metadata.parquet` (per-batch setpoints, duration, fault rate) to
the output directory. Options: `--interval-minutes` (default 15).

## Tests

```
python -m pytest
```

## Layout

```
src/myflora/generator/
    config.py      # SensorSpec / GeneratorConfig, default sensor parameters
    ou_process.py  # Ornstein-Uhlenbeck simulation + diurnal light envelope
    faults.py      # dropout / spike / stuck-value fault injection
    batches.py     # per-batch setpoints + deterministic RNG seeding
    simulate.py    # orchestrates batches -> readings, Parquet I/O
    cli.py         # `myflora-generate` entry point
tests/generator/   # mirrors the module layout above
```
=======

## Validate a dataset

```
python -c "
import pandas as pd
from myflora.validation.plausibility import find_plausibility_violations
from myflora.validation.reproducibility import hash_dataframe

readings = pd.read_parquet('data/raw/readings.parquet')
metadata = pd.read_parquet('data/raw/batch_metadata.parquet')
print(find_plausibility_violations(readings, metadata))
print(hash_dataframe(readings))
"
```

`myflora.validation.plausibility` flags (batch, sensor) combinations whose
NaN or out-of-bounds count exceeds what fault injection alone could
plausibly produce (a Poisson-quantile threshold on `n * fault_rate / 3`,
the dropout/spike share of fault events) -- an empty result means the
dataset is clean. `myflora.validation.reproducibility` hashes DataFrames
(`hash_dataframe`) and can call a generator function twice to confirm
bit-exact reproducibility (`is_bit_exact_reproducible`).

```
src/myflora/validation/
    plausibility.py     # bounds/NaN-rate checks vs. each batch's fault_rate
    reproducibility.py  # DataFrame hashing + bit-exact reproducibility check
tests/validation/        # mirrors the module layout above
```

## Engineer batch-level features

```
python -m myflora.features.cli --readings-path data/raw/readings.parquet --metadata-path data/raw/batch_metadata.parquet --output-path data/interim/batch_features.parquet
```

or the installed console script: `myflora-features` (same options).

Writes one row per `batch_id` to `batch_features.parquet`, ready to join
with a yield label on `batch_id`. Per sensor: rolling-window mean/std
(`{sensor}_rolling_mean`/`_std`, window in `FeatureConfig.rolling_window_hours`,
default 24h), stress-event count/fraction outside the sensor's optimal
band (`{sensor}_stress_event_count`/`_fraction`, using
`SensorSpec.target_low`/`target_high`), and fraction of readings within
`FeatureConfig.near_target_sigma_multiplier` (default 2) multiples of
`SensorSpec.sigma` of the batch's own target (`{sensor}_frac_near_target`).
Plus `degree_days_above_optimal`/`degree_days_below_optimal` (temperature
exposure past `target_high`/`target_low`, in degC-days) and
`light_integral_mol_m2` (cumulative PPFD exposure). All thresholds reuse
existing `SensorSpec` fields rather than introducing new magic numbers.
NaN (dropout-faulted) readings are excluded rather than counted as
stress. The sampling interval is inferred from each batch's own
timestamps, so it works regardless of `--interval-minutes` at generation
time.

Note: since PPFD's optimal band is the same `[target_low, target_high]`
used for setpoint targets, nightly near-zero PPFD (by design, not a
fault) also counts toward `ppfd_stress_event_count` -- expect that
feature to run high and read it as "time outside the target light band"
rather than literally "sensor malfunction."

```
src/myflora/features/
    config.py    # FeatureConfig: rolling window size, near-target tolerance
    engineer.py  # compute_batch_features / compute_feature_table
    io.py        # Parquet read/write helpers
    cli.py       # `myflora-features` entry point
tests/features/  # mirrors the module layout above
```

## Generate yield labels and train a model

```
python -m myflora.model.cli --features-path data/interim/batch_features.parquet
```

or the installed console script: `myflora-train` (`--test-fraction`,
`--master-seed`, and output-path options available; see `--help`).

There's no real yield data, so labels are synthetic:
`yield = base_yield + max_yield_gain * closeness_score + noise`, where
`closeness_score` is the (optionally sensor-weighted) average across
sensors of `1 - {sensor}_stress_event_fraction` -- i.e. how much of each
batch's time was spent inside that sensor's optimal range, reusing the
Phase 3 feature directly rather than recomputing anything from raw
readings. `noise_sigma` defaults to 8 (`YieldLabelConfig`), calibrated
against the *default* generator/feature config: its fault rates and OU
mean-reversion keep batches fairly uniformly well-controlled, so
`closeness_score` only varies by about 0.05 (std) batch-to-batch across a
default 60-batch run -- the original illustrative guess of 30 drowned
that signal out entirely (oracle in-sample R² ~0.1 for a linear fit of
yield on closeness_score alone, i.e. unlearnable by any model). At 8, the
oracle fit reaches R² ~0.7. If you regenerate data with very different
fault rates or OU parameters, re-check this calibration.

The model is a gradient boosting regressor (`xgboost.XGBRegressor`),
trained on every numeric feature column except `batch_id`/`yield`/
`closeness_score`. `TrainConfig`'s defaults are deliberately conservative
(50 shallow trees, depth 2, `reg_lambda=5`) rather than XGBoost's
out-of-the-box defaults: with ~48 training rows against ~28 features (the
default dataset's 80/20 split), a larger/deeper ensemble memorizes the
training set (train R² ~0.999) and generalizes worse than predicting the
mean (test R² < 0) -- verified empirically while tuning these. With the
current defaults, test R² typically lands around 0.2-0.4 depending on the
train/test split seed (`--master-seed`) -- a small, noisy dataset, so
expect that kind of run-to-run spread rather than a single "true" number.
The train/test split itself (`myflora.model.split.split_batches`) uses
`np.random.default_rng` directly rather than scikit-learn's
`train_test_split`, keeping the project's "Generator API only" convention
consistent through every pipeline stage, not just data generation.

Each run appends one JSON record (config, feature columns, train/test
metrics, and the saved model artifact's path) to
`models/experiment_log.jsonl` -- read it back with
`myflora.model.tracking.read_runs(log_path)`. That log entry doubles as a
lightweight model registry: it's the pointer from a run's config/metrics
back to its saved artifact (`models/yield_xgb.json`, XGBoost's native
format, loadable via `myflora.model.train.load_model`).

**Out of scope for this pass:** SPEC.md's Phase 4 also calls for a simple
RNN/transformer trained directly on the raw per-batch time series, for
comparison against this gradient-boosting-on-aggregated-features
baseline. Not implemented here -- see the module docstring in
`src/myflora/model/train.py`.

```
src/myflora/model/
    config.py     # YieldLabelConfig, TrainConfig
    labels.py     # compute_yield_labels (synthetic ground truth)
    split.py       # split_batches (Generator-based train/test split)
    train.py        # train_yield_model, metrics, save_model/load_model
    tracking.py      # build_run_record, log_run, read_runs (experiment log)
    cli.py            # `myflora-train` entry point
tests/model/      # mirrors the module layout above
```

