# myflora-dna-portfolio

Synthetic cultivation data generation and crop yield prediction pipeline.
See [SPEC.md](SPEC.md) for the full project plan. Phases 1-3 (synthetic
data generator, validation suite, feature engineering) are implemented;
later phases are not yet built.

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

