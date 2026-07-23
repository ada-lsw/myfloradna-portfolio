# myflora-dna-portfolio

Synthetic cultivation data generation and crop yield prediction pipeline.
See [SPEC.md](SPEC.md) for the full project plan. Phase 1 (synthetic data
generator) is implemented; later phases are not yet built.

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

