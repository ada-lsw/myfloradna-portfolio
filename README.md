<<<<<<< HEAD
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
# myfloradna-portfolio
>>>>>>> 0c9fc07262b3f81d08ebca28a0ce26e0c286f125
