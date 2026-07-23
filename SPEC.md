# Synthetic Cultivation Data & Crop Yield Prediction — Project Spec

## Purpose

Portfolio project built to demonstrate the skills required for the MyFloraDNA
Data Science / Software Engineering Internship: synthetic environmental data
generation, validation, feature engineering, and predictive modeling for
agronomic time-series data.

## Overview

Simulate a controlled indoor cultivation facility running many independent
grow **batches**. Each batch produces multivariate sensor time-series data
(temperature, humidity, light, CO2, soil moisture) with realistic physical
structure — diurnal cycles, mean-reverting drift, and occasional sensor
faults. This synthetic dataset then feeds a full ML pipeline: validation,
feature engineering, and yield prediction.

---

## Phase 1: Synthetic Data Generator

### Batch structure

- Simulate **50–100 independent batches**, each representing one grow cycle
  in one room, lasting ~60–90 days.
- Each batch is assigned **randomly-drawn target setpoints** within
  realistic agronomic ranges, e.g.:
  - Target temperature: 22–26°C
  - Target humidity: 40–70% RH
  - Target PPFD (light intensity): 400–800 μmol/m²/s
  - Target CO2: 800–1500 ppm
- These per-batch targets are what create meaningful variation for the
  downstream yield model to learn from — without varied targets, every
  batch would look the same and there'd be nothing to predict.

### Sensor time-series generation

For each sensor channel, per batch:

1. **Mean-reverting drift** around the batch's target setpoint, using an
   **Ornstein-Uhlenbeck (OU) process**. Conceptually: a random walk with a
   "spring" pulling values back toward the target, so readings wander
   realistically without drifting to implausible values.
2. **Diurnal cycles** for light and temperature — e.g., an 18-hour "lights
   on" period with ramp-up/plateau/ramp-down, near-zero PPFD during "night,"
   and a temperature that loosely tracks the light cycle.
3. **Occasional sensor faults** — dropouts (missing/NaN readings), spikes
   (implausible outliers), or stuck values — injected at a configurable
   rate that can vary per batch (some batches simulate a flakier sensor).

### Reproducibility

- Use NumPy's modern **`Generator`** API (`np.random.default_rng`), not the
  legacy `RandomState`.
- Derive each batch's seed deterministically from a single master seed +
  batch index, so the entire dataset is **bit-exact reproducible** from one
  seed value, while each batch is still a distinct random draw.

### Output format

**Sensor readings (long format), Parquet:**

| batch_id | timestamp | temperature | humidity | ppfd | co2 | soil_moisture |
|---|---|---|---|---|---|---|

**Batch metadata table, Parquet:**

| batch_id | target_temp | target_humidity | target_ppfd | target_co2 | duration_days | fault_rate |
|---|---|---|---|---|---|---|

---

## Phase 2: Validation Suite

Using `pytest`:

- **Physical plausibility checks** — enforce realistic bounds per sensor
  (e.g., humidity ∈ [0, 100]%, PPFD ≥ 0, temperature within a sane range),
  applied to generated data, treating injected faults as expected
  exceptions rather than bugs.
- **Bit-exact reproducibility checks** — regenerate a dataset from the same
  master seed twice and confirm identical output (e.g., via hash comparison
  of the resulting arrays/dataframes).

---

## Phase 3: Feature Engineering

From raw per-batch time-series, engineer batch-level features:

- Rolling means/standard deviations per sensor channel
- **Degree-days** — cumulative temperature exposure above/below a
  reference threshold (a real agronomy concept)
- Light integral (cumulative PPFD exposure over the batch)
- **Stress-event counts** — number of readings outside the optimal range
  for each sensor
- Fraction of time near target setpoint (a proxy for "control quality")

Output: one feature row per batch, ready to join with a yield label.

---

## Phase 4: Yield Prediction Model

- Define a **synthetic ground-truth yield function**: yield as a function
  of how close each batch's actual conditions stayed to a defined
  "optimal" biological zone, plus noise — used to generate labels since
  there's no real yield data.
- Train and compare model architectures suited to this feature set:
  - Gradient boosting (XGBoost or LightGBM) as a baseline
  - A simple RNN or transformer on the raw time-series, for comparison
- Build a reproducible pipeline: versioned datasets, experiment tracking
  (lightweight — structured logs/CSVs are fine), basic model registry
  concept (saved model artifacts + metadata).

---

## Phase 5: Documentation

- README covering: data generation parameters, data lineage (how
  batch metadata + labels were produced), and model evaluation results.
- Keep documentation aligned with what the job posting explicitly asks
  for: "documenting data lineage, generation parameters, and model
  evaluation results for internal R&D review."

---

## Tech Stack & Conventions

- Python 3.11+, `numpy`, `pandas`, `pyarrow` (Parquet I/O), `scikit-learn`,
  `pytest`, `matplotlib`
- Phase 4 adds `xgboost` and/or `pytorch`
- Use NumPy's `Generator` API exclusively — never `np.random.seed()` /
  `RandomState`
- Type hints and docstrings on all public functions
- `src/` layout with a `tests/` directory mirroring module structure
