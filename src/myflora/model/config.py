"""Configuration for synthetic yield labels and gradient boosting training."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class YieldLabelConfig:
    """Knobs for the synthetic ground-truth yield-label function.

    There is no real yield data for this project; these constants are
    illustrative choices (arbitrary yield units, e.g. g/plant) meant to
    produce a plausible-looking, learnable label -- not literature-derived
    agronomic values.

    Attributes:
        base_yield: Yield at a hypothetical zero-closeness batch (every
            reading outside every sensor's optimal range, the whole time).
        max_yield_gain: Additional yield at perfect closeness (every
            reading inside every sensor's optimal range, the whole time),
            on top of base_yield.
        noise_sigma: Standard deviation of additive Gaussian label noise.
            Calibrated against the *default* generator/feature config: with
            its fault rates and OU mean-reversion strength, batches all end
            up fairly well-controlled, so closeness_score varies only
            slightly batch-to-batch (empirically std ~0.05 across a default
            60-batch run). noise_sigma=8 keeps that modest real signal
            (oracle in-sample R^2 ~0.7 for a linear fit of yield on
            closeness_score alone) learnable rather than drowned out --
            the original noise_sigma=30 pick made even the oracle linear
            fit's R^2 ~0.1, i.e. unlearnable regardless of model choice.
        sensor_weights: Optional {sensor_name: weight} map controlling
            each sensor's contribution to the closeness score. Sensors
            present in the feature table but missing from this map get
            weight 0. None (default) means equal weight across every
            sensor with a "*_stress_event_fraction" column.
        master_seed: Seed for the label-noise RNG, so labels are
            reproducible from a given feature table + config.
    """

    base_yield: float = 400.0
    max_yield_gain: float = 250.0
    noise_sigma: float = 8.0
    sensor_weights: dict[str, float] | None = None
    master_seed: int = 2024


@dataclass(frozen=True)
class TrainConfig:
    """Knobs for the gradient boosting yield model and its train/test split.

    Attributes:
        test_fraction: Fraction of batches held out for the test set.
        master_seed: Seed for the train/test split RNG.
        n_estimators: Number of boosting rounds.
        max_depth: Max tree depth.
        learning_rate: Boosting learning rate (shrinkage).
        subsample: Row subsample ratio per boosting round.
        colsample_bytree: Column subsample ratio per tree.
        reg_lambda: L2 regularization term on leaf weights.
        xgb_random_state: Seed forwarded to XGBRegressor itself (its
            internal row/column subsampling), kept separate from
            master_seed (the train/test split) so each is independently
            reproducible.

    Defaults are deliberately conservative (shallow, few trees, strong L2)
    rather than XGBoost's own out-of-the-box defaults: with a default
    60-batch dataset an 80/20 split leaves ~48 training rows against ~28
    features, so a deeper/larger ensemble memorizes the training set
    (train R^2 ~0.999) and generalizes worse than predicting the mean
    (test R^2 < 0) -- classic small-N overfitting, confirmed empirically
    against the real generator/feature pipeline while tuning these.
    """

    test_fraction: float = 0.2
    master_seed: int = 7
    n_estimators: int = 50
    max_depth: int = 2
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 5.0
    xgb_random_state: int = 7
