import numpy as np

from myflora.generator.faults import inject_faults


def test_zero_fault_rate_returns_unchanged_copy():
    values = np.linspace(20.0, 25.0, 1000)
    out = inject_faults(np.random.default_rng(0), values, fault_rate=0.0, sigma=1.0)
    np.testing.assert_array_equal(out, values)
    assert out is not values


def test_faults_do_not_mutate_input():
    values = np.full(500, 25.0)
    original = values.copy()
    inject_faults(np.random.default_rng(0), values, fault_rate=0.05, sigma=1.0)
    np.testing.assert_array_equal(values, original)


def test_high_fault_rate_introduces_nans_and_deviations():
    values = np.full(5000, 25.0)
    out = inject_faults(np.random.default_rng(1), values, fault_rate=0.05, sigma=1.0)
    assert np.isnan(out).any()
    non_nan = out[~np.isnan(out)]
    assert not np.allclose(non_nan, 25.0)


def test_stuck_fault_holds_a_constant_run():
    # Force every reading to be an event, and force it to be a "stuck" event,
    # by using a fault_rate of 1 and inspecting for held runs directly.
    values = np.arange(200, dtype=float)
    out = inject_faults(np.random.default_rng(2), values, fault_rate=1.0, sigma=0.1, stuck_run_range=(5, 5))
    # With every index eligible, there should be at least one run of >=2
    # identical consecutive values that don't match the clean ramp (a hold).
    diffs = np.diff(out[~np.isnan(out)])
    assert (diffs == 0).any()


def test_fault_rate_of_one_still_produces_finite_output_shape():
    values = np.full(50, 10.0)
    out = inject_faults(np.random.default_rng(3), values, fault_rate=1.0, sigma=1.0)
    assert out.shape == values.shape
