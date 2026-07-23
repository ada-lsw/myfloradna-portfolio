import numpy as np
import pytest

from myflora.generator.ou_process import light_envelope, ou_process


def test_ou_process_reverts_toward_constant_mean():
    rng = np.random.default_rng(0)
    x = ou_process(rng, mu=25.0, theta=1.0, sigma=0.05, dt_hours=0.25, n_steps=2000, x0=10.0)
    # Low sigma + many steps => should settle close to mu regardless of x0.
    assert abs(np.mean(x[-200:]) - 25.0) < 0.5


def test_ou_process_is_reproducible_given_same_rng_state():
    x1 = ou_process(np.random.default_rng(7), mu=25.0, theta=0.5, sigma=1.0, dt_hours=0.25, n_steps=500)
    x2 = ou_process(np.random.default_rng(7), mu=25.0, theta=0.5, sigma=1.0, dt_hours=0.25, n_steps=500)
    np.testing.assert_array_equal(x1, x2)


def test_ou_process_accepts_time_varying_mu():
    rng = np.random.default_rng(1)
    mu = np.concatenate([np.zeros(100), np.full(100, 800.0)])
    x = ou_process(rng, mu=mu, theta=2.0, sigma=1.0, dt_hours=0.25, n_steps=200, x0=0.0)
    assert x[:100].mean() < x[100:].mean()


def test_ou_process_rejects_nonpositive_theta():
    with pytest.raises(ValueError):
        ou_process(np.random.default_rng(0), mu=1.0, theta=0.0, sigma=1.0, dt_hours=1.0, n_steps=10)


def test_ou_process_rejects_mismatched_mu_length():
    with pytest.raises(ValueError):
        ou_process(np.random.default_rng(0), mu=np.zeros(5), theta=1.0, sigma=1.0, dt_hours=1.0, n_steps=10)


def test_light_envelope_is_zero_at_night_and_one_at_plateau():
    # Lights on 6am-midnight (18h). Check 2am (deep night), 6am (start), noon (plateau).
    minute_of_day = np.array([2 * 60, 6 * 60, 12 * 60])
    envelope = light_envelope(minute_of_day, lights_on_hours=18.0, ramp_minutes=30.0, lights_on_start_hour=6.0)
    assert envelope[0] == pytest.approx(0.0)  # 2am: lights off
    assert envelope[2] == pytest.approx(1.0)  # noon: full plateau

    assert (envelope >= 0.0).all() and (envelope <= 1.0).all()


def test_light_envelope_ramps_up_linearly():
    minute_of_day = np.array([6 * 60, 6 * 60 + 15])  # lights-on start, +15min into a 30min ramp
    envelope = light_envelope(minute_of_day, lights_on_hours=18.0, ramp_minutes=30.0, lights_on_start_hour=6.0)
    assert envelope[0] == pytest.approx(0.0)
    assert envelope[1] == pytest.approx(0.5)

def test_light_envelope_ramps_down_symmetrically():
    # Lights on 6am-midnight (18h, 30min ramps). Ramp-down window is
    # 11:30pm (relative=1050min) to midnight (relative=1080min).
    minute_of_day = np.array([23 * 60 + 30, 23 * 60 + 45, 23 * 60 + 59])
    envelope = light_envelope(minute_of_day, lights_on_hours=18.0, ramp_minutes=30.0, lights_on_start_hour=6.0)
    assert envelope[0] == pytest.approx(1.0)          # 11:30pm: still full plateau
    assert envelope[1] == pytest.approx(0.5)           # 11:45pm: halfway down
    assert envelope[2] == pytest.approx(1 / 30, abs=1e-6)  # 11:59pm: nearly off