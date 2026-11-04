"""Unit tests for the estimators. Synthetic data with known properties:
a test must be able to fail for the right reason, not just execute code."""
import numpy as np
import pytest
from scipy import stats

from var_engine import methods

RNG = np.random.default_rng(7)
NORMAL = RNG.standard_normal(20_000) * 0.01
FAT = RNG.standard_t(4, 20_000) * 0.008  # fat tails, df=4


def test_historical_known_values():
    r = np.array([-0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04])
    var, es = methods.historical_var_es(r, 0.90)
    assert var == pytest.approx(0.041, abs=0.002)  # 10th pct of returns ~ -0.041
    assert es >= var


def test_es_never_below_var_all_methods():
    for fn in (methods.historical_var_es, methods.student_t_var_es, methods.cornish_fisher_var_es):
        v, e = fn(FAT, 0.99)
        assert e >= v > 0


def test_gaussian_matches_scipy_formula():
    mu, sd = 0.001, 0.02
    v, e = methods.gaussian_var_es(mu, sd, 0.99)
    z = stats.norm.ppf(0.99)
    assert v == pytest.approx(-mu + z * sd)
    assert e == pytest.approx(-mu + sd * stats.norm.pdf(z) / 0.01)


def test_student_t_tail_heavier_than_gaussian_on_fat_data():
    _, e_t = methods.student_t_var_es(FAT, 0.99)
    _, e_g = methods.gaussian_var_es(float(FAT.mean()), float(FAT.std(ddof=1)), 0.99)
    assert e_t > e_g  # the central empirical claim of the whole project


def test_cornish_fisher_exceeds_gaussian_under_negative_skew():
    rng = np.random.default_rng(3)
    skewed = rng.standard_normal(20_000) * 0.01 - np.abs(rng.standard_normal(20_000)) * 0.004
    v_cf, _ = methods.cornish_fisher_var_es(skewed, 0.99)
    v_g, _ = methods.gaussian_var_es(float(skewed.mean()), float(skewed.std(ddof=1)), 0.99)
    assert v_cf > v_g


def test_ewma_recursion_matches_manual():
    r = RNG.standard_normal(300) * 0.01
    lam = 0.94
    var = np.var(r[:50])
    for x in r[50:]:
        var = lam * var + (1 - lam) * x**2
    manual = np.sqrt(lam * var + (1 - lam) * r[-1] ** 2)
    assert methods.ewma_forecast_sd(r, lam) == pytest.approx(manual)


def test_monte_carlo_reproducible_and_close_to_gaussian_for_normal_inputs():
    mu = np.zeros(3); cov = np.diag([0.0001, 0.0004, 0.000225]); w = np.array([0.5, 0.3, 0.2])
    v1, _ = methods.monte_carlo_var_es(mu, cov, w, 0.99, dist="normal", seed=11)
    v2, _ = methods.monte_carlo_var_es(mu, cov, w, 0.99, dist="normal", seed=11)
    assert v1 == v2  # seeded -> reproducible
    port_sd = float(np.sqrt(w @ cov @ w))
    v_g, _ = methods.gaussian_var_es(0.0, port_sd, 0.99)
    assert v1 == pytest.approx(v_g, rel=0.06)


def test_evt_recovers_tail_of_t_distribution():
    out = methods.evt_pot_var_es(FAT, 0.99)
    v_h, e_h = methods.historical_var_es(FAT, 0.99)
    assert out["var"] == pytest.approx(v_h, rel=0.25)  # EVT should land near the empirical tail
    assert out["es"] >= out["var"] and out["n_exceed"] >= 30


def test_garch_fhs_runs_and_is_positive():
    out = methods.garch_fhs_var_es(NORMAL, 0.99, model="GARCH-t")
    assert out["var"] > 0 and out["es"] >= out["var"]
    assert 0.8 < out["persistence"] < 1.05
