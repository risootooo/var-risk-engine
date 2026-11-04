import numpy as np
import pytest

from var_engine import backtest

RNG = np.random.default_rng(21)


def test_kupiec_accepts_correct_rate_rejects_wrong_rate():
    _, p_ok = backtest.kupiec(1000, 12, 0.01)      # 1.2% vs 1% expected: plausible
    _, p_bad = backtest.kupiec(1000, 40, 0.01)     # 4% vs 1%: reject
    assert p_ok > 0.05 and p_bad < 0.001


def test_kupiec_zero_exceptions_edge():
    lr, p = backtest.kupiec(250, 0, 0.01)
    assert lr == pytest.approx(-2 * 250 * np.log(0.99)) and 0 < p < 1


def test_christoffersen_detects_clustering():
    iid = (RNG.random(2000) < 0.01).astype(int)
    _, p_iid = backtest.christoffersen(iid)
    clustered = np.zeros(2000, dtype=int); clustered[500:510] = 1; clustered[1200:1205] = 1
    _, p_cl = backtest.christoffersen(clustered)
    assert p_cl < p_iid  # clustered exceptions must look worse (seeded, deterministic)


def test_basel_zones_boundaries():
    assert backtest.basel_zone(0) == "GREEN" and backtest.basel_zone(4) == "GREEN"
    assert backtest.basel_zone(5) == "YELLOW" and backtest.basel_zone(9) == "YELLOW"
    assert backtest.basel_zone(10) == "RED" and backtest.basel_zone(37) == "RED"


def test_acerbi_zero_when_es_equals_realised_tail_mean():
    rng = np.random.default_rng(5)
    losses = np.abs(rng.standard_normal(5000)) * 0.01
    var = np.full(5000, np.quantile(losses, 0.99))
    es_val = losses[losses > var[0]].mean()
    es = np.full(5000, es_val)
    az = backtest.acerbi_szekely(losses, var, es, 0.99)
    assert abs(az["Z1"]) < 0.05 and abs(az["Z2"]) < 0.35


def test_acerbi_flags_understated_es():
    rng = np.random.default_rng(5)
    losses = np.abs(rng.standard_normal(5000)) * 0.01
    var = np.full(5000, np.quantile(losses, 0.99))
    es = np.full(5000, losses[losses > var[0]].mean() * 0.6)  # ES understated by 40%
    az = backtest.acerbi_szekely(losses, var, es, 0.99)
    assert az["Z1"] > 0.3  # must point the right way
