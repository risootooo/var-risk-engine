import numpy as np
import pandas as pd
import pytest

from var_engine import analysis, data as data_mod

RNG = np.random.default_rng(13)


def _fake_prices(days=600):
    idx = pd.bdate_range("2020-01-01", periods=days)
    base = 100 * np.exp(np.cumsum(RNG.standard_normal((days, 5)) * 0.01, axis=0))
    return pd.DataFrame(base, index=idx, columns=data_mod.TICKERS)


def test_ensure_order_fixes_alphabetical_columns():
    px = _fake_prices()
    shuffled = px[sorted(px.columns)]  # what yfinance actually hands back
    fixed = data_mod.ensure_order(shuffled, data_mod.TICKERS)
    assert list(fixed.columns) == data_mod.TICKERS
    assert fixed["SPY"].iloc[0] == px["SPY"].iloc[0]  # values follow their own ticker


def test_ensure_order_raises_on_missing_ticker():
    px = _fake_prices().drop(columns=["GLD"])
    with pytest.raises(ValueError):
        data_mod.ensure_order(px, data_mod.TICKERS)


def test_portfolio_weights_must_sum_to_one():
    rets = data_mod.log_returns(_fake_prices())
    with pytest.raises(ValueError):
        data_mod.portfolio_returns(rets, np.array([0.5, 0.5, 0.5, 0.0, 0.0]))


def test_component_es_sums_to_portfolio_es():
    rets = data_mod.log_returns(_fake_prices(3000))
    comp = analysis.component_es(rets, data_mod.WEIGHTS, 0.99)
    port = rets.values @ data_mod.WEIGHTS
    var = -np.quantile(port, 0.01)
    es = -port[port <= -var].mean()
    assert comp["component_ES_pct"].sum() == pytest.approx(es * 100, rel=1e-6)  # Euler identity
    assert comp["share_of_ES_pct"].sum() == pytest.approx(100.0, abs=0.6)
