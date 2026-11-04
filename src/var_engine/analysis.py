"""Regime analysis, stressed VaR and component (Euler) risk allocation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import methods

REGIMES: dict[str, tuple[str, str]] = {
    "GFC & aftermath (2007-2009)": ("2007-01-01", "2009-12-31"),
    "Euro crisis & QE (2010-2015)": ("2010-01-01", "2015-12-31"),
    "Late cycle (2016-2019)": ("2016-01-01", "2019-12-31"),
    "Covid & ZIRP (2020-2021)": ("2020-01-01", "2021-12-31"),
    "Rate shock (2022)": ("2022-01-01", "2022-12-31"),
    "AI bull market (2023-2026)": ("2023-01-01", "2100-01-01"),
}

STRESS_WINDOWS: dict[str, tuple[str, str]] = {
    "GFC crash (Sep 2008 - Mar 2009)": ("2008-09-01", "2009-03-31"),
    "Covid crash (Feb - Mar 2020)": ("2020-02-19", "2020-03-23"),
    "Rate shock (Jan - Oct 2022)": ("2022-01-01", "2022-10-31"),
    "Euro sovereign crisis (2011)": ("2011-07-01", "2011-11-30"),
}


def regime_table(port: pd.Series, alpha: float = 0.99) -> pd.DataFrame:
    rows = []
    for name, (a, b) in REGIMES.items():
        seg = port[(port.index >= a) & (port.index < b)]
        if len(seg) < 60:
            continue
        v, e = methods.historical_var_es(seg.values, alpha)
        nu = float(__import__("scipy").stats.t.fit(seg.values)[0])
        rows.append({"regime": name, "days": len(seg),
                     "ann_vol_pct": round(float(seg.std() * np.sqrt(252) * 100), 2),
                     "hist_VaR99_pct": round(v * 100, 2), "hist_ES99_pct": round(e * 100, 2),
                     "worst_day_pct": round(float(seg.min() * 100), 2), "fitted_t_nu": round(nu, 2)})
    return pd.DataFrame(rows)


def stressed_var(port: pd.Series, alpha: float = 0.99) -> pd.DataFrame:
    """Basel-style stressed VaR: the same estimator applied to stress windows,
    plus the worst-volatility rolling 12 months in the sample (the Basel 2.5
    sVaR idea: calibrate to a period of significant financial stress)."""
    rows = []
    for name, (a, b) in STRESS_WINDOWS.items():
        seg = port[(port.index >= a) & (port.index <= b)]
        if len(seg) < 20:
            continue
        v, e = methods.historical_var_es(seg.values, alpha)
        rows.append({"window": name, "days": len(seg), "stressed_VaR99_pct": round(v * 100, 2),
                     "stressed_ES99_pct": round(e * 100, 2)})
    roll_vol = port.rolling(252).std()
    if roll_vol.notna().any():
        end = roll_vol.idxmax(); start = end - pd.Timedelta(days=370)
        seg = port[(port.index > start) & (port.index <= end)]
        v, e = methods.historical_var_es(seg.values, alpha)
        rows.append({"window": f"Worst-vol 12m (ending {end.date()})", "days": len(seg),
                     "stressed_VaR99_pct": round(v * 100, 2), "stressed_ES99_pct": round(e * 100, 2)})
    return pd.DataFrame(rows)


def component_es(asset_returns: pd.DataFrame, weights: np.ndarray, alpha: float = 0.99) -> pd.DataFrame:
    """Euler allocation of historical ES: each asset's contribution is its
    weight times its mean return on the portfolio's tail days. Contributions
    sum exactly to portfolio ES - the allocation identity a risk desk needs
    to say *which position* owns the tail."""
    port = asset_returns.values @ weights
    var = float(-np.quantile(port, 1 - alpha))
    tail_mask = port <= -var
    contrib = -weights * asset_returns.values[tail_mask].mean(axis=0)
    total = float(contrib.sum())
    # Full precision is kept (rounding is a presentation concern): the Euler
    # identity sum(components) == portfolio ES then holds to machine precision
    # and is asserted in the test suite.
    return pd.DataFrame({
        "asset": list(asset_returns.columns), "weight_pct": weights * 100,
        "component_ES_pct": contrib * 100,
        "share_of_ES_pct": (contrib / total * 100) if total else np.zeros(len(weights)),
    })
