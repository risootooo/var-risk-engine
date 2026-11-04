"""VaR / ES backtesting - the model-validation layer.

Implemented tests
-----------------
* Kupiec (1995) POF - unconditional coverage, LR ~ chi2(1).
* Christoffersen (1998) independence - LR ~ chi2(1); conditional coverage
  (LRuc + LRind) ~ chi2(2).
* Basel traffic light (BCBS): exceptions over a 250-day window at 99%:
  green 0-4, yellow 5-9, red 10+.  Reported, as supervisors do, alongside -
  not instead of - the statistical tests (they legitimately disagree:
  6 exceptions fails Basel-yellow while Kupiec does not reject).
* Acerbi-Szekely (2014) ES diagnostics Z1 / Z2, reported as signed
  statistics: under a correct model both are ~0; Z2 < 0 (equivalently
  exception losses averaging above forecast ES) means the model understates
  tail severity.  No p-value is attached to Z2 - its critical values require
  simulation of the predictive distribution, and pretending otherwise would
  be false precision.

Rolling one-step-ahead backtests use a 250-day estimation window for
historical / gaussian / EWMA.  The GARCH-FHS backtest refits annually on a
750-day window (parameters frozen between refits) - a documented compromise
between full daily refits and a single in-sample fit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import methods


def kupiec(n: int, x: int, p: float) -> tuple[float, float]:
    """LR statistic and p-value for H0: exception rate == p."""
    if x == 0:
        lr = -2 * n * np.log(1 - p)  # LR = -2 ln[ (1-p)^n ] when x = 0
    elif x == n:
        lr = -2 * n * np.log(p)
    else:
        phat = x / n
        lr = -2 * (np.log(((1 - p) ** (n - x)) * (p**x)) - np.log(((1 - phat) ** (n - x)) * (phat**x)))
    return float(lr), float(1 - stats.chi2.cdf(lr, 1))


def christoffersen(hits: np.ndarray) -> tuple[float, float]:
    h = np.asarray(hits, dtype=int)
    n00 = int(np.sum((h[:-1] == 0) & (h[1:] == 0))); n01 = int(np.sum((h[:-1] == 0) & (h[1:] == 1)))
    n10 = int(np.sum((h[:-1] == 1) & (h[1:] == 0))); n11 = int(np.sum((h[:-1] == 1) & (h[1:] == 1)))

    def ll(prob: float, a: int, b: int) -> float:
        out = 0.0
        if a > 0 and prob > 0: out += a * np.log(prob)
        if b > 0 and prob < 1: out += b * np.log(1 - prob)
        return out

    p = (n01 + n11) / max(len(h) - 1, 1)
    p01 = n01 / max(n00 + n01, 1); p11 = n11 / max(n10 + n11, 1)
    lr = -2 * (ll(p, n01 + n11, n00 + n10) - (ll(p01, n01, n00) + ll(p11, n11, n10)))
    return float(lr), float(1 - stats.chi2.cdf(lr, 1))


def basel_zone(exceptions: int) -> str:
    """Basel traffic light for 250 trading days at 99% VaR."""
    if exceptions <= 4: return "GREEN"
    if exceptions <= 9: return "YELLOW"
    return "RED"


def acerbi_szekely(losses: np.ndarray, var: np.ndarray, es: np.ndarray, alpha: float) -> dict:
    """Z1 = mean(exception loss / ES) - 1; Z2 = 1 - sum(exception loss/ES)/(T*p).
    Both ~0 under a correct model; positive Z1 / negative Z2 = tail understated."""
    exc = losses > var
    n_exc = int(exc.sum()); T = len(losses); p = 1 - alpha
    if n_exc == 0:
        return {"Z1": float("nan"), "Z2": float("nan"), "exceptions": 0}
    ratios = losses[exc] / es[exc]
    return {"Z1": float(ratios.mean() - 1), "Z2": float(1 - ratios.sum() / (T * p)),
            "exceptions": n_exc}


def _rolling_series(r: np.ndarray, alpha: float, method: str, window: int = 250):
    """One-step-ahead VaR/ES series for window-based methods."""
    n = len(r); var_s = np.full(n, np.nan); es_s = np.full(n, np.nan)
    for i in range(window, n):
        w = r[i - window:i]
        if method == "historical":
            v, e = methods.historical_var_es(w, alpha)
        elif method == "gaussian":
            v, e = methods.gaussian_var_es(float(w.mean()), float(w.std(ddof=1)), alpha)
        elif method == "ewma":
            v, e = methods.ewma_var_es(w, alpha)
        else:
            raise ValueError(method)
        var_s[i], es_s[i] = v, e
    return var_s, es_s


def _garch_series(r: np.ndarray, alpha: float, window: int = 750, refit_every: int = 250):
    """One-step-ahead GARCH-FHS VaR/ES: refit every ``refit_every`` days on the
    trailing ``window``; between refits, forecast vol by the model's recursion
    approximated with the fitted last conditional vol updated by realised
    returns (standard one-step GARCH forecast recursion)."""
    from arch import arch_model

    n = len(r); var_s = np.full(n, np.nan); es_s = np.full(n, np.nan)
    i = window
    while i < n:
        fit_r = r[i - window:i]
        am = arch_model(fit_r * 100, vol="Garch", p=1, q=1, dist="t", rescale=False)
        res = am.fit(disp="off")
        omega = float(res.params.get("omega", 0.0)); a1 = float(res.params.get("alpha[1]", 0.0))
        b1 = float(res.params.get("beta[1]", 0.0))
        zresid = res.std_resid
        q = float(np.quantile(zresid, 1 - alpha))
        tail = zresid[zresid <= q]; tail_mean = float(tail.mean()) if len(tail) else q
        vol2 = float(res.conditional_volatility[-1] ** 2)
        last_ret = fit_r[-1] * 100
        for j in range(i, min(i + refit_every, n)):
            # one-step forecast: sigma2_t = omega + a1 * eps2_{t-1} + b1 * sigma2_{t-1}
            eps2 = (last_ret - 0.0) ** 2
            vol2 = omega + a1 * eps2 + b1 * vol2
            vol = np.sqrt(vol2) / 100
            var_s[j] = -vol * q; es_s[j] = -vol * tail_mean
            last_ret = r[j] * 100
        i += refit_every
    return var_s, es_s


def run_backtests(port: pd.Series, alpha: float = 0.99, window: int = 250) -> pd.DataFrame:
    """Full rolling backtest for historical / gaussian / ewma / garch_fhs."""
    r = port.values; losses = -r
    rows = []
    series = {}
    for method in ("historical", "gaussian", "ewma"):
        series[method] = _rolling_series(r, alpha, method, window)
    series["garch_fhs"] = _garch_series(r, alpha)
    for method, (var_s, es_s) in series.items():
        mask = ~np.isnan(var_s)
        hits = (losses[mask] > var_s[mask]).astype(int)
        n = int(mask.sum()); x = int(hits.sum())
        lr_k, p_k = kupiec(n, x, 1 - alpha)
        lr_c, p_c = christoffersen(hits)
        lr_cc = lr_k + lr_c; p_cc = float(1 - stats.chi2.cdf(lr_cc, 2))
        az = acerbi_szekely(losses[mask], var_s[mask], es_s[mask], alpha)
        # Basel zone on the most recent 250 out-of-sample days
        recent_hits = hits[-250:]
        rows.append({
            "method": method, "alpha": alpha, "n": n, "exceptions": x,
            "expected": round(n * (1 - alpha), 1), "exception_rate_pct": round(x / n * 100, 2),
            "Kupiec_LR": round(lr_k, 3), "Kupiec_p": round(p_k, 4),
            "Christoffersen_LR": round(lr_c, 3), "Christoffersen_p": round(p_c, 4),
            "CondCoverage_p": round(p_cc, 4),
            "Acerbi_Z1": round(az["Z1"], 3) if az["Z1"] == az["Z1"] else None,
            "Acerbi_Z2": round(az["Z2"], 3) if az["Z2"] == az["Z2"] else None,
            "Basel_zone_last250": basel_zone(int(recent_hits.sum())) if alpha == 0.99 else "n/a (99% only)",
            "exceptions_last250": int(recent_hits.sum()),
        })
    return pd.DataFrame(rows)
