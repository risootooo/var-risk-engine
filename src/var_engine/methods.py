"""VaR / Expected Shortfall estimators.

Conventions
-----------
* Inputs are *returns* (log returns, decimals).  Outputs are *losses*:
  positive numbers in percent units are produced by callers, functions here
  return decimal losses (0.021 == 2.1%).
* alpha is the confidence level (0.99 == 99% VaR).
* Every estimator returns ``(VaR, ES)`` so methods are drop-in comparable.

Methods: historical, gaussian, student_t, cornish_fisher, ewma (RiskMetrics),
monte_carlo (multivariate normal / t), garch_fhs (GARCH / EGARCH / GJR with
filtered historical simulation), evt_pot (peaks-over-threshold GPD, McNeil &
Frey style tail).  See RESEARCH.md for sources and formula references.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def _tail(r: np.ndarray, var: float) -> float:
    tail = r[r <= -var]
    return float(-tail.mean()) if len(tail) else float(var)


def historical_var_es(r: np.ndarray, alpha: float) -> tuple[float, float]:
    var = float(-np.quantile(r, 1 - alpha))
    return var, _tail(r, var)


def gaussian_var_es(mu: float, sd: float, alpha: float) -> tuple[float, float]:
    z = stats.norm.ppf(alpha)
    var = float(-mu + z * sd)
    es = float(-mu + sd * stats.norm.pdf(z) / (1 - alpha))
    return var, es


def student_t_var_es(r: np.ndarray, alpha: float) -> tuple[float, float]:
    """Student-t fitted by MLE (nu, loc, scale); ES via the t tail formula."""
    nu, loc, scale = stats.t.fit(r)
    q = stats.t.ppf(1 - alpha, nu)
    var = float(-(loc + q * scale))
    g = stats.t.pdf(q, nu)
    es = float(-loc + scale * (g / (1 - alpha)) * ((nu + q**2) / (nu - 1)))
    return var, es


def cornish_fisher_var_es(r: np.ndarray, alpha: float) -> tuple[float, float]:
    """Cornish-Fisher expanded quantile; ES by numerical integration of the
    expanded quantile function over the tail (200-point grid)."""
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    s = float(stats.skew(r)); k = float(stats.kurtosis(r))  # excess kurtosis

    def cf_quantile(p: float) -> float:
        z = stats.norm.ppf(p)
        zcf = (z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24
               - (2 * z**3 - 5 * z) * (s**2) / 36)
        return mu + zcf * sd

    var = float(-cf_quantile(1 - alpha))
    ps = np.linspace(0, 1 - alpha, 200)
    tail_q = np.array([cf_quantile(max(p, 1e-6)) for p in ps])
    es = float(-tail_q.mean())
    return var, max(es, var)


def ewma_forecast_sd(r: np.ndarray, lam: float = 0.94) -> float:
    """RiskMetrics EWMA one-step volatility forecast (mu = 0 convention)."""
    var = float(np.var(r[:50]))
    for x in r[50:]:
        var = lam * var + (1 - lam) * x**2
    return float(np.sqrt(lam * var + (1 - lam) * r[-1] ** 2))


def ewma_var_es(r: np.ndarray, alpha: float, lam: float = 0.94) -> tuple[float, float]:
    sd = ewma_forecast_sd(r, lam)
    return gaussian_var_es(0.0, sd, alpha)


def monte_carlo_var_es(
    mu_vec: np.ndarray,
    cov: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    n: int = 200_000,
    dist: str = "t",
    nu: float = 6.0,
    seed: int = 42,
) -> tuple[float, float]:
    """Multivariate simulation via Cholesky; 't' scales normals by a common
    chi-square draw (multivariate Student-t dependence)."""
    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(cov)
    z = rng.standard_normal((n, len(weights)))
    sims = mu_vec + z @ chol.T
    if dist == "t":
        chi = rng.chisquare(nu, n) / nu
        sims = mu_vec + (z @ chol.T) / np.sqrt(chi)[:, None]
    port = sims @ weights
    return historical_var_es(port, alpha)


_GARCH_SPECS = {
    "GARCH-t": dict(vol="Garch", p=1, q=1, dist="t"),
    "GARCH-skewt": dict(vol="Garch", p=1, q=1, dist="skewt"),
    "EGARCH-t": dict(vol="EGARCH", p=1, o=1, q=1, dist="t"),
    "GJR-GARCH-t": dict(vol="Garch", p=1, o=1, q=1, dist="t"),
}


def garch_fhs_var_es(
    r: np.ndarray,
    alpha: float,
    model: str = "GARCH-t",
) -> dict:
    """Filtered historical simulation: fit a GARCH-family model (``arch``
    library), scale the empirical quantiles of standardised residuals by the
    last conditional volatility. Returns VaR/ES plus fitted parameters."""
    from arch import arch_model

    if model not in _GARCH_SPECS:
        raise ValueError(f"Unknown GARCH spec {model!r}; choose from {list(_GARCH_SPECS)}")
    am = arch_model(r * 100, rescale=False, **_GARCH_SPECS[model])
    res = am.fit(disp="off")
    last_vol = float(res.conditional_volatility[-1]) / 100
    zresid = res.std_resid
    q = float(np.quantile(zresid, 1 - alpha))
    var = float(-last_vol * q)
    tail = zresid[zresid <= q]
    es = float(-last_vol * tail.mean()) if len(tail) else var
    params = {k: float(v) for k, v in res.params.items()}
    persistence = params.get("alpha[1]", 0.0) + params.get("beta[1]", 0.0)
    return {"var": var, "es": es, "last_vol": last_vol, "params": params,
            "persistence": float(persistence)}


def evt_pot_var_es(r: np.ndarray, alpha: float, threshold_q: float = 0.95) -> dict:
    """Peaks-over-threshold: fit a Generalised Pareto Distribution to losses
    above the ``threshold_q`` quantile (MLE, location fixed at 0) and invert
    the GPD tail for VaR; ES from the GPD mean-excess formula."""
    losses = -np.asarray(r)
    u = float(np.quantile(losses, threshold_q))
    exceed = losses[losses > u] - u
    n, nu_count = len(losses), len(exceed)
    if nu_count < 30:
        raise ValueError("Too few exceedances for a stable GPD fit")
    xi, _, beta = stats.genpareto.fit(exceed, floc=0)
    xi, beta = float(xi), float(beta)
    var = u + (beta / xi) * (((1 - alpha) / (nu_count / n)) ** (-xi) - 1) if abs(xi) > 1e-6 else \
        u - beta * np.log((1 - alpha) / (nu_count / n))
    var = float(var)
    if xi >= 1:
        es = float("inf")
    else:
        mean_excess = (beta + xi * (var - u)) / (1 - xi)
        es = float(var + mean_excess)
    return {"var": var, "es": es, "xi": xi, "beta": beta, "threshold": u,
            "n_exceed": int(nu_count)}
