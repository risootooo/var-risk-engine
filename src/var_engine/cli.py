"""Pipeline: data -> methods -> backtests -> regimes/stress/components ->
Riskfolio-Lib cross-check -> results/ + Plotly dashboard + QuantStats tearsheet.

Usage:  python -m var_engine.cli [--refresh-data]
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import analysis, backtest, data as data_mod, methods
from .dashboard import build_dashboard

log = logging.getLogger("var_engine")
ROOT = Path(__file__).resolve().parents[2]
ALPHAS = (0.95, 0.975, 0.99)  # 97.5 included: it is the FRTB ES level


def compute_methods(asset_rets: pd.DataFrame, port: pd.Series) -> pd.DataFrame:
    r = port.values
    mu_vec, cov = asset_rets.mean().values, asset_rets.cov().values
    rows = []
    garch_main: dict = {}
    for alpha in ALPHAS:
        fixed = {
            "Historical": methods.historical_var_es(r, alpha),
            "Gaussian": methods.gaussian_var_es(float(r.mean()), float(r.std(ddof=1)), alpha),
            "Student-t": methods.student_t_var_es(r, alpha),
            "Cornish-Fisher": methods.cornish_fisher_var_es(r, alpha),
            "EWMA (RiskMetrics)": methods.ewma_var_es(r, alpha),
            "Monte Carlo (multiv-t)": methods.monte_carlo_var_es(mu_vec, cov, data_mod.WEIGHTS, alpha),
            "Monte Carlo (multiv-normal)": methods.monte_carlo_var_es(mu_vec, cov, data_mod.WEIGHTS, alpha, dist="normal"),
        }
        for name, (v, e) in fixed.items():
            rows.append({"method": name, "alpha": alpha, "VaR_pct": round(v * 100, 3), "ES_pct": round(e * 100, 3)})
        for spec in ("GARCH-t", "EGARCH-t", "GJR-GARCH-t"):
            g = methods.garch_fhs_var_es(r, alpha=alpha, model=spec)
            if spec == "GARCH-t" and alpha == 0.99:
                garch_main = g
            rows.append({"method": f"FHS {spec}", "alpha": alpha, "VaR_pct": round(g["var"] * 100, 3),
                         "ES_pct": round(g["es"] * 100, 3)})
        evt = methods.evt_pot_var_es(r, alpha)
        rows.append({"method": "EVT POT-GPD", "alpha": alpha, "VaR_pct": round(evt["var"] * 100, 3),
                     "ES_pct": round(evt["es"] * 100, 3)})
    return pd.DataFrame(rows), garch_main, evt


def riskfolio_crosscheck(port: pd.Series) -> dict:
    """Independent implementation check: Riskfolio-Lib's RiskFunctions.
    Returns max abs difference (pct pts) vs this engine's Historical numbers."""
    try:
        import riskfolio as rp

        rets = port.to_frame("portfolio")
        out = {}
        for alpha in (0.95, 0.99):
            # Riskfolio's `alpha` is the TAIL probability (significance), not
            # the confidence level - passing 0.99 returns a nonsense "VaR" of
            # the wrong tail (verified by direct comparison, see summary).
            v = float(np.ravel(rp.RiskFunctions.VaR_Hist(rets, alpha=1 - alpha))[0])
            c = float(np.ravel(rp.RiskFunctions.CVaR_Hist(rets, alpha=1 - alpha))[0])
            hv, he = methods.historical_var_es(port.values, alpha)
            out[str(alpha)] = {"riskfolio_VaR_pct": round(v * 100, 3), "riskfolio_CVaR_pct": round(c * 100, 3),
                               "engine_VaR_pct": round(hv * 100, 3), "engine_ES_pct": round(he * 100, 3),
                               "diff_VaR": round(abs(v - hv) * 100, 4), "diff_ES": round(abs(c - he) * 100, 4)}
        return out
    except Exception as exc:  # cross-check is a bonus, never a silent failure
        log.warning("Riskfolio cross-check unavailable: %s", exc)
        return {"error": str(exc)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-data", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    prices = data_mod.load_prices(ROOT / "data" / "prices_2007.csv", refresh=args.refresh_data)
    asset_rets = data_mod.log_returns(prices)
    port = data_mod.portfolio_returns(asset_rets, data_mod.WEIGHTS)

    methods_df, garch_main, evt_main = compute_methods(asset_rets, port)
    bt99 = backtest.run_backtests(port, alpha=0.99)
    bt95 = backtest.run_backtests(port, alpha=0.95)
    regimes = analysis.regime_table(port)
    stress = analysis.stressed_var(port)
    components = analysis.component_es(asset_rets, data_mod.WEIGHTS)

    # Rolling frame for the dashboard chart (historical 99, window 250)
    r = port.values
    roll_var = pd.Series(r).rolling(250).quantile(0.01) * -100
    rolling = pd.DataFrame({"date": port.index, "return_pct": r * 100, "var99_hist_pct": roll_var.values})
    rolling["exception"] = (rolling["return_pct"] < -rolling["var99_hist_pct"]).astype(int)

    cross = riskfolio_crosscheck(port)
    max_diff = 0.0
    if "error" not in cross:
        max_diff = max(max(v["diff_VaR"], v["diff_ES"]) for v in cross.values())

    meta = {"n_days": int(len(port)), "start": str(port.index.min().date()), "end": str(port.index.max().date()),
            "worst_day_pct": round(float(port.min() * 100), 2),
            "ann_vol_pct": round(float(port.std() * np.sqrt(252) * 100), 2),
            "t_nu": float(stats.t.fit(r)[0]), "riskfolio_diff_pct": max_diff,
            "garch_persistence": garch_main.get("persistence"), "evt_xi": evt_main.get("xi")}

    res = ROOT / "results"; res.mkdir(exist_ok=True)
    methods_df.to_csv(res / "methods_comparison.csv", index=False)
    bt99.to_csv(res / "backtest_99.csv", index=False)
    bt95.to_csv(res / "backtest_95.csv", index=False)
    regimes.to_csv(res / "regimes.csv", index=False)
    stress.to_csv(res / "stress.csv", index=False)
    components.to_csv(res / "component_es.csv", index=False)
    json.dump({"meta": meta, "riskfolio_crosscheck": cross}, open(res / "summary.json", "w"), indent=2)

    html = build_dashboard(port, methods_df, bt99, bt95, regimes, stress, components, rolling, meta)
    (ROOT / "dashboard").mkdir(exist_ok=True)
    (ROOT / "dashboard" / "dashboard.html").write_text(html)

    try:  # QuantStats tearsheet - professional performance reporting, optional
        import quantstats as qs

        qs.reports.html(port.rename("VaR portfolio"), benchmark=asset_rets["SPY"],
                        output=str(res / "quantstats_tearsheet.html"), title="VaR Portfolio Tearsheet")
        log.info("QuantStats tearsheet written")
    except Exception as exc:
        log.warning("QuantStats tearsheet skipped: %s", exc)

    print(json.dumps(meta, indent=2))
    print(methods_df[methods_df.alpha == 0.99].to_string(index=False))
    print(bt99.to_string(index=False))


if __name__ == "__main__":
    main()
