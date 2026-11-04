"""Professional Plotly dashboard export.

One self-contained HTML file (Plotly inlined - works offline, no server):
KPI cards, method comparison, returns-vs-VaR with exception markers,
distribution with VaR/ES levels, regime and component-ES charts, stress
table and the full backtest verdict table with Basel zones.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CSS = """
body{margin:0;font-family:'Segoe UI',system-ui,Arial;background:#0b111d;color:#e8eef7}
header{padding:28px 34px 4px}h1{margin:0;font-size:27px}h2{font-size:18px;margin:26px 0 6px}
.sub{color:#9db2cc;max-width:1080px;line-height:1.55}.wrap{max-width:1240px;margin:auto;padding:8px 30px 40px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:16px 0}
.kpi{background:#141f31;border:1px solid #263650;border-radius:12px;padding:14px}
.kpi .v{font-size:24px;font-weight:700}.kpi .l{color:#9db2cc;font-size:11.5px;text-transform:uppercase;letter-spacing:.05em}
.card{background:#111a29;border:1px solid #263650;border-radius:12px;padding:16px;margin:12px 0}
table{border-collapse:collapse;width:100%;font-size:13.5px}th,td{border-bottom:1px solid #263650;padding:6px 8px;text-align:right}
th:first-child,td:first-child{text-align:left}th{color:#9db2cc}
.GREEN{color:#3ecf8e;font-weight:700}.YELLOW{color:#ffc857;font-weight:700}.RED{color:#ff6b6b;font-weight:700}
.note{color:#9db2cc;font-size:13.5px;line-height:1.6}footer{color:#7186a3;font-size:12.5px;padding:0 30px 28px;max-width:1240px;margin:auto;line-height:1.6}
"""


_PLOTLY_EMBEDDED = False


def _fig(fig: go.Figure) -> str:
    global _PLOTLY_EMBEDDED
    fig.update_layout(template="plotly_dark", paper_bgcolor="#111a29", plot_bgcolor="#0d1523",
                      font=dict(size=12), margin=dict(l=46, r=16, t=36, b=36), height=340,
                      legend=dict(orientation="h", y=1.12))
    # Inline Plotly.js once (first figure) so the file works fully offline.
    include = False if _PLOTLY_EMBEDDED else "inline"
    _PLOTLY_EMBEDDED = True
    return fig.to_html(full_html=False, include_plotlyjs=include)


def _table(df: pd.DataFrame, zone_col: str | None = None) -> str:
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    body = ""
    for _, row in df.iterrows():
        cells = ""
        for c in df.columns:
            v = row[c]
            if zone_col and c == zone_col and isinstance(v, str):
                cells += f'<td class="{v}">{v}</td>'
            else:
                cells += f"<td>{v}</td>"
        body += f"<tr>{cells}</tr>"
    return f"<table><tr>{head}</tr>{body}</table>"


def build_dashboard(
    port: pd.Series,
    methods_df: pd.DataFrame,
    backtest99: pd.DataFrame,
    backtest95: pd.DataFrame,
    regimes: pd.DataFrame,
    stress: pd.DataFrame,
    components: pd.DataFrame,
    rolling: pd.DataFrame,
    meta: dict,
) -> str:
    m99 = methods_df[methods_df.alpha == 0.99].set_index("method")
    hist_es = m99.loc["Historical", "ES_pct"]; norm_es = m99.loc["Gaussian", "ES_pct"]
    evt_es = m99.loc["EVT POT-GPD", "ES_pct"]
    garch_bt = backtest99[backtest99.method == "garch_fhs"].iloc[0]
    hist_bt = backtest99[backtest99.method == "historical"].iloc[0]

    # Fig 1: returns + rolling VaR + exceptions
    f1 = go.Figure()
    f1.add_trace(go.Scatter(x=rolling["date"], y=rolling["return_pct"], mode="lines",
                            name="Daily return %", line=dict(width=0.7, color="#7d94b5")))
    f1.add_trace(go.Scatter(x=rolling["date"], y=-rolling["var99_hist_pct"], mode="lines",
                            name="-VaR 99% (rolling hist)", line=dict(width=1.2, color="#ff6b6b")))
    exc = rolling[rolling["exception"] == 1]
    f1.add_trace(go.Scatter(x=exc["date"], y=exc["return_pct"], mode="markers",
                            name="VaR exception", marker=dict(color="#ffc857", size=5)))
    f1.update_layout(title="Portfolio returns vs rolling 250-day 99% VaR, with exceptions")

    # Fig 2: method comparison at 99
    f2 = go.Figure()
    f2.add_trace(go.Bar(x=m99.index, y=m99["VaR_pct"], name="VaR 99%", marker_color="#4da3ff"))
    f2.add_trace(go.Bar(x=m99.index, y=m99["ES_pct"], name="ES 99%", marker_color="#ff8c42"))
    f2.update_layout(title="VaR / ES at 99% by method (full sample, % of portfolio value)", barmode="group")

    # Fig 3: distribution
    f3 = go.Figure()
    f3.add_trace(go.Histogram(x=port.values * 100, nbinsx=90, name="Daily returns %",
                              marker_color="#4da3ff", opacity=0.75))
    for val, nm, col in [(m99.loc["Historical", "VaR_pct"], "Hist VaR99", "#ff6b6b"),
                         (hist_es, "Hist ES99", "#ffc857"), (evt_es, "EVT ES99", "#3ecf8e")]:
        f3.add_vline(x=-val, line_color=col, line_width=2,
                     annotation_text=f"{nm} -{val:.2f}%", annotation_position="top")
    f3.update_layout(title="Return distribution (log returns, %) with tail levels")

    # Fig 4: regimes + components
    f4 = make_subplots(rows=1, cols=2, subplot_titles=("ES 99% by regime (%)", "Component ES - who owns the tail (%)"))
    f4.add_trace(go.Bar(x=regimes["hist_ES99_pct"], y=regimes["regime"], orientation="h",
                        marker_color="#ff8c42", name="Regime ES"), row=1, col=1)
    f4.add_trace(go.Bar(x=components["component_ES_pct"], y=components["asset"], orientation="h",
                        marker_color="#4da3ff", name="Component ES"), row=1, col=2)
    f4.update_layout(title="Risk is a regime story, and a position story", showlegend=False, height=360)

    bt_show = backtest99.drop(columns=[c for c in ("alpha",) if c in backtest99.columns])
    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VaR Engine Pro - Market Risk Dashboard</title>
<style>{CSS}</style></head><body>
<header><h1>VaR Engine Pro - Multi-Method Market Risk, 2007-2026</h1>
<p class="sub">Portfolio: SPY 35% / QQQ 20% / TLT 20% / LQD 10% / GLD 15%, daily log returns,
{meta['n_days']} trading days ({meta['start']} to {meta['end']}) covering the GFC, euro crisis, Covid crash,
the 2022 rate shock and the AI bull market. Nine estimators, a regulatory-grade backtest suite
(Kupiec, Christoffersen, conditional coverage, Acerbi-Szekely ES diagnostics, Basel traffic light),
stressed VaR and Euler component ES. Cross-checked against Riskfolio-Lib's independent
VaR_Hist / CVaR_Hist implementation (agreement within {meta['riskfolio_diff_pct']} pct pts).</p></header>
<div class="wrap">
<div class="kpis">
<div class="kpi"><div class="l">Historical ES 99%</div><div class="v">{hist_es:.2f}%</div></div>
<div class="kpi"><div class="l">Gaussian ES 99% (the trap)</div><div class="v">{norm_es:.2f}%</div></div>
<div class="kpi"><div class="l">EVT (GPD) ES 99%</div><div class="v">{evt_es:.2f}%</div></div>
<div class="kpi"><div class="l">Worst day in sample</div><div class="v">{meta['worst_day_pct']:.2f}%</div></div>
<div class="kpi"><div class="l">GARCH-FHS backtest - Basel zone (last 250d)</div><div class="v {garch_bt['Basel_zone_last250']}">{garch_bt['Basel_zone_last250']}</div></div>
<div class="kpi"><div class="l">Historical backtest - Basel zone (last 250d)</div><div class="v {hist_bt['Basel_zone_last250']}">{hist_bt['Basel_zone_last250']}</div></div>
</div>
<div class="card">{_fig(f1)}</div>
<div class="card">{_fig(f2)}
<p class="note"><b>Reading this:</b> the Gaussian method prices the 99% tail at barely half the EVT level.
Under FRTB, capital follows ES - a desk running Gaussian ES would hold roughly
{abs(norm_es/evt_es*100-100):.0f}% less capital against the same book than the tail evidence supports.
The fitted Student-t degrees of freedom on the full sample is {meta['t_nu']:.1f} - the data itself rejects normality.
At the other extreme, Cornish-Fisher overshoots wildly on this sample - with kurtosis this large the expansion
itself breaks down, which is why it is kept in the table as a demonstrated failure mode, not a candidate.</p></div>
<div class="card">{_fig(f3)}</div>
<div class="card">{_fig(f4)}</div>
<h2>Backtest verdicts - 99% (rolling, one-step-ahead)</h2>
<div class="card">{_table(bt_show, zone_col="Basel_zone_last250")}
<p class="note">How to read: Kupiec tests the exception <i>rate</i>; Christoffersen tests whether exceptions
<i>cluster</i>; conditional coverage is the joint test. Acerbi-Szekely Z1/Z2 are signed ES diagnostics
(~0 = calibrated; Z2 below 0 = realised tail losses ran above forecast ES). Basel zones are the
supervisor's lens on the last 250 days and can disagree with the statistical tests - that disagreement
is itself reported in professional validation. Full 95% backtests and all method tables are in results/.</p></div>
<h2>Stressed VaR (Basel 2.5 concept)</h2>
<div class="card">{_table(stress)}
<p class="note">Current-window VaR is a fair-weather number. Stressed VaR re-prices the same book on the
worst windows - it is routinely 2-3x the full-sample figure here, which is exactly why Basel made it a
separate capital component after 2008.</p></div>
<h2>Methodology and limitations</h2>
<p class="note">Log returns on adjusted prices; fixed weights rebalanced daily (documented simplification -
no transaction costs on the risk numbers). GARCH backtest refits annually on 750 days with parameters
frozen between refits. EVT threshold at the 95th percentile of losses (249 exceedances), GPD fitted by
MLE with location 0; threshold choice is the known soft spot of POT and is stress-tested in the notebook.
Monte Carlo: 200,000 draws, multivariate-t (nu=6) via Cholesky - dependence is elliptical, so tail
dependence is symmetric by construction; a t-copula variant is the documented next step. ES backtesting
remains genuinely hard (ES is not elicitable alone) - hence diagnostics, not verdicts, for ES.
Repo: src/var_engine (data, methods, backtest, analysis), tests/, notebooks/, RESEARCH.md.</p>
</div>
<footer>Project 01 of the Advanced Finance Projects programme (US/EU). Author: Samrat Banerjee.
First draft built with AI assistance (disclosed); research, verification and conclusions by the author.
Data: Yahoo Finance. Libraries: arch, Riskfolio-Lib (cross-check), QuantStats (tearsheet in results/), Plotly, scipy.
Not investment advice; not a regulatory capital tool.</footer></body></html>"""
    return html
