"""Streamlit front-end for var_engine (optional extra: pip install streamlit).

Run:  streamlit run app/streamlit_app.py
The app recomputes Historical / Gaussian / Student-t / EWMA VaR-ES live from
the cached price history whenever weights or the confidence level change;
the heavy backtest suite runs from the CLI (var_engine.cli) and its verdicts
are read from results/.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from var_engine import data as data_mod, methods  # noqa: E402

st.set_page_config(page_title="VaR Engine Pro", layout="wide")
st.title("VaR Engine Pro - multi-method market risk")
st.caption("SPY / QQQ / TLT / LQD / GLD - log returns - cached Yahoo Finance history")

prices = data_mod.load_prices(Path(__file__).resolve().parents[1] / "data" / "prices_2007.csv")
asset_rets = data_mod.log_returns(prices)

with st.sidebar:
    st.header("Portfolio")
    raw = [st.slider(t, 0, 60, int(w * 100)) for t, w in zip(data_mod.TICKERS, data_mod.WEIGHTS)]
    total = sum(raw) or 1
    weights = np.array([x / total for x in raw])
    alpha = st.selectbox("Confidence level", [0.95, 0.975, 0.99], index=2)
    st.write("Normalised weights:", {t: f"{w:.1%}" for t, w in zip(data_mod.TICKERS, weights)})

port = data_mod.portfolio_returns(asset_rets, weights)
r = port.values
rows = {
    "Historical": methods.historical_var_es(r, alpha),
    "Gaussian": methods.gaussian_var_es(float(r.mean()), float(r.std(ddof=1)), alpha),
    "Student-t": methods.student_t_var_es(r, alpha),
    "Cornish-Fisher": methods.cornish_fisher_var_es(r, alpha),
    "EWMA": methods.ewma_var_es(r, alpha),
}
tbl = pd.DataFrame(rows, index=["VaR", "ES"]).T.mul(100).round(2)
c1, c2 = st.columns([1, 2])
c1.subheader(f"VaR / ES at {alpha:.1%} (%)")
c1.dataframe(tbl)
c1.metric("Historical ES", f"{tbl.loc['Historical', 'ES']:.2f}%")
c1.metric("Gaussian ES", f"{tbl.loc['Gaussian', 'ES']:.2f}%")

fig = go.Figure()
roll = port.rolling(250).quantile(1 - alpha) * -100
fig.add_trace(go.Scatter(x=port.index, y=port.values * 100, name="Return %", line=dict(width=0.7)))
fig.add_trace(go.Scatter(x=port.index, y=-roll, name=f"-VaR {alpha:.0%}", line=dict(color="#ff6b6b")))
fig.update_layout(template="plotly_dark", title="Returns vs rolling VaR", height=420)
c2.plotly_chart(fig, use_container_width=True)
st.info("Full validation (Kupiec / Christoffersen / Acerbi-Szekely / Basel zones, stressed VaR, "
        "component ES) is produced by `python -m var_engine.cli` - see results/ and dashboard/dashboard.html.")
