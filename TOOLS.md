# Tooling provenance
Cloned 2026-11-02 from GitHub (depth-1) into ~/workspace/tools-github/ and
installed into a dedicated venv (~/workspace/tools-github/venv-risk):
- bashtage/arch @ 8164ef8 - GARCH family (used directly by the engine)
- dcajasn/Riskfolio-Lib @ 7c5429f - independent VaR/CVaR cross-check
- ranaroussi/quantstats @ 9ef4c6d - tearsheet reporting
Installed versions in the build venv: arch 8.0.0, riskfolio-lib 7.4.0,
quantstats 0.0.86, plotly 7.1.0, streamlit 1.66.1, yfinance 1.7.0.
Anyone else: `make setup` reproduces the environment from requirements.txt.
