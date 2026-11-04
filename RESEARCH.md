# Research basis - what "professional standard" means for this project

Researched 2 Nov 2026 before rebuilding. Sources are the primary/standard
references a market-risk hiring manager would expect a candidate to know.

## Regulatory framing (why these methods and tests)

1. **Basel traffic light (BCBS, MAR32).** Internal-model VaR is backtested on
   250 trading days at 99%: green 0-4 exceptions, yellow 5-9, red 10+.
   Expected exceptions = 2.5/yr. Yellow/red raise the capital multiplier and
   can lead to model disallowance. The engine reports the zone *alongside*
   the statistical tests - they legitimately disagree (6 exceptions: Kupiec
   does not reject at 5%, Basel is already yellow), and professional
   validation reports both rather than hiding the disagreement.
2. **FRTB (Basel III/IV).** Capital moves from 99% VaR to **97.5% Expected
   Shortfall** (97.5% ES ≈ 99% VaR under normality, so the change bites only
   through fat tails - which is the point). VaR survives as the *backtesting*
   instrument because ES cannot be validated by counting exceptions. Hence
   the engine computes 95 / 97.5 / 99 levels and backtests VaR while treating
   ES with diagnostics, not verdicts.
3. **Stressed VaR (Basel 2.5, post-2008).** VaR must also be calibrated to a
   12-month period of significant financial stress. Implemented as named
   stress windows + the worst-volatility rolling 12 months in sample.
4. **Acerbi & Szekely (2014), "Backtesting Expected Shortfall".** Z1/Z2
   statistics: ~0 under a correct model; Z2 < 0 (in return convention) when
   realised tail losses exceed forecast ES. Their critical values require
   simulating the predictive distribution, so this project reports Z1/Z2 as
   signed diagnostics **without p-values** - attaching one would be false
   precision, and saying so is the professional choice.
5. **Kupiec (1995)** POF and **Christoffersen (1998)** independence /
   conditional coverage: the standard VaR backtest pair, LR tests vs chi2.

## Methods (why nine estimators)

* Historical and Gaussian are the baselines every desk still runs - and the
  demonstration matters: on 2007-2026 data the Gaussian 99% ES (1.91%) prices
  the tail a third below the EVT level (2.97%), and its rolling backtest
  produces 108 exceptions against 47 expected.
* Student-t (MLE) and Cornish-Fisher: cheap parametric answers to fat tails
  and skewness, with opposite failure modes (CF can misbehave far in the
  tail; the t forces symmetry).
* EWMA/RiskMetrics (lambda 0.94): the industry workhorse - reactive, but
  Gaussian-conditional, so it understates tail *shape* while tracking vol.
* Monte Carlo (multivariate normal and t via Cholesky): the flexible
  benchmark; the t version's common chi-square shock is what creates joint
  tail events, at the cost of symmetric tail dependence.
* GARCH-family filtered historical simulation (Barone-Adesi et al.):
  GARCH-t, EGARCH-t (leverage), GJR-GARCH-t (asymmetry), residuals read
  empirically (FHS) rather than from the fitted density. Recent large-sample
  evidence (39 markets, 2005-2025) finds FHS repairs most of historical
  simulation's independence failures and prices ES nearly without bias -
  the reason FHS gets a full one-step-ahead backtest here (annual refits),
  not just a point estimate.
* EVT peaks-over-threshold (McNeil & Frey 2000): GPD fitted to losses above
  the 95th percentile; the theoretically grounded tail estimator, sensitive
  to threshold choice - which the notebook stress-tests explicitly.

## Tooling (GitHub, all free / open source)

* **bashtage/arch** - the standard GARCH library (cloned and used for all
  GARCH-family fits).
* **dcajasn/Riskfolio-Lib** - 20+ risk measures; used here as an
  *independent implementation* to cross-check this engine's Historical
  VaR/CVaR (`RiskFunctions.VaR_Hist` / `CVaR_Hist`). Agreement is reported
  in results/summary.json - a second codebase agreeing is stronger evidence
  than a second run of the same code.
* **ranaroussi/quantstats** - professional HTML tearsheets; the pipeline
  emits one for the portfolio (results/quantstats_tearsheet.html).
* **plotly** - the dashboarding layer (interactive, single-file export).
* **streamlit** - the optional app front-end (app/streamlit_app.py).
* Data: Yahoo Finance via yfinance (free). No paid data anywhere.

## What was deliberately NOT done

* No 10-day sqrt-time scaling presented as analysis: it assumes iid returns
  that this project's own backtests reject; FRTB uses liquidity horizons,
  not sqrt(10). Mentioned here so the omission reads as a decision.
* No ES "pass/fail" verdicts (see Acerbi-Szekely note above).
* No machine-learning VaR in v2: the 39-market study shows a tuned LSTM
  still trails FHS on ES bias; adding one before the classical suite is
  bulletproof would be decoration, not engineering. It is the documented
  v3 candidate, against the FHS benchmark built here.
