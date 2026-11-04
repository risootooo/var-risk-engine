# Benchmark — how this project compares with professional / recruiter-facing exemplars

researched 3 Nov 2026. Filter method: (a) library-grade open-source finance
projects a risk hiring manager actually knows, (b) individual candidate
projects closest to this one's scale that present themselves the way
recruiters reward, (c) one platform-scale reference. Star counts via the
GitHub API on the day; live links are the point, not stars — the two best
individual exemplars have 0–1 stars each, which is itself the lesson:
**presentation and a live link beat repository popularity at candidate scale.**

## The filtered exemplars

| # | Project | Link | Scale | What sets the bar |
|---|---|---|---|---|
| 1 | Riskfolio-Lib | https://github.com/dcajasn/Riskfolio-Lib | 4,584★ library | Docs site, worked examples, versioned releases, academic paper. The "serious library" standard. |
| 2 | QuantStats | https://github.com/ranaroussi/quantstats | 7,784★ library | One function produces a beautiful, complete HTML tearsheet — output quality IS the product. |
| 3 | arch | https://github.com/bashtage/arch | 1,665★ library | Deep docs, examples, tests; boring in the best way — it reads as infrastructure. |
| 4 | Credit Risk Scoreboard (individual) | https://github.com/Keshav-spec/credit-risk-scoreboard | 1★, live apps | README opens with LIVE Streamlit + Tableau links and screenshots; bank-grade framing (WoE, KS/Gini, PSI monitoring, approval simulation). |
| 5 | Stock Market Dashboard (individual) | https://github.com/Dev2943/stock-market-dashboard | 0★, live on Google Cloud Run | One integrated 9-tab app incl. a VaR & Stress tab (Kupiec backtest, 2008/2020 replays) behind a public URL. Thin tests/CI — weaker engineering than this project, stronger product. |
| 6 | Quant Trading Framework (individual) | https://github.com/wangyunfeiphilip/quant-trading-framework | 1★ | Explicitly honest positioning ("no fabricated performance numbers"), dashboard pages, data-freshness checks, deploy instructions for a public Streamlit URL. |
| 7 | OpenBB (platform reference) | https://github.com/OpenBB-finance/OpenBB | ~60k★ platform | The visual/product ceiling: hosted workspace, brand, docs, community. Not a fair like-for-like for a solo project — included so the gap is understood, not chased. |

Recruiter-side evidence (hiring-manager write-ups reviewed in the same pass):
the consistent pattern is *one real problem, shipped, with a live artefact and
a 2-minute story* — projects are screened in the first minute on whether
something opens and works; depth is only read after that gate is passed.

## Rating — Project 01 (v2, as delivered 2 Nov 2026)

Scored /10 per dimension against the exemplar median above.

| Dimension | Exemplar median | P01 v2 | Evidence |
|---|---|---|---|
| First impression / visual design | 7 | **3** | Default dark Plotly export; dense; no brand, no narrative layout, no screenshots |
| Live interactivity / deployed | 8 | **2** | No public link; Streamlit needed a local install; the dashboard the user actually opened was static |
| Analytical depth | 6 | **8** | 9+ estimators incl. EVT/GPD, EGARCH/GJR — beyond every individual exemplar |
| Validation rigour | 5 | **9** | Kupiec + Christoffersen + CC + Acerbi–Székely + Basel zones + independent-library cross-check; no exemplar matches this |
| Code engineering | 4–6 | **8** | Package, 19 tests, CI, Docker — the app exemplars have no tests or CI at all |
| Documentation / storytelling | 7 | **6** | Strong README/RESEARCH text, but no live link or screenshots at the top where recruiters look |
| Data realism | 7 | **7** | Real 19-year data incl. GFC; same free-data ceiling as everyone |
| Original insight | 5 | **8** | "GARCH-FHS passes where Historical fails; TLT is a hedge in the tail" is an interview-grade finding |
| **Total** | **~51/80** | **51/80** | Substance ≈ exemplar median overall — but distributed exactly wrong |

**Verdict.** On substance this is a B− project (51/80). As a *delivered
product* — the only thing a recruiter or Samrat actually experiences — it is
a **D, and on first impression an F**: the two dimensions that gate everything
(first impression 3, live interactivity 2) are the two it failed. Samrat's
"F tier" call is correct for the deliverable as received; it is not correct
for the analytics underneath, which scored above the exemplar median.
Both things are true, and the fix targets the presentation layer, not the math.

## The v3 fix (this folder, dashboard/interactive.html)

Built against the exemplar pattern: a branded, light, product-style
interface; a verdict headline up top; **live controls** (weights +
95/97.5/99) recomputing VaR/ES from the full embedded 2007–2026 history in
the browser; tabbed Backtest / Regimes / Model-Card sections using the
verified engine tables. Projected re-score: first impression 7, live
interactivity 6 (single-file interactive; still no public hosted URL),
documentation 7 → **60/80 (B)**. The two remaining steps to A-tier
presentation both need Samrat's accounts, not more code: (1) push the repo
to his GitHub, (2) deploy the Streamlit app to Streamlit Community Cloud
(free) so the README can lead with a live link like exemplar #4.
