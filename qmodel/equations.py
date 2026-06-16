"""Single source of truth for every adjustable number in the model.

The website's "equation editor" reads DEFAULT_PARAMS to render the textboxes
and posts overrides back. Keeping it all here means the math and the UI never
drift apart. Each factor maps to a fiscal.ai ratioId (or a computed series).
"""
from __future__ import annotations
import copy

# Each fundamental factor: the fiscal.ai ratioId, a human label, the screen
# threshold, the comparison direction, whether higher raw values are "better"
# (used for z-score sign), and its weight in the composite.
FACTORS = {
    "roic": {
        "label": "ROIC", "ratio_id": "ratio_return_on_invested_capital",
        "threshold": 0.25, "op": ">=", "higher_better": True, "weight": 0.0,
        "unit": "ratio", "desc": "Return on invested capital (>= 25%)",
    },
    "eps": {
        "label": "Diluted EPS", "ratio_id": "ratio_diluted_eps",
        "threshold": 5.0, "op": ">=", "higher_better": True, "weight": 0.0,
        "unit": "usd", "desc": "Diluted earnings per share (>= $5)",
    },
    "pe": {
        "label": "P/E", "ratio_id": "ratio_price_to_earnings",
        "threshold": 30.0, "op": "<=", "higher_better": False, "weight": 0.0,
        "unit": "ratio", "desc": "Price / earnings (<= 30)",
    },
    "fcf_margin": {
        "label": "FCF Margin", "ratio_id": "ratio_fcf_margin",
        "threshold": 0.15, "op": ">=", "higher_better": True, "weight": 0.0,
        "unit": "ratio", "desc": "Free cash flow margin (>= 15%)",
    },
    "rev_growth": {
        "label": "Revenue Growth 1Y", "ratio_id": "growth_revenue_1y",
        "threshold": 0.15, "op": ">=", "higher_better": True, "weight": 1.5,
        "unit": "ratio", "desc": "Year-over-year revenue growth (>= 15%, HARD filter)",
    },
    # Price-derived factors (computed, no ratioId). Screens applied on computed values.
    "momentum": {
        "label": "Momentum 12-1", "ratio_id": None,
        "threshold": 0.0, "op": ">=", "higher_better": True, "weight": 1.0,
        "unit": "ratio", "desc": "12-month return skipping most recent month",
    },
    "sharpe": {
        "label": "Sharpe", "ratio_id": None,
        "threshold": 1.2, "op": ">=", "higher_better": True, "weight": 1.25,
        "unit": "ratio", "desc": "Annualized Sharpe ratio (>= 1.2)",
    },
    "sortino": {
        "label": "Sortino", "ratio_id": None,
        "threshold": 1.0, "op": ">=", "higher_better": True, "weight": 0.0,
        "unit": "ratio", "desc": "Annualized Sortino ratio (>= 1.0)",
    },
    "mom_accel": {
        "label": "Momentum Accel.", "ratio_id": None,
        "threshold": 0.0, "op": ">=", "higher_better": True, "weight": 1.5,
        "unit": "ratio", "desc": "Change in momentum: recent 6M return minus prior 6M return (acceleration)",
    },
    "mom_52w_high": {
        "label": "52-Week High", "ratio_id": None,
        "threshold": 0.0, "op": ">=", "higher_better": True, "weight": 0.0,
        "unit": "ratio", "desc": "Nearness to 52-week high (price / 252-day max)",
    },
    "resid_mom": {
        "label": "Residual Momentum", "ratio_id": None,
        "threshold": 0.0, "op": ">=", "higher_better": True, "weight": 0.0,
        "unit": "ratio", "desc": "Risk-adjusted residual (beta-neutral) 12-1 momentum",
    },
    "reversal_1m": {
        "label": "Short-Term Reversal", "ratio_id": None,
        "threshold": 0.0, "op": ">=", "higher_better": False, "weight": 0.0,
        "unit": "ratio", "desc": "Last 1-month return; lower-is-better (recent losers bounce)",
    },
    # Variance penalty: higher realized variance subtracts from the score.
    "volatility": {
        "label": "Volatility", "ratio_id": None,
        "threshold": 0.40, "op": "<=", "higher_better": False, "weight": 0.0,
        "unit": "ratio", "desc": "Annualized realized volatility (<= 40%); neutral in the aggressive default",
    },
    # Size: small-cap premium (most statistically significant factor at quarterly).
    "size": {
        "label": "Size", "ratio_id": "calculated_market_cap",
        "threshold": 1e15, "op": "<=", "higher_better": False, "weight": 0.5,
        "unit": "number", "desc": "Market cap; small-is-better. Weight 0 by default — the small-cap tilt badly underperformed the mega-cap regime. Raise it for a cleaner decile staircase + defensiveness.",
    },
    "shareholder_yield": {
        "label": "Shareholder Yield", "ratio_id": "ratio_shareholder_yield",
        "threshold": 0.0, "op": ">=", "higher_better": True, "weight": 0.0,
        "unit": "ratio", "desc": "Cash returned to shareholders / market cap",
    },
    "fcf_yield": {
        "label": "FCF Yield", "ratio_id": "ratio_fcf_yield",
        "threshold": 0.0, "op": ">=", "higher_better": True, "weight": 0.0,
        "unit": "ratio", "desc": "Free cash flow / enterprise value",
    },
}

# Presentation metadata for the equation editor: how each factor reads as a
# formula, a plain-English explanation, and how to display its threshold.
# display_scale: multiply the stored (raw) threshold by this for display, and
# divide the user's input by it on the way back (so a 0.25 ratio reads as "25 %").
EQUATION_META = {
    "roic": {
        "symbol": "ROIC", "formula": r"ROIC = NOPAT / Invested&nbsp;Capital",
        "display_scale": 100, "display_unit": "%",
        "explain": "Return on invested capital — profit generated per dollar of capital deployed. "
                   "Our headline quality screen; high ROIC signals a durable competitive edge.",
    },
    "eps": {
        "symbol": "EPS", "formula": r"EPS = Net&nbsp;Income / Diluted&nbsp;Shares",
        "display_scale": 1, "display_unit": "$",
        "explain": "Diluted earnings per share — absolute per-share earnings power. "
                   "Screens out tiny or unprofitable names.",
    },
    "pe": {
        "symbol": "P/E", "formula": r"P/E = Price / EPS",
        "display_scale": 1, "display_unit": "×",
        "explain": "Price-to-earnings — what you pay per dollar of earnings. "
                   "Capped so the model doesn't overpay for growth.",
    },
    "fcf_margin": {
        "symbol": "FCF&nbsp;margin", "formula": r"FCF&nbsp;margin = Free&nbsp;Cash&nbsp;Flow / Revenue",
        "display_scale": 100, "display_unit": "%",
        "explain": "Share of revenue that converts to free cash flow — the quality of earnings, "
                   "since cash is harder to manipulate than accounting profit.",
    },
    "rev_growth": {
        "symbol": "g", "formula": r"g = Rev<sub>t</sub> / Rev<sub>t-1</sub> &minus; 1",
        "display_scale": 100, "display_unit": "%",
        "explain": "Year-over-year revenue growth — the top-line expansion rate.",
    },
    "momentum": {
        "symbol": "MOM", "formula": r"MOM = P<sub>t-1m</sub> / P<sub>t-12m</sub> &minus; 1",
        "display_scale": 100, "display_unit": "%",
        "explain": "12-month price return skipping the most recent month (the classic 12-1). "
                   "Skipping a month avoids short-term mean reversion.",
    },
    "sharpe": {
        "symbol": "Sharpe", "formula": r"Sharpe = &radic;252 &middot; (r&#772; &minus; r<sub>f</sub>) / &sigma;<sub>r</sub>",
        "display_scale": 1, "display_unit": "",
        "explain": "Annualized excess return per unit of total volatility — risk-adjusted performance.",
    },
    "sortino": {
        "symbol": "Sortino", "formula": r"Sortino = &radic;252 &middot; (r&#772; &minus; r<sub>f</sub>) / &sigma;<sub>down</sub>",
        "display_scale": 1, "display_unit": "",
        "explain": "Like Sharpe, but only downside volatility is penalized — upside swings don't count as risk.",
    },
    "mom_accel": {
        "symbol": "&Delta;MOM", "formula": r"&Delta;MOM = (P<sub>t</sub>/P<sub>t-6m</sub> &minus; 1) &minus; (P<sub>t-6m</sub>/P<sub>t-12m</sub> &minus; 1)",
        "display_scale": 100, "display_unit": "%",
        "explain": "Momentum acceleration (Ardila-Sornette 2020) — whether momentum is increasing "
                   "(recent 6-month return above the prior 6-month return). Captures stocks whose trend "
                   "is strengthening, not just stocks already up; beat plain momentum in 2/3 of academic tests.",
    },
    "mom_52w_high": {
        "symbol": "52wH", "formula": r"52wH = P<sub>t</sub> / max(P over trailing 252d)",
        "display_scale": 100, "display_unit": "%",
        "explain": "Nearness to the 52-week high (George-Hwang 2004) — stocks near their 1-year high "
                   "keep outperforming; it dominates and improves on plain past-return momentum.",
    },
    "reversal_1m": {
        "symbol": "REV", "formula": r"REV = P<sub>t</sub>/P<sub>t-1m</sub> &minus; 1",
        "display_scale": 100, "display_unit": "%",
        "explain": "Short-term reversal (Jegadeesh 1990): last 1-month return, scored low-is-better — "
                   "recent losers tend to bounce next month. Strongest untested factor (t=-2.3, clean "
                   "staircase) and complements 12-1 momentum, which skips this most-recent month.",
    },
    "resid_mom": {
        "symbol": "RMOM", "formula": r"RMOM = &Sigma; residual returns<sub>(12-1)</sub> / &sigma;<sub>residual</sub>",
        "display_scale": 1, "display_unit": "",
        "explain": "Residual momentum (Blitz-Huij-Martens 2011): 12-1 momentum of the part of returns "
                   "NOT explained by the market (residuals from a market regression), scaled by residual "
                   "vol. Strips out beta noise — in our scan it had a significant IC (t=2.4) AND a near-perfect "
                   "decile staircase (+0.94), unlike raw momentum.",
    },
    "volatility": {
        "symbol": "&sigma;", "formula": r"&sigma; = &radic;252 &middot; stdev(daily&nbsp;returns)",
        "display_scale": 100, "display_unit": "%",
        "explain": "Annualized realized volatility. Neutral (weight 0) by default: in this 2005-2026 "
                   "sample high-volatility stocks actually outperformed, so penalizing variance hurt "
                   "returns. Raise its weight for a low-risk tilt (lower CAGR, higher Sharpe).",
    },
    "size": {
        "symbol": "Size", "formula": r"Size = log(Market&nbsp;Cap)",
        "display_scale": 1, "display_unit": "",
        "explain": "Market cap, scored small-is-better (subtracted). The size premium — smaller "
                   "companies have historically out-returned larger ones, and it was the most "
                   "statistically significant, most monotonic factor in our scan.",
    },
    "shareholder_yield": {
        "symbol": "SH&nbsp;yield", "formula": r"= (buybacks + dividends + debt&nbsp;paydown) / mcap",
        "display_scale": 100, "display_unit": "%",
        "explain": "Total cash returned to shareholders as a share of market cap — a robust "
                   "value/discipline signal that held up in the scan.",
    },
    "fcf_yield": {
        "symbol": "FCF&nbsp;yield", "formula": r"FCF&nbsp;yield = Free&nbsp;Cash&nbsp;Flow / Enterprise&nbsp;Value",
        "display_scale": 100, "display_unit": "%",
        "explain": "Cheapness measured in cash, not accounting earnings — value done right, and a "
                   "positive-IC factor in the scan.",
    },
}

# Extra fundamental factors pulled for context / future weighting (not screened
# by default, weight 0 so they don't affect the composite until you turn them on).
EXTRA_RATIOS = {
    "earnings_yield": "ratio_earnings_yield",
    "ev_ebitda": "ratio_ev_to_ebitda",
    "gross_margin": "ratio_gross_profit_margin",
    "roe": "ratio_return_on_equity",
    "net_margin": "ratio_net_profit_margin",
    "debt_to_equity": "ratio_debt_to_equity",
    "shareholder_yield": "ratio_shareholder_yield",
    "fcf_yield": "ratio_fcf_yield",
    "eps_growth": "growth_diluted_eps_1y",
    "market_cap": "calculated_market_cap",
}

# Knobs that aren't per-factor.
SETTINGS = {
    "winsor_pct": 0.01,        # clip factors at 1st / 99th percentile
    "sector_neutral": True,     # z-score within sector
    "normalize": "rank",       # "rank" (robust, cleaner deciles) or "zscore"
    "screen_mode": "soft",     # "hard" = must pass all screens; "soft" = rank, screens informational
    "min_pass_fraction": 0.5,   # in soft mode, keep names passing >= this fraction of screens
    "hard_filters": ["rev_growth"],  # factors that are HARD requirements (applied in backtest + live)
    "hmm_states": 3,            # market regimes for the HMM
    "gold_weight_bull": 0.08,   # gold sleeve weight by regime
    "gold_weight_neutral": 0.14,
    "gold_weight_bear": 0.25,
    "rebalance_freq": "M",     # monthly rebalance for backtest

    # Stage 3 — weight scheme: "equal" | "inverse_vol" | "min_var" | "max_sharpe"
    "weight_scheme": "equal",
    "cov_window": 126,          # trailing days for the covariance estimate
    "max_weight": 0.30,         # per-name cap in the optimizer

    # Stage 4 — HMM regime overlay: scale equity exposure, rest -> gold
    "regime_overlay": False,    # off by default; turn on to de-risk in bear regimes
    "exposure_bull": 1.0,
    "exposure_neutral": 0.85,
    "exposure_bear": 0.50,

    # Stage 5 — Turnover buffer (hold-band). Keep a current holding as long as it
    # stays in the top-K ranked names; only replace it once it drops out of top-K.
    # Cuts ~20-25% of turnover with no loss of edge. 0 = off (plain top-N).
    "turnover_band": 15,

    # Stage 6 — Crash overlay: daily vol-targeting (Barroso-Santa-Clara 2015).
    # Scales equity exposure by target_vol / trailing EWMA daily-vol forecast; the
    # un-invested fraction earns cash. Cuts max drawdown ~-58% -> ~-44% and lifts
    # risk-adjusted return. Supersedes the regime exposure overlay when on.
    "vol_target": True,            # de-risk into rising-volatility regimes
    "vol_target_annual": 0.27,     # target annualized vol (~strategy full-sample)
    "vol_ewma_lambda": 0.94,       # EWMA decay for the daily-vol forecast
    "vol_exposure_cap": 1.0,       # max exposure (1.0 = de-risk only, no leverage)
}


def default_params() -> dict:
    """Fresh, mutable copy of all defaults for a request."""
    return {
        "factors": copy.deepcopy(FACTORS),
        "extra_ratios": copy.deepcopy(EXTRA_RATIOS),
        "settings": copy.deepcopy(SETTINGS),
    }


def merge_overrides(overrides: dict | None) -> dict:
    """Merge user overrides (from the website) onto the defaults."""
    params = default_params()
    if not overrides:
        return params
    for fname, fvals in (overrides.get("factors") or {}).items():
        if fname in params["factors"]:
            params["factors"][fname].update(
                {k: v for k, v in fvals.items() if k in ("threshold", "weight", "op")}
            )
    for k, v in (overrides.get("settings") or {}).items():
        if k in params["settings"]:
            params["settings"][k] = v
    return params


# All ratioIds we need from fiscal.ai in a single batched call per company.
def all_ratio_ids() -> list[str]:
    ids = [f["ratio_id"] for f in FACTORS.values() if f["ratio_id"]]
    ids += list(EXTRA_RATIOS.values())
    # de-dup preserving order
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i); out.append(i)
    return out
