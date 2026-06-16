"""Turn scored candidates into the final 7-holding portfolio (6 stocks + gold).

Selection: apply the screen (hard or soft), rank by composite, take the top 6,
then add a gold sleeve whose weight is set by the current HMM regime. Equity
weights are proportional to a positive-shifted composite so higher-conviction
names get more capital, without letting one name dominate.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import config
from qmodel import scoring
from qmodel.hmm_regime import gold_weight_for


def select(df_scored: pd.DataFrame, params: dict, regime: str,
           price_data: dict | None = None,
           current_holdings: list | None = None) -> dict:
    settings = params["settings"]
    df = scoring.screen_flags(df_scored, params)

    mode = settings.get("screen_mode", "soft")
    if mode == "hard":
        eligible = df[df["pass_fraction"] >= 0.999]
        if len(eligible) < config.N_STOCKS:      # graceful fallback
            eligible = df[df["pass_fraction"] >= settings["min_pass_fraction"]]
    else:
        eligible = df[df["pass_fraction"] >= settings["min_pass_fraction"]]
    if len(eligible) < config.N_STOCKS:
        eligible = df                              # last resort: rank everything

    ranked = scoring.apply_hard_filters(eligible, params).sort_values("composite", ascending=False)
    # Stage 5: hold-band -- if we know the current book, keep names still inside the
    # top-`band` and only replace those that dropped out (cuts needless turnover).
    band = int(settings.get("turnover_band", 0) or 0)
    if band and current_holdings:
        order = list(ranked["company_key"])
        rank_of = {ck: i for i, ck in enumerate(order)}
        prev = set(current_holdings)
        sel = [ck for ck in order if ck in prev and rank_of[ck] < band]
        for ck in order:
            if len(sel) >= config.N_STOCKS:
                break
            if ck not in sel:
                sel.append(ck)
        picks = ranked.set_index("company_key").loc[sel[:config.N_STOCKS]].reset_index().copy()
    else:
        picks = ranked.head(config.N_STOCKS).copy()
    comp = picks["composite"].values.astype(float)
    cks = list(picks["company_key"])

    # Stage 3: equity weights by scheme (default = composite-proportional)
    scheme = settings.get("weight_scheme", "equal")
    eq_w = None
    if scheme in ("inverse_vol", "min_var", "max_sharpe") and price_data:
        from qmodel import optimize
        cov_w = int(settings.get("cov_window", 126))
        R = []
        for ck in cks:
            s = price_data.get(ck)
            r = s.pct_change().dropna().values[-cov_w:] if s is not None else None
            if r is None or len(r) < cov_w:
                R = None; break
            R.append(r)
        if R and len(R) > 1:
            cov = np.cov(np.vstack(R))
            eq_w = optimize.weights_for(scheme, comp, cov, float(settings.get("max_weight", 0.30)))
    if eq_w is None:
        shifted = comp - comp.min() + 0.1
        eq_w = shifted / shifted.sum()

    # Stage 4: regime overlay scales equity exposure (rest -> gold), else gold sleeve
    if settings.get("regime_overlay"):
        exp_map = {"bull": settings.get("exposure_bull", 1.0),
                   "neutral": settings.get("exposure_neutral", 0.85),
                   "bear": settings.get("exposure_bear", 0.50)}
        eq_scale = float(exp_map.get(regime, 0.85))
        gold_w = 1.0 - eq_scale
    else:
        gold_w = gold_weight_for(regime, settings)
        eq_scale = 1.0 - gold_w

    # Stage 6: crash overlay -- scale equity exposure by the book's own recent
    # volatility (vol-targeting). The de-risked remainder goes to cash.
    cash_w, vol_exposure = 0.0, 1.0
    if settings.get("vol_target") and price_data:
        vol_exposure = _vol_target_exposure(cks, eq_w, price_data, settings)
        cash_w = eq_scale * (1.0 - vol_exposure)
        eq_scale = eq_scale * vol_exposure

    holdings = []
    for (_, r), w in zip(picks.iterrows(), eq_w):
        holdings.append({
            "ticker": r["ticker"], "name": r["name"], "sector": r["sector"],
            "company_key": r["company_key"],
            "weight": round(float(w * eq_scale), 4),
            "composite": round(float(r["composite"]), 3),
            "screens_passed": int(r["screens_passed"]),
            "screens_total": int(r["screens_total"]),
            "last_price": None if pd.isna(r.get("last_price")) else round(float(r["last_price"]), 2),
            "returns": r.get("returns", {}),
            "metrics": {
                "roic": _num(r.get("roic")), "eps": _num(r.get("eps")),
                "pe": _num(r.get("pe")), "fcf_margin": _num(r.get("fcf_margin")),
                "rev_growth": _num(r.get("rev_growth")), "momentum": _num(r.get("momentum")),
                "sharpe": _num(r.get("sharpe")), "sortino": _num(r.get("sortino")),
                "volatility": _num(r.get("volatility")),
            },
            "kind": "stock",
        })

    holdings.append({
        "ticker": config.GOLD_SYMBOL, "name": "Gold (SPDR Gold Shares)",
        "sector": "Commodity", "company_key": config.GOLD_SYMBOL,
        "weight": round(float(gold_w), 4), "composite": None,
        "kind": "gold", "regime": regime,
    })
    if cash_w > 1e-4:
        holdings.append({
            "ticker": "CASH", "name": "Cash (risk-free)", "sector": "Cash",
            "company_key": "CASH", "weight": round(float(cash_w), 4),
            "composite": None, "kind": "cash",
        })
    return {"holdings": holdings, "regime": regime, "gold_weight": gold_w,
            "cash_weight": round(float(cash_w), 4),
            "vol_exposure": round(float(vol_exposure), 3),
            "n_eligible": int(len(eligible))}


def _vol_target_exposure(cks: list, eq_w, price_data: dict, settings: dict) -> float:
    """Equity exposure for the crash overlay: target_vol / the selected book's own
    trailing EWMA volatility, capped at vol_exposure_cap (1.0 = de-risk only)."""
    lam = float(settings.get("vol_ewma_lambda", 0.94))
    tgt = float(settings.get("vol_target_annual", 0.27))
    cap = float(settings.get("vol_exposure_cap", 1.0))
    series = []
    for ck in cks:
        s = price_data.get(ck) if price_data else None
        if s is None or len(s) < 60:
            return 1.0
        series.append(s.pct_change())
    R = pd.concat(series, axis=1).dropna()
    if len(R) < 60:
        return 1.0
    wv = np.asarray(eq_w, dtype=float); wv = wv / wv.sum()
    book = R.tail(config.TRADING_DAYS).to_numpy() @ wv
    var = float(np.var(book[:21])) if len(book) > 21 else float(np.var(book))
    for x in book[1:]:
        var = lam * var + (1.0 - lam) * x * x
    vol = (var * config.TRADING_DAYS) ** 0.5
    return float(min(tgt / vol, cap)) if vol > 0 else 1.0


def _num(v):
    try:
        f = float(v)
        return None if np.isnan(f) else round(f, 4)
    except (TypeError, ValueError):
        return None
