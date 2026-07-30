"""Can a FUNDAMENTAL screen cut the drawdown that vol targeting can't reach?

THE PRIOR, stated up front so the result can contradict it. Vol targeting acts on
EXPOSURE -- it scales the whole book, so it attacks drawdown mechanically
whatever the cause. A fundamental screen acts on SELECTION -- it can only help if
the names it removes are the ones that crash. This model's worst drawdown
(2021-08 -> 2023-10) was a momentum UNWIND: the whole factor went down together,
with the S&P above its 200dMA most of the way. Screening within momentum names
does not obviously help with that. But it is testable, so test it.

A TRAP THIS CONTROLS FOR: `df[df.pe <= 30]` silently drops every company with a
NaN or negative P/E -- i.e. everything unprofitable. That is not a valuation cap,
it is a profitability screen wearing one, and on a momentum book (biotech,
pre-earnings growth) it removes a large slice of the universe. Each screen below
reports how much of the eligible field it actually removes, and the P/E caps are
run BOTH ways: keeping unprofitable names and dropping them.
"""
import numpy as np, pandas as pd
import edge_lib as E

S = E.EDGE_SPEC
SIG = tuple(sorted(S["signal"].items()))
M = E.D.daily_return_matrix()
RD = E._ma200_daily_state()
pan = E.load_edge_panel(**E.clock_spec(S["hold"]))


def q(df, col, frac, keep="low", drop_na=False):
    """Keep the `frac` fraction of the field on the chosen side of `col`."""
    if col not in df.columns:
        return df
    v = pd.to_numeric(df[col], errors="coerce")
    ok = v.notna()
    if not ok.any():
        return df
    cut = v[ok].quantile(frac if keep == "low" else 1 - frac)
    sel = (v <= cut) if keep == "low" else (v >= cut)
    return df[sel | (~ok if not drop_na else False)]


SCREENS = {
    "SHIPPED (no fundamental screen)": None,
    "P/E <= 40 (keep unprofitable)":   lambda d: d[(pd.to_numeric(d.get("pe"), errors="coerce") <= 40) | pd.to_numeric(d.get("pe"), errors="coerce").isna()],
    "P/E <= 40 (DROP unprofitable)":   lambda d: d[pd.to_numeric(d.get("pe"), errors="coerce").between(0, 40)],
    "cheapest 70% by P/E":             lambda d: q(d, "pe", 0.70, "low"),
    "FCF margin > 0":                  lambda d: d[pd.to_numeric(d.get("fcf_margin"), errors="coerce") > 0],
    "FCF margin top 70%":              lambda d: q(d, "fcf_margin", 0.70, "high"),
    "GP/assets top 60% (Novy-Marx)":   lambda d: q(d, "gp_assets", 0.60, "high"),
    "debt/EBITDA <= 4":                lambda d: d[(pd.to_numeric(d.get("debt_ebitda"), errors="coerce") <= 4) | pd.to_numeric(d.get("debt_ebitda"), errors="coerce").isna()],
    "no net share issuance":           lambda d: d[(pd.to_numeric(d.get("share_issuance"), errors="coerce") <= 0) | pd.to_numeric(d.get("share_issuance"), errors="coerce").isna()],
    "low accruals (best 70%)":         lambda d: q(d, "accruals", 0.70, "low"),
    "GP/assets 60% + P/E<=40":         lambda d: q(d[(pd.to_numeric(d.get("pe"), errors="coerce") <= 40) | pd.to_numeric(d.get("pe"), errors="coerce").isna()], "gp_assets", 0.60, "high"),
}


def evaluate(screen):
    holds, turn, prev, kept, tot = [], [], None, 0, 0
    for i, df in enumerate(pan.panels):
        d = E._candidates(df, S["mcap_floor"], False)
        tot += len(d)
        if screen is not None:
            d = screen(d)
        kept += len(d)
        cks = E._blend_select(d, pan.bdates[i], S["n"], dict(SIG), S["corr_cap"],
                              S["corr_lookback"], 0.0, S["growth_thresh"],
                              fcf_screen=False, sector_cap=S["sector_cap"])
        cur = set(cks)
        turn.append(1 - len(cur & prev) / len(cur) if prev and cur else (1.0 if cur else 0.0))
        prev = cur; holds.append(cks)
    ser = E._sleeve_daily(pan, holds, np.array(turn), M, S["hold"],
                          S["regime_expo"], S["cost_bps"], RD).dropna()
    model = E._apply_vol_target(ser, S["vol_target"], S["vol_lookback"],
                                S["vol_cap"], S["cost_bps"])
    r = model.to_numpy()
    c, dd, sh = E._perf_daily(r)
    eq = np.cumprod(1 + np.nan_to_num(r)); d_ = eq / np.maximum.accumulate(eq) - 1
    return c, sh, E._sortino_daily(r), dd, float((d_ < -0.20).mean()), kept / max(tot, 1)


print(f"{'screen':<34}{'CAGR':>8}{'Sharpe':>8}{'Sortino':>9}{'maxDD':>9}"
      f"{'MAR':>6}{'%<-20%':>8}{'univ kept':>10}")
for name, fn in SCREENS.items():
    try:
        c, sh, so, dd, pain, frac = evaluate(fn)
        print(f"{name:<34}{c*100:7.2f}%{sh:8.3f}{so:9.3f}{dd*100:8.2f}%"
              f"{c/abs(dd):6.2f}{pain*100:7.1f}%{frac*100:9.0f}%")
    except Exception as e:
        print(f"{name:<34}  FAILED {type(e).__name__}: {e}")
