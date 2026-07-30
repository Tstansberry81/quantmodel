"""Stage 2: do the winning screens SURVIVE, or did I just pick the best of ten?

I ran ten screens and three looked good on the full sample. That is the textbook
setup for a false positive (RESEARCH_RULES #3). A real effect shows up in
independent sub-periods; a data-mined one is carried by one era.

Two tests:
  A. SUB-ERA STABILITY -- split the history into four independent stretches and
     require the screen to help in most of them, not just on aggregate.
  B. INTERACTION WITH THE REAL LEVER -- vol targeting already cuts drawdown hard.
     A screen that only helps at vol 25% and adds nothing at vol 15% is
     redundant, not additive.
"""
import numpy as np, pandas as pd
import edge_lib as E

S = E.EDGE_SPEC
SIG = tuple(sorted(S["signal"].items()))
M = E.D.daily_return_matrix()
RD = E._ma200_daily_state()
pan = E.load_edge_panel(**E.clock_spec(S["hold"]))

def num(d, c):
    return pd.to_numeric(d.get(c), errors="coerce")

SCREENS = {
    "none (shipped)":      None,
    "FCF margin > 0":      lambda d: d[num(d, "fcf_margin") > 0],
    "debt/EBITDA <= 4":    lambda d: d[(num(d, "debt_ebitda") <= 4) | num(d, "debt_ebitda").isna()],
    "no share issuance":   lambda d: d[(num(d, "share_issuance") <= 0) | num(d, "share_issuance").isna()],
    "FCF>0 + debt<=4":     lambda d: d[(num(d, "fcf_margin") > 0) & ((num(d, "debt_ebitda") <= 4) | num(d, "debt_ebitda").isna())],
}

def curve(screen, vol_target):
    holds, turn, prev = [], [], None
    for i, df in enumerate(pan.panels):
        d = E._candidates(df, S["mcap_floor"], False)
        if screen is not None:
            d = screen(d)
        cks = E._blend_select(d, pan.bdates[i], S["n"], dict(SIG), S["corr_cap"],
                              S["corr_lookback"], 0.0, S["growth_thresh"],
                              fcf_screen=False, sector_cap=S["sector_cap"])
        cur = set(cks)
        turn.append(1 - len(cur & prev) / len(cur) if prev and cur else (1.0 if cur else 0.0))
        prev = cur; holds.append(cks)
    ser = E._sleeve_daily(pan, holds, np.array(turn), M, S["hold"],
                          S["regime_expo"], S["cost_bps"], RD).dropna()
    if vol_target:
        ser = E._apply_vol_target(ser, vol_target, S["vol_lookback"], S["vol_cap"], S["cost_bps"])
    return ser

def dd(r):
    eq = np.cumprod(1 + np.nan_to_num(r)); return float((eq/np.maximum.accumulate(eq)-1).min())

ERAS = [("1999-2006", "1999-01-01", "2006-12-31"), ("2007-2012", "2007-01-01", "2012-12-31"),
        ("2013-2019", "2013-01-01", "2019-12-31"), ("2020-2026", "2020-01-01", "2026-12-31")]

print("A. SUB-ERA maxDD (shipped vol 25%) — a real screen helps in MOST eras\n")
base = curve(None, S["vol_target"])
hdr = "".join(f"{e[0]:>12}" for e in ERAS)
print(f"{'screen':<22}{hdr}{'  full':>9}")
for name, fn in SCREENS.items():
    s = curve(fn, S["vol_target"])
    cells, wins = "", 0
    for lbl, a, b in ERAS:
        seg = s[(s.index >= a) & (s.index <= b)].to_numpy()
        bseg = base[(base.index >= a) & (base.index <= b)].to_numpy()
        d_, bd_ = dd(seg), dd(bseg)
        wins += (d_ > bd_ + 1e-9)
        cells += f"{d_*100:11.1f}%"
    tag = "" if name.startswith("none") else f"   ({wins}/4 eras better)"
    print(f"{name:<22}{cells}{dd(s.to_numpy())*100:8.1f}%{tag}")

print("\nB. STILL ADDITIVE once vol targeting does its job?\n")
print(f"{'screen':<22}{'vol 25% CAGR/DD':>22}{'vol 15% CAGR/DD':>22}")
for name, fn in SCREENS.items():
    out = []
    for vt in (0.25, 0.15):
        s = curve(fn, vt).to_numpy()
        c, d_, _ = E._perf_daily(s)
        out.append(f"{c*100:6.2f}% / {d_*100:6.1f}%")
    print(f"{name:<22}{out[0]:>22}{out[1]:>22}")
