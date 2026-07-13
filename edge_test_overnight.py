"""The Edge -- overnight vs intraday return decomposition (Lou-Polk-Skouras 2019, JFE).

Finding to test: momentum accrues ENTIRELY overnight; the intraday leg is flat-to-
negative. Documented to STRENGTHEN in the largest stocks (our universe) -- the only
literature idea in the sweep with a large-cap tailwind rather than headwind.

Uses data/cache/ohlc.pkl (adjusted open+close, from build_ohlc.py):
    overnight_t = open_t / close_{t-1} - 1        (close-to-open)
    intraday_t  = close_t / open_t    - 1         (open-to-close)
    close-to-close = (1+overnight)(1+intraday) - 1

Signals per name at each rebalance (cumulative over trailing windows, no lookahead):
    on_63 / id_63   cumulative overnight / intraday return, last 63d (3-mo momentum split)
    on_21 / id_21   same over last 21d (1-mo)
    on_accel        overnight acceleration = on(last 63d) - on(prior 63d)
                    [the paper's core claim => this should carry accel's predictive power]

Tests (reuse edge_signal_ic scorecard): IC vs fwd_ret, decile monotonicity, and
INCREMENTAL IC after removing accel -- does the overnight leg ADD to the product?
Held to t>3 (Harvey-Liu-Zhu). hold=42 (product clock) + 21. Then, if any overnight
signal clears IC, a net portfolio test vs the accel baseline.

No edge_lib/qmodel changes. Not committed.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import os
import numpy as np, pandas as pd
import edge_lib as E
from edge_signal_ic import spearman, t_stat, resid_of, deciles

OHLC = os.path.join("data", "cache", "ohlc.pkl")


def load_on_id():
    """Per-name overnight & intraday daily-return series from the OHLC cache."""
    blob = pd.read_pickle(OHLC)
    on, idn = {}, {}
    for ck, df in blob.items():
        df = df.sort_index()
        o = df["open"].to_numpy(float); c = df["close"].to_numpy(float)
        idx = df.index
        overnight = np.full(len(c), np.nan); intraday = np.full(len(c), np.nan)
        overnight[1:] = o[1:] / c[:-1] - 1        # close_{t-1} -> open_t
        intraday = c / o - 1                        # open_t -> close_t
        on[ck] = pd.Series(overnight, index=idx)
        idn[ck] = pd.Series(intraday, index=idx)
    return on, idn


def cum(series, pos, k):
    """Cumulative compounded return of a daily-return series over [pos-k+1, pos]."""
    w = series.to_numpy()[pos - k + 1: pos + 1]
    w = w[np.isfinite(w)]
    return float(np.prod(1 + w) - 1) if len(w) else np.nan


def sig_at(on, idn, d):
    """Overnight/intraday cumulative signals for one name at rebalance date d."""
    if on is None:
        return {}
    pos = int(on.index.searchsorted(d, side="right")) - 1
    if pos < 130:
        return {}
    o63 = cum(on, pos, 63); o63p = cum(on, pos - 63, 63)
    return {
        "on_63": o63, "id_63": cum(idn, pos, 63),
        "on_21": cum(on, pos, 21), "id_21": cum(idn, pos, 21),
        "on_accel": (o63 - o63p) if np.isfinite(o63) and np.isfinite(o63p) else np.nan,
    }


def build(pan, on, idn):
    panels = []
    for i, df in enumerate(pan.panels):
        d = pan.bdates[i]
        recs = {ck: sig_at(on.get(ck), idn.get(ck), d) for ck in df["company_key"]}
        add = pd.DataFrame.from_dict(recs, orient="index")
        panels.append(df.set_index("company_key").join(add).reset_index())
        if (i + 1) % 60 == 0:
            print(f"    overnight signals {i+1}/{pan.T}")
    return panels


def report(panels, col):
    ic = np.array([spearman(df[col], df["fwd_ret"]) for df in panels], float)
    inc = np.array([spearman(resid_of(df[col].to_numpy(), df["accel"].to_numpy()),
                             df["fwd_ret"].to_numpy()) for df in panels], float)
    _dm, mono, spread = deciles(panels, col)
    print(f"   {col:<10} IC {np.nanmean(ic)*100:+6.2f}%  t={t_stat(ic):+5.2f}   "
          f"mono {mono:+.2f}   D10-D1 {spread*100:+5.2f}%   "
          f"incrIC(vs accel) {np.nanmean(inc)*100:+5.2f}% t={t_stat(inc):+5.2f}")
    return np.nanmean(ic), t_stat(ic)


def main():
    if not os.path.exists(OHLC):
        print(f"{OHLC} not present -- run build_ohlc.py first."); return
    print("Loading overnight/intraday returns ...")
    on, idn = load_on_id()
    print(f"  {len(on)} names with OHLC")
    for hold in (42, 21):
        pan = E.load_edge_panel(hold=hold)
        print(f"\n{'='*82}\nhold={hold} — {pan.T} rebalances "
              f"{pan.bdates[0].date()} -> {pan.bdates[-1].date()}\n{'='*82}")
        panels = build(pan, on, idn)
        print("\nOvernight/intraday signals [Lou-Polk-Skouras: overnight carries momentum]")
        for col in ("on_63", "id_63", "on_21", "id_21", "on_accel"):
            report(panels, col)
        print("   (reference) accel baseline:")
        report(panels, "accel")
    print("\nDONE.  overnight IC positive & t>3 & incremental => carry to a net portfolio test.")


if __name__ == "__main__":
    main()
