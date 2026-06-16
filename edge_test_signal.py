"""The Edge -- short-horizon signal search.

Find the market-data-only signal (+ holding period + basket size) that
maximizes RISK-ADJUSTED performance (Sharpe) with controlled drawdown across
1Y / 2Y / 5Y / MAX windows. Does NOT modify edge_lib.py or qmodel/.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import edge_lib as E

WINDOWS = (("1Y", 12), ("2Y", 24), ("5Y", 60), ("MAX", None))


def add_risk_adj(pan):
    """Add a derived risk-adjusted-momentum column ramom = ret_63 / vol_21."""
    for df in pan.panels:
        if "ramom" not in df.columns:
            v = pd.to_numeric(df["vol_21"], errors="coerce")
            df["ramom"] = pd.to_numeric(df["ret_63"], errors="coerce") / v.replace(0, np.nan)
            df["ramom21"] = pd.to_numeric(df["ret_21"], errors="coerce") / v.replace(0, np.nan)


def window_stats(pan, rets):
    """Return dict window -> (cagr, dd, sharpe) and S&P cagr."""
    rets = np.asarray(rets, float)
    out = {}
    for wname, k in WINDOWS:
        kk = len(rets) if k is None else min(k, len(rets))
        c, dd, sh = E.perf(rets[-kk:])
        cs, _, _ = E.perf(pan.spxf[-kk:])
        out[wname] = (c, dd, sh, cs)
    return out


def eval_spec(pan, weights, n=10, regime_cash=False, holdperiod=None):
    holds = [E.select_topN(df, n=n, weights=weights) for df in pan.panels]
    rets = E.period_returns(pan, holds, regime_cash=regime_cash)
    return window_stats(pan, rets)


def fmt_row(label, st):
    parts = [f"{label:<26}"]
    for wname, _ in WINDOWS:
        c, dd, sh, cs = st[wname]
        parts.append(f"{wname}: {c*100:6.1f}%/{sh:4.2f}/{dd*100:4.0f}%")
    return "  ".join(parts)


def main():
    print("Loading hold=21 panel ...")
    pan = E.load_edge_panel(hold=21)
    add_risk_adj(pan)
    print(f"Panel: {pan.T} rebalances, {pan.bdates[0].date()} -> {pan.bdates[-1].date()}\n")

    # ---- Task 1: single signals and blends ----
    specs = {
        "ret_21 (1mo)":           {"ret_21": 1.0},
        "ret_63 (3mo) BASELINE":  {"ret_63": 1.0},
        "ret_126 (6mo)":          {"ret_126": 1.0},
        "accel":                  {"accel": 1.0},
        "hi_252":                 {"hi_252": 1.0},
        "rs_63":                  {"rs_63": 1.0},
        "ramom (ret63/vol)":      {"ramom": 1.0},
        "ramom21 (ret21/vol)":    {"ramom21": 1.0},
        "ret63+accel":            {"ret_63": 1.0, "accel": 1.0},
        "ret63+hi252":            {"ret_63": 1.0, "hi_252": 1.0},
        "ret63+rs63":             {"ret_63": 1.0, "rs_63": 1.0},
        "ret63+ramom":            {"ret_63": 1.0, "ramom": 1.0},
        "ret63+accel+hi252":      {"ret_63": 1.0, "accel": 1.0, "hi_252": 1.0},
        "ramom+hi252":            {"ramom": 1.0, "hi_252": 1.0},
        "ramom+accel":            {"ramom": 1.0, "accel": 1.0},
        "ret126+accel":           {"ret_126": 1.0, "accel": 1.0},
    }
    print("=" * 130)
    print("TASK 1 -- single signals + blends (hold=21, n=10)   [CAGR / Sharpe / maxDD per window]")
    print("=" * 130)
    results = {}
    for label, w in specs.items():
        st = eval_spec(pan, w, n=10)
        results[label] = st
        print(fmt_row(label, st))

    # S&P reference
    cs = {wn: window_stats(pan, pan.spxf)[wn] for wn, _ in WINDOWS}
    print("-" * 130)
    sp = window_stats(pan, pan.spxf)
    print(fmt_row("S&P 500 (ref)", {wn: (sp[wn][0], sp[wn][1], sp[wn][2], 0) for wn, _ in WINDOWS}))

    # Rank by average Sharpe across 1Y/2Y/5Y
    def avg_sharpe(st):
        return np.mean([st[w][2] for w in ("1Y", "2Y", "5Y")])
    ranked = sorted(results.items(), key=lambda kv: avg_sharpe(kv[1]), reverse=True)
    print("\n  Ranked by avg(1Y,2Y,5Y Sharpe):")
    for label, st in ranked:
        print(f"    {label:<26} avgSharpe={avg_sharpe(st):4.2f}  "
              f"1Y={st['1Y'][2]:.2f} 2Y={st['2Y'][2]:.2f} 5Y={st['5Y'][2]:.2f} MAX={st['MAX'][2]:.2f}  "
              f"maxDD(MAX)={st['MAX'][1]*100:.0f}%")

    top2 = [lbl for lbl, _ in ranked[:2]]
    top_specs = {lbl: specs[lbl] for lbl in top2}

    # ---- Task 2: holding period for best 1-2 signals ----
    print("\n" + "=" * 130)
    print("TASK 2 -- holding period (hold=21/42/63) for top signals (n=10)")
    print("=" * 130)
    hold_results = {}
    for hold in (21, 42, 63):
        p = E.load_edge_panel(hold=hold)
        add_risk_adj(p)
        for lbl, w in top_specs.items():
            holds = [E.select_topN(df, n=10, weights=w) for df in p.panels]
            rets = E.period_returns(p, holds)
            # reproject perf with this panel's own ppy via E.perf default constants;
            # E.perf uses module PPY=12; use per-panel ppy for correct annualization
            ppy = 252.0 / hold
            st = {}
            rets = np.asarray(rets, float)
            for wname, k in WINDOWS:
                # window in periods: scale k (months) by clock; k is in 21d periods originally.
                # convert: 1Y = 252/hold periods, etc.
                if k is None:
                    kk = len(rets)
                else:
                    kk = min(int(round(k * 21 / hold)), len(rets))
                    kk = max(kk, 2)
                r = rets[-kk:]
                eq = np.cumprod(1 + np.nan_to_num(r))
                cagr = eq[-1] ** (ppy / len(r)) - 1 if len(r) and eq[-1] > 0 else -1
                dd = float((eq / np.maximum.accumulate(eq) - 1).min())
                sh = float(np.sqrt(ppy) * np.nanmean(r) / np.nanstd(r)) if np.nanstd(r) > 0 else 0.0
                st[wname] = (cagr, dd, sh, 0)
            hold_results[(lbl, hold)] = st
            print(fmt_row(f"{lbl} h={hold}", st))

    # ---- Task 3: basket size for the winner ----
    print("\n" + "=" * 130)
    print("TASK 3 -- basket size n in {10,15,20,25} for winner (hold=21)")
    print("=" * 130)
    winner = top2[0]
    wspec = specs[winner]
    n_results = {}
    for n in (10, 15, 20, 25):
        st = eval_spec(pan, wspec, n=n)
        n_results[n] = st
        print(fmt_row(f"{winner} n={n}", st))

    # ---- Task 3b: basket size at the best clocks (hold=42, 63) for accel ----
    print("\n" + "=" * 130)
    print("TASK 3b -- accel basket size at hold=42 and hold=63   (n periods shown for sanity)")
    print("=" * 130)
    for hold in (42, 63):
        p = E.load_edge_panel(hold=hold)
        add_risk_adj(p)
        ppy = 252.0 / hold
        for n in (10, 15, 20, 25):
            holds = [E.select_topN(df, n=n, weights={"accel": 1.0}) for df in p.panels]
            rets = np.asarray(E.period_returns(p, holds), float)
            st = {}
            for wname, k in WINDOWS:
                kk = len(rets) if k is None else max(2, min(int(round(k * 21 / hold)), len(rets)))
                r = rets[-kk:]
                eq = np.cumprod(1 + np.nan_to_num(r))
                cagr = eq[-1] ** (ppy / len(r)) - 1 if len(r) and eq[-1] > 0 else -1
                dd = float((eq / np.maximum.accumulate(eq) - 1).min())
                sh = float(np.sqrt(ppy) * np.nanmean(r) / np.nanstd(r)) if np.nanstd(r) > 0 else 0.0
                st[wname] = (cagr, dd, sh, 0)
            print(fmt_row(f"accel h={hold} n={n}", st) + f"   [T={p.T}]")

    print("\nDONE.")
    return results, hold_results, n_results


if __name__ == "__main__":
    main()
