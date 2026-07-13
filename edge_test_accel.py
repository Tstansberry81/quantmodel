"""The Edge -- improved acceleration geometry + trend quality (GPT audit 5.1 / 5.2).

Our accel is a crude two-block simple-return difference. GPT's suggestions:
  * log form (additive scale, no arbitrary block-boundary artifact)
  * fit the log-price path to a quadratic  log P(t-j) = a + b1*j + b2*j^2 :
        trend strength = b1 (slope),  ACCELERATION = b2 (curvature),  quality = R^2
  * trend efficiency = |P_t - P_{t-L}| / sum|daily moves|  (Kaufman ER)

All price-only, computed on close t (signal date). Scored the honest way: rank-IC
over the whole cross-section, decile monotonicity, and -- the decisive test --
INCREMENTAL IC after removing the current accel (does the new geometry ADD
anything?). Held to t>3 (Harvey-Liu-Zhu). hold=42 (product clock) + 21.

Reuses edge_signal_ic helpers. Does NOT modify edge_lib. Not committed.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import edge_lib as E
from edge_signal_ic import _raw_prep, spearman, t_stat, resid_of, deciles


def _fit_quad(logp):
    """Quadratic fit of a log-price window (oldest->newest). Returns (slope b1,
    curvature b2, R^2). x scaled to [0,1] so coefficients are comparable."""
    n = len(logp)
    if n < 20 or not np.all(np.isfinite(logp)):
        return np.nan, np.nan, np.nan
    x = np.linspace(0.0, 1.0, n)
    c2, c1, c0 = np.polyfit(x, logp, 2)
    fit = c2 * x * x + c1 * x + c0
    ss_res = np.sum((logp - fit) ** 2); ss_tot = np.sum((logp - logp.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(c1), float(c2), float(r2)


def sig_at(P, pos):
    arr = P["arr"]
    out = {}
    # log form of the current 3m-minus-prior-3m acceleration
    a, b, c = arr[pos], arr[pos - 63], arr[pos - 126]
    if a > 0 and b > 0 and c > 0:
        out["accel_log"] = (np.log(a / b)) - (np.log(b / c))
    else:
        out["accel_log"] = np.nan
    # quadratic geometry over trailing 126d and 252d
    for L in (126, 252):
        seg = arr[pos - L + 1: pos + 1]
        if len(seg) == L and np.all(seg > 0):
            b1, b2, r2 = _fit_quad(np.log(seg))
        else:
            b1 = b2 = r2 = np.nan
        out[f"trend{L}"] = b1        # trend strength
        out[f"curv{L}"] = b2         # curvature = acceleration
        out[f"r2_{L}"] = r2          # trend quality
    # Kaufman trend efficiency over 126d: net move / sum of abs daily moves
    seg = arr[pos - 126: pos + 1]
    if len(seg) == 127 and np.all(seg > 0):
        net = abs(seg[-1] - seg[0]); path = np.sum(np.abs(np.diff(seg)))
        out["trend_eff"] = float(net / path) if path > 0 else np.nan
    else:
        out["trend_eff"] = np.nan
    # curvature gated to positive trend (interaction: accel matters when trending up)
    out["curv126_up"] = out["curv126"] if (out["trend126"] == out["trend126"]
                                           and out["trend126"] > 0) else 0.0
    return out


def build(pan, prep):
    panels = []
    for i, df in enumerate(pan.panels):
        dt = np.datetime64(pd.Timestamp(pan.bdates[i]), "ns")
        recs = {}
        for ck in df["company_key"]:
            Pp = prep.get(ck)
            if Pp is None:
                continue
            pos = int(np.searchsorted(Pp["pidx"], dt, side="right")) - 1
            if pos < 252 or pos >= len(Pp["arr"]):
                continue
            recs[ck] = sig_at(Pp, pos)
        add = pd.DataFrame.from_dict(recs, orient="index")
        panels.append(df.set_index("company_key").join(add).reset_index())
        if (i + 1) % 60 == 0:
            print(f"    accel geometry {i+1}/{pan.T}")
    return panels


def report(panels, col):
    ic = np.array([spearman(df[col], df["fwd_ret"]) for df in panels], float)
    inc = np.array([spearman(resid_of(df[col].to_numpy(), df["accel"].to_numpy()),
                             df["fwd_ret"].to_numpy()) for df in panels], float)
    _dm, mono, spread = deciles(panels, col)
    print(f"   {col:<11} IC {np.nanmean(ic)*100:+6.2f}%  t={t_stat(ic):+5.2f}   "
          f"mono {mono:+.2f}   D10-D1 {spread*100:+5.2f}%   "
          f"incrIC(vs accel) {np.nanmean(inc)*100:+5.2f}% t={t_stat(inc):+5.2f}")


def main():
    prep = _raw_prep()
    for hold in (42, 21):
        pan = E.load_edge_panel(hold=hold)
        print(f"\n{'='*84}\nhold={hold} — {pan.T} rebalances "
              f"{pan.bdates[0].date()} -> {pan.bdates[-1].date()}\n{'='*84}")
        panels = build(pan, prep)
        print("\nAccel-geometry candidates [must be POSITIVE IC and INCREMENTAL to accel]")
        for col in ("accel_log", "curv126", "curv252", "trend126", "trend252",
                    "r2_126", "trend_eff", "curv126_up"):
            report(panels, col)
        print("   (reference) current accel:")
        report(panels, "accel")
    print("\nDONE.  A candidate graduates only if incrIC t>~3 AND it beats accel in a net portfolio.")


if __name__ == "__main__":
    main()
