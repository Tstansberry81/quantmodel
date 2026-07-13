"""The Edge -- penalized-linear cross-sectional meta-model (elastic net), walk-forward.

The "switch the model" test (literature #5). Instead of ranking on accel alone, let a
REGULARIZED LINEAR model weight our whole feature panel. Per the skeptic literature
(deep nets need ~10x our data; ML alpha concentrates in hard-to-arbitrage names but the
LONG side survives) we use elastic net, not neural nets, and evaluate long-only net.

CRITICAL DISCIPLINE -- walk-forward / expanding window: every out-of-sample prediction
for rebalance i is made by a model trained STRICTLY on periods < i. No full-sample fit
(the classic ML-backtest cheat). Features z-scored per period cross-sectionally; target
= per-period cross-sectionally-demeaned forward return (predict RELATIVE performance).

Two framings tested:
  A. FULL cross-section: elastic net ranks all $2B-floored names -> top-n book.
  B. META-LABELING: restrict to the accel top-K pool (the only place we found alpha),
     let the model pick the final n WITHIN it. This is the structurally-different bet.

Both compared head-to-head vs the accel baseline over the SAME out-of-sample window,
net of 10bps, at hold=42 (product clock), n in {7,10}. Reuses edge_lib for returns.

No edge_lib/qmodel changes. Not committed.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import edge_lib as E
from edge_signal_ic import spearman, t_stat
from sklearn.linear_model import ElasticNetCV

FEATURES = ["accel", "ret_21", "ret_63", "ret_126", "hi_252", "rs_63",
            "vol_21", "beta", "log_mcap", "rev_growth"]
FLOOR, CAP, CLB, EXPO, BPS = 2e9, 0.50, 126, 0.25, 10.0
BURN = 30            # periods of history before the first OOS prediction
ACCEL_POOL = 40      # meta-labeling: candidate pool = accel top-K


def prep(pan):
    """Per-period standardized feature matrix + demeaned target, $2B-floored."""
    periods = []
    for i, df in enumerate(pan.panels):
        d = df[pd.to_numeric(df["pit_mcap"], errors="coerce") >= FLOOR].copy()
        d = d.dropna(subset=["fwd_ret"])
        if len(d) < 40:
            periods.append(None); continue
        d["log_mcap"] = np.log(pd.to_numeric(d["pit_mcap"], errors="coerce"))
        X = np.zeros((len(d), len(FEATURES)))
        for j, c in enumerate(FEATURES):
            v = pd.to_numeric(d[c], errors="coerce")
            mu, sd = v.mean(), v.std()
            X[:, j] = ((v - mu) / sd).fillna(0.0).to_numpy() if sd and sd > 0 else 0.0
        y = pd.to_numeric(d["fwd_ret"], errors="coerce")
        y = (y - y.mean()).to_numpy()
        periods.append({"cks": d["company_key"].to_numpy(), "X": X, "y": y,
                        "accel": pd.to_numeric(d["accel"], errors="coerce").to_numpy(),
                        "fwd": pd.to_numeric(d["fwd_ret"], errors="coerce").to_numpy()})
    return periods


def walk_forward(periods):
    """Expanding-window elastic-net OOS predictions per period (>= BURN)."""
    preds = [None] * len(periods)
    coefs = []
    valid = [i for i, p in enumerate(periods) if p is not None]
    for i in valid:
        if i < BURN:
            continue
        tr = [periods[j] for j in valid if j < i]
        if len(tr) < BURN:
            continue
        Xtr = np.vstack([p["X"] for p in tr]); ytr = np.concatenate([p["y"] for p in tr])
        m = ElasticNetCV(l1_ratio=[.2, .5, .8], alphas=20, cv=3, max_iter=2000, n_jobs=-1)
        m.fit(Xtr, ytr)
        preds[i] = m.predict(periods[i]["X"])
        coefs.append(m.coef_)
    return preds, np.array(coefs)


def book_from_scores(periods, preds, n, meta=False):
    """Top-n company_keys per OOS period by score (full X-section or accel-pool meta)."""
    holds, idx = [], []
    for i, p in enumerate(periods):
        if p is None or preds[i] is None:
            continue
        score = preds[i]
        cks, accel = p["cks"], p["accel"]
        if meta:
            pool = np.argsort(-accel)[:ACCEL_POOL]     # accel top-K candidate pool
            order = pool[np.argsort(-score[pool])]
        else:
            order = np.argsort(-score)
        holds.append(list(cks[order[:n]])); idx.append(i)
    return holds, idx


def net_from_holds(pan, holds, idx):
    """Equal-weight net returns for an OOS holds path (regime + 10bps), matched window."""
    turn, prev, sel_by_i = [], None, {}
    for h, i in zip(holds, idx):
        sel_by_i[i] = h
        cur = set(h)
        turn.append(1 - len(cur & prev) / len(cur) if prev and cur else (1.0 if cur else 0.0))
        prev = cur
    # returns per OOS period
    r = []
    for h, i in zip(holds, idx):
        sub = pan.panels[i][pan.panels[i]["company_key"].isin(h)]
        r.append(float(sub["fwd_ret"].mean()) if len(sub) else 0.0)
    r = np.array(r)
    ma = pan.ma200_on[idx]
    r = np.where(ma, r, EXPO * r + (1 - EXPO) * pan.rf_per)
    r = r - (BPS / 1e4) * np.array(turn)
    return r, float(np.mean(turn))


def perf_row(label, r, ppy, spxf_slice):
    c, dd, sh = E.perf(r, ppy); cs, _, _ = E.perf(spxf_slice, ppy)
    print(f"   {label:<34} CAGR {c*100:6.1f}%  Sharpe {sh:5.2f}  maxDD {dd*100:5.0f}%  "
          f"excess {(c-cs)*100:+6.1f}%  ")


def main():
    hold = 42
    pan = E.load_edge_panel(hold=hold); ppy = pan.ppy
    print(f"hold={hold} — {pan.T} rebalances; walk-forward elastic net (burn-in {BURN})")
    periods = prep(pan)
    print("Fitting expanding-window models (one per OOS rebalance) ...")
    preds, coefs = walk_forward(periods)

    # ---- OOS IC: elastic net vs accel over the SAME OOS periods ----
    ic_ml, ic_ac = [], []
    for i, p in enumerate(periods):
        if p is None or preds[i] is None:
            continue
        ic_ml.append(spearman(preds[i], p["fwd"]))
        ic_ac.append(spearman(p["accel"], p["fwd"]))
    ic_ml, ic_ac = np.array(ic_ml), np.array(ic_ac)
    oos_idx = [i for i, p in enumerate(periods) if p is not None and preds[i] is not None]
    print(f"\nOUT-OF-SAMPLE window: {len(oos_idx)} rebalances "
          f"{pan.bdates[oos_idx[0]].date()} -> {pan.bdates[oos_idx[-1]].date()}")
    print(f"   elastic-net OOS IC: {np.nanmean(ic_ml)*100:+5.2f}%  t={t_stat(ic_ml):+5.2f}")
    print(f"   accel       OOS IC: {np.nanmean(ic_ac)*100:+5.2f}%  t={t_stat(ic_ac):+5.2f}")

    # ---- average standardized coefficients (what did the model learn?) ----
    print("\nAvg elastic-net coefficients (standardized features):")
    order = np.argsort(-np.abs(coefs.mean(0)))
    for j in order:
        print(f"   {FEATURES[j]:<10} {coefs.mean(0)[j]:+.4f}   (nonzero {100*np.mean(coefs[:,j]!=0):.0f}% of refits)")

    # ---- portfolio: ML full X-section, ML meta-label, vs accel baseline ----
    spx = pan.spxf[oos_idx]
    print(f"\nNET portfolios over the OOS window (regime+10bps), vs accel baseline:")
    for n in (10, 7):
        print(f"  n={n}:")
        # accel baseline (same selection edge_lib uses, restricted to OOS window)
        ac_holds, ac_idx = [], []
        for i in oos_idx:
            d = pan.panels[i][pan.panels[i]["pit_mcap"] >= FLOOR]
            ac_holds.append(E.corr_cap_select(d, asof=pan.bdates[i], n=n, weights={"accel": 1.0},
                                              cap=CAP, lookback=CLB))
            ac_idx.append(i)
        r_ac, to_ac = net_from_holds(pan, ac_holds, ac_idx)
        h_ml, i_ml = book_from_scores(periods, preds, n, meta=False)
        r_ml, to_ml = net_from_holds(pan, h_ml, i_ml)
        h_me, i_me = book_from_scores(periods, preds, n, meta=True)
        r_me, to_me = net_from_holds(pan, h_me, i_me)
        perf_row(f"accel baseline (to {to_ac:.2f})", r_ac, ppy, spx)
        perf_row(f"elastic-net full X-sec (to {to_ml:.2f})", r_ml, ppy, spx)
        perf_row(f"elastic-net META on accel-top{ACCEL_POOL} (to {to_me:.2f})", r_me, ppy, spx)

    print("\nDONE.  OOS = strictly walk-forward. Beat the accel baseline net, or it's noise.")


if __name__ == "__main__":
    main()
