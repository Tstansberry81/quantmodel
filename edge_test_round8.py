"""The Edge -- round 8: the four remaining dials, drawdown-first.

All at the validated candidate (n=7, hold=42, mix=75, cap=0.50, expo=25, 10bps):

  1. IN-BASKET WEIGHTING  equal vs rank-linear vs top-heavy (1/rank) vs
     inverse-vol (the DD-focused variant). Same picks, different sizing.
  2. LIQUIDITY FLOOR      $2B vs $5B vs $10B.
  3. REGIME DEFINITION    200dMA (baseline) vs 100dMA vs 2-of-3 vote
     (200dMA, 100dMA, 63d market return > 0). Same 25% exposure when off.
  4. EXIT ASYMMETRY       full rebuild (baseline) vs hold-band (keep incumbents
     still in the top-M by score, M=14/21) vs keep-winners (keep incumbents
     whose just-completed period return was positive -- known at decision time,
     no lookahead).

Output: per-window CAGR/Sharpe/maxDD per variant + a final table of ALL variants
ranked by MAX-window maxDD (shallowest first; user priority), with 2008 shown.
No edge_lib/qmodel changes. Not committed.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import edge_lib as E
import edge_data as engine   # self-contained Edge data layer (was qmodel.engine)

W = {"accel": 1.0}
SPEC = dict(n=7, mix=0.75, thresh=0.15, cap=0.50, lb=126, floor=2e9, expo=0.25, bps=10.0)

pan = E.load_edge_panel(hold=42)
PPY = pan.ppy
YRS = np.array([d.year for d in pan.bdates])
RESULTS = []   # (label, net) for the final DD-ranked table


def eligible(df, floor):
    return df[pd.to_numeric(df["pit_mcap"], errors="coerce") >= floor]


def select_all(floor=SPEC["floor"]):
    return [E._blend_select(eligible(df, floor), pan.bdates[i], SPEC["n"], W,
                            SPEC["cap"], SPEC["lb"], SPEC["mix"], SPEC["thresh"])
            for i, df in enumerate(pan.panels)]


def net_from(holds, weight_fn=None, regime_mask=None, expo=SPEC["expo"]):
    """Per-period net returns for a holdings path; optional in-basket weighting
    and alternative regime mask. Turnover kept name-count-based for cost parity."""
    grossL, turn, prev = [], [], None
    for i, df in enumerate(pan.panels):
        cks = holds[i]
        sub = df[df["company_key"].isin(cks)].dropna(subset=["fwd_ret"])
        if len(sub) == 0:
            r = 0.0
        elif weight_fn is None:
            r = float(sub["fwd_ret"].mean())
        else:
            wts = weight_fn(sub)
            wts = wts / wts.sum() if wts.sum() > 0 else pd.Series(1 / len(sub), index=sub.index)
            r = float((sub["fwd_ret"] * wts).sum())
        grossL.append(r)
        cur = set(cks)
        turn.append(1 - len(cur & prev) / len(cur) if prev and cur else (1.0 if cur else 0.0))
        prev = cur
    gross = np.array(grossL)
    mask = pan.ma200_on if regime_mask is None else regime_mask
    gross = np.where(mask, gross, expo * gross + (1 - expo) * pan.rf_per)
    return gross - (SPEC["bps"] / 1e4) * np.array(turn), float(np.mean(turn))


def report(label, net, avg_to):
    parts = [f"{label:<26}"]
    T = len(net)
    for wname, yrs_ in (("1Y", 1), ("2Y", 2), ("5Y", 5), ("MAX", None)):
        k = T if yrs_ is None else min(int(round(yrs_ * PPY)), T)
        c, dd, sh = E.perf(net[-k:], PPY)
        parts.append(f"{wname}: {c*100:6.1f}%/{sh:4.2f}/{dd*100:4.0f}%")
    print("  ".join(parts) + f"  to={avg_to:.2f}")
    RESULTS.append((label, net))


# ---------------- 1. in-basket weighting (same picks, different sizing) --------
print("=" * 112)
print("1. IN-BASKET WEIGHTING at the candidate spec (same 7 picks, different sizes)")
print("=" * 112)
base_holds = select_all()

def w_rank_linear(sub):
    r = pd.to_numeric(sub["accel"], errors="coerce").rank(ascending=False)
    return (len(sub) + 1 - r)

def w_top_heavy(sub):
    r = pd.to_numeric(sub["accel"], errors="coerce").rank(ascending=False)
    return 1.0 / r

def w_inv_vol(sub):
    v = pd.to_numeric(sub["vol_21"], errors="coerce")
    v = v.fillna(v.median() if v.notna().any() else 1.0).clip(lower=1e-6)
    return 1.0 / v

net, to = net_from(base_holds);                     report("equal (BASELINE)", net, to)
net, to = net_from(base_holds, w_rank_linear);      report("rank-linear", net, to)
net, to = net_from(base_holds, w_top_heavy);        report("top-heavy (1/rank)", net, to)
net, to = net_from(base_holds, w_inv_vol);          report("inverse-vol", net, to)

# ---------------- 2. liquidity floor -------------------------------------------
print("\n" + "=" * 112)
print("2. LIQUIDITY FLOOR sweep ($2B baseline / $5B / $10B)")
print("=" * 112)
for floor in (2e9, 5e9, 10e9):
    holds = base_holds if floor == 2e9 else select_all(floor)
    net, to = net_from(holds)
    report(f"floor=${floor/1e9:.0f}B" + (" (BASELINE)" if floor == 2e9 else ""), net, to)

# ---------------- 3. regime definition -----------------------------------------
print("\n" + "=" * 112)
print("3. REGIME DEFINITION (same 25% exposure when risk-off)")
print("=" * 112)
bm = engine._benchmarks_cached(); spx = bm["SP500"].dropna()
ma200 = spx.rolling(200).mean(); ma100 = spx.rolling(100).mean()
idx = spx.index.values.astype("datetime64[ns]")
pos = np.array([int(np.searchsorted(idx, np.datetime64(pd.Timestamp(d), "ns"), side="right")) - 1
                for d in pan.bdates])
m200 = np.array([bool(spx.iloc[p] > ma200.iloc[p]) if not np.isnan(ma200.iloc[p]) else True for p in pos])
m100 = np.array([bool(spx.iloc[p] > ma100.iloc[p]) if not np.isnan(ma100.iloc[p]) else True for p in pos])
mom63 = np.array([bool(spx.iloc[p] / spx.iloc[p - 63] - 1 > 0) if p >= 63 else True for p in pos])
vote = (m200.astype(int) + m100.astype(int) + mom63.astype(int)) >= 2
mismatch = int((m200 != pan.ma200_on).sum())
print(f"[sanity: recomputed 200dMA mask matches panel on {len(m200)-mismatch}/{len(m200)} dates]")
net, to = net_from(base_holds, regime_mask=m200);   report("200dMA (BASELINE)", net, to)
net, to = net_from(base_holds, regime_mask=m100);   report("100dMA", net, to)
net, to = net_from(base_holds, regime_mask=vote);   report("2-of-3 vote", net, to)

# ---------------- 4. exit asymmetry --------------------------------------------
print("\n" + "=" * 112)
print("4. EXIT ASYMMETRY (reduce churn; keeps relax the growth mix by design)")
print("=" * 112)

def holds_band(M):
    out, prev = [], []
    for i, df in enumerate(pan.panels):
        d = eligible(df, SPEC["floor"]).dropna(subset=["fwd_ret"])
        sc = E.score(d, W)
        order = list(d.assign(_c=sc).sort_values("_c", ascending=False)["company_key"])
        topM = set(order[:M])
        keep = [ck for ck in prev if ck in topM]
        sel = keep[:SPEC["n"]]
        for ck in E._blend_select(d, pan.bdates[i], SPEC["n"], W, SPEC["cap"],
                                  SPEC["lb"], SPEC["mix"], SPEC["thresh"]):
            if len(sel) >= SPEC["n"]:
                break
            if ck not in sel:
                sel.append(ck)
        out.append(sel); prev = sel
    return out

def holds_keepwin():
    out, prev = [], []
    for i, df in enumerate(pan.panels):
        d = eligible(df, SPEC["floor"]).dropna(subset=["fwd_ret"])
        elig = set(d["company_key"])
        keep = []
        if i > 0 and prev:
            prevdf = pan.panels[i - 1].set_index("company_key")
            for ck in prev:
                if ck in elig and ck in prevdf.index:
                    r = prevdf.loc[ck, "fwd_ret"]   # just-completed period, known now
                    if pd.notna(r) and float(r) > 0:
                        keep.append(ck)
        sel = keep[:SPEC["n"]]
        for ck in E._blend_select(d, pan.bdates[i], SPEC["n"], W, SPEC["cap"],
                                  SPEC["lb"], SPEC["mix"], SPEC["thresh"]):
            if len(sel) >= SPEC["n"]:
                break
            if ck not in sel:
                sel.append(ck)
        out.append(sel); prev = sel
    return out

net, to = net_from(base_holds);                     report("full rebuild (BASELINE)", net, to)
for M in (14, 21):
    net, to = net_from(holds_band(M));              report(f"hold-band top-{M}", net, to)
net, to = net_from(holds_keepwin());                report("keep-winners", net, to)

# ---------------- final: DD-ranked table ----------------------------------------
print("\n" + "=" * 112)
print("ALL VARIANTS ranked by MAX-window maxDD (shallowest first -- user priority), 2008 shown")
print("=" * 112)
rows = []
for label, net in RESULTS:
    c, dd, sh = E.perf(net, PPY)
    m08 = YRS == 2008
    y08 = float(np.cumprod(1 + np.nan_to_num(net[m08]))[-1] - 1) if m08.any() else np.nan
    rows.append((dd, label, c, sh, y08))
rows.sort(key=lambda r: -r[0])   # dd is negative; shallowest (closest to 0) first
print(f"{'variant':<28}{'maxDD':>8}{'CAGR':>9}{'Sharpe':>8}{'2008':>9}")
for dd, label, c, sh, y08 in rows:
    print(f"{label:<28}{dd*100:7.0f}%{c*100:8.1f}%{sh:8.2f}{y08*100:+8.1f}%")
print("\nDONE.")
