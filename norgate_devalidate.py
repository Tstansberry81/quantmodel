"""De-biased Edge validation on the REAL point-in-time Russell-1000 (Norgate, ~2yr).

Runs the Edge's core (acceleration signal -> correlation cap -> 200dMA regime ->
costs, growth-mix OFF since the Norgate Stocks package has no fundamentals) on the
survivorship-free universe built by norgate_ingest.py, and compares it to the
PROXY Edge (our yfinance-cached universe) over the same window. The gap is the
real survivorship effect — names that were in the R1000 and later delisted are
now present, held, and priced through their delisting.
"""
import warnings; warnings.filterwarnings("ignore")
import pickle
import numpy as np, pandas as pd
import config
import edge_lib as E

HOLD, N, CAP, LB, REG, COST = 42, 10, 0.50, 126, 0.25, 10.0
PPY = 252.0 / HOLD
RF_PER = config.RISK_FREE_ANNUAL / PPY

d = pickle.load(open(config.CACHE_DIR / "norgate_r1000.pkl", "rb"))
cal = d["cal"]; T = len(cal)
spx = d["spx_close"]; sptr = d["sptr"]
syms = list(d["price"].keys())
P = np.column_stack([d["price"][s] for s in syms])          # T x S total-return close (NaN off-listing)
M = np.column_stack([d["member"][s] for s in syms])         # T x S membership bool
RET = np.full_like(P, np.nan); RET[1:] = P[1:] / P[:-1] - 1.0
ma200 = pd.Series(spx).rolling(200).mean().to_numpy()
delisted = np.array(["-" in s and s.rsplit("-", 1)[-1].isdigit() for s in syms])

print(f"Universe: {len(syms)} R1000 current+past names ({int(delisted.sum())} delisted), "
      f"{cal.min().date()}..{cal.max().date()}\n")


def accel(pos, j):
    a, b, c = P[pos, j], P[pos - 63, j], P[pos - 126, j]
    if not (a > 0 and b > 0 and c > 0):
        return np.nan
    return (a / b - 1.0) - (b / c - 1.0)


def corr_cap_pick(pos, cand, scores, n, cap):
    """Greedy: best accel first, admit if |corr| (trailing 126d) with all held <= cap."""
    order = [cand[k] for k in np.argsort(-scores)]
    win = RET[pos - LB + 1: pos + 1]                        # 126 x S
    chosen = []
    for j in order:
        if not np.isfinite(win[:, j]).sum() > 60:
            continue
        if not chosen:
            chosen.append(j); continue
        rj = win[:, j]
        ok = True
        for h in chosen:
            m = np.isfinite(rj) & np.isfinite(win[:, h])
            if m.sum() > 30:
                c = np.corrcoef(rj[m], win[m, h])[0, 1]
                if np.isfinite(c) and abs(c) > cap:
                    ok = False; break
        if ok:
            chosen.append(j)
        if len(chosen) >= n:
            break
    return chosen[:n]


def fwd(pos, j):
    seg = P[pos: pos + HOLD + 1, j]
    v = seg[np.isfinite(seg)]
    return (v[-1] / v[0] - 1.0) if len(v) >= 2 and v[0] > 0 else np.nan


rebals = list(range(LB + 4, T - 2, HOLD))
rets, bench, prev, held_delisted = [], [], set(), 0
for pos in rebals:
    cand = [j for j in range(len(syms))
            if M[pos, j] and np.isfinite(P[pos, j]) and np.isfinite(P[pos - 126, j])]
    sc = np.array([accel(pos, j) for j in cand])
    keep = np.isfinite(sc)
    cand = [c for c, k in zip(cand, keep) if k]; sc = sc[keep]
    if len(cand) < N:
        continue
    pick = corr_cap_pick(pos, cand, sc, N, CAP)
    fr = np.array([fwd(pos, j) for j in pick])
    r = float(np.nanmean(fr))
    if not (spx[pos] > ma200[pos]) and np.isfinite(ma200[pos]):
        r = REG * r + (1 - REG) * RF_PER                    # 200dMA regime de-risk
    to = 1 - len(set(pick) & prev) / len(pick) if prev else 1.0
    prev = set(pick)
    rets.append(r - (COST / 1e4) * to)
    held_delisted += int(delisted[pick].sum())
    # S&P TR forward return over the same window
    b = sptr[pos: pos + HOLD + 1]; b = b[np.isfinite(b)]
    bench.append(b[-1] / b[0] - 1.0 if len(b) >= 2 else np.nan)

rets = np.array(rets); bench = np.array(bench)


def perf(r):
    eq = np.cumprod(1 + np.nan_to_num(r))
    cg = eq[-1] ** (PPY / len(r)) - 1 if len(r) and eq[-1] > 0 else -1
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    sh = float(np.sqrt(PPY) * (np.nanmean(r) - RF_PER) / np.nanstd(r)) if np.nanstd(r) > 0 else 0.0
    return cg, dd, sh


nc, nd_, ns = perf(rets); bc, bd, bs = perf(bench)
print("=" * 70)
print(f"DE-BIASED Edge on the REAL PIT Russell-1000 (Norgate, {len(rets)} rebalances)")
print("=" * 70)
print(f"  Edge (real PIT):  CAGR {nc*100:6.1f}%   Sharpe {ns:.2f}   maxDD {nd_*100:.0f}%")
print(f"  S&P 500 (TR):     CAGR {bc*100:6.1f}%   Sharpe {bs:.2f}   maxDD {bd*100:.0f}%")
print(f"  alpha vs S&P:     {(nc-bc)*100:+.1f}%/yr")
print(f"  times a later-delisted name was held: {held_delisted}")

# ---- proxy Edge over the same ~2yr window (pure momentum, n=10) ----
try:
    r = E.run_edge_backtest("2Y", spec={"growth_mix": 0.0, "n": N, "hold": HOLD})
    m, sp = r["performance"]["model"], r["performance"]["sp500"]
    print("\n  --- PROXY Edge (yfinance universe), same 2Y window, mix=off ---")
    print(f"  Edge (proxy):     CAGR {m['cagr']*100:6.1f}%   Sharpe {m['sharpe']:.2f}   maxDD {m['max_drawdown']*100:.0f}%")
    print(f"  SURVIVORSHIP GAP (proxy - real): {(m['cagr']-nc)*100:+.1f}%/yr")
except Exception as e:
    print("  proxy compare failed:", repr(e)[:80])
print("\nNote: trial = 2yr only, ~8 rebalances -> directional, not statistically tight.")
print("Growth mix OFF (Norgate has no fundamentals). Full-history de-bias needs paid Platinum.")
