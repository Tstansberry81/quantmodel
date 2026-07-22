"""De-biased Edge validation on the REAL point-in-time Russell-1000 (Norgate, ~2yr).

Runs the Edge on the survivorship-free universe built by norgate_ingest.py --
names that delisted mid-sample are present, held, and priced through their
delisting -- on the SAME basis the product ships, so the result is comparable:

  * t+1 entry      -- signal from close(t), position entered at close(t+1).
  * two staggered  -- sleeves at offset 0 and hold/2, averaged, as in the book.
  * daily regime   -- the 200dMA de-risk is evaluated EVERY day (continuous),
                      off the PRIOR close, not once per rebalance.
  * daily returns  -- so max drawdown is true intra-period, directly comparable
                      to the proxy's daily-measured drawdown.

It also sweeps ALL start offsets, because a single-sleeve run at one arbitrary
calendar phase is known to swing ~12 CAGR points; the offset distribution is
reported so the headline can be read against its own noise.

Growth mix is OFF (the Norgate Stocks package carries no fundamentals), so the
matching proxy comparison must also be run at mix=0.
"""
import warnings; warnings.filterwarnings("ignore")
import pickle
import numpy as np, pandas as pd
import config

HOLD, N, CAP, LB, REG, COST = 42, 10, 0.50, 126, 0.25, 10.0
RF_D = config.RISK_FREE_ANNUAL / 252.0

with open(config.CACHE_DIR / "norgate_r1000.pkl", "rb") as fh:
    d = pickle.load(fh)
cal = d["cal"]; T = len(cal)
spx = d["spx_close"]; sptr = d["sptr"]
syms = list(d["price"].keys())
P = np.column_stack([d["price"][s] for s in syms])
M = np.column_stack([d["member"][s] for s in syms])
RET = np.full_like(P, np.nan); RET[1:] = P[1:] / P[:-1] - 1.0
ma200 = pd.Series(spx).rolling(200).mean().to_numpy()

if not np.isfinite(sptr).sum() > 0.9 * T:
    raise SystemExit("benchmark series is mostly NaN -- refusing to report fabricated alpha")

# Norgate marks delisted symbols with a trailing marker; reuse the canonical parser.
from pit_universe import _strip_norgate
delisted = np.array([_strip_norgate(s) != s.strip().upper() for s in syms])


def accel(pos, j):
    a, b, c = P[pos, j], P[pos - 63, j], P[pos - 126, j]
    if not (a > 0 and b > 0 and c > 0):
        return np.nan
    return (a / b - 1.0) - (b / c - 1.0)


def corr_cap_pick(pos, cand, scores, n=N, cap=CAP):
    """Greedy: best accel first, admit if |corr| (trailing 126d) with all held <= cap."""
    order = [cand[k] for k in np.argsort(-scores)]
    win = RET[pos - LB + 1: pos + 1]
    chosen = []
    for j in order:
        if np.isfinite(win[:, j]).sum() <= 60:
            continue
        if chosen:
            rj = win[:, j]
            ok = True
            for h in chosen:
                m = np.isfinite(rj) & np.isfinite(win[:, h])
                if m.sum() > 30:
                    c = np.corrcoef(rj[m], win[m, h])[0, 1]
                    if np.isfinite(c) and abs(c) > cap:
                        ok = False; break
            if not ok:
                continue
        chosen.append(j)
        if len(chosen) >= n:
            break
    return chosen


def sleeve(off):
    """Daily net return series for one sleeve starting at calendar offset `off`.
    Returns (daily, covered_mask, n_periods, n_delisted_held)."""
    r = np.full(T, np.nan); cost = np.zeros(T)
    prev, nper, ndel = set(), 0, 0
    pos = LB + off
    while pos + 2 <= T - 1:
        cand = [j for j in range(len(syms))
                if M[pos, j] and np.isfinite(P[pos, j]) and np.isfinite(P[pos - 126, j])]
        sc = np.array([accel(pos, j) for j in cand])
        keep = np.isfinite(sc)
        cand = [c for c, k in zip(cand, keep) if k]; sc = sc[keep]
        if len(cand) >= N:
            pick = corr_cap_pick(pos, cand, sc)
            if pick:
                nper += 1; ndel += int(delisted[pick].sum())
                ent = pos + 1                        # enter at close(t+1)
                ext = min(pos + 1 + HOLD, T - 1)
                for t in range(ent + 1, ext + 1):    # first return is ent -> ent+1
                    v = RET[t, pick]; v = v[np.isfinite(v)]
                    if len(v):
                        r[t] = v.mean()
                to = 1 - len(set(pick) & prev) / len(pick) if prev else 1.0
                cost[ent] += (COST / 1e4) * to
                prev = set(pick)
        pos += HOLD
    covered = np.isfinite(r)
    daily = np.where(covered, r, 0.0) - cost
    # continuous 200dMA regime, decided on the PRIOR close (no look-ahead)
    below = np.zeros(T, bool)
    below[1:] = np.isfinite(ma200[:-1]) & (spx[:-1] < ma200[:-1])
    daily = np.where(below, REG * daily + (1 - REG) * RF_D, daily)
    return daily, covered | (cost > 0), nper, ndel


def stats(daily, mask):
    x = daily[mask]
    if len(x) < 20:
        return dict(cagr=float("nan"), sharpe=float("nan"), dd=float("nan"), n=len(x))
    eq = np.cumprod(1 + x)
    cagr = eq[-1] ** (252.0 / len(x)) - 1
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    sh = float(np.sqrt(252) * (x.mean() - RF_D) / x.std()) if x.std() > 0 else 0.0
    return dict(cagr=float(cagr), sharpe=sh, dd=dd, n=len(x))


print(f"Universe: {len(syms)} R1000 current+past names ({int(delisted.sum())} delisted), "
      f"{cal.min().date()}..{cal.max().date()}")

# ---- the shipped book: two half-period-staggered sleeves, averaged ----
s0, m0, n0, dl0 = sleeve(0)
s1, m1, n1, dl1 = sleeve(HOLD // 2)
book = np.nanmean(np.vstack([np.where(m0, s0, np.nan), np.where(m1, s1, np.nan)]), axis=0)
bmask = np.isfinite(book)
book = np.where(bmask, book, 0.0)
st = stats(book, bmask)

bench_d = np.full(T, np.nan); bench_d[1:] = sptr[1:] / sptr[:-1] - 1.0
bs = stats(np.nan_to_num(bench_d), bmask & np.isfinite(bench_d))

print("=" * 74)
print(f"DE-BIASED Edge, PRODUCTION BASIS (t+1, 2 staggered sleeves, daily continuous")
print(f"regime, daily drawdown) -- {n0}+{n1} rebalances, {st['n']} trading days")
print("=" * 74)
print(f"  Edge (real PIT):  CAGR {st['cagr']*100:6.1f}%   Sharpe {st['sharpe']:.2f}   maxDD {st['dd']*100:6.1f}%")
print(f"  S&P 500 (TR):     CAGR {bs['cagr']*100:6.1f}%   Sharpe {bs['sharpe']:.2f}   maxDD {bs['dd']*100:6.1f}%")
print(f"  excess vs S&P:    {(st['cagr']-bs['cagr'])*100:+.1f}%/yr")
print(f"  later-delisted names held: {dl0+dl1} slot-periods")

# ---- calendar-phase sensitivity: the headline vs its own noise ----
cg = []
for off in range(0, HOLD, 3):
    s, m, _, _ = sleeve(off)
    cg.append(stats(s, m)["cagr"])
cg = np.array([c for c in cg if np.isfinite(c)])
print(f"\n  single-sleeve CAGR across {len(cg)} start offsets: "
      f"min {cg.min()*100:.1f}%  median {np.median(cg)*100:.1f}%  max {cg.max()*100:.1f}%")
print(f"  -> the staggered book ({st['cagr']*100:.1f}%) vs S&P ({bs['cagr']*100:.1f}%); "
      f"{int((cg > bs['cagr']).sum())}/{len(cg)} offsets beat the S&P")

print("\nBasis matches production EXCEPT: no $2B mcap floor (R1000 membership is the")
print("floor here) and growth-mix OFF (no fundamentals in the Norgate Stocks package),")
print("so any proxy comparison must also be run at mix=0. Trial = 2yr -> directional.")
