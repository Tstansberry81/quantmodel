"""Carhart 4-factor + robustness attribution of the stock-selection strategy.

Extends tech_bias_test.py. Answers the harder question: once we control for the
MOMENTUM and SIZE premia (which are buyable, NOT stock-picking skill), does any
real selection alpha survive? Plus survivorship, costs, and proper daily-vol
risk-scaling.

Tasks (all run end-to-end, labeled output):
  1. Carhart 4-factor (Mkt-RF, SMB, HML, UMD) + sectors. KEY: alpha after mom+size.
  2. Survivorship sensitivity (active-only universe).
  3. Net-of-cost alpha (per-rebalance turnover * 10bps one-way).
  4. Proper daily-vol vol-scaling (Barroso-Santa-Clara 2015).
  5. (optional) sector-ETF cross-check via yfinance.

FF FACTOR SOURCING
------------------
We first TRY to download the canonical Ken French daily CSVs (Mkt-RF/SMB/HML and
the momentum factor) and compound them over each holding window. If the machine
has no outbound network (this sandbox does not), we FALL BACK to constructing the
four factors *from the strategy's own point-in-time top-500 universe*:
  Mkt-RF = S&P500-TR forward return - rf   (the strategy's actual benchmark)
  SMB    = small-cap tercile fwd ret - large-cap tercile fwd ret  (by pit mcap)
  HML    = cheap tercile - expensive tercile  (value = earnings_yield E/P)
  UMD    = winner tercile - loser tercile      (12-1 momentum)
These are arguably the MORE relevant test here: they are formed from exactly the
buyable names the strategy chooses among, so "is the edge just buyable factor
exposure?" is asked against factors the strategy could literally have traded.
The script PRINTS which source was used.
"""
import warnings; warnings.filterwarnings("ignore")
import io, zipfile, urllib.request
import numpy as np, pandas as pd
import copy
import config
from qmodel import engine, scoring, backtest as bt
from qmodel.equations import FACTORS, default_params

N = config.N_STOCKS; HOLD = 21; LB = 251; SKIP = 21
PPY = 252.0 / HOLD
RF_PER = config.RISK_FREE_ANNUAL / PPY
RF_DAILY = config.RISK_FREE_ANNUAL / 252.0
FILING_LAG = np.timedelta64(bt.FILING_LAG_DAYS, "D")

print("Loading backtest artifacts ...")
data = engine._load_bt_data(); bm = engine._benchmarks_cached()
mkt = bm["SP500"].dropna(); mret_full = mkt.pct_change()
ndx = bm["NASDAQ"].dropna()
fund_ids = {f: FACTORS[f]["ratio_id"] for f in FACTORS if FACTORS[f]["ratio_id"]}
VALUE_RID = "ratio_earnings_yield"   # E/P, higher = cheaper -> the HML value signal

prep = {}
for ck, blob in data.items():
    pr = blob.get("prices")
    if pr is None or len(pr) < LB + HOLD + 2:
        continue
    arr = np.asarray(pr.values, float); pidx = np.asarray(pr.index.values, "datetime64[ns]")
    rets = np.empty(len(arr)); rets[0] = np.nan; rets[1:] = arr[1:] / arr[:-1] - 1
    mret = np.asarray(mret_full.reindex(pr.index, method="ffill").values, float)
    fh = blob.get("fund_hist"); fidx = None; fcols = {}; mcap = None; val = None
    if fh is not None and not fh.empty:
        fidx = np.asarray(fh.index.values, "datetime64[ns]")
        for fn, rid in fund_ids.items():
            fcols[fn] = np.asarray(fh[rid].values, float) if rid in fh.columns else None
        if "calculated_market_cap" in fh.columns:
            mcap = np.asarray(fh["calculated_market_cap"].values, float)
        if VALUE_RID in fh.columns:
            val = np.asarray(fh[VALUE_RID].values, float)
    prep[ck] = {"arr": arr, "pidx": pidx, "rets": rets, "mret": mret, "fidx": fidx,
                "fcols": fcols, "mcap": mcap, "val": val,
                "sector": blob.get("meta", {}).get("sector", "Unknown"),
                "status": blob.get("meta", {}).get("trading_status", "Active")}


def asof(fidx, varr, dlag):
    if fidx is None or varr is None:
        return np.nan
    p = int(np.searchsorted(fidx, dlag, side="right")) - 1
    return varr[p] if p >= 0 else np.nan


def bench_fwd(series, d):
    past = series[series.index <= d]; fwd = series[series.index > d]
    if past.empty or len(fwd) < 1:
        return np.nan
    return float(fwd.iloc[min(HOLD, len(fwd)) - 1] / past.iloc[-1] - 1)


# ---- build the rebalance panel once -----------------------------------------
spx = mkt
monthly = spx.index[spx.index >= spx.index.min() + pd.Timedelta(days=400)]
rebal = list(monthly[:-HOLD][::HOLD])

panels = []        # per-period candidate DataFrame
bdates = []        # rebalance date
fwd_dates = []     # the date HOLD trading days ahead (window end) -- for FF alignment
spxf, ndxf = [], []
sector_ret = []    # per-period {sector: cap-weighted fwd return}
ff_panel = []      # per-period internally-built {Mkt-RF,SMB,HML,UMD}

for d in rebal:
    dt = np.datetime64(pd.Timestamp(d), "ns"); dlag = dt - FILING_LAG
    cand = []
    for ck, P in prep.items():
        pos = int(np.searchsorted(P["pidx"], dt, side="right")) - 1
        if pos < LB or (len(P["arr"]) - 1 - pos) < HOLD:
            continue
        mc = asof(P["fidx"], P["mcap"], dlag)
        if mc is None or np.isnan(mc):
            continue
        cand.append((mc, ck, pos))
    if len(cand) < 50:
        continue
    cand.sort(key=lambda x: x[0], reverse=True); cand = cand[:config.BT_UNIVERSE_SIZE]
    rows = []
    for mc, ck, pos in cand:
        P = prep[ck]; arr = P["arr"]
        w = P["rets"][pos - LB + 1:pos + 1]; wm = P["mret"][pos - LB + 1:pos + 1]
        sd = w.std(ddof=1); neg = w[w < 0]; dd = neg.std(ddof=1) if neg.size > 1 else np.nan
        p6, p12 = arr[pos - 125], arr[pos - 251]; hi = arr[pos - 251:pos + 1].max()
        ok = ~np.isnan(w) & ~np.isnan(wm); rm = np.nan; beta = np.nan
        if ok.sum() > 60:
            rr, mm = w[ok], wm[ok]
            if mm.var() > 0:
                beta = float(np.cov(rr, mm)[0, 1] / mm.var())
                resid = rr - (rr.mean() - beta * mm.mean()) - beta * mm
                rv = resid.std(ddof=1)
                if rv > 0 and resid.size > SKIP:
                    rm = resid[:-SKIP].sum() / rv
        fwd = arr[pos + HOLD] / arr[pos] - 1
        mom121 = arr[pos - 20] / arr[pos - 251] - 1
        row = {"company_key": ck, "sector": P["sector"], "pit_mcap": mc, "beta": beta,
               "status": P["status"], "pos": pos,
               "momentum": mom121, "resid_mom": rm,
               "mom_accel": ((arr[pos] / p6 - 1) - (p6 / p12 - 1)) if p6 > 0 and p12 > 0 else np.nan,
               "mom_52w_high": arr[pos] / hi if hi > 0 else np.nan,
               "sharpe": (np.sqrt(252) * (w.mean() - config.RISK_FREE_ANNUAL / 252) / sd) if sd > 0 else np.nan,
               "sortino": (np.sqrt(252) * (w.mean() - config.RISK_FREE_ANNUAL / 252) / dd) if dd and dd > 0 else np.nan,
               "volatility": np.sqrt(252) * sd, "reversal_1m": arr[pos] / arr[pos - 20] - 1,
               "ep": asof(P["fidx"], P["val"], dlag),     # earnings yield -> value signal
               "fwd_ret": fwd}
        for fn in fund_ids:
            row[fn] = asof(P["fidx"], P["fcols"].get(fn), dlag)
        rows.append(row)
    df = pd.DataFrame(rows)

    # cap-weighted sector forward returns
    sr = {}
    for sec, g in df.dropna(subset=["fwd_ret"]).groupby("sector"):
        wcap = g["pit_mcap"].to_numpy(float)
        if wcap.sum() > 0:
            sr[sec] = float(np.average(g["fwd_ret"].to_numpy(float), weights=wcap))

    # internally-built Carhart factors from THIS period's buyable universe (terciles)
    def long_short(sig_col, fwd_col="fwd_ret", high_minus_low=True):
        g = df.dropna(subset=[sig_col, fwd_col])
        if len(g) < 30:
            return np.nan
        s = g[sig_col].to_numpy(float); fwd = g[fwd_col].to_numpy(float)
        lo, hi = np.nanpercentile(s, 33.3), np.nanpercentile(s, 66.7)
        top = fwd[s >= hi]; bot = fwd[s <= lo]
        if top.size == 0 or bot.size == 0:
            return np.nan
        d_ = top.mean() - bot.mean()
        return float(d_ if high_minus_low else -d_)
    smb = long_short("pit_mcap", high_minus_low=False)   # small minus big
    hml = long_short("ep", high_minus_low=True)           # high E/P (cheap) minus low
    umd = long_short("momentum", high_minus_low=True)     # winners minus losers
    ff_panel.append({"SMB": smb, "HML": hml, "UMD": umd})

    panels.append(df)
    sector_ret.append(sr)
    bdates.append(pd.Timestamp(d))
    # window-end date for FF alignment = HOLD trading days after d
    fwd_idx = spx.index[spx.index > d]
    fwd_dates.append(fwd_idx[min(HOLD, len(fwd_idx)) - 1] if len(fwd_idx) else d)
    spxf.append(bench_fwd(spx, d)); ndxf.append(bench_fwd(ndx, d))

bdates = pd.DatetimeIndex(bdates)
fwd_dates = pd.DatetimeIndex(fwd_dates)
spxf = np.array(spxf); ndxf = np.array(ndxf)
T = len(panels)
print(f"Built {T} monthly rebalances, {bdates[0].date()} -> {bdates[-1].date()}\n")

ALL_SECTORS = sorted({s for sr in sector_ret for s in sr})
SECTORS = [s for s in ALL_SECTORS if s.lower() != "unknown"]


# =============================================================================
# Carhart factors: try canonical Ken French download, else internal construction
# =============================================================================
def _try_download_ff():
    """Return (mktrf, smb, hml, umd) daily pd.Series in DECIMAL, or None on failure."""
    base = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    def grab(fn):
        req = urllib.request.Request(base + fn, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=20).read()
        z = zipfile.ZipFile(io.BytesIO(raw))
        txt = z.read(z.namelist()[0]).decode("latin-1")
        return txt
    def parse(txt, cols):
        lines = txt.splitlines()
        recs = []
        for ln in lines:
            p = ln.split(",")
            if len(p) < 2:
                continue
            k = p[0].strip()
            if len(k) == 8 and k.isdigit():
                try:
                    recs.append([k] + [float(x) for x in p[1:len(cols) + 1]])
                except ValueError:
                    continue
        df = pd.DataFrame(recs, columns=["date"] + cols)
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        return df.set_index("date")[cols] / 100.0
    f3 = parse(grab("F-F_Research_Data_Factors_daily_CSV.zip"), ["MktRF", "SMB", "HML", "RF"])
    mom = parse(grab("F-F_Momentum_Factor_daily_CSV.zip"), ["UMD"])
    return f3["MktRF"], f3["SMB"], f3["HML"], mom["UMD"].reindex(f3.index)


def _compound_window(daily, d0, d1):
    seg = daily[(daily.index > d0) & (daily.index <= d1)]
    if seg.empty:
        return np.nan
    return float(np.prod(1.0 + seg.values) - 1.0)


FF_SOURCE = None
MKTRF = SMB = HML = UMD = None
try:
    dly_mkt, dly_smb, dly_hml, dly_umd = _try_download_ff()
    MKTRF = np.array([_compound_window(dly_mkt, bdates[i], fwd_dates[i]) for i in range(T)])
    SMB = np.array([_compound_window(dly_smb, bdates[i], fwd_dates[i]) for i in range(T)])
    HML = np.array([_compound_window(dly_hml, bdates[i], fwd_dates[i]) for i in range(T)])
    UMD = np.array([_compound_window(dly_umd, bdates[i], fwd_dates[i]) for i in range(T)])
    if np.isfinite(MKTRF).sum() > T * 0.8:
        FF_SOURCE = "Ken French daily CSVs (canonical, compounded over each holding window)"
    else:
        raise RuntimeError("insufficient FF coverage after alignment")
except Exception as e:
    FF_SOURCE = ("INTERNAL construction from the strategy's own point-in-time top-500 "
                 "universe (Ken French download unavailable: %s)" % (repr(e)[:60]))
    MKTRF = spxf - RF_PER
    SMB = np.array([p["SMB"] for p in ff_panel])
    HML = np.array([p["HML"] for p in ff_panel])
    UMD = np.array([p["UMD"] for p in ff_panel])

# fill any nan factor values with 0 (period contributes no tilt that month)
for a in (MKTRF, SMB, HML, UMD):
    a[~np.isfinite(a)] = 0.0
print(f"Carhart factor source: {FF_SOURCE}")
print(f"  factor means (per-period): Mkt-RF {MKTRF.mean()*100:+.2f}%  SMB {SMB.mean()*100:+.2f}%  "
      f"HML {HML.mean()*100:+.2f}%  UMD {UMD.mean()*100:+.2f}%\n")


# ---- model-return builder (raw / coarse vol-scale / daily vol-scale) --------
def run_variant(params, active_only=False, vol_mode=None, target_vol=None,
                lev_cap=1.0, vol_win_months=6, daily_win=42):
    """Return dict of arrays. vol_mode in {None,'monthly','daily'}.

    Also returns per-period turnover and (for daily mode) the daily return series.
    daily_win: trailing trading days for the daily vol forecast (~21-63).
    """
    hard = params["settings"].get("hard_filters", []) or []
    min_g = FACTORS["rev_growth"]["threshold"] if "rev_growth" in hard else None
    mret, tech_w, bbeta, turn = [], [], [], []
    daily_rets = []                 # equal-weighted daily returns within each window
    prev_top = None
    for ti, df0 in enumerate(panels):
        df = df0[df0["status"] == "Active"] if active_only else df0
        cs = scoring.compute_scores(df, params)
        valid = cs.dropna(subset=["composite", "fwd_ret"]).copy()
        pool = valid
        if min_g is not None:
            gg = pd.to_numeric(valid["rev_growth"], errors="coerce")
            sub = valid[(gg >= min_g)]
            pool = sub if len(sub) >= N else valid
        top = pool.sort_values("composite", ascending=False).head(N)
        mret.append(float(top["fwd_ret"].mean()))
        tw = (top["sector"].str.lower().str.contains("tech|information technology|communication",
              regex=True, na=False)).mean()
        tech_w.append(float(tw)); bbeta.append(float(top["beta"].mean()))
        # one-way turnover vs previous book
        cur = set(top["company_key"])
        turn.append(1.0 - len(cur & prev_top) / len(cur) if prev_top and cur else (0.0 if prev_top is None else 1.0))
        prev_top = cur
        # equal-weighted DAILY returns over this holding window (for daily vol scaling)
        if vol_mode == "daily":
            mats = []
            for ck in top["company_key"]:
                P = prep[ck]; pos = int(df.loc[df["company_key"] == ck, "pos"].iloc[0])
                seg = P["rets"][pos + 1: pos + 1 + HOLD]   # the HOLD daily fwd returns
                if len(seg) == HOLD:
                    mats.append(seg)
            daily_rets.append(np.nanmean(np.vstack(mats), axis=0) if mats else np.full(HOLD, np.nan))

    mret = np.array(mret); tech_w = np.array(tech_w); bbeta = np.array(bbeta); turn = np.array(turn)

    scaled = mret.copy(); lev_used = np.ones(T)
    if vol_mode == "monthly":
        tv = target_vol if target_vol is not None else np.nanstd(mret) * np.sqrt(PPY)
        for t in range(T):
            if t < vol_win_months:
                continue
            rv = np.nanstd(mret[t - vol_win_months:t]) * np.sqrt(PPY)
            lev = min((tv / rv) if rv > 0 else 1.0, lev_cap)
            lev_used[t] = lev
            scaled[t] = lev * mret[t] + (1 - lev) * RF_PER
    elif vol_mode == "daily":
        # build a continuous daily return stream (non-overlapping windows)
        stream = np.concatenate([dr for dr in daily_rets])
        tv = target_vol if target_vol is not None else (np.nanstd(stream) * np.sqrt(252))
        for t in range(T):
            # forecast vol from the trailing `daily_win` daily returns ENDING just
            # before this window starts -> strictly out-of-sample
            end = t * HOLD                      # index into stream where window t begins
            start = max(0, end - daily_win)
            hist = stream[start:end]
            if hist.size < min(daily_win, 21):
                continue
            rv = np.nanstd(hist) * np.sqrt(252)
            lev = min((tv / rv) if rv > 0 else 1.0, lev_cap)
            lev_used[t] = lev
            scaled[t] = lev * mret[t] + (1 - lev) * RF_PER

    out = {"raw": mret, "scaled": scaled, "tech_w": tech_w, "beta": bbeta,
           "turn": turn, "lev": lev_used}
    return out


# ---- OLS + Newey-West -------------------------------------------------------
def ols_nw(y, X, L=6):
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    S = (X * resid[:, None]).T @ (X * resid[:, None])
    for l in range(1, L + 1):
        wgt = 1.0 - l / (L + 1)
        Xe = X * resid[:, None]
        G = Xe[l:].T @ Xe[:-l]
        S += wgt * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    return beta, se, beta / se


def build_X(carhart=True, use_sectors=False):
    cols = [np.ones(T), MKTRF if carhart else (spxf - RF_PER)]
    names = ["alpha", "Mkt-RF"]
    if carhart:
        cols += [SMB, HML, UMD]; names += ["SMB", "HML", "UMD"]
    if use_sectors:
        for s in SECTORS[:-1]:
            v = np.array([sr.get(s, spxf[i]) for i, sr in enumerate(sector_ret)])
            cols.append(v - spxf); names.append("sec:" + s[:10])
    return np.column_stack(cols), names


def attribute(label, mret, carhart=True, use_sectors=False, mask=None, show_loadings=False):
    y = mret - RF_PER
    X, names = build_X(carhart, use_sectors)
    if mask is not None:
        y, X = y[mask], X[mask]
    beta, se, t = ols_nw(y, X)
    a_ann = (1 + beta[0]) ** PPY - 1
    tag = "** survives" if abs(t[0]) > 2 else "not significant" if abs(t[0]) < 1.65 else "~ marginal"
    print(f"  [{label}]  (n={len(y)})")
    print(f"    alpha = {beta[0]*100:+.3f}%/period  ann {a_ann*100:+.2f}%   t = {t[0]:+.2f}   {tag}")
    if show_loadings:
        for j in range(1, min(len(names), 5)):   # show the 4 factor loadings
            print(f"      {names[j]:7s} loading = {beta[j]:+.3f}   t = {t[j]:+.2f}")
    return a_ann, t[0], beta


def perf(mret):
    eq = np.cumprod(1 + np.nan_to_num(mret))
    cagr = eq[-1] ** (PPY / len(mret)) - 1 if eq[-1] > 0 else -1
    dd = (eq / np.maximum.accumulate(eq) - 1).min()
    shp = np.sqrt(PPY) * np.nanmean(mret) / np.nanstd(mret)
    return cagr, dd, shp


base = default_params()
yr = bdates.year
half = T // 2
m_all = run_variant(base)                       # full universe, raw
m_act = run_variant(base, active_only=True)     # active-only universe, raw
m_dvs = run_variant(base, vol_mode="daily", daily_win=42, lev_cap=1.0)
m_dvsl = run_variant(base, vol_mode="daily", daily_win=42, lev_cap=1.5)
m_mvs = run_variant(base, vol_mode="monthly", lev_cap=1.0)


def sub(name, arr, carhart, sectors):
    attribute(f"{name}: FULL", arr, carhart, sectors, show_loadings=carhart)
    attribute(f"{name}: 1st half {bdates[0].date()}..{bdates[half-1].date()}",
              arr, carhart, sectors, mask=np.arange(T) < half)
    attribute(f"{name}: 2nd half {bdates[half].date()}..{bdates[-1].date()}",
              arr, carhart, sectors, mask=np.arange(T) >= half)
    attribute(f"{name}: EXCL 2020", arr, carhart, sectors, mask=(yr != 2020))


print("=" * 80)
print("1. CARHART 4-FACTOR ATTRIBUTION  (the key test: alpha after MOM + SIZE)")
print("=" * 80)
print("  Strategy excess return ~ Mkt-RF + SMB + HML + UMD  (Newey-West L=6 t).")
print("  If the 'alpha' was just buyable momentum/size premia, it collapses here.\n")
print("  --- market-only baseline (for reference) ---")
sub("Mkt only", m_all["raw"], carhart=False, sectors=False)
print("\n  --- Carhart 4-factor (NO sectors) ---")
sub("Carhart", m_all["raw"], carhart=True, sectors=False)
print("\n  --- Carhart 4-factor + sector tilts ---")
sub("Carhart+sectors", m_all["raw"], carhart=True, sectors=True)

print()
print("=" * 80)
print("2. SURVIVORSHIP SENSITIVITY  (active-only universe vs full)")
print("=" * 80)
print(f"  Universe: 1000 active + 111 inactive names. Re-running on active-only.\n")
a_full, t_full, _ = attribute("full universe   (Carhart+sectors)", m_all["raw"], True, True)
a_act, t_act, _ = attribute("active-only      (Carhart+sectors)", m_act["raw"], True, True)
print(f"  -> active-only alpha delta = {(a_act - a_full)*100:+.2f} pp/yr "
      f"(t {t_full:+.2f} -> {t_act:+.2f})")
ca, _, sa = perf(m_all["raw"]); cb, _, sb = perf(m_act["raw"])
print(f"  -> CAGR {ca*100:.1f}% (full) vs {cb*100:.1f}% (active-only); "
      f"Sharpe {sa:.2f} vs {sb:.2f}")

print()
print("=" * 80)
print("3. NET-OF-COST ALPHA  (2 x one-way turnover x 10bps per rebalance)")
print("=" * 80)
turn = m_all["turn"]
cost = 2.0 * turn * 0.0010
m_net = m_all["raw"] - cost
print(f"  avg one-way turnover/rebalance = {np.nanmean(turn)*100:.1f}%  "
      f"(annualized {np.nanmean(turn)*PPY*100:.0f}%)")
print(f"  avg cost drag/rebalance       = {np.nanmean(cost)*1e4:.1f} bps  "
      f"(annualized {np.nanmean(cost)*PPY*100:.2f}%)\n")
print("  GROSS:")
attribute("gross (Carhart+sectors)", m_all["raw"], True, True)
print("  NET of costs:")
attribute("net   (Carhart+sectors)", m_net, True, True)
print("  Sub-sample stability NET of costs:")
sub("net Carhart+sec", m_net, carhart=True, sectors=True)

print()
print("=" * 80)
print("4. PROPER DAILY-VOL VOL-SCALING  (Barroso-Santa-Clara 2015)")
print("=" * 80)
print("  Forecast vol from trailing 42 daily strategy returns (out-of-sample),")
print("  scale next window toward full-sample realized vol; rest -> risk-free.\n")
print(f"  {'variant':<26}  {'CAGR':>6}  {'Sharpe':>6}  {'maxDD':>6}")
rows = [("raw", m_all["raw"]),
        ("coarse monthly-scaled", m_mvs["scaled"]),
        ("daily-scaled (cap 1.0)", m_dvs["scaled"]),
        ("daily-scaled (cap 1.5)", m_dvsl["scaled"])]
for nm, mr in rows:
    c, dd, sh = perf(mr)
    print(f"  {nm:<26}  {c*100:5.1f}%  {sh:6.2f}  {dd*100:5.0f}%")
print("\n  crisis-year returns (compounded within calendar year):")
print(f"  {'variant':<26}  {'2008':>7}  {'2009':>7}  {'2020':>7}  {'2022':>7}")
for nm, mr in rows:
    cells = []
    for y in (2008, 2009, 2020, 2022):
        mk = yr == y
        cells.append(f"{(np.prod(1+np.nan_to_num(mr[mk]))-1)*100:+6.1f}%" if mk.sum() else "   n/a ")
    print(f"  {nm:<26}  " + "  ".join(cells))
print("\n  alpha of daily-scaled variant (Carhart+sectors):")
attribute("daily-scaled cap1.0", m_dvs["scaled"], True, True, show_loadings=True)

print()
print("=" * 80)
print("5. SECTOR-ETF CROSS-CHECK (optional; yfinance)")
print("=" * 80)
try:
    import yfinance as yf
    etfs = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLRE", "XLU", "XLC"]
    px = yf.download(etfs, start="2005-01-01", progress=False)["Close"]
    if px is None or px.dropna(how="all").empty:
        raise RuntimeError("no data returned")
    print(f"  pulled {px.shape[1]} sector ETFs, {px.dropna(how='all').shape[0]} days")
    print("  (validation of universe-built sector factors vs tradeable SPDRs would go here)")
except Exception as e:
    print(f"  SKIPPED: yfinance unavailable / no network in this environment ({repr(e)[:70]})")

print("\nDone.")
