"""The Edge -- does a VOLUME signal add to price-momentum?

Tests whether volume signals (volume growth, dollar volume, volume-surge
z-score, OBV trend) add incremental value over the price-momentum baseline
(ret_63) used by The Edge short-term trading model.

Does NOT modify edge_lib.py or qmodel/. Pulls daily volume via yfinance, caches
to data/cache/volume.pkl, builds as-of volume signals per rebalance (no
look-ahead), then tests volume-only / momentum-only / blends over 1Y/2Y/5Y/MAX.

Run:  .venv\\Scripts\\python.exe edge_test_volume.py
"""
from __future__ import annotations
import os, sys, time, pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import yfinance as yf

import edge_lib as E
from qmodel import engine

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "cache", "volume.pkl")
YEARS = 8                       # recent window that covers 1Y/2Y/5Y
START = (pd.Timestamp.today() - pd.Timedelta(days=int(YEARS * 365.25 + 30))).strftime("%Y-%m-%d")


# --------------------------------------------------------------------------- #
# 1. Pull + cache daily volume                                                #
# --------------------------------------------------------------------------- #
def build_volume_cache():
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            vol = pickle.load(f)
        print(f"[cache] loaded {CACHE}: {vol.shape[1]} tickers, "
              f"{vol.index.min().date()} -> {vol.index.max().date()}")
        return vol

    data = engine._load_bt_data()
    # map ticker -> company_key (skip dupes; keep first)
    tk2ck = {}
    for ck, blob in data.items():
        tk = blob.get("meta", {}).get("ticker")
        if tk and tk not in tk2ck:
            tk2ck[tk] = ck
    tickers = sorted(tk2ck)
    print(f"[pull] {len(tickers)} unique tickers, since {START} (chunks of 150)")

    frames = []
    CH = 150
    for i in range(0, len(tickers), CH):
        chunk = tickers[i:i + CH]
        for attempt in range(3):
            try:
                df = yf.download(chunk, start=START, auto_adjust=False,
                                 progress=False, threads=True, group_by="column")
                break
            except Exception as e:
                print(f"  chunk {i//CH} attempt {attempt} failed: {e}")
                time.sleep(5)
        else:
            print(f"  chunk {i//CH} permanently failed, skipping")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            v = df["Volume"] if "Volume" in df.columns.get_level_values(0) else None
        else:  # single ticker -> flat columns
            v = df[["Volume"]].rename(columns={"Volume": chunk[0]})
        if v is not None:
            frames.append(v)
        print(f"  chunk {i//CH+1}/{(len(tickers)+CH-1)//CH} done ({len(chunk)} tickers)")

    vol = pd.concat(frames, axis=1)
    vol = vol.loc[:, ~vol.columns.duplicated()].sort_index()
    # rename ticker columns -> company_key
    vol = vol.rename(columns={tk: tk2ck[tk] for tk in vol.columns if tk in tk2ck})
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump(vol, f)
    print(f"[cache] wrote {CACHE}: {vol.shape[1]} cols, "
          f"{vol.index.min().date()} -> {vol.index.max().date()}")
    return vol


# --------------------------------------------------------------------------- #
# 2. As-of volume signals per rebalance (no look-ahead)                       #
# --------------------------------------------------------------------------- #
def attach_volume_signals(pan, vol):
    """Add vol_growth, dollar_vol, vol_surge_z, obv_trend columns to each panel
    DataFrame, computed using ONLY volume data up to (<=) the rebalance date."""
    data = engine._load_bt_data()
    prices = {ck: data[ck]["prices"] for ck in data if data[ck].get("prices") is not None}
    vidx = vol.index.values.astype("datetime64[ns]")
    vcols = set(vol.columns)
    cov_hits = cov_tot = 0

    for i, d in enumerate(pan.panels):
        dt = np.datetime64(pd.Timestamp(pan.bdates[i]), "ns")
        vpos = int(np.searchsorted(vidx, dt, side="right"))   # rows strictly <= dt
        g = np.full(len(d), np.nan); dv = np.full(len(d), np.nan)
        vz = np.full(len(d), np.nan); ob = np.full(len(d), np.nan)
        for j, ck in enumerate(d["company_key"].values):
            cov_tot += 1
            if ck not in vcols or vpos < 64:
                continue
            vser = vol[ck].values[:vpos]                       # only up to date
            v = vser[-126:]
            v = v[~np.isnan(v)]
            if len(v) < 64:
                continue
            cov_hits += 1
            v21 = v[-21:]; v63 = v[-63:]
            a21, a63 = v21.mean(), v63.mean()
            g[j] = a21 / a63 - 1 if a63 > 0 else np.nan
            # surge z: most-recent-5d avg vs trailing-63d dist
            base = v63[:-5] if len(v63) > 5 else v63
            mu, sd = base.mean(), base.std(ddof=1)
            vz[j] = (v[-5:].mean() - mu) / sd if sd > 0 else np.nan
            # dollar volume (liquidity): price * avg 21d volume
            pr = prices.get(ck)
            if pr is not None:
                ppos = int(np.searchsorted(pr.index.values.astype("datetime64[ns]"), dt, side="right")) - 1
                if ppos >= 0:
                    dv[j] = float(pr.values[ppos]) * a21
            # OBV trend: sign(price chg)*volume cumulated, slope over last 21d
            if pr is not None and ppos >= 21:
                pidx = pr.index.values.astype("datetime64[ns]")
                # align volume to last ~63 price days up to date
                obv = []
                pv = pr.values
                start = max(1, ppos - 62)
                cum = 0.0
                for t in range(start, ppos + 1):
                    # find volume at that price date
                    vp = int(np.searchsorted(vidx, pidx[t], side="right")) - 1
                    if vp < 0 or vp >= len(vol):
                        continue
                    vt = vol[ck].values[vp]
                    if np.isnan(vt):
                        continue
                    cum += np.sign(pv[t] - pv[t - 1]) * vt
                    obv.append(cum)
                if len(obv) >= 10:
                    y = np.array(obv); x = np.arange(len(y))
                    sl = np.polyfit(x, y, 1)[0]
                    scale = np.abs(y).mean() + 1e-9
                    ob[j] = sl / scale * len(y)   # normalized trend
        d["vol_growth"] = g
        d["dollar_vol"] = dv
        d["vol_surge_z"] = vz
        d["obv_trend"] = ob
        d["log_dollar_vol"] = np.log(np.where(dv > 0, dv, np.nan))
    print(f"[signals] volume coverage: {cov_hits}/{cov_tot} "
          f"({100*cov_hits/max(cov_tot,1):.0f}%) of (name,rebalance) cells")
    return pan


# --------------------------------------------------------------------------- #
# 3. Test rankers                                                             #
# --------------------------------------------------------------------------- #
def run(pan, weights, label, n=10):
    holds = [E.select_topN(df, n, weights) for df in pan.panels]
    rets = E.period_returns(pan, holds)
    return rets, label


def table_row(pan, rets, label):
    out = {"label": label}
    for wname, k in (("1Y", 12), ("2Y", 24), ("5Y", 60), ("MAX", None)):
        kk = len(rets) if k is None else min(k, len(rets))
        c, dd, sh = E.perf(rets[-kk:])
        out[wname] = (c, sh, dd)
    return out


def fmt_table(rows):
    cols = ["1Y", "2Y", "5Y", "MAX"]
    hdr = f"{'ranker':<34}" + "".join(f"{c+' CAGR/Sh/DD':>22}" for c in cols)
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        line = f"{r['label']:<34}"
        for c in cols:
            cg, sh, dd = r[c]
            line += f"{cg*100:7.1f}% {sh:5.2f} {dd*100:4.0f}%".rjust(22)
        lines.append(line)
    return "\n".join(lines)


def main():
    vol = build_volume_cache()
    print("\n[panel] loading edge panel (~1-2 min) ...")
    pan = E.load_edge_panel()
    print(f"[panel] {pan.T} rebalances, {pan.bdates[0].date()} -> {pan.bdates[-1].date()}")
    attach_volume_signals(pan, vol)

    rows = []
    # baseline: price momentum only
    r, l = run(pan, {"ret_63": 1.0}, "momentum only (ret_63) [BASELINE]")
    rows.append(table_row(pan, r, l))

    # volume signals alone
    for col, lab in [("vol_growth", "vol_growth only"),
                     ("vol_surge_z", "vol_surge_z only"),
                     ("obv_trend", "obv_trend only"),
                     ("log_dollar_vol", "dollar_vol(liquidity) only")]:
        r, l = run(pan, {col: 1.0}, lab)
        rows.append(table_row(pan, r, l))

    # blends: momentum + volume_growth
    for w in (0.25, 0.5, 1.0):
        r, l = run(pan, {"ret_63": 1.0, "vol_growth": w}, f"mom + {w:g}*vol_growth")
        rows.append(table_row(pan, r, l))
    # momentum + surge z
    for w in (0.25, 0.5):
        r, l = run(pan, {"ret_63": 1.0, "vol_surge_z": w}, f"mom + {w:g}*vol_surge_z")
        rows.append(table_row(pan, r, l))
    # momentum + obv
    r, l = run(pan, {"ret_63": 1.0, "obv_trend": 0.5}, "mom + 0.5*obv_trend")
    rows.append(table_row(pan, r, l))

    # S&P reference
    spx_row = {"label": "S&P 500 (reference)"}
    for wname, k in (("1Y", 12), ("2Y", 24), ("5Y", 60), ("MAX", None)):
        kk = len(pan.spxf) if k is None else min(k, len(pan.spxf))
        c, dd, sh = E.perf(pan.spxf[-kk:])
        spx_row[wname] = (c, sh, dd)
    rows.append(spx_row)

    print("\n" + "=" * 120)
    print("VOLUME vs MOMENTUM -- top-10, equal-weight, no overlays  (CAGR / Sharpe / maxDD)")
    print("=" * 120)
    print(fmt_table(rows))

    # cross-sectional correlation of vol_growth signal with ret_63 (redundancy check)
    cors = []
    for df in pan.panels:
        a = pd.to_numeric(df["ret_63"], errors="coerce")
        b = pd.to_numeric(df["vol_growth"], errors="coerce")
        m = a.notna() & b.notna()
        if m.sum() > 30:
            cors.append(np.corrcoef(a[m], b[m])[0, 1])
    print(f"\n[redundancy] mean cross-sectional corr(ret_63, vol_growth) = {np.nanmean(cors):+.3f}")


if __name__ == "__main__":
    main()
