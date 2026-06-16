"""External-factor validation of the momentum/growth stock-selection strategy.

WHY THIS SCRIPT EXISTS
----------------------
We already attributed the strategy's return to IN-UNIVERSE factors (size/value/
momentum built from the same point-in-time top-500 universe the strategy picks
from) and found a full-control alpha of only ~+6.3%/yr, t=1.71 (marginal). A
skeptic could call in-universe factors circular -- "you built the controls from
the same pond you fish in." This script answers that objection by validating
against REAL, EXTERNAL, PUBLISHED factors and TRADEABLE sector ETFs:

  1. Ken French published Carhart factors (Mkt-RF, SMB, HML, RF + Mom/UMD),
     MONTHLY, downloaded live from the Dartmouth Data Library.
  2. Strategy monthly returns compounded to CALENDAR months and regressed on the
     four Carhart factors (Newey-West HAC t).
  3. Tradeable SPDR sector ETFs (XLK, XLF, ... XLC) added as extra regressors.

If the French download is blocked, we fall back to monthly factor proxies built
from liquid yfinance ETFs (clearly labeled). Either way the script PRINTS the
source actually used.

Does NOT modify tech_bias_lib.py or any live-model file.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import io, zipfile, urllib.request
import numpy as np, pandas as pd

import tech_bias_lib
from qmodel.equations import default_params

FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
PPY_MONTHLY = 12.0   # we work in calendar months here, not 21-day rebalances


# ---------------------------------------------------------------------------
# 1. Published Carhart factors from Ken French (MONTHLY), with ETF fallback
# ---------------------------------------------------------------------------
def _grab_zip_csv(fn: str) -> str:
    req = urllib.request.Request(FRENCH_BASE + fn, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=30).read()
    z = zipfile.ZipFile(io.BytesIO(raw))
    return z.read(z.namelist()[0]).decode("latin-1")


def _parse_monthly_section(txt: str, cols: list[str]) -> pd.DataFrame:
    """Parse the MONTHLY section of a Ken French CSV.

    Monthly rows have a 6-digit YYYYMM key. The file later has an annual section
    with 4-digit YYYY keys; we stop at the first blank line AFTER we've started
    collecting monthly rows so we never bleed into the annual block.
    """
    recs, started = [], False
    for ln in txt.splitlines():
        p = [x.strip() for x in ln.split(",")]
        k = p[0]
        if k.isdigit() and len(k) == 6:           # YYYYMM monthly row
            try:
                vals = [float(x) for x in p[1:len(cols) + 1]]
            except ValueError:
                continue
            recs.append([k] + vals); started = True
        elif started and (ln.strip() == "" or (k.isdigit() and len(k) == 4)):
            break                                  # reached blank line / annual block
    df = pd.DataFrame(recs, columns=["ym"] + cols)
    df["ym"] = pd.PeriodIndex(df["ym"], freq="M")
    return df.set_index("ym")[cols].astype(float) / 100.0   # percent -> decimal


def load_french_monthly():
    """Return (DataFrame[Mkt-RF,SMB,HML,RF,UMD] monthly, source_label) or raise."""
    f3 = _parse_monthly_section(_grab_zip_csv("F-F_Research_Data_Factors_CSV.zip"),
                                ["Mkt-RF", "SMB", "HML", "RF"])
    mom = _parse_monthly_section(_grab_zip_csv("F-F_Momentum_Factor_CSV.zip"), ["Mom"])
    df = f3.join(mom, how="inner").rename(columns={"Mom": "UMD"})
    if len(df) < 100:
        raise RuntimeError(f"French parse returned only {len(df)} months")
    return df, "Ken French Data Library (REAL published Carhart factors, MONTHLY)"


def load_etf_proxy_monthly():
    """Fallback: monthly Carhart-style factor PROXIES from liquid ETFs (labeled)."""
    import yfinance as yf
    tk = ["^SP500TR", "SPY", "IWM", "IWD", "IWF", "MTUM", "^IRX"]
    px = yf.download(tk, start="2003-01-01", progress=False, auto_adjust=True)["Close"]
    m = (1 + px.pct_change()).resample("ME").prod() - 1
    rf = (px["^IRX"].resample("ME").last() / 100.0) / 12.0     # 13wk T-bill, annualized %
    out = pd.DataFrame(index=m.index)
    mkt = m["^SP500TR"].where(m["^SP500TR"].notna(), m["SPY"])
    out["Mkt-RF"] = mkt - rf
    out["SMB"] = m["IWM"] - m["SPY"]          # small minus big
    out["HML"] = m["IWD"] - m["IWF"]          # value minus growth
    out["UMD"] = m["MTUM"] - m["SPY"]         # momentum minus market
    out["RF"] = rf
    out.index = out.index.to_period("M")
    return out.dropna(), "yfinance ETF PROXY (SPY/IWM/IWD/IWF/MTUM) -- NOT the real French factors"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def annualize_alpha(a_monthly: float) -> float:
    return (1.0 + a_monthly) ** PPY_MONTHLY - 1.0


def regress(y, regressors: dict, L: int = 6):
    """y on [const] + regressors via Newey-West. Returns (beta, se, t, names, r2)."""
    names = ["alpha"] + list(regressors)
    X = np.column_stack([np.ones(len(y))] + [regressors[k] for k in regressors])
    beta, se, t = tech_bias_lib.ols_nw(y, X, L=L)
    resid = y - X @ beta
    ss_res = float(resid @ resid); ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return beta, se, t, names, r2


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("EXTERNAL-FACTOR VALIDATION (real Ken French Carhart + tradeable sector ETFs)")
    print("=" * 78)

    # --- factors ---
    try:
        fac, FAC_SOURCE = load_french_monthly()
    except Exception as e:
        print(f"  [French download failed: {repr(e)[:80]}] -> falling back to ETF proxy")
        fac, FAC_SOURCE = load_etf_proxy_monthly()
    print(f"\n(a) FACTOR SOURCE: {FAC_SOURCE}")
    print(f"    coverage: {fac.index.min()} .. {fac.index.max()}  ({len(fac)} months)")

    # --- strategy monthly returns (calendar months) ---
    print("\n  loading rebalance panel (~1 min) ...")
    pan = tech_bias_lib.load_panel()
    daily = tech_bias_lib.daily_strategy_returns(pan, default_params())
    strat_m = (1.0 + daily).resample("ME").prod() - 1.0
    strat_m.index = strat_m.index.to_period("M")
    strat_m = strat_m[strat_m != 0.0].dropna()    # drop empty leading/trailing months

    # --- align by year-month ---
    common = strat_m.index.intersection(fac.index)
    common = common.sort_values()
    s = strat_m.reindex(common)
    F = fac.reindex(common)
    y = (s - F["RF"]).to_numpy(float)             # strategy EXCESS return
    reg4 = {"Mkt-RF": F["Mkt-RF"].to_numpy(float), "SMB": F["SMB"].to_numpy(float),
            "HML": F["HML"].to_numpy(float), "UMD": F["UMD"].to_numpy(float)}
    print(f"    aligned strategy months: {common.min()} .. {common.max()}  (n={len(common)})")
    print(f"    strategy mean excess/mo {y.mean()*100:+.2f}%   ann ~{annualize_alpha(y.mean())*100:+.1f}%")

    # --- (b) Carhart 4-factor regression ---
    print("\n" + "-" * 78)
    print("(b) CARHART 4-FACTOR REGRESSION  (strategy excess ~ Mkt-RF + SMB + HML + UMD)")
    print("-" * 78)
    beta, se, t, names, r2 = regress(y, reg4)
    a_ann = annualize_alpha(beta[0])
    tag = "** survives (t>2)" if abs(t[0]) > 2 else "NOT significant (t<1.65)" if abs(t[0]) < 1.65 else "~ marginal"
    print(f"    alpha = {beta[0]*100:+.3f}%/mo   ANNUALIZED {a_ann*100:+.2f}%/yr   t={t[0]:+.2f}   {tag}")
    for i, nm in enumerate(names[1:], 1):
        print(f"      {nm:<8} loading={beta[i]:+.3f}   t={t[i]:+.2f}")
    print(f"    R^2 = {r2:.3f}")
    carhart_alpha, carhart_t = a_ann, float(t[0])

    # --- (c) add tradeable SPDR sector ETFs ---
    print("\n" + "-" * 78)
    print("(c) + TRADEABLE SPDR SECTOR ETFs  (excess-vs-market; late-inception sectors dropped)")
    print("-" * 78)
    sec_alpha = sec_t = None
    try:
        import yfinance as yf
        etfs = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE", "XLC"]
        spy = yf.download("SPY", start="2003-01-01", progress=False, auto_adjust=True)["Close"]
        epx = yf.download(etfs, start="2003-01-01", progress=False, auto_adjust=True)["Close"]
        # NB: resample(...).prod() returns 1.0 (->0% return) for all-NaN months,
        # which would silently fabricate pre-inception data for XLRE (2015) and
        # XLC (2018). Mask any month with no actual daily observation back to NaN
        # so dropna() correctly trims each ETF to its true live history.
        def monthly_ret(px):
            dr = px.pct_change()
            r = (1 + dr).resample("ME").prod() - 1
            valid = dr.notna().resample("ME").sum() > 0
            r = r.where(valid)
            r.index = r.index.to_period("M")
            return r
        spy_m = monthly_ret(spy); etf_m = monthly_ret(epx)
        if isinstance(spy_m, pd.DataFrame):
            spy_m = spy_m.iloc[:, 0]

        # Excess-vs-market sector returns. We drop XLRE (Oct-2015) and XLC (Jun-2018)
        # for TWO reasons: (i) they only exist for part of the sample, so keeping them
        # would force the regression onto a short post-2018 window; (ii) the 11 cap-
        # weighted sectors are collinear (they sum to ~market), so dropping some avoids
        # the dummy trap. The remaining 9 sectors span the full 2005-2026 window.
        avail = [e for e in etfs if e in etf_m.columns]
        drop = [e for e in ("XLRE", "XLC") if e in avail] or [avail[-1]]
        use_etfs = [e for e in avail if e not in drop]
        sec_reg = dict(reg4)
        # align on the months where every used ETF has data AND we have a factor row
        emask_idx = etf_m[use_etfs].dropna().index
        common2 = common.intersection(emask_idx)
        if len(common2) < 24:
            raise RuntimeError(f"too few overlapping ETF months ({len(common2)})")
        s2 = strat_m.reindex(common2); F2 = fac.reindex(common2)
        y2 = (s2 - F2["RF"]).to_numpy(float)
        sec_reg = {"Mkt-RF": F2["Mkt-RF"].to_numpy(float), "SMB": F2["SMB"].to_numpy(float),
                   "HML": F2["HML"].to_numpy(float), "UMD": F2["UMD"].to_numpy(float)}
        for e in use_etfs:
            sec_reg[f"se_{e}"] = (etf_m[e].reindex(common2) - spy_m.reindex(common2)).to_numpy(float)
        b2, se2, t2, n2, r2b = regress(y2, sec_reg)
        sec_alpha = annualize_alpha(b2[0]); sec_t = float(t2[0])
        tag2 = "** survives (t>2)" if abs(t2[0]) > 2 else "NOT significant (t<1.65)" if abs(t2[0]) < 1.65 else "~ marginal"
        print(f"    used {len(use_etfs)} sector ETFs (dropped {drop}); n={len(common2)} months")
        print(f"    alpha = {b2[0]*100:+.3f}%/mo   ANNUALIZED {sec_alpha*100:+.2f}%/yr   t={t2[0]:+.2f}   {tag2}")
        print("    Carhart loadings after adding sectors:")
        for i, nm in enumerate(n2[1:5], 1):
            print(f"      {nm:<8} loading={b2[i]:+.3f}   t={t2[i]:+.2f}")
        print(f"    R^2 = {r2b:.3f}  (vs {r2:.3f} Carhart-only)")
    except Exception as e:
        print(f"    SKIPPED sector-ETF cross-check: {repr(e)[:90]}")

    # --- (d) verdict vs in-universe ---
    print("\n" + "=" * 78)
    print("(d) VERDICT -- external vs in-universe attribution")
    print("=" * 78)
    print(f"    in-universe full-control alpha (prior work):  +6.30%/yr,  t=1.71  (marginal)")
    print(f"    EXTERNAL Carhart alpha (real French):        {carhart_alpha*100:+5.2f}%/yr,  t={carhart_t:+.2f}")
    if sec_alpha is not None:
        print(f"    EXTERNAL Carhart + sector ETFs:              {sec_alpha*100:+5.2f}%/yr,  t={sec_t:+.2f}")
    agree = abs(carhart_t) < 2.0
    print()
    if agree:
        print("    => External attribution CONFIRMS the in-universe finding: after controlling")
        print("       for REAL published momentum/size/value premia, the leftover stock-picking")
        print("       alpha is weak/marginal (t<2). The edge is largely buyable factor exposure,")
        print("       not circular in-universe construction.")
    else:
        print("    => External attribution DISAGREES: alpha survives real published factors (t>2),")
        print("       suggesting the in-universe controls were over-absorbing the edge.")


if __name__ == "__main__":
    main()
