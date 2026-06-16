"""Stacked test: turnover buffer (hold-band K=15) + daily crash rule together.

Compares four versions on the same panel so we can see the combined effect
before wiring it into the live model:
  - current        : plain top-10, no overlays (the live default today)
  - buffer only    : hold-band K=15 (only sell a name once it drops out of top-15)
  - crash only     : daily EWMA(0.94) vol-targeting, de-risk-only (cap 1.0)
  - BOTH stacked   : buffer + crash together  <- the proposed improved model

Reads cached artifacts via tech_bias_lib. Does not touch the live model.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import config
import tech_bias_lib as L
from qmodel import scoring
from qmodel.equations import default_params, FACTORS

RF_D = config.RISK_FREE_ANNUAL / 252.0
print("Building panel ...")
pan = L.load_panel()
P = default_params()
F = L.carhart_factors(pan); TILTS = L.sector_tilts(pan)
FULL = {"MKT": F["MKT_RF"], "SMB": F["SMB"], "HML": F["HML"], "UMD": F["UMD"], **TILTS}
yr = pan.bdates.year
M = L.daily_return_matrix(); CAL = M.index


# ---- selection variants -----------------------------------------------------
def _scored_pool(df):
    cs = scoring.compute_scores(df, P)
    valid = cs.dropna(subset=["composite", "fwd_ret"]).copy()
    hard = P["settings"].get("hard_filters", []) or []
    if "rev_growth" in hard:
        gg = pd.to_numeric(valid["rev_growth"], errors="coerce")
        sub = valid[gg >= FACTORS["rev_growth"]["threshold"]]
        if len(sub) >= L.N:
            valid = sub
    return valid.sort_values("composite", ascending=False).reset_index(drop=True)


def holdings_plain():
    return [list(_scored_pool(df)["company_key"].head(L.N)) for df in pan.panels]


def holdings_band(K=15):
    """Hold-band: keep a name while it stays in the top-K; fill the rest from the
    highest-ranked names not already held."""
    out, prev = [], []
    for df in pan.panels:
        ranked = _scored_pool(df)
        order = list(ranked["company_key"])
        rank_of = {ck: i for i, ck in enumerate(order)}
        if not prev:
            sel = order[:L.N]
        else:
            sel = [ck for ck in prev if rank_of.get(ck, 10**9) < K]
            for ck in order:
                if len(sel) >= L.N:
                    break
                if ck not in sel:
                    sel.append(ck)
            sel = sel[:L.N]
        out.append(sel); prev = sel
    return out


# ---- daily series + period returns from a holdings list ---------------------
def daily_from_holdings(holds):
    pieces = []
    for i, d in enumerate(pan.bdates):
        cks = [c for c in holds[i] if c in M.columns]
        loc = int(CAL.searchsorted(d, side="right"))
        win = CAL[loc:loc + L.HOLD]
        if len(win) == 0 or not cks:
            continue
        pieces.append(M.loc[win, cks].mean(axis=1))
    s = pd.concat(pieces)
    return s[~s.index.duplicated(keep="first")].fillna(0.0)


def volscale_daily(ds, lam=0.94, cap=1.0, target=None):
    """Barroso-Santa-Clara: scale each day by yesterday's EWMA vol forecast."""
    r = ds.values.astype(float)
    var = np.empty(len(r)); var[0] = np.nanvar(r[:21]) if len(r) > 21 else np.nanvar(r)
    for t in range(1, len(r)):
        var[t] = lam * var[t - 1] + (1 - lam) * r[t - 1] ** 2
    fvol = np.sqrt(var * 252.0)
    tv = target if target is not None else np.nanstd(r) * np.sqrt(252.0)
    lev = np.minimum(np.where(fvol > 0, tv / fvol, 1.0), cap)
    lev = np.nan_to_num(lev, nan=1.0)
    return pd.Series(lev * r + (1 - lev) * RF_D, index=ds.index)


def to_period(ds):
    """Compound a daily series into the T period-grid returns aligned to bdates."""
    out = []
    for d in pan.bdates:
        loc = int(CAL.searchsorted(d, side="right"))
        win = CAL[loc:loc + L.HOLD]
        seg = ds.reindex(win).fillna(0.0)
        out.append((1 + seg).prod() - 1)
    return np.array(out)


def turnover_of(holds):
    tos, prev = [], None
    for h in holds:
        cur = set(h)
        if prev is not None and cur:
            tos.append(1 - len(cur & prev) / len(cur))
        prev = cur
    return float(np.mean(tos)) * L.PPY      # annualized one-way


# ---- build the four variants ------------------------------------------------
h_plain, h_band = holdings_plain(), holdings_band(15)
to_plain, to_band = turnover_of(h_plain), turnover_of(h_band)

ds_plain, ds_band = daily_from_holdings(h_plain), daily_from_holdings(h_band)
tv = np.nanstd(ds_plain.values) * np.sqrt(252.0)           # common vol target

variants = {
    "current (top-10)":     (to_period(ds_plain), to_plain),
    "buffer only (K=15)":   (to_period(ds_band), to_band),
    "crash only (vol 1.0)": (to_period(volscale_daily(ds_plain, target=tv)), to_plain),
    "BOTH stacked":         (to_period(volscale_daily(ds_band, target=tv)), to_band),
}

print("\n" + "=" * 92)
print("STACKED COMPARISON  (period grid; alpha = full Carhart + sector tilts, Newey-West)")
print("=" * 92)
hdr = (f"{'variant':<22}{'CAGR':>7}{'Sharpe':>8}{'maxDD':>7}{'turn/yr':>9}"
       f"{'net@10':>8}{'gross a (t)':>14}{'  crisis 08/09/20/22'}")
print(hdr)
for nm, (r, to) in variants.items():
    cagr, dd, sh = L.perf(r)
    net = L.perf(r - (10 / 1e4) * (to / L.PPY))[0]          # 10bps one-way on per-period turnover
    a, ta, _ = L.attribution(nm, r, FULL, verbose=False)
    cy = []
    for y in (2008, 2009, 2020, 2022):
        m = yr == y
        cy.append(np.prod(1 + np.nan_to_num(r[m])) - 1)
    print(f"{nm:<22}{cagr*100:6.1f}%{sh:8.2f}{dd*100:6.0f}%{to*100:8.0f}%"
          f"{net*100:7.1f}%{a*100:>+8.1f}% ({ta:+.2f}){'   '}"
          f"{cy[0]*100:+.0f}/{cy[1]*100:+.0f}/{cy[2]*100:+.0f}/{cy[3]*100:+.0f}")

print("\nNotes: net@10 charges 10bps one-way on selection turnover (vol-scaling adds")
print("minor extra exposure-trading cost not modeled here). Crisis = calendar-year return.")
print("Done.")
