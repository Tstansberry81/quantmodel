"""Composite scoring: winsorize -> (sector-neutral) z-score -> weighted blend.

Tuned weights are the fast road to a backtest that looks great and works never,
so the default is an equal-weight composite. Weights are still adjustable from
the website for experimentation.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def winsorize(s: pd.Series, pct: float) -> pd.Series:
    if s.dropna().empty or pct <= 0:
        return s
    lo, hi = s.quantile(pct), s.quantile(1 - pct)
    return s.clip(lo, hi)


def zscore(s: pd.Series) -> pd.Series:
    mu, sd = s.mean(), s.std()
    if not sd or np.isnan(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


def _factor_z(df: pd.DataFrame, col: str, higher_better: bool,
              winsor: float, sector_neutral: bool, normalize: str = "zscore") -> pd.Series:
    raw = pd.to_numeric(df[col], errors="coerce")
    if normalize == "rank":
        # rank-normalize: robust to outliers and skew, gives cleaner deciles
        if sector_neutral and "sector" in df:
            base = raw.groupby(df["sector"]).rank(pct=True)
        else:
            base = raw.rank(pct=True)
        z = zscore(base.fillna(0.5))
    else:
        raw = winsorize(raw, winsor)
        if sector_neutral and "sector" in df:
            z = raw.groupby(df["sector"]).transform(zscore)
        else:
            z = zscore(raw)
    z = z.fillna(0.0)
    return z if higher_better else -z


def compute_scores(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Add per-factor z-score columns and the weighted composite score."""
    factors = params["factors"]
    settings = params["settings"]
    winsor = settings["winsor_pct"]
    sector_neutral = settings["sector_neutral"]
    normalize = settings.get("normalize", "zscore")

    df = df.copy()
    zcols, weights = [], []
    for fname, f in factors.items():
        if fname not in df.columns:
            continue
        zc = f"z_{fname}"
        df[zc] = _factor_z(df, fname, f["higher_better"], winsor, sector_neutral, normalize)
        zcols.append(zc)
        weights.append(float(f.get("weight", 0.0)))

    w = np.array(weights, dtype=float)
    if w.sum() == 0:
        w = np.ones_like(w)
    w = w / w.sum()
    df["composite"] = df[zcols].values @ w
    return df


def apply_hard_filters(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Keep only rows passing every factor listed in settings['hard_filters']
    (hard requirements). Falls back to the unfiltered df if too few survive."""
    import config
    hard = params["settings"].get("hard_filters", []) or []
    factors = params["factors"]
    out = df
    for fname in hard:
        f = factors.get(fname)
        if not f or fname not in out.columns:
            continue
        col = pd.to_numeric(out[fname], errors="coerce")
        thr, op = f["threshold"], f["op"]
        keep = (col >= thr) if op == ">=" else (col <= thr) if op == "<=" else (col == thr)
        out = out[keep.fillna(False)]
    return out if len(out) >= config.N_STOCKS else df


def screen_flags(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Boolean pass/fail per factor screen + count of screens passed."""
    factors = params["factors"]
    df = df.copy()
    pass_cols = []
    for fname, f in factors.items():
        if fname not in df.columns:
            continue
        col = pd.to_numeric(df[fname], errors="coerce")
        thr, op = f["threshold"], f["op"]
        if op == ">=":
            flag = col >= thr
        elif op == "<=":
            flag = col <= thr
        else:
            flag = col == thr
        pc = f"pass_{fname}"
        df[pc] = flag.fillna(False)
        pass_cols.append(pc)
    df["screens_passed"] = df[pass_cols].sum(axis=1)
    df["screens_total"] = len(pass_cols)
    df["pass_fraction"] = df["screens_passed"] / max(len(pass_cols), 1)
    return df
