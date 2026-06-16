"""Turnover-reduction research for the top-N stock-selection model.

The strategy's biggest weakness is TURNOVER (~42% one-way/month => ~506%/yr).
At ~10 bps cost this erodes net return and the leftover alpha loses significance.

This script implements several turnover reducers, each driven off the SAME
per-period scored/filtered candidate pools the live model uses, then compares
NET-of-cost performance and Carhart+sector alpha against the baseline.

Reducers implemented:
  1. Hold-band / hysteresis (K = 15, 20, 25): keep a holding while it stays in
     the top-K of the ranking; only drop it (for the best-ranked non-holding)
     once it falls out of top-K.
  2. Partial rebalance: rebalance only every 2 or 3 months (skip rebalances),
     holding the prior book in between.
  3. Max-trades cap: at most M names swapped per period (M = 2, 3).
  4. (optional) Composite EMA smoothing: rank on an EMA of each name's composite
     over the last few periods to stabilize selection.

Selection / cost mechanics mirror tech_bias_lib.model_returns exactly:
  compute_scores -> drop NaN composite/fwd_ret -> rev_growth hard filter
  (fallback to full pool if < N pass) -> rank by composite desc.
Net return per period = mean fwd_ret of book - cost_bps/1e4 * one-way turnover,
with turnover = 1 - |held ∩ prev| / |held|.

Run:  .venv/Scripts/python.exe tech_bias_turnover.py
Does NOT modify tech_bias_lib.py or any live-model files.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

import tech_bias_lib as L
from qmodel import scoring
from qmodel.equations import default_params, FACTORS

N = L.N
PPY = L.PPY


# ---------------------------------------------------------------------------
# Build the per-period RANKED pool once (same logic as model_returns/holdings).
# Each entry is the filtered candidate frame for one rebalance, sorted by
# composite desc, indexed by company_key, carrying fwd_ret / sector / beta /
# composite. Selection rules below operate purely on these ranked frames.
# ---------------------------------------------------------------------------
def build_ranked_pools(pan, params):
    hard = params["settings"].get("hard_filters", []) or []
    min_g = FACTORS["rev_growth"]["threshold"] if "rev_growth" in hard else None
    pools = []
    for df in pan.panels:
        cs = scoring.compute_scores(df, params)
        valid = cs.dropna(subset=["composite", "fwd_ret"]).copy()
        pool = valid
        if min_g is not None:
            gg = pd.to_numeric(valid["rev_growth"], errors="coerce")
            sub = valid[gg >= min_g]
            pool = sub if len(sub) >= N else valid
        pool = pool.sort_values("composite", ascending=False).reset_index(drop=True)
        pool["rank"] = np.arange(len(pool))  # 0 = best
        pools.append(pool.set_index("company_key"))
    return pools


def _tech_w(frame):
    return float(frame["sector"].str.lower().str.contains(
        "information technology|communication", regex=True, na=False).mean())


def _period_stats(held_keys, pool):
    """Return (mean fwd_ret, tech weight, mean beta) for an equal-weight book."""
    sub = pool.loc[[k for k in held_keys if k in pool.index]]
    return float(sub["fwd_ret"].mean()), _tech_w(sub), float(sub["beta"].mean())


def _series_from_holdings(pools, holdings_per_period, cost_bps_for_ret=0.0):
    """Given the list of held company_key sets per period, compute the
    gross/net return series and turnover series (mirrors model_returns)."""
    ret, turn, tw, bb = [], [], [], []
    prev = set()
    for pool, held in zip(pools, holdings_per_period):
        cur = set(held)
        to = 1.0 - len(cur & prev) / len(cur) if cur and prev else (1.0 if cur else 0.0)
        prev = cur
        r, w, b = _period_stats(held, pool)
        ret.append(r - (cost_bps_for_ret / 1e4) * to)
        turn.append(to); tw.append(w); bb.append(b)
    return {"ret": np.array(ret), "turnover": np.array(turn),
            "tech_w": np.array(tw), "beta": np.array(bb)}


# ---------------------------------------------------------------------------
# Selection rules: each returns a list (per period) of held company_key lists.
# ---------------------------------------------------------------------------
def sel_baseline(pools):
    """Plain top-N each period (the current model)."""
    return [list(p.index[:N]) for p in pools]


def sel_holdband(pools, K):
    """Hysteresis: keep a holding while its rank stays < K; replace dropouts
    with the best-ranked non-holdings to refill to N."""
    rank_of = [dict(zip(p.index, p["rank"])) for p in pools]
    order = [list(p.index) for p in pools]  # already rank-sorted
    held = []
    prev = []
    for t, (p, ro, od) in enumerate(zip(pools, rank_of, order)):
        if not prev:
            cur = od[:N]
        else:
            # keep prior names that are still present AND within top-K
            keep = [k for k in prev if k in ro and ro[k] < K]
            cur = list(keep)
            # refill with best-ranked names not already held
            for k in od:
                if len(cur) >= N:
                    break
                if k not in cur:
                    cur.append(k)
            cur = cur[:N]
        held.append(cur); prev = cur
    return held


def sel_partial_rebal(pools, every):
    """Rebalance only every `every` periods; hold the prior book in between.
    On a held period the book is unchanged (intersected with names still
    present in the pool so fwd_ret/turnover are well-defined)."""
    held = []
    prev = []
    for t, p in enumerate(pools):
        if t % every == 0 or not prev:
            cur = list(p.index[:N])
        else:
            # carry prior book forward; keep only names still in this pool
            cur = [k for k in prev if k in p.index]
            # if names dropped out of the universe, top up from best-ranked
            if len(cur) < N:
                for k in p.index:
                    if len(cur) >= N:
                        break
                    if k not in cur:
                        cur.append(k)
        held.append(cur); prev = cur
    return held


def sel_max_trades(pools, max_trades):
    """At most `max_trades` names swapped per period. Among current holdings,
    sell the worst-ranked ones (up to max_trades) only if a better-ranked
    non-holding exists to replace them; buy the best-ranked non-holdings."""
    held = []
    prev = []
    for t, p in enumerate(pools):
        ro = dict(zip(p.index, p["rank"]))
        if not prev:
            cur = list(p.index[:N])
        else:
            # current names still present, ordered by current rank (best first)
            present = [k for k in prev if k in ro]
            present.sort(key=lambda k: ro[k])
            # candidate buys: best-ranked names not currently held
            buys = [k for k in p.index if k not in present]
            # first restore book to N if names left the universe (free top-ups)
            cur = list(present)
            bi = 0
            while len(cur) < N and bi < len(buys):
                cur.append(buys[bi]); bi += 1
            # now perform up to max_trades discretionary swaps:
            # replace worst-held with best-available iff it improves rank
            swaps = 0
            held_sorted = sorted(cur, key=lambda k: ro[k])  # best..worst
            while swaps < max_trades and bi < len(buys):
                worst = held_sorted[-1]
                cand = buys[bi]
                if ro[cand] < ro[worst]:
                    cur.remove(worst); cur.append(cand)
                    held_sorted = sorted(cur, key=lambda k: ro[k])
                    bi += 1; swaps += 1
                else:
                    break
            cur = cur[:N]
        held.append(cur); prev = cur
    return held


def build_ema_pools(pan, params, span):
    """Rebuild ranked pools where the ranking key is an EMA of each name's
    composite over the trailing `span` periods (names re-rank on smoothed
    composite). Missing periods reset the EMA for that name."""
    hard = params["settings"].get("hard_filters", []) or []
    min_g = FACTORS["rev_growth"]["threshold"] if "rev_growth" in hard else None
    alpha = 2.0 / (span + 1.0)
    ema = {}  # company_key -> running EMA of composite
    pools = []
    for df in pan.panels:
        cs = scoring.compute_scores(df, params)
        valid = cs.dropna(subset=["composite", "fwd_ret"]).copy()
        # update EMA for every scored candidate this period
        comp = pd.to_numeric(valid["composite"], errors="coerce")
        sm = {}
        present = set(valid["company_key"])
        for ck, c in zip(valid["company_key"], comp):
            if ck in ema:
                ema[ck] = alpha * c + (1 - alpha) * ema[ck]
            else:
                ema[ck] = c
            sm[ck] = ema[ck]
        # forget names not present (so a re-entry restarts cleanly)
        for ck in list(ema):
            if ck not in present:
                del ema[ck]
        valid["composite_ema"] = valid["company_key"].map(sm)
        pool = valid
        if min_g is not None:
            gg = pd.to_numeric(valid["rev_growth"], errors="coerce")
            sub = valid[gg >= min_g]
            pool = sub if len(sub) >= N else valid
        pool = pool.sort_values("composite_ema", ascending=False).reset_index(drop=True)
        pool["rank"] = np.arange(len(pool))
        pools.append(pool.set_index("company_key"))
    return pools


# ---------------------------------------------------------------------------
def main():
    print("Loading panel (~1 min)...")
    pan = L.load_panel()
    params = default_params()
    print(f"  {pan.T} rebalances, {pan.bdates[0].date()} .. {pan.bdates[-1].date()}\n")

    # Regressors: full Carhart + sector controls
    cf = L.carhart_factors(pan)
    st = L.sector_tilts(pan)
    regressors = {**cf, **st}

    pools = build_ranked_pools(pan, params)

    # sanity: our replicated baseline should match model_returns exactly
    base_mr = L.model_returns(pan, params, cost_bps=0.0)
    base_held = sel_baseline(pools)
    base_rep = _series_from_holdings(pools, base_held, cost_bps_for_ret=0.0)
    drift = float(np.nanmax(np.abs(base_rep["ret"] - base_mr["ret"])))
    print(f"baseline replication check: max |ret diff| = {drift:.2e} "
          f"(turnover diff = {np.nanmax(np.abs(base_rep['turnover']-base_mr['turnover'])):.2e})\n")

    # Define all variants -> held-key lists
    variants = {}
    variants["Baseline (top-10)"] = base_held
    for K in (15, 20, 25):
        variants[f"Hold-band K={K}"] = sel_holdband(pools, K)
    for ev in (2, 3):
        variants[f"Rebalance every {ev}m"] = sel_partial_rebal(pools, ev)
    for m in (3, 2):
        variants[f"Max {m} trades/mo"] = sel_max_trades(pools, m)
    for span in (2, 3):
        ep = build_ema_pools(pan, params, span)
        variants[f"Composite EMA span={span}"] = sel_baseline(ep)
        # stash the ema pools so return calc uses the right fwd_ret frames
        variants[f"Composite EMA span={span}"] = (sel_baseline(ep), ep)

    # ---- compute comparison table -------------------------------------------
    rows = []
    for name, held in variants.items():
        if isinstance(held, tuple):
            held_list, use_pools = held
        else:
            held_list, use_pools = held, pools
        gross = _series_from_holdings(use_pools, held_list, cost_bps_for_ret=0.0)
        g = gross["ret"]; to = gross["turnover"]
        ann_to = float(np.nanmean(to)) * PPY
        cagr, mdd, shp = L.perf(g)
        net10 = g - (10 / 1e4) * to
        net20 = g - (20 / 1e4) * to
        c10 = L.perf(net10)[0]
        c20 = L.perf(net20)[0]
        # alpha (gross and net@10bps) with full Carhart + sector controls
        ga, gt, _ = L.attribution(name + " gross", g, regressors, verbose=False)
        na, nt, _ = L.attribution(name + " net10", net10, regressors, verbose=False)
        rows.append({
            "variant": name, "ann_turn": ann_to, "gross_CAGR": cagr,
            "Sharpe": shp, "maxDD": mdd, "net10_CAGR": c10, "net20_CAGR": c20,
            "gross_alpha": ga * 100, "gross_t": gt,
            "net10_alpha": na * 100, "net10_t": nt,
        })

    df = pd.DataFrame(rows)

    # ---- print table --------------------------------------------------------
    hdr = (f"{'variant':<22} {'ann turn':>8} {'grCAGR':>7} {'Shrp':>5} "
           f"{'maxDD':>7} {'net@10':>7} {'net@20':>7} "
           f"{'grAlpha(t)':>14} {'netAlpha@10(t)':>16}")
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for _, r in df.iterrows():
        print(f"{r['variant']:<22} "
              f"{r['ann_turn']*100:>7.0f}% "
              f"{r['gross_CAGR']*100:>6.1f}% "
              f"{r['Sharpe']:>5.2f} "
              f"{r['maxDD']*100:>6.0f}% "
              f"{r['net10_CAGR']*100:>6.1f}% "
              f"{r['net20_CAGR']*100:>6.1f}% "
              f"{r['gross_alpha']:>7.2f} ({r['gross_t']:>+4.2f}) "
              f"{r['net10_alpha']:>8.2f} ({r['net10_t']:>+4.2f})")
    print("=" * len(hdr))

    # ---- pick the best NET-of-cost variant ----------------------------------
    base = df[df["variant"] == "Baseline (top-10)"].iloc[0]
    best10 = df.sort_values("net10_CAGR", ascending=False).iloc[0]
    best20 = df.sort_values("net20_CAGR", ascending=False).iloc[0]
    print(f"\nBaseline: ann turnover {base['ann_turn']*100:.0f}%, "
          f"net@10 CAGR {base['net10_CAGR']*100:.1f}%, "
          f"net@20 CAGR {base['net20_CAGR']*100:.1f}%, "
          f"gross alpha {base['gross_alpha']:.2f}%/yr (t={base['gross_t']:+.2f}), "
          f"net@10 alpha {base['net10_alpha']:.2f}%/yr (t={base['net10_t']:+.2f})")
    print(f"\nBest NET@10bps: {best10['variant']}  ->  net@10 CAGR "
          f"{best10['net10_CAGR']*100:.1f}% (vs base {base['net10_CAGR']*100:.1f}%), "
          f"turnover {best10['ann_turn']*100:.0f}% "
          f"(vs base {base['ann_turn']*100:.0f}%, "
          f"-{(1-best10['ann_turn']/base['ann_turn'])*100:.0f}%), "
          f"gross alpha {best10['gross_alpha']:.2f}%/yr (t={best10['gross_t']:+.2f})")
    print(f"Best NET@20bps: {best20['variant']}  ->  net@20 CAGR "
          f"{best20['net20_CAGR']*100:.1f}% (vs base {base['net20_CAGR']*100:.1f}%)")

    df.to_csv("turnover_comparison.csv", index=False)
    print("\nsaved turnover_comparison.csv")


if __name__ == "__main__":
    main()
