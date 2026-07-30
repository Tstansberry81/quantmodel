"""Recompute every factual claim the Edge page makes, on the SHIPPED config.

The page's caveat block carries hard numbers -- delisted counts, worst single
positions, the share of positions affected by delisting. All of them were
measured on the OLD 42-day grid, which had 164 rebalances. The monthly grid has
330. Any per-position statistic is therefore suspect until recomputed.
"""
import pickle, numpy as np, pandas as pd
import config, edge_lib as E

art = pickle.load(open(config.ARTIFACT_DIR / "backtest_data.pkl", "rb"))
n_tot = len(art)
dead = [ck for ck, b in art.items() if b.get("meta", {}).get("trading_status") == "Inactive"]
print(f"CLAIM 'delisted companies'      -> artifact has {len(dead):,} of {n_tot:,} "
      f"({len(dead)/n_tot*100:.0f}%)   page says 7,826 / 64%")

pan = E.load_edge_panel(**E.clock_spec(E.EDGE_SPEC["hold"]))
S = E.EDGE_SPEC
sig = tuple(sorted(S["signal"].items()))
_, holds, _ = E._select_holds_cached(
    S["hold"], 0, S["n"], sig, S["mcap_floor"], S["corr_cap"], S["corr_lookback"],
    0.0, S["growth_thresh"], False, S["sector_cap"], False, S["rebal_months"])

# every position actually taken, with its realized window return
rows = []
for i, cks in enumerate(holds):
    df = pan.panels[i].set_index("company_key")
    for ck in cks:
        if ck in df.index:
            rows.append((float(df.loc[ck, "fwd_ret"]), ck, pan.bdates[i]))
rows = [r for r in rows if r[0] == r[0]]
rows.sort()
n_pos = len(rows)
print(f"\npositions taken                 -> {n_pos:,} across {len(holds)} rebalances "
      f"(page's 1.1% figure was measured on 1,640)")
print("CLAIM worst single positions    -> page says -63%, -62%, -62%")
for r, ck, d in rows[:5]:
    tk = art[ck]["meta"]["ticker"]; st = art[ck]["meta"]["trading_status"]
    print(f"    {r*100:7.1f}%  {tk:<6} opened {d.date()}  {'DELISTED' if st=='Inactive' else 'active'}")
worst10 = rows[:10]
n_dead10 = sum(1 for _, ck, _ in worst10 if art[ck]["meta"]["trading_status"] == "Inactive")
print(f"CLAIM 'five of the ten worst later delisted' -> {n_dead10} of the 10 worst are delisted")

# how many positions are truncated-at-delisting (the "1.1%" claim)
spx_end = np.datetime64(pd.Timestamp(E.D.benchmarks()['SP500'].dropna().index[-1]), 'ns')
trunc = 0
for i, cks in enumerate(holds):
    for ck in cks:
        pr = art[ck].get("prices")
        if pr is None or len(pr) == 0: continue
        last = np.datetime64(pd.Timestamp(pr.index[-1]), 'ns')
        nxt = pan.next_dates[i] if pan.next_dates is not None else None
        if nxt is not None and pd.notna(nxt) and last < np.datetime64(pd.Timestamp(nxt), 'ns') \
           and last < spx_end - np.timedelta64(7, 'D'):
            trunc += 1
print(f"CLAIM 'capped at 1.1% of positions' -> {trunc} of {n_pos:,} = {trunc/max(n_pos,1)*100:.2f}%")

# names that ever cleared the floor
ever = set()
for df in pan.panels:
    ever |= set(df.loc[df["pit_mcap"] >= S["mcap_floor"], "company_key"])
ever_dead = sum(1 for ck in ever if art.get(ck, {}).get("meta", {}).get("trading_status") == "Inactive")
print(f"CLAIM '590 once cleared the $10B floor' -> {ever_dead} delisted names cleared it "
      f"(of {len(ever):,} that ever did)")
