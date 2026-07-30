"""Where the cold backtest time ACTUALLY goes. Three components, measured."""
import time, pickle, numpy as np, pandas as pd
import config, edge_lib as E

S = E.EDGE_SPEC; SIG = tuple(sorted(S["signal"].items()))
path = config.CACHE_DIR / "panel_h21_u1000_o0_m1.pkl"
print(f"panel file: {path.stat().st_size/1e6:.0f} MB")

t0 = time.time()
with open(path, "rb") as fh:
    blob = pickle.load(fh)
t_unpickle = time.time() - t0
pan = blob["panel"]
print(f"1. UNPICKLE          {t_unpickle:7.1f}s")

t0 = time.time()
holds, turn = E._select_holds(pan, S["n"], dict(SIG), S["mcap_floor"], S["corr_cap"],
                              S["corr_lookback"], 0.0, S["growth_thresh"],
                              fcf_screen=False, sector_cap=S["sector_cap"],
                              fcf_positive=S["fcf_positive"],
                              debt_ebitda_max=S["debt_ebitda_max"])
t_select = time.time() - t0
print(f"2. SELECTION         {t_select:7.1f}s   ({len(holds)} rebalances)")

M = E.D.daily_return_matrix(); RD = E._ma200_daily_state()
t0 = time.time()
E._sleeve_daily(pan, holds, turn, M, S["hold"], S["regime_expo"], S["cost_bps"], RD)
t_sleeve = time.time() - t0
print(f"3. SLEEVE (x2/run)   {t_sleeve:7.1f}s")
print(f"   TOTAL COLD        {t_unpickle+t_select+2*t_sleeve:7.1f}s")

# what makes the unpickle slow? count the Python objects it must rebuild.
df = pan.panels[0]
obj_cols = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype) == "str"]
n_obj = sum(len(p) for p in pan.panels) * len(obj_cols)
print(f"\nobject-dtype columns per panel: {obj_cols}")
print(f"string objects the unpickle must rebuild: {n_obj:,}")
print(f"  -> {t_unpickle/max(n_obj,1)*1e6:.2f} microseconds each")
