"""Baseline capture + timing for the sleeve. Run BEFORE and AFTER optimizing.

Correctness first: the daily return array must match to floating-point noise.
A backtest that got faster and moved a number is not an optimization.
"""
import sys, time, pathlib, numpy as np
import edge_lib as E

S = E.EDGE_SPEC; SIG = tuple(sorted(S["signal"].items()))
OUT = pathlib.Path("/private/tmp/claude-501/-Users-travelerstansberry-claude/f0c14cf3-6b39-4eb7-95dd-eefbb259923a/scratchpad/baseline.npz")

t0 = time.time(); pan = E.load_edge_panel(**E.clock_spec(S["hold"])); tl = time.time()-t0
M = E.D.daily_return_matrix(); RD = E._ma200_daily_state()
_, holds, turn = E._select_holds_cached(
    S["hold"], 0, S["n"], SIG, S["mcap_floor"], S["corr_cap"], S["corr_lookback"],
    0.0, S["growth_thresh"], False, S["sector_cap"], False, S["rebal_months"],
    S["fcf_positive"], S["debt_ebitda_max"])
print(f"panel load {tl:.1f}s · selection cached")

reps = 3
t0 = time.time()
for _ in range(reps):
    ser = E._sleeve_daily(pan, holds, turn, M, S["hold"], S["regime_expo"],
                          S["cost_bps"], RD)
el = (time.time()-t0)/reps
arr = ser.dropna().to_numpy()
print(f"_sleeve_daily: {el:.2f}s per call  ({len(arr)} days)")

mode = sys.argv[1] if len(sys.argv) > 1 else "save"
if mode == "save":
    np.savez(OUT, arr=arr); print(f"baseline saved ({len(arr)} days)")
else:
    base = np.load(OUT)["arr"]
    if len(base) != len(arr):
        print(f"FAIL length {len(base)} -> {len(arr)}"); sys.exit(1)
    d = np.abs(base - arr); print(f"max abs diff vs baseline: {d.max():.3e}")
    print("PASS — identical to fp noise" if d.max() < 1e-12 else "FAIL — numbers moved")
