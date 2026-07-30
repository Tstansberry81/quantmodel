"""Where does a backtest actually spend its time? Measure before optimizing."""
import cProfile, pstats, io, time
import edge_lib as E

# warm the panel first so we profile the BACKTEST, not the unpickle
t0 = time.time(); E.load_edge_panel(**E.clock_spec(E.EDGE_SPEC["hold"]))
print(f"panel load: {time.time()-t0:.1f}s (excluded from the profile below)")

E._edge_daily.cache_clear(); E._select_holds_cached.cache_clear()
pr = cProfile.Profile(); pr.enable()
t0 = time.time()
bt = E.run_edge_backtest("MAX")
el = time.time() - t0
pr.disable()
print(f"run_edge_backtest (cold caches): {el:.1f}s\n")
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(18)
for ln in s.getvalue().splitlines():
    if "edge_lib" in ln or "edge_data" in ln or "{method" in ln or "cumtime" in ln:
        print(ln[:150])
