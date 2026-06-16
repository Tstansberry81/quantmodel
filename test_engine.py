from qmodel import engine

print("=== META ===", engine.meta())

p = engine.compute_portfolio({})
print("\n=== PORTFOLIO (live) ===")
print("ok:", p.get("ok"), "| regime:", p.get("regime", {}).get("current"),
      "| n_eligible:", p.get("n_eligible"), "| ranked:", len(p.get("ranked", [])))
# verify NO inactive names leaked into the live ranked table
df = engine.factor_dataframe()
n_inactive_total = int((df["trading_status"] != "Active").sum()) if "trading_status" in df else 0
live_keys = {r["company_key"] for r in p["ranked"]}
inactive_keys = set(df[df["trading_status"] != "Active"]["company_key"]) if "trading_status" in df else set()
print(f"inactive in pool: {n_inactive_total} | inactive leaked into live ranked: {len(live_keys & inactive_keys)}")
print("holdings:", [h["ticker"] for h in p["portfolio"]])

print("\n=== BACKTEST (MAX, point-in-time universe) ===")
b = engine.compute_backtest({}, window="MAX")
print("ok:", b.get("ok"), "rebalances:", b.get("n_rebalances"), "period:", b.get("period"))
print("pool:", b.get("pool"))
print("IC:", b.get("ic"))
print("perf model:", b["performance"]["model"])
print("perf sp500:", b["performance"]["sp500"])
