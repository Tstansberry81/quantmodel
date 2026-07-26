"""Same model, different data: fiscal.ai vs Sharadar.

The Edge's backtest excess has never been trustworthy in LEVEL because the
fiscal.ai artifact is survivorship-inflated: 1,111 names, 111 delisted, and
every one of those died in 2024 or later -- i.e. no delisted history at all
before 2024, so the backtest literally could not hold a stock through its
death. Sharadar carries 15,634 delisted tickers back to 1998.

This runs the IDENTICAL product spec on both panels. The model is held fixed,
so the difference is data quality alone -- which is the number we could only
bound (+2.2% to +8.5%/yr) before.

Run:  .venv-mac/bin/python vendor_ab.py            # product config
      .venv-mac/bin/python vendor_ab.py 21         # different clock
"""
from __future__ import annotations

import subprocess
import sys

WINDOWS = ("5Y", "10Y", "20Y", "MAX")

# Each vendor runs in its OWN process: edge_data caches the artifact and
# edge_lib caches panels/selection above it, and the vendor is chosen by an
# env var read at import. A subprocess is the only way to be certain that
# variant B cannot see variant A's warm caches.
_CHILD = r'''
import os, sys, json
import edge_data as D
import edge_lib as E
hold = int(sys.argv[1])
bt = E.run_edge_backtest(window="MAX", spec={"hold": hold, "growth_mix": 0.75, "n": 10})
if not bt.get("ok"):
    print(json.dumps({"ok": False, "reason": bt.get("reason")})); raise SystemExit
rows = {w["w"]: {"cagr": w["cagr"], "sharpe": w["sharpe"], "dd": w["dd"],
                 "excess": w["excess"]} for w in bt["windows"]}
data = D.load_bt_data()
dead = sum(1 for b in data.values()
           if (b.get("meta") or {}).get("trading_status") != "Active")
print(json.dumps({"ok": True, "windows": rows, "n_names": len(data),
                  "n_dead": dead, "source": D.meta().get("source", "fiscal"),
                  "artifact": D.ARTIFACT_FILE}))
'''


def run(artifact: str, hold: int) -> dict:
    import json
    env = dict(**__import__("os").environ, EDGE_ARTIFACT=artifact)
    # Hold the UNIVERSE RULE fixed at the top-N-by-PIT-mcap proxy for both runs.
    # Without this the Sharadar panel would auto-activate pit_universe's
    # SharadarAdapter (its meta says source=sharadar), swapping the universe
    # DEFINITION at the same time as the data -- two changes at once, and the
    # comparison would no longer isolate data quality. The same rule on
    # survivorship-free data is exactly the honest version of that rule.
    env["EDGE_USE_PIT_UNIVERSE"] = "0"
    p = subprocess.run([sys.executable, "-c", _CHILD, str(hold)],
                       capture_output=True, text=True, env=env)
    line = [l for l in p.stdout.splitlines() if l.startswith("{")]
    if not line:
        return {"ok": False, "reason": (p.stderr or p.stdout)[-400:]}
    return json.loads(line[-1])


def main(argv: list[str]) -> int:
    hold = int(argv[1]) if len(argv) > 1 else 42
    print(f"Edge product spec — {hold}d clock / 10 names / 75% growth mix, "
          f"net of 10bps, t+1 execution\n")
    results = {}
    for label, art in (("fiscal.ai", "backtest_data.fiscal.pkl"),
                       ("Sharadar", "backtest_data.pkl")):
        r = run(art, hold)
        results[label] = r
        if not r.get("ok"):
            print(f"{label:10} FAILED: {r.get('reason')}")
            continue
        # Label by the ARTIFACT actually loaded, not meta.json's `source`:
        # meta.json describes whichever panel was built last, so it reports the
        # same vendor for both runs and hides which file each one really read.
        print(f"{label:10} artifact={r['artifact']:26} names={r['n_names']:,} "
              f"delisted={r['n_dead']:,} ({r['n_dead']/max(r['n_names'],1)*100:.0f}%)")
    print()
    header = f"{'window':<8}" + "".join(f"{lbl:>26}" for lbl in results)
    print(header)
    print("-" * len(header))
    for w in WINDOWS:
        cells = ""
        for lbl, r in results.items():
            if not r.get("ok") or w not in r["windows"]:
                cells += f"{'--':>26}"; continue
            d = r["windows"][w]
            cells += f"{d['cagr']*100:>10.1f}% xs{d['excess']*100:>+7.1f}% Sh{d['sharpe']:>5.2f}"
        print(f"{w:<8}{cells}")

    ok = [r for r in results.values() if r.get("ok")]
    if len(ok) == 2:
        f, s = results["fiscal.ai"], results["Sharadar"]
        print("\nSurvivorship cost of the fiscal.ai panel (fiscal excess - Sharadar excess):")
        for w in WINDOWS:
            if w in f["windows"] and w in s["windows"]:
                gap = (f["windows"][w]["excess"] - s["windows"][w]["excess"]) * 100
                print(f"  {w:<5} {gap:+.1f} pts/yr")
    print("\nDisclosures — data: Sharadar SEP/SF1 point-in-time (ART, datekey) vs "
          "fiscal.ai; universe: delisted INCLUDED on the Sharadar panel, "
          "effectively survivors-only pre-2024 on fiscal.ai; filing lag: +1 "
          f"trading day; rebalance: {hold} trading days.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
