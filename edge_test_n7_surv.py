"""The Edge -- final gate for the n=7 candidate: survivorship at low n.

The 'full-vs-active gap ~ 0' finding was measured on 10-20-name books. A 7-name
book concentrates harder on exactly the hot names that blow up and vanish from a
survivorship-biased universe, so the gap must be re-measured at n=7 before the
candidate can be recommended. Also reports the share of basket slots that go to
now-inactive names at each n (concentration of dead-name exposure).

Method mirrors edge_mix_validate.py: full pool vs active-only ("today's
survivors"); the gap is a LOWER BOUND on survivorship inflation because the
universe is missing ~89% of truly-dead names either way.

Full spec net (floor + corr-cap 0.5 + regime 25% + 10bps + mix 75) at hold=42.
No edge_lib/qmodel changes. Not committed.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import edge_lib as E
import edge_test_skip as SK
from edge_test_skip import run_spec
import edge_data as engine   # self-contained Edge data layer (was qmodel.engine)

W = {"accel": 1.0}
data = engine._load_bt_data()
INACTIVE = {ck for ck, b in data.items()
            if b.get("meta", {}).get("trading_status", "Active") != "Active"}


def run(n, active_only):
    """Full-spec net run; active_only drops now-inactive names from the pool
    (the 'today's survivors' book = the MOST survivorship-biased version)."""
    SK.NBASKET = n; SK.GROWTH_MIX = 0.75
    pan = E.load_edge_panel(hold=42)
    holds, turn, prev = [], [], None
    for i, df in enumerate(pan.panels):
        d = df[df["pit_mcap"] >= SK.MCAP_FLOOR]
        if active_only:
            d = d[~d["company_key"].isin(INACTIVE)]
        cks = E._blend_select(d, pan.bdates[i], n, W, SK.CORR_CAP, SK.CORR_LB,
                              SK.GROWTH_MIX, SK.GROWTH_THRESH)
        cur = set(cks)
        turn.append(1 - len(cur & prev) / len(cur) if prev and cur else (1.0 if cur else 0.0))
        prev = cur; holds.append(cks)
    gross = E.period_returns(pan, holds)
    gross = np.where(pan.ma200_on, gross, SK.REGIME_EXPO * gross + (1 - SK.REGIME_EXPO) * pan.rf_per)
    net = gross - (SK.COST_BPS / 1e4) * np.array(turn)
    slot_share = np.mean([len(set(h) & INACTIVE) / max(len(h), 1) for h in holds])
    return pan, net, slot_share


def main():
    print(f"inactive names in pool: {len(INACTIVE)} of {len(data)}")
    print(f"\n{'':<10}{'full pool':>22}{'active-only':>22}{'gap (full - active)':>22}")
    for n in (7, 10):
        pan, full, share_f = run(n, active_only=False)
        _, act, share_a = run(n, active_only=True)
        ppy = pan.ppy; T = len(full)
        print(f"\nn={n}  (inactive slot share, full pool: {share_f*100:.1f}%)")
        for wn, yrs in (("1Y", 1), ("2Y", 2), ("5Y", 5), ("MAX", None)):
            k = T if yrs is None else min(int(round(yrs * ppy)), T)
            cf, _, shf = E.perf(full[-k:], ppy)
            ca, _, sha = E.perf(act[-k:], ppy)
            print(f"  {wn:<6}{cf*100:14.1f}%/{shf:4.2f}{ca*100:15.1f}%/{sha:4.2f}"
                  f"{(cf-ca)*100:+18.1f}%")
    print("\nReading it: gap>0 = dead names HELPED the full-pool book (bias risk if they'd")
    print("been missing); gap<~0 = selection doesn't lean on dead names. Either way the")
    print("~89%-missing-dead-names haircut (~3%/yr) still applies ON TOP.")


if __name__ == "__main__":
    main()
