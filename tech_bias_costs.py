"""Transaction-cost stress test for the momentum/growth selection strategy.

Question: the gross alpha after Carhart + sector-tilt controls is already weak
(~+6.3%/yr, t=1.71 full-sample, not significant in subsamples). Do realistic
trading costs kill it?

We:
  1. report average annual one-way turnover,
  2. for one-way cost levels {0,5,10,20,35,50} bps, compute NET CAGR / Sharpe
     and the NET alpha + t-stat from the FULL controls regression (Carhart +
     sector tilts),
  3. find the break-even cost (where alpha t drops below 1.65 and below ~0).
"""
import numpy as np
import tech_bias_lib as tbl
from qmodel.equations import default_params

PPY = tbl.PPY


def main():
    pan = tbl.load_panel()
    params = default_params()

    # full control set: Carhart (MKT/SMB/HML/UMD) + sector tilts
    carhart = tbl.carhart_factors(pan)
    tilts = tbl.sector_tilts(pan)
    controls = {**carhart, **tilts}

    # 1) average annual one-way turnover (period turnover annualized by PPY)
    base = tbl.model_returns(pan, params, cost_bps=0.0)
    avg_period_to = float(np.nanmean(base["turnover"]))
    ann_turnover = avg_period_to * PPY
    print(f"Periods: {pan.T}   PPY={PPY:.2f}")
    print(f"Avg one-way turnover: {avg_period_to*100:.1f}%/rebalance "
          f"-> {ann_turnover*100:.0f}%/yr (one-way)\n")

    # 2) cost sweep
    cost_levels = [0, 5, 10, 20, 35, 50]
    rows = []
    print("Full-controls regression (Carhart + sector tilts), NET of costs:")
    for bps in cost_levels:
        mr = tbl.model_returns(pan, params, cost_bps=float(bps))
        cagr, mdd, shp = tbl.perf(mr["ret"])
        a_ann, a_t, _ = tbl.attribution(f"cost={bps}bps", mr["ret"], controls,
                                        verbose=False)
        rows.append((bps, cagr, shp, a_ann, a_t))

    print("\n cost_bps | net CAGR | net Sharpe | net alpha %/yr | alpha t")
    print(" ---------+----------+------------+----------------+--------")
    for bps, cagr, shp, a_ann, a_t in rows:
        print(f"   {bps:>4}   |  {cagr*100:6.2f}% |    {shp:5.2f}   |"
              f"    {a_ann*100:+7.2f}     | {a_t:+6.2f}")

    # 3) break-even: linear interpolation of alpha-t vs bps crossing 1.65 and 0
    bps_arr = np.array([r[0] for r in rows], float)
    t_arr = np.array([r[4] for r in rows], float)

    def crossing(level):
        for i in range(len(t_arr) - 1):
            t0, t1 = t_arr[i], t_arr[i + 1]
            if (t0 - level) * (t1 - level) <= 0 and t0 != t1:
                b0, b1 = bps_arr[i], bps_arr[i + 1]
                return b0 + (level - t0) * (b1 - b0) / (t1 - t0)
        return None

    be_165 = crossing(1.65)
    be_0 = crossing(0.0)
    print("\nBreak-even (one-way bps):")
    print(f"  alpha t < 1.65 (loses marginal signif): "
          f"{'%.1f bps' % be_165 if be_165 is not None else 'already <1.65 at 0 bps' if t_arr[0] < 1.65 else '>50 bps'}")
    print(f"  alpha t < 0    (alpha turns negative):  "
          f"{'%.1f bps' % be_0 if be_0 is not None else 'n/a in range'}")


if __name__ == "__main__":
    main()
