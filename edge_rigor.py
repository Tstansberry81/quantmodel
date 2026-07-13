"""The Edge -- overfitting-adjusted rigor metrics (GPT audit 3.1 / 6).

Puts honest confidence bounds on the product's numbers, accounting for the fact
that we screened ~150 configs/variants across this campaign:

  * Newey-West t-stat  -- significance of the excess vs S&P, corrected for the
    autocorrelation that overlapping/held positions induce.
  * Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014) -- the probability the
    true Sharpe > 0 AFTER deflating for the number of trials and for non-normal
    (skewed / fat-tailed) returns. DSR > 0.95 = survives the multiple-testing bar.
  * PBO (Probability of Backtest Overfitting, CSCV) -- across a grid of candidate
    configs, how often the in-sample winner underperforms out-of-sample. > 0.5 =
    the selection process is overfit.

Operates on the ACTUAL product return series (single-sleeve period returns for the
significance tests; the staggered/daily product is what the site reports). No
edge_lib changes. Run: .venv/Scripts/python.exe edge_rigor.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import itertools, os
import numpy as np, pandas as pd
from scipy.stats import norm, skew, kurtosis
import edge_lib as E

S = E.EDGE_SPEC
EULER = 0.5772156649015329


def _edge_periods(n, mix):
    """Single-sleeve per-period NET returns + aligned S&P (for the stats tests)."""
    bdates, gross, net, turn, spxf, ndxf, holds = E._edge_full(
        S["hold"], n, S["mcap_floor"], S["corr_cap"], S["corr_lookback"],
        S["regime_expo"], S["cost_bps"], (("accel", 1.0),), mix, S["growth_thresh"])
    return np.asarray(net, float), np.asarray(spxf, float)


def newey_west_t(r, L=None):
    """t-stat for H0: mean(r)=0 with Newey-West (HAC) standard error."""
    r = np.asarray(r, float); r = r[np.isfinite(r)]; T = len(r); m = r.mean()
    if L is None:
        L = int(np.floor(4 * (T / 100) ** (2 / 9)))
    e = r - m; var = np.mean(e * e)
    for l in range(1, L + 1):
        var += 2 * (1 - l / (L + 1)) * np.mean(e[l:] * e[:-l])
    se = np.sqrt(var / T)
    return (m / se if se > 0 else 0.0), T, L


def deflated_sharpe(r, n_trials, var_sr_trials):
    """Deflated Sharpe Ratio. r = per-period returns; var_sr_trials = variance of
    per-period Sharpes across the trials we ran."""
    r = np.asarray(r, float); r = r[np.isfinite(r)]; T = len(r)
    sr = r.mean() / r.std(ddof=1)
    g3 = float(skew(r)); g4 = float(kurtosis(r, fisher=False))
    sr_star = np.sqrt(var_sr_trials) * (
        (1 - EULER) * norm.ppf(1 - 1.0 / n_trials)
        + EULER * norm.ppf(1 - 1.0 / (n_trials * np.e)))
    den = np.sqrt(max(1e-9, 1 - g3 * sr + (g4 - 1) / 4.0 * sr ** 2))
    dsr = float(norm.cdf((sr - sr_star) * np.sqrt(T - 1) / den))
    return dsr, sr, float(sr_star), g3, g4, T


def pbo_cscv(R, S_blocks=8):
    """Probability of Backtest Overfitting via combinatorially symmetric CV.
    R = T x N matrix of per-period returns (N candidate configs)."""
    T, N = R.shape
    b = T // S_blocks
    blocks = [np.arange(i * b, (i + 1) * b if i < S_blocks - 1 else T) for i in range(S_blocks)]
    sr = lambda X: X.mean(0) / (X.std(0, ddof=1) + 1e-12)
    logits = []
    for combo in itertools.combinations(range(S_blocks), S_blocks // 2):
        is_idx = np.concatenate([blocks[i] for i in combo])
        oos_idx = np.concatenate([blocks[i] for i in range(S_blocks) if i not in combo])
        n_star = int(np.argmax(sr(R[is_idx])))               # in-sample winner
        oos_sr = sr(R[oos_idx])
        rank = (np.argsort(np.argsort(oos_sr))[n_star] + 1) / (N + 1)  # OOS rank in (0,1)
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(np.log(rank / (1 - rank)))
    logits = np.array(logits)
    return float(np.mean(logits <= 0)), logits      # PBO = P(winner lands in OOS lower half)


def main():
    n = 10
    net, spx = _edge_periods(n, S["growth_mix"])
    excess = net - spx
    ppy = 252.0 / S["hold"]

    print("=" * 78)
    print(f"RIGOR METRICS — product config (n={n}, mix={S['growth_mix']}, hold={S['hold']})")
    print("=" * 78)

    # ---- Newey-West t on the excess vs S&P ----
    t_raw = net.mean() and (excess.mean() / (excess.std(ddof=1) / np.sqrt(len(excess))))
    t_nw, T, L = newey_west_t(excess)
    print(f"\nExcess vs S&P (per {S['hold']}d period): mean {excess.mean()*100:+.2f}%/period "
          f"(~{excess.mean()*ppy*100:+.1f}%/yr)")
    print(f"  t-stat  raw {t_raw:+.2f}   Newey-West(L={L}) {t_nw:+.2f}   "
          f"({'significant' if abs(t_nw) > 2 else 'NOT significant'} at |t|>2)")

    # ---- trial-Sharpe dispersion from the 120-config sweep (if present) ----
    csv = os.path.join(os.environ.get("EDGE_SWEEP_OUT",
          "C:/Users/WYATTK~1/AppData/Local/Temp/claude/C--Users-Wyatt-Kelly-quant-model/"
          "6f3ae0f6-272c-4c22-8e42-3ce1430e536d/scratchpad"), "edge_configsweep.csv")
    if os.path.exists(csv):
        g = pd.read_csv(csv); g = g[g.window == "MAX"]
        sr_ann = g["sharpe"].dropna().to_numpy()
        var_sr = float(np.var(sr_ann / np.sqrt(ppy)))        # -> per-period Sharpe variance
        n_trials = int(len(g)) + 30                          # grid + other campaign variants
    else:
        var_sr = float(np.var(np.array([0.9, 1.0, 1.1, 0.8, 1.2]) / np.sqrt(ppy)))
        n_trials = 150
    print(f"\nTrials screened this campaign ~{n_trials}; per-period Sharpe dispersion "
          f"Var(SR)={var_sr:.4f}")

    # ---- Deflated Sharpe Ratio ----
    dsr, sr, sr_star, g3, g4, T = deflated_sharpe(net, n_trials, var_sr)
    print(f"\nDeflated Sharpe Ratio:")
    print(f"  observed per-period SR {sr:.3f} (~annual {sr*np.sqrt(ppy):.2f}); skew {g3:+.2f}, kurt {g4:.1f}")
    print(f"  benchmark SR* (max under {n_trials} nulls) {sr_star:.3f}")
    print(f"  DSR = {dsr:.3f}   ({'PASSES' if dsr > 0.95 else 'FAILS'} the >0.95 overfitting-adjusted bar)")

    # ---- PBO over the n x mix grid ----
    grid = [(nn, mm) for nn in (5, 6, 7, 8, 9, 10) for mm in (0.0, 0.5, 0.75, 1.0)]
    cols = []
    for (nn, mm) in grid:
        r, _ = _edge_periods(nn, mm)
        cols.append(r[-min(len(r), 124):])
    Tm = min(len(c) for c in cols)
    R = np.column_stack([c[-Tm:] for c in cols])
    pbo, logits = pbo_cscv(R, S_blocks=8)
    print(f"\nPBO (CSCV over {R.shape[1]} n×mix configs, {R.shape[0]} periods):")
    print(f"  PBO = {pbo:.2f}   ({'OVERFIT risk high' if pbo > 0.5 else 'acceptable'} — <0.5 is good)")
    print("\nDONE.")


if __name__ == "__main__":
    main()
