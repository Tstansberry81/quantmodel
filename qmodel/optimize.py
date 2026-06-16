"""Portfolio weight schemes, including mean-variance optimization.

All return a long-only weight vector summing to 1, with a per-name cap to avoid
concentration. Covariance is estimated from trailing returns; expected returns
for max-Sharpe use the composite score as a relative expected-return proxy.
"""
from __future__ import annotations
import numpy as np


def equal_weights(n: int) -> np.ndarray:
    return np.ones(n) / n if n > 0 else np.array([])


def inverse_vol_weights(cov: np.ndarray) -> np.ndarray:
    vol = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    w = 1.0 / vol
    return w / w.sum()


def _solve(objective, n: int, wmax: float) -> np.ndarray:
    from scipy.optimize import minimize
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bnds = [(0.0, wmax)] * n
    x0 = np.ones(n) / n
    res = minimize(objective, x0, method="SLSQP", bounds=bnds, constraints=cons,
                   options={"maxiter": 200, "ftol": 1e-9})
    w = res.x if res.success else x0
    w = np.clip(w, 0, None)
    return w / w.sum() if w.sum() > 0 else x0


def min_variance_weights(cov: np.ndarray, wmax: float = 0.30) -> np.ndarray:
    n = len(cov)
    if n == 1:
        return np.array([1.0])
    return _solve(lambda w: float(w @ cov @ w), n, wmax)


def max_sharpe_weights(mu: np.ndarray, cov: np.ndarray, wmax: float = 0.30,
                       rf: float = 0.0) -> np.ndarray:
    n = len(mu)
    if n == 1:
        return np.array([1.0])
    # shift composite scores to positive expected-return proxies
    mu = np.asarray(mu, float)
    mu = mu - mu.min() + 0.5

    def neg_sharpe(w):
        ret = float(w @ mu)
        vol = float(np.sqrt(max(w @ cov @ w, 1e-12)))
        return -(ret - rf) / vol
    return _solve(neg_sharpe, n, wmax)


def weights_for(scheme: str, mu: np.ndarray, cov: np.ndarray, wmax: float = 0.30) -> np.ndarray:
    """Dispatch by scheme name. Falls back to equal weight on any failure."""
    n = len(mu)
    if n == 0:
        return np.array([])
    try:
        if scheme == "inverse_vol":
            return inverse_vol_weights(cov)
        if scheme == "min_var":
            return min_variance_weights(cov, wmax)
        if scheme == "max_sharpe":
            return max_sharpe_weights(mu, cov, wmax)
    except Exception:
        pass
    return equal_weights(n)
