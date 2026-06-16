"""Hidden Markov Model market-regime detection.

We fit a Gaussian HMM on S&P 500 weekly log-returns to infer latent regimes,
then label states bull / neutral / bear by their mean return. The current
regime drives the gold sleeve weight and is reported as a confidence overlay.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _label_states(means: np.ndarray) -> dict[int, str]:
    order = np.argsort(means)              # ascending by mean return
    labels = {}
    if len(order) == 1:
        labels[order[0]] = "neutral"
    elif len(order) == 2:
        labels[order[0]] = "bear"; labels[order[1]] = "bull"
    else:
        labels[order[0]] = "bear"
        labels[order[-1]] = "bull"
        for i in order[1:-1]:
            labels[i] = "neutral"
    return labels


def fit_regime(sp500_prices: pd.Series, n_states: int = 3, seed: int = 42) -> dict:
    """Return current regime, per-date regime series, and state stats."""
    prices = sp500_prices.dropna()
    if len(prices) < 200:
        return {"ok": False, "current": "neutral", "reason": "insufficient history"}

    weekly = prices.resample("W").last().dropna()
    rets = np.log(weekly).diff().dropna()
    X = rets.values.reshape(-1, 1)

    try:
        import warnings, logging
        logging.getLogger("hmmlearn").setLevel(logging.ERROR)  # silence EM convergence chatter
        from hmmlearn.hmm import GaussianHMM
        model = GaussianHMM(n_components=n_states, covariance_type="full",
                            n_iter=200, tol=1e-3, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # benign EM convergence chatter
            model.fit(X)
        hidden = model.predict(X)
        means = model.means_.flatten()
    except Exception as e:
        return {"ok": False, "current": "neutral", "reason": f"hmm failed: {e}"}

    labels = _label_states(means)
    regime_series = pd.Series([labels[h] for h in hidden], index=rets.index)
    current = regime_series.iloc[-1]

    # transition-implied persistence + state annualized stats
    state_stats = {}
    for s in range(n_states):
        mask = hidden == s
        if mask.sum() == 0:
            continue
        state_stats[labels[s]] = {
            "ann_return": float(means[s] * 52),
            "ann_vol": float(np.sqrt(model.covars_[s].flatten()[0] * 52)),
            "weeks": int(mask.sum()),
        }
    # start of the current contiguous regime run
    since = regime_series.index[-1]
    for i in range(len(regime_series) - 1, -1, -1):
        if regime_series.iloc[i] == current:
            since = regime_series.index[i]
        else:
            break

    return {
        "ok": True,
        "current": current,
        "n_states": n_states,
        "regime_series": regime_series,
        "state_stats": state_stats,
        "current_since": str(since.date()),
    }


def gold_weight_for(regime: str, settings: dict) -> float:
    return {
        "bull": settings["gold_weight_bull"],
        "neutral": settings["gold_weight_neutral"],
        "bear": settings["gold_weight_bear"],
    }.get(regime, settings["gold_weight_neutral"])
