"""Central configuration for the quant model.

Most things you'll want to tweak live here or in qmodel/equations.py.
The equation/threshold values here are the *defaults*; the website's
equation editor can override them at request time.
"""
import os
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent


# ----------------------------------------------------------------------------
# Local secrets
# ----------------------------------------------------------------------------
def load_dotenv() -> None:
    """Load KEY=VALUE pairs from the local .env (gitignored) into the process
    environment. Lives HERE rather than in app.py because every entrypoint --
    the site, build_data.py, and the research scripts -- imports config, so a
    key dropped in .env now works everywhere instead of only in the web app.
    Real environment variables always win (Render sets them for real)."""
    p = ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


load_dotenv()
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
ARTIFACT_DIR = DATA_DIR / "artifacts"
for _d in (DATA_DIR, CACHE_DIR, ARTIFACT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Fiscal.ai API
# ----------------------------------------------------------------------------
# Set via the FISCAL_API_KEY environment variable. Only needed by build_data.py
# when (re)fetching fundamentals — NOT required to serve the site from prebuilt
# artifacts. (The previous hard-coded key was removed for security; rotate it at
# fiscal.ai since it was committed to history.)
FISCAL_API_KEY = os.environ.get("FISCAL_API_KEY", "")
FISCAL_BASE = "https://api.fiscal.ai"
# Free tier is 50 req/min. Stay safely under it. Seconds between calls:
FISCAL_MIN_INTERVAL = 1.25
CACHE_TTL_DAYS = 3  # re-fetch fiscal data older than this

# ----------------------------------------------------------------------------
# Universe
# ----------------------------------------------------------------------------
# companies-list is returned sorted by market cap descending, so "top N" is
# simply the first N rows after filtering to US-listed common stocks.
UNIVERSE_SIZE = int(os.environ.get("UNIVERSE_SIZE", "1000"))
ALLOWED_EXCHANGES = {"NASDAQ", "NYSE"}
ALLOWED_COUNTRY = {"US"}
REQUIRED_DATASETS = {"financials", "stock_prices"}
# Include delisted/inactive names in the pool. STALE NOTE (pre-2026-07): under
# fiscal.ai this only "reduced" survivorship bias, because that vendor carried
# ~111 inactive names and ALL of them died in 2024+. The live artifact is now
# built by sharadar_ingest.py: 12,164 names of which 7,851 are delisted, back to
# 1998 — survivorship bias is genuinely eliminated, not merely reduced.
INCLUDE_INACTIVE = True

# Backtest universe is rebuilt at each rebalance as the top-N names by
# POINT-IN-TIME market cap (not today's), so we hold what was actually large then.
BT_UNIVERSE_SIZE = int(os.environ.get("BT_UNIVERSE_SIZE", "500"))

# ----------------------------------------------------------------------------
# Benchmarks (yfinance — Sharadar's SFP fund table isn't ingested; only these
# few index/ETF series are needed). GOLD is legacy: the Edge holds no gold
# sleeve, it survives because edge_data.benchmarks() still returns the key.
# Returns are measured on a TOTAL-RETURN basis (dividends reinvested), so the
# benchmarks are total-return series for a fair, like-for-like comparison.
# ----------------------------------------------------------------------------
BENCH_SP500 = "^SP500TR"      # S&P 500 Total Return index
BENCH_NASDAQ = "QQQ"          # Nasdaq-100 total return (adj close)
BENCH_SP500_LABEL = "S&P 500 (TR)"
BENCH_NASDAQ_LABEL = "Nasdaq-100 (TR)"
GOLD_SYMBOL = "GLD"           # SPDR Gold Shares (tradeable proxy for gold)
RISK_FREE_ANNUAL = 0.04       # for Sharpe/Sortino

# Price/return basis: TOTAL RETURN (splits + dividends reinvested). Stock prices
# and fundamentals both now come from SHARADAR via sharadar_ingest.py — SEP
# `closeadj` for prices, SF1/ART (filing-dated) plus DAILY for point-in-time
# multiples. The FISCAL_* settings above are retained only for the legacy
# research scripts; the shipped model does not read fiscal.ai.
RETURN_BASIS = "total"

# Portfolio shape
PORTFOLIO_SIZE = 11          # total holdings
N_STOCKS = 10                # equities; plus the gold sleeve (regime-driven)

# Time intervals used everywhere (months). 3,6,12 months + 2,5,10 years.
INTERVALS = {
    "3M": 63, "6M": 126, "12M": 252,
    "2Y": 504, "5Y": 1260, "10Y": 2520,
}
TRADING_DAYS = 252
