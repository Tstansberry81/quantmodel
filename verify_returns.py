"""Independent cross-check of the model's 1Y and 5Y returns.

Model returns come from fiscal.ai split-adjusted prices. We re-derive the same
windows from Yahoo Finance (yfinance) -- a fully independent source -- over the
*identical* start/end dates, comparing split-adjusted price return like-for-like,
and also reporting dividend-adjusted total return for context.
"""
import sys, warnings
warnings.filterwarnings("ignore")
import pandas as pd
import yfinance as yf

from qmodel import engine, factors as factmod
import config

OFFSETS = {"12M": 252, "5Y": 1260}

def yahoo_sym(ticker):
    return ticker.replace(".", "-")  # BRK.A -> BRK-A etc.

def nearest(series, date):
    """value in series on/just-before date."""
    s = series[series.index <= date]
    return (s.index[-1], float(s.iloc[-1])) if len(s) else (None, None)

def main():
    p = engine.compute_portfolio({})
    top = p["ranked"][:10]
    print(f"Top 10 by composite (universe = {p['meta'].get('n_names')} names)\n")

    # preload yahoo data
    print("Downloading Yahoo Finance history (independent source)...\n")
    rows = []
    for r in top:
        ck, tk = r["company_key"], r["ticker"]
        fser = factmod.price_series(ck).dropna()
        if len(fser) < 1300:
            print(f"  {tk}: insufficient fiscal history ({len(fser)} days), skipping 5Y")
        ysym = yahoo_sym(tk)
        ydf = yf.download(ysym, period="7y", interval="1d",
                          progress=False, auto_adjust=False)
        if ydf is None or len(ydf) == 0:
            print(f"  {tk}: NO Yahoo data for {ysym}"); continue
        yclose = ydf["Close"];  yadj = ydf["Adj Close"]
        if isinstance(yclose, pd.DataFrame): yclose = yclose.iloc[:, 0]
        if isinstance(yadj, pd.DataFrame): yadj = yadj.iloc[:, 0]
        yclose.index = pd.to_datetime(yclose.index); yadj.index = pd.to_datetime(yadj.index)

        end_date = fser.index[-1]
        for label, off in OFFSETS.items():
            if len(fser) <= off:
                continue
            start_date = fser.index[-1 - off]
            model_ret = fser.iloc[-1] / fser.iloc[-1 - off] - 1.0

            # Yahoo over identical dates
            d0, c0 = nearest(yclose, start_date); d1, c1 = nearest(yclose, end_date)
            _, a0 = nearest(yadj, start_date); _, a1 = nearest(yadj, end_date)
            y_price = (c1 / c0 - 1.0) if c0 else None
            y_total = (a1 / a0 - 1.0) if a0 else None
            diff = (model_ret - y_price) if y_price is not None else None
            rows.append({
                "ticker": tk, "horizon": label,
                "model_%": round(model_ret*100, 1),
                "yahoo_price_%": None if y_price is None else round(y_price*100, 1),
                "diff_pp": None if diff is None else round(diff*100, 2),
                "yahoo_total_%": None if y_total is None else round(y_total*100, 1),
                "window": f"{start_date.date()}->{end_date.date()}",
                "y_window": f"{d0.date() if d0 else '?'}->{d1.date() if d1 else '?'}",
            })

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    for h in ["12M", "5Y"]:
        sub = df[df["horizon"] == h]
        if sub.empty: continue
        print(f"\n===== {h} =====")
        print(sub[["ticker","model_%","yahoo_price_%","diff_pp","yahoo_total_%","window","y_window"]]
              .to_string(index=False))
        d = sub["diff_pp"].dropna().abs()
        if len(d):
            print(f"  abs diff vs Yahoo price-return: mean {d.mean():.2f}pp, max {d.max():.2f}pp")

if __name__ == "__main__":
    sys.exit(main())
