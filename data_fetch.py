"""Pulls AAPL price data plus a handful of free macro/cross-asset series from Yahoo Finance."""
import pandas as pd
import yfinance as yf

TICKER = "AAPL"

# Free proxies for the macro/cross-asset data that actually move AAPL:
#   ^GSPC  S&P 500          - overall market beta
#   QQQ    Nasdaq-100 ETF   - tech-sector beta (AAPL is ~8% of it)
#   ^VIX   CBOE volatility  - risk appetite
#   ^TNX   10Y Treasury yld - discount-rate proxy, hits growth-stock valuations
#   DX-Y.NYB  US Dollar idx - AAPL earns ~60% revenue overseas, dollar strength is a headwind
#   SMH    Semiconductor ETF - supply-chain/component-cost proxy
MACRO_TICKERS = {
    "spx": "^GSPC",
    "qqq": "QQQ",
    "vix": "^VIX",
    "tnx": "^TNX",
    "dxy": "DX-Y.NYB",
    "smh": "SMH",
}


def fetch_all(period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Fetch AAPL OHLCV plus macro closes, aligned on trading date."""
    aapl = yf.download(TICKER, period=period, interval=interval, progress=False)
    aapl.columns = aapl.columns.get_level_values(0)  # drop the ticker level
    aapl = aapl.rename(columns=str.lower)

    frame = aapl[["open", "high", "low", "close", "volume"]].copy()

    for name, symbol in MACRO_TICKERS.items():
        raw = yf.download(symbol, period=period, interval=interval, progress=False)
        raw.columns = raw.columns.get_level_values(0)
        frame[name] = raw["Close"].reindex(frame.index).ffill()

    frame = frame.dropna()
    return frame


if __name__ == "__main__":
    df = fetch_all(period="1y")
    print(df.tail())
    print(f"\n{len(df)} rows, columns: {list(df.columns)}")
