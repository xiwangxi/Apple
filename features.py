"""Turns raw price/macro columns into model features. All features use only
information available as of the close of day t, so they can legitimately
predict day t+1's return without lookahead leakage."""
import numpy as np
import pandas as pd


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    close = df["close"]
    ret1 = close.pct_change()

    # --- price/technical features ---
    out["ret_1d"] = ret1
    out["ret_5d"] = close.pct_change(5)
    out["ret_10d"] = close.pct_change(10)
    out["sma5_gap"] = close / close.rolling(5).mean() - 1
    out["sma20_gap"] = close / close.rolling(20).mean() - 1
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    out["macd"] = macd
    out["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    out["rsi14"] = _rsi(close)
    out["vol10"] = ret1.rolling(10).std()
    out["vol20"] = ret1.rolling(20).std()
    out["volume_chg"] = df["volume"].pct_change(5)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    out["bb_width"] = (4 * bb_std) / bb_mid

    # --- cross-asset / macro features ---
    out["spx_ret1"] = df["spx"].pct_change()
    out["qqq_ret1"] = df["qqq"].pct_change()
    out["smh_ret1"] = df["smh"].pct_change()
    out["vix_level"] = df["vix"]
    out["vix_chg"] = df["vix"].pct_change()
    out["tnx_chg"] = df["tnx"].diff()  # yield is already in %, diff not pct_change
    out["dxy_ret1"] = df["dxy"].pct_change()
    # relative strength vs the tech sector it trades inside
    out["excess_vs_qqq"] = ret1 - out["qqq_ret1"]

    # --- targets (next-day, so must NOT be used as a feature) ---
    out["target_ret_next"] = ret1.shift(-1)
    out["target_up_next"] = (out["target_ret_next"] > 0).astype(int)

    # Drop rows with NaN features (rolling-window warmup period), but keep the
    # final row even though its target is unknown -- that's the row we need
    # for actually predicting tomorrow.
    feature_cols = [c for c in out.columns if not c.startswith("target_")]
    out = out.dropna(subset=feature_cols)
    return out


if __name__ == "__main__":
    from data_fetch import fetch_all

    raw = fetch_all(period="2y")
    feat = build_features(raw)
    print(feat.tail())
    print(f"\n{len(feat)} rows, {feat.shape[1]} columns")
