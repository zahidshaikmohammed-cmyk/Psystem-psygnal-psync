"""Leakage-free feature engineering, shared by LIVE and HISTORICAL/BACKTEST
modes (same function, same formulas — the constitution requires this).

Every column produced here at row T is computable using only candles
`[0..T]`. This is enforced structurally: every building block (EMA/RSI/
MACD/ATR/ADX/Bollinger — see indicators/common.py) is a forward-only
recursive or rolling-window computation, and the one component that is
naturally lookahead-sensitive (fractal swing structure) is re-derived here
per-row using only swings whose confirmation window has already closed by
T (see `_confirmed_trend_structure_series`). `tests/test_leakage.py`
verifies this directly by asserting that truncating the input frame at T
does not change the feature row computed for T.
"""

from __future__ import annotations

import bisect

import numpy as np
import pandas as pd

from psygnal import config
from psygnal.indicators.adx import adx
from psygnal.indicators.atr import atr
from psygnal.indicators.ema import compute_emas
from psygnal.indicators.macd import macd
from psygnal.indicators.roc import roc
from psygnal.indicators.rsi import rsi
from psygnal.indicators.volume import volume_features
from psygnal.intelligence.candle_features import compute_candle_features
from psygnal.intelligence.structure import classify_trend_structure, find_swings

FEATURE_WINDOWS: tuple[int, ...] = (6, 12, 24)

STRUCTURE_LABEL_TO_SCORE = {"UPTREND": 1.0, "DOWNTREND": -1.0, "RANGE": 0.0, "UNDEFINED": 0.0}


def _rolling_relative_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_centered = x - x_mean
    x_var = float((x_centered**2).sum())

    def _slope(y: np.ndarray) -> float:
        y_mean = y.mean()
        if not np.isfinite(y_mean) or y_mean == 0:
            return 0.0
        raw_slope = float((x_centered * (y - y_mean)).sum() / x_var)
        return raw_slope / abs(y_mean)

    return series.rolling(window=window, min_periods=window).apply(_slope, raw=True)


def _ema_alignment_series(emas: pd.DataFrame) -> pd.Series:
    cols = [f"ema_{p}" for p in config.EMA_PERIODS]
    values = emas[cols].to_numpy()
    n_pairs = len(cols) - 1
    bullish = np.zeros(len(values))
    bearish = np.zeros(len(values))
    valid = ~np.isnan(values).any(axis=1)
    for i in range(n_pairs):
        a, b = values[:, i], values[:, i + 1]
        bullish += (a > b).astype(float)
        bearish += (a < b).astype(float)
    score = (bullish - bearish) / n_pairs
    score[~valid] = np.nan
    return pd.Series(score, index=emas.index)


def _confirmed_trend_structure_series(df: pd.DataFrame, lookback: int = config.SWING_LOOKBACK) -> pd.Series:
    """Trend-structure label as of each row T, using only swing points whose
    confirmation window (`index + lookback`) has already closed by T."""
    n = len(df)
    swing_highs, swing_lows = find_swings(df, lookback=lookback)

    high_confirmed_at = [sp.index + lookback for sp in swing_highs]
    low_confirmed_at = [sp.index + lookback for sp in swing_lows]

    labels = np.full(n, "UNDEFINED", dtype=object)
    for t in range(n):
        h_count = bisect.bisect_right(high_confirmed_at, t)
        l_count = bisect.bisect_right(low_confirmed_at, t)
        visible_highs = swing_highs[:h_count]
        visible_lows = swing_lows[:l_count]
        labels[t] = classify_trend_structure(visible_highs, visible_lows)
    return pd.Series(labels, index=df.index)


def build_feature_frame(df_m5: pd.DataFrame) -> pd.DataFrame:
    """Build the full causal feature frame for an M5 OHLCV DataFrame.

    Used both for a single live snapshot (caller takes `.iloc[-1]`) and for
    historical dataset generation (caller uses every row).
    """
    if df_m5.empty:
        return pd.DataFrame(index=df_m5.index)

    close, high, low = df_m5["close"], df_m5["high"], df_m5["low"]
    candle_feats = compute_candle_features(df_m5)

    out = pd.DataFrame(index=df_m5.index)
    out["return"] = candle_feats["return"]
    out["body_pct"] = candle_feats["body_pct"]
    out["wick_to_body"] = candle_feats["wick_to_body"]
    out["close_location_value"] = candle_feats["close_location_value"]
    out["directional_strength"] = candle_feats["directional_strength"]
    out["range_percentile"] = candle_feats["range_percentile"]

    rsi_series = rsi(close, config.RSI_PERIOD)
    macd_df = macd(close, config.MACD_FAST, config.MACD_SLOW, config.MACD_SIGNAL)
    roc_series = roc(close, config.ROC_PERIOD)
    adx_df = adx(high, low, close, config.ADX_PERIOD)
    atr_series = atr(high, low, close, config.ATR_PERIOD)
    vol_feats = volume_features(df_m5["volume"], config.VOLUME_MA_PERIOD)

    out["rsi_component"] = (rsi_series - 50.0) / 50.0
    out["macd_component"] = np.clip(100 * macd_df["histogram"] / close * 5, -1, 1)
    out["roc_component"] = np.clip(roc_series / 2.0, -1, 1)
    out["momentum_component"] = (out["rsi_component"] + out["macd_component"] + out["roc_component"]) / 3.0
    out["adx_value"] = adx_df["adx"]
    out["di_diff"] = (adx_df["plus_di"] - adx_df["minus_di"]) / 100.0
    out["atr_pct_of_price"] = atr_series / close
    out["relative_volume"] = vol_feats["relative_volume"]
    out["volume_percentile"] = vol_feats["volume_percentile"]

    emas = compute_emas(close)
    out["ema_alignment"] = _ema_alignment_series(emas)
    ema200 = emas["ema_200"]
    out["price_vs_ema200"] = np.where(close > ema200, 1.0, -1.0)
    out.loc[ema200.isna(), "price_vs_ema200"] = np.nan

    structure_labels = _confirmed_trend_structure_series(df_m5)
    out["structure_component"] = structure_labels.map(STRUCTURE_LABEL_TO_SCORE).astype(float)

    for window in FEATURE_WINDOWS:
        ret_window = out["return"].rolling(window=window, min_periods=window)
        out[f"return_mean_w{window}"] = ret_window.mean()
        out[f"return_std_w{window}"] = ret_window.std(ddof=0)
        direction = np.sign(out["return"]).fillna(0.0)
        pos = direction.rolling(window=window, min_periods=window).apply(lambda a: (a > 0).sum(), raw=True)
        neg = direction.rolling(window=window, min_periods=window).apply(lambda a: (a < 0).sum(), raw=True)
        out[f"trend_persistence_w{window}"] = (pos - neg).abs() / window
        out[f"volatility_trend_w{window}"] = _rolling_relative_slope(atr_series, window)
        out[f"volume_trend_w{window}"] = _rolling_relative_slope(df_m5["volume"].astype(float), window)
        half = window // 2
        first_half_mean = out["return"].rolling(window=half, min_periods=half).mean()
        second_half_mean = out["return"].rolling(window=window - half, min_periods=window - half).mean()
        out[f"acceleration_w{window}"] = second_half_mean - first_half_mean.shift(window - half)

    return out
