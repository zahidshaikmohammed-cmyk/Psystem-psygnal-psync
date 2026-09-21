"""Shared numerical primitives for the indicator engine.

Implements the two recursive smoothing schemes every other indicator is
built from: classic EMA and Wilder smoothing. Both gracefully handle a
series with leading NaNs (e.g. the MACD line before its slow EMA has
warmed up) by starting the recursion at the first valid observation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average.

    alpha = 2 / (period + 1)
    EMA_t = alpha * price_t + (1 - alpha) * EMA_(t-1)

    The first `period - 1` values (after the series' first valid value) are
    NaN, signalling insufficient warm-up data rather than a fabricated
    number.
    """
    values = series.to_numpy(dtype=float)
    out = np.full(values.shape, np.nan)
    valid_mask = ~np.isnan(values)
    if not valid_mask.any():
        return pd.Series(out, index=series.index)

    first_idx = int(np.argmax(valid_mask))
    sub = values[first_idx:]
    alpha = 2.0 / (period + 1)
    sub_out = np.full(sub.shape, np.nan)
    sub_out[0] = sub[0]
    for i in range(1, len(sub)):
        prev = sub_out[i - 1]
        cur = sub[i]
        if np.isnan(cur):
            sub_out[i] = prev
            continue
        if np.isnan(prev):
            sub_out[i] = cur
            continue
        sub_out[i] = alpha * cur + (1 - alpha) * prev

    if len(sub) >= period:
        sub_out[: period - 1] = np.nan
    else:
        sub_out[:] = np.nan

    out[first_idx:] = sub_out
    return pd.Series(out, index=series.index)


def wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder-style smoothed average, used by RSI/ATR/ADX.

    Seed = simple average of the first `period` valid observations.
    Then: avg_t = (avg_(t-1) * (period - 1) + value_t) / period
    """
    values = series.to_numpy(dtype=float)
    out = np.full(values.shape, np.nan)
    valid_mask = ~np.isnan(values)
    if not valid_mask.any():
        return pd.Series(out, index=series.index)

    first_idx = int(np.argmax(valid_mask))
    sub = values[first_idx:]
    n = len(sub)
    sub_out = np.full(n, np.nan)
    if n < period:
        out[first_idx:] = sub_out
        return pd.Series(out, index=series.index)

    seed = float(np.mean(sub[:period]))
    sub_out[period - 1] = seed
    for i in range(period, n):
        sub_out[i] = (sub_out[i - 1] * (period - 1) + sub[i]) / period

    out[first_idx:] = sub_out
    return pd.Series(out, index=series.index)


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def rolling_std(series: pd.Series, period: int, ddof: int = 0) -> pd.Series:
    return series.rolling(window=period, min_periods=period).std(ddof=ddof)
