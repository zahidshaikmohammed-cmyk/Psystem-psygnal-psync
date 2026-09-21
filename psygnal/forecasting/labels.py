"""Forward-looking labels for the ~60-minute (12 x M5) forecast horizon.

Labels are built strictly from `[T+1 .. T+horizon]` and must never be
merged back into anything computed as-of T. `dataset.py` is the only
place that joins features (as-of T) with labels (from T's future).

Volatility-adjusted thresholds (in ATR units) are used instead of a fixed
pip threshold, so the same label definition is meaningful across
instruments with very different tick sizes (e.g. XAUUSD vs EURUSD).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from psygnal import config


def build_forward_labels(
    df_m5: pd.DataFrame,
    atr_series: pd.Series,
    horizon: int = config.FORECAST_HORIZON_M5_CANDLES,
    threshold_atr_mult: float = 0.5,
) -> pd.DataFrame:
    close = df_m5["close"]
    high = df_m5["high"]
    low = df_m5["low"]
    n = len(df_m5)

    forward_close = close.shift(-horizon)
    forward_return = (forward_close - close) / close

    forward_high = high.shift(-1).rolling(window=horizon, min_periods=horizon).max().shift(-(horizon - 1))
    forward_low = low.shift(-1).rolling(window=horizon, min_periods=horizon).min().shift(-(horizon - 1))

    atr_at_t = atr_series
    forward_move_atr = (forward_close - close) / atr_at_t.replace(0.0, np.nan)

    label = pd.Series(np.full(n, None, dtype=object), index=df_m5.index)
    valid = forward_move_atr.notna()
    label[valid & (forward_move_atr >= threshold_atr_mult)] = "UP"
    label[valid & (forward_move_atr <= -threshold_atr_mult)] = "DOWN"
    label[valid & (forward_move_atr.abs() < threshold_atr_mult)] = "NEUTRAL"

    # Rows within `horizon` bars of the end of the available data have no
    # complete future window — they must never receive a label.
    label.iloc[max(0, n - horizon):] = None
    forward_return.iloc[max(0, n - horizon):] = np.nan
    forward_high.iloc[max(0, n - horizon):] = np.nan
    forward_low.iloc[max(0, n - horizon):] = np.nan

    mfe = np.where(label == "UP", forward_high - close, np.where(label == "DOWN", close - forward_low, np.nan))
    mae = np.where(label == "UP", close - forward_low, np.where(label == "DOWN", forward_high - close, np.nan))

    return pd.DataFrame(
        {
            "forward_return": forward_return,
            "forward_move_atr": forward_move_atr,
            "forward_high": forward_high,
            "forward_low": forward_low,
            "label": label,
            "mfe": mfe,
            "mae": mae,
        },
        index=df_m5.index,
    )
