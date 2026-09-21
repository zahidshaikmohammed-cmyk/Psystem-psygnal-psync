"""Per-candle anatomy features and behavioural-state classification.

Named candlestick patterns are intentionally de-emphasised: the engine
computes continuous, context-aware measurements (body %, wick dominance,
range percentile, close-location) and only derives a coarse behavioural
label from them, because the constitution explicitly warns that "context
matters more than pattern names."
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from psygnal.indicators.atr import true_range

RANGE_PERCENTILE_LOOKBACK = 100


def compute_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """`df` must have open/high/low/close/volume columns, ascending time index."""
    if df.empty:
        return df.copy()

    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    out = pd.DataFrame(index=df.index)
    out["open"], out["high"], out["low"], out["close"], out["volume"] = o, h, l, c, v

    prev_close = c.shift(1)
    out["return"] = (c - prev_close) / prev_close
    out["log_return"] = np.log(c / prev_close)

    candle_range = (h - l)
    safe_range = candle_range.replace(0.0, np.nan)

    out["body"] = (c - o).abs()
    out["body_signed"] = c - o
    out["body_pct"] = out["body"] / safe_range
    out["upper_wick"] = h - pd.concat([o, c], axis=1).max(axis=1)
    out["lower_wick"] = pd.concat([o, c], axis=1).min(axis=1) - l
    out["wick_to_body"] = (out["upper_wick"] + out["lower_wick"]) / out["body"].replace(0.0, np.nan)

    out["true_range"] = true_range(h, l, c)
    out["range"] = candle_range
    out["close_location_value"] = ((c - l) - (h - c)) / safe_range
    out["directional_strength"] = out["body"] / out["true_range"].replace(0.0, np.nan)

    lookback = min(RANGE_PERCENTILE_LOOKBACK, len(df))
    min_periods = min(lookback, max(1, lookback // 4))
    out["range_percentile"] = out["true_range"].rolling(
        window=lookback, min_periods=min_periods
    ).rank(pct=True)
    out["body_pct_percentile"] = out["body_pct"].rolling(
        window=lookback, min_periods=min_periods
    ).rank(pct=True)

    out["is_inside"] = (h <= h.shift(1)) & (l >= l.shift(1))
    out["is_outside"] = (h >= h.shift(1)) & (l <= l.shift(1))
    out["direction"] = np.sign(out["body_signed"]).fillna(0.0)

    out["behavior"] = _classify_behavior(out)
    return out


def _classify_behavior(f: pd.DataFrame) -> pd.Series:
    n = len(f)
    labels = np.full(n, "neutral", dtype=object)

    body_pct = f["body_pct"].to_numpy()
    range_pctl = f["range_percentile"].to_numpy()
    direction = f["direction"].to_numpy()
    is_inside = f["is_inside"].to_numpy()
    is_outside = f["is_outside"].to_numpy()
    upper_wick = f["upper_wick"].to_numpy()
    lower_wick = f["lower_wick"].to_numpy()
    body = f["body"].to_numpy()

    for i in range(n):
        bp = body_pct[i]
        rp = range_pctl[i]
        if np.isnan(bp) or np.isnan(rp):
            labels[i] = "insufficient_data"
            continue

        # Displacement/rejection/exhaustion are the strongest directional
        # signals and take priority over the purely structural inside/
        # outside classification, which would otherwise mask them (a huge
        # displacement candle is almost always also an "outside" candle).
        if rp >= 0.70 and bp >= 0.65:
            labels[i] = "strong_bullish_displacement" if direction[i] > 0 else "strong_bearish_displacement"
            continue

        if rp >= 0.70 and bp <= 0.35:
            body_i = body[i] if body[i] > 0 else 1e-12
            if upper_wick[i] > 2 * body_i and upper_wick[i] > lower_wick[i]:
                labels[i] = "rejection_bearish"
            elif lower_wick[i] > 2 * body_i and lower_wick[i] > upper_wick[i]:
                labels[i] = "rejection_bullish"
            else:
                labels[i] = "exhaustion"
            continue

        if is_inside[i]:
            labels[i] = "inside_candle"
            continue
        if is_outside[i]:
            labels[i] = "outside_candle"
            continue

        if rp <= 0.20:
            labels[i] = "compression"
            continue

        if rp >= 0.55:
            labels[i] = "expansion"
            continue

        labels[i] = "neutral"

    return pd.Series(labels, index=f.index)
