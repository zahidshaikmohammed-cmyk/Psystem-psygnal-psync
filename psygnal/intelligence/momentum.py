"""Momentum engine.

RSI/MACD/ROC are correlated measurements of the same underlying
phenomenon (rate of directional price change), so they are blended into a
single momentum score rather than treated as independent confirmations.
ADX's rate of change is used separately, as a genuinely distinct signal:
whether trend *strength* itself is accelerating or decelerating.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from psygnal import config
from psygnal.indicators.adx import adx
from psygnal.indicators.macd import macd
from psygnal.indicators.roc import roc
from psygnal.indicators.rsi import rsi
from psygnal.intelligence.structure import SwingPoint


@dataclass
class MomentumState:
    label: str
    momentum_score: float
    accelerating: bool
    divergence_bullish: bool
    divergence_bearish: bool
    rsi_value: float | None
    macd_histogram: float | None
    roc_value: float | None
    adx_value: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "momentum_score": round(self.momentum_score, 4) if self.momentum_score is not None else None,
            "accelerating": self.accelerating,
            "divergence_bullish": self.divergence_bullish,
            "divergence_bearish": self.divergence_bearish,
            "rsi": self.rsi_value,
            "macd_histogram": self.macd_histogram,
            "roc": self.roc_value,
            "adx": self.adx_value,
        }


def _normalized_macd_hist(macd_hist: pd.Series, close: pd.Series) -> pd.Series:
    # Normalize by price so the value is comparable across instruments.
    return 100 * macd_hist / close


def detect_divergence(
    close: pd.Series, rsi_series: pd.Series, swing_highs: list[SwingPoint], swing_lows: list[SwingPoint]
) -> tuple[bool, bool]:
    bullish = False
    bearish = False
    if len(swing_highs) >= 2:
        a, b = swing_highs[-2], swing_highs[-1]
        if b.price > a.price:
            rsi_a, rsi_b = rsi_series.iloc[a.index], rsi_series.iloc[b.index]
            if not (np.isnan(rsi_a) or np.isnan(rsi_b)) and rsi_b < rsi_a:
                bearish = True
    if len(swing_lows) >= 2:
        a, b = swing_lows[-2], swing_lows[-1]
        if b.price < a.price:
            rsi_a, rsi_b = rsi_series.iloc[a.index], rsi_series.iloc[b.index]
            if not (np.isnan(rsi_a) or np.isnan(rsi_b)) and rsi_b > rsi_a:
                bullish = True
    return bullish, bearish


def analyze_momentum(
    df: pd.DataFrame,
    swing_highs: list[SwingPoint] | None = None,
    swing_lows: list[SwingPoint] | None = None,
) -> MomentumState:
    min_len = config.MACD_SLOW + config.MACD_SIGNAL + 5
    if df.empty or len(df) < min_len:
        return MomentumState("NEUTRAL", 0.0, False, False, False, None, None, None, None)

    close, high, low = df["close"], df["high"], df["low"]

    rsi_series = rsi(close, config.RSI_PERIOD)
    macd_df = macd(close, config.MACD_FAST, config.MACD_SLOW, config.MACD_SIGNAL)
    roc_series = roc(close, config.ROC_PERIOD)
    adx_df = adx(high, low, close, config.ADX_PERIOD)

    rsi_component = (rsi_series - 50.0) / 50.0
    macd_component = np.clip(_normalized_macd_hist(macd_df["histogram"], close) * 5, -1, 1)
    roc_component = np.clip(roc_series / 2.0, -1, 1)

    momentum_score_series = (rsi_component + macd_component + roc_component) / 3.0
    momentum_score = float(momentum_score_series.iloc[-1]) if not np.isnan(momentum_score_series.iloc[-1]) else 0.0

    lookback = 5
    prior_score = momentum_score_series.iloc[-1 - lookback] if len(momentum_score_series) > lookback else np.nan
    accelerating = bool(not np.isnan(prior_score) and abs(momentum_score) > abs(float(prior_score)))

    div_bullish = div_bearish = False
    if swing_highs is not None and swing_lows is not None:
        div_bullish, div_bearish = detect_divergence(close, rsi_series, swing_highs, swing_lows)

    if momentum_score > 0.15:
        label = "ACCELERATING_BULLISH" if accelerating else "DECELERATING_BULLISH"
    elif momentum_score < -0.15:
        label = "ACCELERATING_BEARISH" if accelerating else "DECELERATING_BEARISH"
    else:
        label = "NEUTRAL"

    last_adx = adx_df["adx"].iloc[-1]

    return MomentumState(
        label=label,
        momentum_score=momentum_score,
        accelerating=accelerating,
        divergence_bullish=div_bullish,
        divergence_bearish=div_bearish,
        rsi_value=float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else None,
        macd_histogram=float(macd_df["histogram"].iloc[-1]) if not np.isnan(macd_df["histogram"].iloc[-1]) else None,
        roc_value=float(roc_series.iloc[-1]) if not np.isnan(roc_series.iloc[-1]) else None,
        adx_value=float(last_adx) if not np.isnan(last_adx) else None,
    )
