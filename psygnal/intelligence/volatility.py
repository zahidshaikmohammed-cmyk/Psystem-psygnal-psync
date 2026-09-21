"""Volatility engine: ATR percentile, realized volatility, range percentile,
and an expected-movement estimate for the 60-minute forecast horizon."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from psygnal import config
from psygnal.indicators.atr import atr


@dataclass
class VolatilityState:
    label: str  # LOW | NORMAL | HIGH
    regime: str  # EXPANSION | COMPRESSION | STABLE
    atr_value: float | None
    atr_percentile: float | None
    realized_volatility: float | None
    range_percentile: float | None
    expected_move_60m: float | None  # in price units, one standard-deviation-ish estimate

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "regime": self.regime,
            "atr": self.atr_value,
            "atr_percentile": self.atr_percentile,
            "realized_volatility": self.realized_volatility,
            "range_percentile": self.range_percentile,
            "expected_move_60m": self.expected_move_60m,
        }


def analyze_volatility(
    df: pd.DataFrame,
    horizon_candles: int = config.FORECAST_HORIZON_M5_CANDLES,
    percentile_lookback: int = config.ATR_PERCENTILE_LOOKBACK,
) -> VolatilityState:
    min_len = config.ATR_PERIOD + 5
    if df.empty or len(df) < min_len:
        return VolatilityState("NORMAL", "STABLE", None, None, None, None, None)

    close, high, low = df["close"], df["high"], df["low"]
    atr_series = atr(high, low, close, config.ATR_PERIOD)
    atr_value = float(atr_series.iloc[-1]) if not np.isnan(atr_series.iloc[-1]) else None

    lookback = min(percentile_lookback, len(atr_series.dropna()))
    atr_percentile = None
    if lookback >= 10:
        recent = atr_series.dropna().iloc[-lookback:]
        atr_percentile = float((recent <= recent.iloc[-1]).mean())

    log_returns = np.log(close / close.shift(1))
    realized_vol_window = min(48, len(log_returns.dropna()))
    realized_volatility = None
    if realized_vol_window >= 10:
        realized_volatility = float(log_returns.dropna().iloc[-realized_vol_window:].std())

    range_pctl = None
    true_range_window = (high - low)
    lookback2 = min(100, len(true_range_window))
    if lookback2 >= 10:
        recent_tr = true_range_window.iloc[-lookback2:]
        range_pctl = float((recent_tr <= recent_tr.iloc[-1]).mean())

    expected_move_60m = None
    if atr_value is not None:
        expected_move_60m = atr_value * math.sqrt(horizon_candles)

    label = "NORMAL"
    if atr_percentile is not None:
        if atr_percentile <= 0.25:
            label = "LOW"
        elif atr_percentile >= 0.75:
            label = "HIGH"

    regime = "STABLE"
    if len(atr_series.dropna()) >= 10:
        recent_atr = atr_series.dropna().iloc[-10:]
        slope = float(np.polyfit(np.arange(len(recent_atr)), recent_atr.to_numpy(), 1)[0])
        mean_atr = float(recent_atr.mean()) or 1e-9
        rel_slope = slope / mean_atr
        if rel_slope > 0.02:
            regime = "EXPANSION"
        elif rel_slope < -0.02:
            regime = "COMPRESSION"

    return VolatilityState(
        label=label,
        regime=regime,
        atr_value=atr_value,
        atr_percentile=atr_percentile,
        realized_volatility=realized_volatility,
        range_percentile=range_pctl,
        expected_move_60m=expected_move_60m,
    )
