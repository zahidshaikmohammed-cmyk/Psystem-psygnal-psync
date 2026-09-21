"""Trend engine: combines structure, EMA alignment/slope, price location and
ADX/DI directional strength into one of five contextual trend states.

Indicators are treated as *contextual features* feeding a composite score,
never as standalone signals (no "EMA9 > EMA20 => bullish").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from psygnal import config
from psygnal.indicators.adx import adx
from psygnal.indicators.ema import compute_emas
from psygnal.intelligence.sequence import _relative_slope
from psygnal.intelligence.structure import StructureState
from psygnal.models import TrendLabel


@dataclass
class TrendState:
    timeframe: str
    label: str
    score: float
    components: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"timeframe": self.timeframe, "label": self.label, "score": round(self.score, 4), "components": {k: round(v, 4) for k, v in self.components.items()}}


def _ema_alignment_score(latest: dict[str, float]) -> float:
    periods = config.EMA_PERIODS
    values = [latest.get(f"ema_{p}") for p in periods]
    if any(v is None or np.isnan(v) for v in values):
        return 0.0
    pairs = list(zip(values, values[1:]))
    bullish_pairs = sum(1 for a, b in pairs if a > b)
    bearish_pairs = sum(1 for a, b in pairs if a < b)
    total = len(pairs)
    return (bullish_pairs - bearish_pairs) / total


def _structure_component(structure_label: str) -> float:
    return {"UPTREND": 1.0, "DOWNTREND": -1.0, "RANGE": 0.0, "UNDEFINED": 0.0}.get(structure_label, 0.0)


def _label_from_score(score: float) -> str:
    if score >= 0.6:
        return TrendLabel.STRONG_BULLISH.value
    if score >= 0.2:
        return TrendLabel.BULLISH.value
    if score > -0.2:
        return TrendLabel.NEUTRAL.value
    if score > -0.6:
        return TrendLabel.BEARISH.value
    return TrendLabel.STRONG_BEARISH.value


def analyze_trend(df: pd.DataFrame, structure_state: StructureState, timeframe: str) -> TrendState:
    min_len = max(config.EMA_PERIODS) + 5
    if df.empty or len(df) < min_len:
        return TrendState(timeframe=timeframe, label=TrendLabel.NEUTRAL.value, score=0.0, components={"insufficient_data": 1.0})

    close, high, low = df["close"], df["high"], df["low"]
    emas = compute_emas(close)
    latest_ema = {col: float(emas[col].iloc[-1]) for col in emas.columns}
    alignment = _ema_alignment_score(latest_ema)

    ema50 = emas["ema_50"].dropna()
    slope = _relative_slope(ema50.iloc[-10:].to_numpy()) if len(ema50) >= 10 else 0.0
    slope_component = float(np.clip(slope * 50, -1, 1))

    price_vs_ema200 = 0.0
    if not np.isnan(latest_ema.get("ema_200", np.nan)):
        price_vs_ema200 = 1.0 if float(close.iloc[-1]) > latest_ema["ema_200"] else -1.0

    adx_df = adx(high, low, close, period=config.ADX_PERIOD)
    last_adx = adx_df.iloc[-1]
    strength_component = 0.0
    if not (np.isnan(last_adx["plus_di"]) or np.isnan(last_adx["minus_di"]) or np.isnan(last_adx["adx"])):
        di_diff = (last_adx["plus_di"] - last_adx["minus_di"]) / 100.0
        strength_gate = min(last_adx["adx"] / 40.0, 1.0)
        strength_component = float(np.clip(di_diff * strength_gate, -1, 1))

    structure_component = _structure_component(structure_state.trend_structure)

    weights = {
        "structure": 0.30,
        "ema_alignment": 0.25,
        "ema_slope": 0.15,
        "price_vs_ema200": 0.10,
        "adx_directional_strength": 0.20,
    }
    components = {
        "structure": structure_component,
        "ema_alignment": alignment,
        "ema_slope": slope_component,
        "price_vs_ema200": price_vs_ema200,
        "adx_directional_strength": strength_component,
    }
    composite = sum(components[k] * weights[k] for k in weights)
    composite = float(np.clip(composite, -1, 1))

    return TrendState(timeframe=timeframe, label=_label_from_score(composite), score=composite, components=components)


def analyze_multi_timeframe_trend(
    frames: dict[str, pd.DataFrame], structures: dict[str, StructureState]
) -> dict[str, TrendState]:
    return {
        tf: analyze_trend(df, structures.get(tf, StructureState(tf, "UNDEFINED", [], [], None, None, False, False, None, None)), tf)
        for tf, df in frames.items()
    }
