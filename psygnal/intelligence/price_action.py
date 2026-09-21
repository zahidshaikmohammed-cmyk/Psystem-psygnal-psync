"""Price-action pattern classification built from candle + sequence
intelligence (structure/liquidity refine this further downstream in
`intelligence/context.py`)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from psygnal.intelligence.sequence import compute_all_window_stats


@dataclass
class PriceActionState:
    label: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"label": self.label, "evidence": self.evidence}


def classify_price_action(features: pd.DataFrame) -> PriceActionState:
    stats = compute_all_window_stats(features)
    s6 = stats.get(6)
    s12 = stats.get(12)
    s24 = stats.get(24)
    s48 = stats.get(48)

    last = features.iloc[-1]
    evidence: dict[str, Any] = {}

    if s6 is None:
        return PriceActionState(label="insufficient_data", evidence={})

    # Boundary is defined from the prior candles only — the breakout candle
    # itself must never be allowed to redefine the level it is breaking.
    boundary_window = features.iloc[-25:-1] if len(features) >= 25 else features.iloc[:-1]
    close = last["close"]
    if not boundary_window.empty:
        range_high = boundary_window["high"].max()
        range_low = boundary_window["low"].min()
        broke_up = bool(close > range_high and last["direction"] > 0)
        broke_down = bool(close < range_low and last["direction"] < 0)
    else:
        range_high = range_low = float("nan")
        broke_up = broke_down = False
    evidence["broke_up"] = broke_up
    evidence["broke_down"] = broke_down

    # A breakout in the last few candles that has since been partly reclaimed.
    prior_window = features.iloc[-24:-3] if len(features) >= 27 else features.iloc[:-3]
    failed_up = False
    failed_down = False
    if not prior_window.empty:
        prior_high = prior_window["high"].max()
        prior_low = prior_window["low"].min()
        broke_up_recently = (features.iloc[-6:-1]["close"] >= prior_high).any()
        broke_down_recently = (features.iloc[-6:-1]["close"] <= prior_low).any()
        failed_up = bool(broke_up_recently and close < prior_high)
        failed_down = bool(broke_down_recently and close > prior_low)
    evidence["failed_up"] = failed_up
    evidence["failed_down"] = failed_down

    label = "consolidation"

    if broke_up or broke_down:
        label = "breakout"
    elif failed_up or failed_down:
        label = "failed_breakout"
    elif s12 and s12.reversal_attempt and s24 and s24.trend_persistence >= 0.65:
        label = "reversal"
    elif s6.dominant_direction != 0 and s24 and s24.dominant_direction == s6.dominant_direction and s24.trend_persistence >= 0.6 and s6.trend_persistence < 0.6:
        label = "pullback"
    elif s6.trend_persistence >= 0.75 and abs(s6.acceleration) >= 0 and last["behavior"] in (
        "strong_bullish_displacement",
        "strong_bearish_displacement",
    ):
        label = "impulse"
    elif s24 and s6.dominant_direction == s24.dominant_direction and s6.trend_persistence >= 0.6 and s24.trend_persistence >= 0.55:
        label = "continuation"
    elif last["behavior"] in ("rejection_bullish", "rejection_bearish"):
        label = "rejection"
    elif last["behavior"] == "exhaustion" or (s6 and s24 and s24.trend_persistence >= 0.6 and s6.acceleration * s24.return_mean < 0):
        label = "exhaustion"
    elif s48 and s48.trend_persistence <= 0.55 and s48.volatility_trend < 0.1:
        label = "consolidation"

    evidence["window_stats"] = {w: s.as_dict() for w, s in stats.items()}
    evidence["last_behavior"] = last["behavior"]

    return PriceActionState(label=label, evidence=evidence)
