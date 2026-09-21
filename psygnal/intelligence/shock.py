"""Market-shock detection.

A MARKET_SHOCK is not itself a trade signal — it tells the reasoning
layer "something important just happened," which regime classification
and the reasoning/contradiction layer then weigh (a shock coinciding
with a scheduled high-impact event, see macro/event_engine.py, is much
stronger evidence than one that doesn't).

Requires at least two independent triggers (abnormal range, abnormal
volume, abnormal single-candle velocity, or a run of consecutive
directional candles) before calling anything a shock — a single elevated
reading is just normal market noise, not "something important just
happened."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

RANGE_PERCENTILE_SHOCK_THRESHOLD = 0.90
RELATIVE_VOLUME_SHOCK_THRESHOLD = 2.0
CONSECUTIVE_DIRECTIONAL_SHOCK_COUNT = 4
VELOCITY_SHOCK_THRESHOLD_ATR = 2.5  # |single-candle body| in ATR units


@dataclass
class ShockState:
    is_shock: bool
    severity: str  # "NONE" | "MODERATE" | "SEVERE"
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"is_shock": self.is_shock, "severity": self.severity, "evidence": self.evidence}


def detect_shock(
    m5_features: pd.DataFrame,
    relative_volume: Optional[float],
    atr_value: Optional[float],
) -> ShockState:
    if m5_features.empty:
        return ShockState(False, "NONE", {"reason": "no data"})

    last = m5_features.iloc[-1]
    evidence: dict[str, Any] = {}
    triggers = 0

    range_pctl = last.get("range_percentile")
    abnormal_range = bool(range_pctl is not None and not pd.isna(range_pctl) and range_pctl >= RANGE_PERCENTILE_SHOCK_THRESHOLD)
    evidence["abnormal_range"] = abnormal_range
    evidence["range_percentile"] = None if range_pctl is None or pd.isna(range_pctl) else float(range_pctl)
    if abnormal_range:
        triggers += 1

    abnormal_volume = bool(relative_volume is not None and relative_volume >= RELATIVE_VOLUME_SHOCK_THRESHOLD)
    evidence["abnormal_volume"] = abnormal_volume
    evidence["relative_volume"] = relative_volume
    if abnormal_volume:
        triggers += 1

    velocity = None
    if atr_value and atr_value > 0:
        velocity = abs(float(last["close"]) - float(last["open"])) / atr_value
    abnormal_velocity = bool(velocity is not None and velocity >= VELOCITY_SHOCK_THRESHOLD_ATR)
    evidence["velocity_atr"] = velocity
    evidence["abnormal_velocity"] = abnormal_velocity
    if abnormal_velocity:
        triggers += 1

    consecutive_directional = False
    if len(m5_features) >= CONSECUTIVE_DIRECTIONAL_SHOCK_COUNT:
        window = m5_features.iloc[-CONSECUTIVE_DIRECTIONAL_SHOCK_COUNT:]
        body_signed = window["body_signed"] if "body_signed" in window.columns else (window["close"] - window["open"])
        directions = np.sign(body_signed.to_numpy())
        consecutive_directional = bool((directions != 0).all() and (directions == directions[0]).all())
    evidence["consecutive_directional"] = consecutive_directional
    if consecutive_directional:
        triggers += 1

    evidence["trigger_count"] = triggers
    is_shock = triggers >= 2
    if not is_shock:
        severity = "NONE"
    elif triggers == 2:
        severity = "MODERATE"
    else:
        severity = "SEVERE"

    return ShockState(is_shock=is_shock, severity=severity, evidence=evidence)
