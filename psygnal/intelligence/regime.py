"""Market-regime classification.

Regime is a major forecasting input (see forecasting/ensemble.py), derived
from a priority cascade over the already-computed intelligence layers
rather than a single indicator threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from psygnal.models import RegimeLabel


@dataclass
class RegimeState:
    label: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"label": self.label, "evidence": self.evidence}


def classify_regime(
    symbol_intel: dict[str, Any],
    macro_state: dict[str, Any],
    reference_trend_timeframe: str = "H1",
) -> RegimeState:
    evidence: dict[str, Any] = {}

    if symbol_intel.get("status") != "OK":
        return RegimeState(RegimeLabel.CHAOTIC.value, {"reason": "insufficient_data"})

    volatility = symbol_intel["volatility"]
    liquidity = symbol_intel["liquidity"]
    price_action = symbol_intel["price_action"]
    structures = symbol_intel["structures"]
    trends = symbol_intel["trends"]

    m5_structure = structures.get("M5")
    ref_trend = trends.get(reference_trend_timeframe) or trends.get("M5")

    recent_high_impact_macro = bool(macro_state.get("recent")) and macro_state.get("status") == "ELEVATED_RISK"
    evidence["recent_high_impact_macro"] = recent_high_impact_macro
    evidence["volatility_label"] = volatility.label
    evidence["volatility_regime"] = volatility.regime
    evidence["price_action_label"] = price_action.label
    evidence["liquidity_sweep_recent"] = any(e.get("recent") for e in liquidity.sweep_events)
    evidence["breakout_retest"] = liquidity.breakout_retest
    evidence["last_choch"] = m5_structure.last_choch if m5_structure else None
    evidence["ref_trend_score"] = ref_trend.score if ref_trend else 0.0

    if recent_high_impact_macro and volatility.label == "HIGH":
        return RegimeState(RegimeLabel.NEWS_SHOCK.value, evidence)

    if evidence["liquidity_sweep_recent"]:
        return RegimeState(RegimeLabel.LIQUIDITY_SWEEP.value, evidence)

    if liquidity.breakout_retest:
        return RegimeState(RegimeLabel.BREAKOUT_RETEST.value, evidence)

    if price_action.label == "breakout" or liquidity.continuation_after_breakout:
        return RegimeState(RegimeLabel.BREAKOUT.value, evidence)

    if price_action.label == "reversal" or evidence["last_choch"] is not None:
        return RegimeState(RegimeLabel.REVERSAL.value, evidence)

    if volatility.regime == "EXPANSION":
        return RegimeState(RegimeLabel.VOLATILITY_EXPANSION.value, evidence)

    if volatility.regime == "COMPRESSION":
        return RegimeState(RegimeLabel.VOLATILITY_CONTRACTION.value, evidence)

    ref_score = evidence["ref_trend_score"]
    structure_label = m5_structure.trend_structure if m5_structure else "UNDEFINED"

    if ref_score >= 0.2 or structure_label == "UPTREND":
        return RegimeState(RegimeLabel.TREND_UP.value, evidence)
    if ref_score <= -0.2 or structure_label == "DOWNTREND":
        return RegimeState(RegimeLabel.TREND_DOWN.value, evidence)

    if abs(ref_score) < 0.15 and volatility.label == "HIGH" and price_action.label in ("failed_breakout", "exhaustion"):
        return RegimeState(RegimeLabel.CHAOTIC.value, evidence)

    return RegimeState(RegimeLabel.RANGE.value, evidence)
