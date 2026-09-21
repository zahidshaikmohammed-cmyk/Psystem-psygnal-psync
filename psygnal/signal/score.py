"""Signal score: transparent, weighted composite of the ensemble output and
every intelligence layer. This is a quality/strength score, NOT a win
probability (see constitution rule #19)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

# Each weight covers a distinct evidence category; correlated indicators
# (e.g. RSI/MACD/ROC) were already pre-aggregated into single components
# upstream (forecasting/ensemble.py, intelligence/momentum.py) so they are
# not double-counted here.
SCORE_WEIGHTS = {
    "forecast_strength": 0.18,
    "historical_pattern_agreement": 0.10,
    "structure_liquidity_alignment": 0.12,
    "trend_alignment": 0.12,
    "momentum_alignment": 0.08,
    "volatility_suitability": 0.08,
    "volume_confirmation": 0.05,
    "session_context": 0.04,
    "cross_market_confirmation": 0.06,
    "macro_news_clarity": 0.04,
    "entry_quality": 0.03,
    "target_quality": 0.05,
    "risk_reward": 0.05,
}
assert abs(sum(SCORE_WEIGHTS.values()) - 1.0) < 1e-9


def compute_rr(entry: float, stop_loss: float, tp1: float) -> Optional[float]:
    risk = abs(entry - stop_loss)
    reward = abs(tp1 - entry)
    if risk <= 0:
        return None
    return reward / risk


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def compute_signal_score(
    direction: str,
    ensemble_result: dict[str, Any],
    pattern_memory: Optional[dict[str, Any]],
    components: dict[str, float],
    volatility_label: str,
    volume_label: str,
    volume_relationship: str,
    session_label: str,
    macro_status: str,
    rr: Optional[float],
) -> dict[str, Any]:
    prob_long = ensemble_result["probability_long"]
    prob_short = ensemble_result["probability_short"]
    forecast_strength = _clip01(abs(prob_long - prob_short) * 2)

    if pattern_memory and pattern_memory.get("status") == "OK":
        pm_prob = pattern_memory["probability_up"] if direction == "LONG" else pattern_memory["probability_down"]
        historical_pattern_agreement = _clip01(pm_prob * 2)
    else:
        historical_pattern_agreement = 0.5  # no evidence either way

    directional_sign = 1.0 if direction == "LONG" else -1.0
    structure_liquidity_alignment = _clip01(0.5 + 0.5 * directional_sign * components.get("structure_liquidity", 0.0))
    trend_alignment = _clip01(0.5 + 0.5 * directional_sign * components.get("trend", 0.0))
    momentum_alignment = _clip01(0.5 + 0.5 * directional_sign * components.get("momentum", 0.0))
    cross_market_confirmation = _clip01(0.5 + 0.5 * directional_sign * components.get("cross_market", 0.0))

    volatility_suitability = {"LOW": 0.3, "NORMAL": 0.75, "HIGH": 0.9}.get(volatility_label, 0.5)

    volume_confirmation = 0.5
    if volume_relationship == "CONFIRMING_EXPANSION":
        volume_confirmation = 0.9
    elif volume_relationship == "DIVERGING":
        volume_confirmation = 0.3
    elif volume_label == "ELEVATED":
        volume_confirmation = 0.7

    session_context = {
        "LONDON_NEW_YORK_OVERLAP": 0.95,
        "LONDON": 0.8,
        "NEW_YORK": 0.8,
        "ASIA": 0.55,
        "OFF_HOURS": 0.35,
    }.get(session_label, 0.5)

    macro_news_clarity = {"CLEAR": 0.85, "ELEVATED_RISK": 0.45, "UNAVAILABLE": 0.6, "DISABLED": 0.6}.get(macro_status, 0.6)

    entry_quality = 0.7  # entries are always structurally anchored; see signal/entry.py
    target_quality = 0.8 if rr is not None else 0.4
    risk_reward_score = _clip01((rr or 0.0) / 3.0)

    sub_scores = {
        "forecast_strength": forecast_strength,
        "historical_pattern_agreement": historical_pattern_agreement,
        "structure_liquidity_alignment": structure_liquidity_alignment,
        "trend_alignment": trend_alignment,
        "momentum_alignment": momentum_alignment,
        "volatility_suitability": volatility_suitability,
        "volume_confirmation": volume_confirmation,
        "session_context": session_context,
        "cross_market_confirmation": cross_market_confirmation,
        "macro_news_clarity": macro_news_clarity,
        "entry_quality": entry_quality,
        "target_quality": target_quality,
        "risk_reward": risk_reward_score,
    }

    score = sum(sub_scores[k] * SCORE_WEIGHTS[k] for k in SCORE_WEIGHTS) * 100.0
    return {"signal_score": round(float(score), 2), "sub_scores": sub_scores}
