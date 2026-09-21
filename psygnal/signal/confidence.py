"""Confidence engine — a distinct concept from signal_score: this measures
how much the engine trusts its own read of the market, not how good the
trade setup looks. Never a guarantee (constitution rule #22/#23: it
reflects uncertainty, it does not remove the directional call)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from psygnal.models import ConfidenceLevel, DataStatus

CONFIDENCE_WEIGHTS = {
    "probability_separation": 0.30,
    "model_agreement": 0.20,
    "pattern_memory_agreement": 0.15,
    "data_completeness": 0.15,
    "regime_clarity": 0.10,
    "cross_market_agreement": 0.10,
}
assert abs(sum(CONFIDENCE_WEIGHTS.values()) - 1.0) < 1e-9

AMBIGUOUS_REGIMES = {"CHAOTIC", "NEWS_SHOCK"}


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def compute_confidence(
    direction: str,
    ensemble_result: dict[str, Any],
    pattern_memory: Optional[dict[str, Any]],
    data_quality_status: str,
    regime_label: str,
    cross_market_component: float,
) -> dict[str, Any]:
    prob_long = ensemble_result["probability_long"]
    prob_short = ensemble_result["probability_short"]
    probability_separation = _clip01(abs(prob_long - prob_short) * 2)

    sources_used = ensemble_result.get("sources_used", ["deterministic"])
    model_agreement = 1.0 if len(sources_used) <= 1 else min(1.0, 0.5 + 0.25 * (len(sources_used) - 1))

    if pattern_memory and pattern_memory.get("status") == "OK":
        pm_prob = pattern_memory["probability_up"] if direction == "LONG" else pattern_memory["probability_down"]
        pattern_memory_agreement = _clip01(pm_prob * 2)
    else:
        pattern_memory_agreement = 0.5

    data_completeness = {DataStatus.OK.value: 1.0, DataStatus.DEGRADED.value: 0.5, DataStatus.UNAVAILABLE.value: 0.0}.get(
        data_quality_status, 0.5
    )

    regime_clarity = 0.4 if regime_label in AMBIGUOUS_REGIMES else 0.9

    directional_sign = 1.0 if direction == "LONG" else -1.0
    cross_market_agreement = _clip01(0.5 + 0.5 * directional_sign * cross_market_component)

    sub_scores = {
        "probability_separation": probability_separation,
        "model_agreement": model_agreement,
        "pattern_memory_agreement": pattern_memory_agreement,
        "data_completeness": data_completeness,
        "regime_clarity": regime_clarity,
        "cross_market_agreement": cross_market_agreement,
    }

    confidence_score = sum(sub_scores[k] * CONFIDENCE_WEIGHTS[k] for k in CONFIDENCE_WEIGHTS) * 100.0

    if confidence_score >= 70:
        level = ConfidenceLevel.HIGH.value
    elif confidence_score >= 40:
        level = ConfidenceLevel.MEDIUM.value
    else:
        level = ConfidenceLevel.LOW.value

    return {"confidence": level, "confidence_score": round(float(confidence_score), 2), "sub_scores": sub_scores}
