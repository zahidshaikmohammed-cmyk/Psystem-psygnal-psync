"""Forecast ensemble.

Combines the deterministic intelligence-based estimate (always available)
with the learned baseline model and pattern-memory lookup (both optional,
used only once enough historical data exists to train/populate them).
Never blindly averages every factor — components that measure the same
underlying phenomenon are pre-aggregated into one score (see the
"*_component" construction below) before the final weighted combination.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from psygnal import config

DIRECTIONAL_WEIGHTS = {
    "trend": 0.35,
    "momentum": 0.20,
    "structure_liquidity": 0.20,
    "price_action": 0.15,
    "cross_market": 0.10,
}

TREND_TIMEFRAME_WEIGHTS = {"H4": 0.35, "H1": 0.35, "M30": 0.20, "M15": 0.10}


def _weighted_trend_component(trends: dict[str, Any]) -> float:
    total_weight = 0.0
    weighted_sum = 0.0
    for tf, weight in TREND_TIMEFRAME_WEIGHTS.items():
        state = trends.get(tf)
        if state is None:
            continue
        weighted_sum += state.score * weight
        total_weight += weight
    if total_weight == 0:
        return 0.0
    return weighted_sum / total_weight


def _structure_liquidity_component(structures: dict[str, Any], liquidity: Any) -> float:
    m5 = structures.get("M5")
    components = []
    if m5 is not None:
        if m5.last_bos == "BULLISH_BOS":
            components.append(1.0)
        elif m5.last_bos == "BEARISH_BOS":
            components.append(-1.0)
        if m5.last_choch == "BULLISH_CHOCH":
            components.append(1.0)
        elif m5.last_choch == "BEARISH_CHOCH":
            components.append(-1.0)

    recent_sweeps = [e for e in liquidity.sweep_events if e.get("recent")]
    if any(e["direction"] == "sweep_low" for e in recent_sweeps):
        components.append(1.0)
    if any(e["direction"] == "sweep_high" for e in recent_sweeps):
        components.append(-1.0)

    if not components:
        return 0.0
    return float(np.mean(components))


def _price_action_component(price_action: Any) -> float:
    evidence = price_action.evidence
    stats = evidence.get("window_stats", {})
    s24 = stats.get(24)
    dominant = s24["dominant_direction"] if s24 else 0

    if price_action.label == "breakout":
        if evidence.get("broke_up"):
            return 1.0
        if evidence.get("broke_down"):
            return -1.0
        return 0.0
    if price_action.label == "reversal":
        return float(-dominant)
    if price_action.label in ("continuation", "impulse", "pullback"):
        return float(dominant)
    return 0.0


def _cross_market_component(symbol: str, cross_market_state: dict[str, Any], gold_state: Optional[dict[str, Any]]) -> float:
    usd_state = cross_market_state.get("usd_composite", {})
    usd_label = usd_state.get("state")
    usd_dir = {"STRENGTHENING": 1.0, "WEAKENING": -1.0}.get(usd_label, 0.0)

    if symbol in config.USD_COMPOSITE_LEGS:
        sign = config.USD_COMPOSITE_LEGS[symbol]
        return sign * usd_dir

    if symbol in (config.GOLD_SYMBOL, config.SILVER_SYMBOL) and gold_state is not None:
        return -usd_dir

    return 0.0


def compute_deterministic_probabilities(
    symbol: str,
    symbol_intel: dict[str, Any],
    cross_market_state: dict[str, Any],
    gold_state: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    trend_component = _weighted_trend_component(symbol_intel["trends"])
    momentum_component = symbol_intel["momentum"].momentum_score
    structure_liquidity_component = _structure_liquidity_component(symbol_intel["structures"], symbol_intel["liquidity"])
    price_action_component = _price_action_component(symbol_intel["price_action"])
    cross_market_component = _cross_market_component(symbol, cross_market_state, gold_state)

    components = {
        "trend": trend_component,
        "momentum": momentum_component,
        "structure_liquidity": structure_liquidity_component,
        "price_action": price_action_component,
        "cross_market": cross_market_component,
    }
    composite = sum(components[k] * DIRECTIONAL_WEIGHTS[k] for k in DIRECTIONAL_WEIGHTS)
    composite = float(np.clip(composite, -1.0, 1.0))

    edge_strength = abs(composite)
    neutral = max(0.05, 0.25 * (1 - edge_strength))
    remaining = 1.0 - neutral
    prob_up = remaining * (0.5 + 0.5 * composite)
    prob_down = remaining - prob_up

    return {
        "probability_long": float(prob_up),
        "probability_short": float(prob_down),
        "probability_neutral": float(neutral),
        "composite_score": composite,
        "components": components,
    }


def combine_forecasts(
    deterministic: dict[str, Any],
    baseline_probs: Optional[dict[str, float]] = None,
    pattern_memory: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    sources: list[tuple[str, dict[str, float], float]] = [
        ("deterministic", {"UP": deterministic["probability_long"], "DOWN": deterministic["probability_short"], "NEUTRAL": deterministic["probability_neutral"]}, 0.45),
    ]

    if baseline_probs is not None:
        sources.append(("baseline_model", baseline_probs, 0.35))

    if pattern_memory is not None and pattern_memory.get("status") == "OK":
        sources.append(
            (
                "pattern_memory",
                {
                    "UP": pattern_memory["probability_up"],
                    "DOWN": pattern_memory["probability_down"],
                    "NEUTRAL": pattern_memory["probability_neutral"],
                },
                0.20,
            )
        )

    total_weight = sum(w for _, _, w in sources)
    blended = {"UP": 0.0, "DOWN": 0.0, "NEUTRAL": 0.0}
    for _, probs, weight in sources:
        norm_weight = weight / total_weight
        for k in blended:
            blended[k] += probs.get(k, 0.0) * norm_weight

    total = sum(blended.values()) or 1.0
    blended = {k: v / total for k, v in blended.items()}

    return {
        "probability_long": blended["UP"],
        "probability_short": blended["DOWN"],
        "probability_neutral": blended["NEUTRAL"],
        "sources_used": [name for name, _, _ in sources],
        "deterministic_detail": deterministic,
    }
