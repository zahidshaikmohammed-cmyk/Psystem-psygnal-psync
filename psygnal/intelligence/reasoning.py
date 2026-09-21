"""Human-like reasoning layer.

Builds an explicit `MarketHypothesis`: the standing directional lean, the
evidence for and against it (weighted by information quality — see
`intelligence/contradiction.py::EvidenceItem` — not just counted), what
would invalidate it, and whether acting on it right now is actually
warranted (TRADEABLE / WAIT / NOT_TRADEABLE).

This is the mechanism behind statements like: "the technical structure is
bearish, but the macro environment is unclear, USD is not confirming, and
a major event is approaching — therefore the correct action is WAIT," or
conversely "multiple independent evidence streams agree — but here is
what would prove this wrong."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from psygnal.intelligence.contradiction import EvidenceItem, detect_contradictions, evidence_score
from psygnal.intelligence.cross_market_confirmation import CrossMarketConfirmationResult
from psygnal.macro.event_engine import EventContext
from psygnal.macro.providers import RatesProvider
from psygnal.models import Tradeability


@dataclass
class MarketHypothesis:
    direction: str
    confirming_evidence: list[EvidenceItem]
    contradicting_evidence: list[EvidenceItem]
    invalidation_conditions: list[str]
    tradeability: str
    tradeability_reasons: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "confirming_evidence": [e.as_dict() for e in self.confirming_evidence],
            "contradicting_evidence": [e.as_dict() for e in self.contradicting_evidence],
            "invalidation_conditions": self.invalidation_conditions,
            "tradeability": self.tradeability,
            "tradeability_reasons": self.tradeability_reasons,
        }


def gather_confirming_evidence(
    direction_sign: int,
    symbol_intel: dict[str, Any],
    cross_market_confirmation: CrossMarketConfirmationResult,
    gold_state: Optional[dict[str, Any]],
) -> list[EvidenceItem]:
    evidence: list[EvidenceItem] = []
    trends = symbol_intel["trends"]

    h4, h1 = trends.get("H4"), trends.get("H1")
    higher_tf_agree = 0
    if h4 is not None and abs(h4.score) > 0.15 and (h4.score > 0) == (direction_sign > 0):
        higher_tf_agree += 1
    if h1 is not None and abs(h1.score) > 0.15 and (h1.score > 0) == (direction_sign > 0):
        higher_tf_agree += 1
    if higher_tf_agree == 2:
        evidence.append(EvidenceItem("H4 and H1 trend both align with this direction.", "HIGH"))
    elif higher_tf_agree == 1:
        evidence.append(EvidenceItem("One higher timeframe (H1 or H4) aligns with this direction.", "MEDIUM"))

    m5_structure = symbol_intel["structures"].get("M5")
    if m5_structure is not None:
        if direction_sign > 0 and m5_structure.last_bos == "BULLISH_BOS":
            evidence.append(EvidenceItem("M5 shows a bullish break of structure.", "MEDIUM"))
        elif direction_sign < 0 and m5_structure.last_bos == "BEARISH_BOS":
            evidence.append(EvidenceItem("M5 shows a bearish break of structure.", "MEDIUM"))

    liquidity = symbol_intel["liquidity"]
    favorable_sweep = any(
        e.get("recent")
        and (
            (direction_sign > 0 and e["direction"] == "sweep_low")
            or (direction_sign < 0 and e["direction"] == "sweep_high")
        )
        for e in liquidity.sweep_events
    )
    if favorable_sweep:
        evidence.append(EvidenceItem("A liquidity sweep in the opposing direction has reversed, favoring this hypothesis.", "HIGH"))

    momentum = symbol_intel["momentum"]
    if (momentum.momentum_score > 0) == (direction_sign > 0) and "ACCELERATING" in momentum.label:
        evidence.append(EvidenceItem("Momentum is accelerating in this direction.", "MEDIUM"))

    if cross_market_confirmation.label in ("MACRO_DRIVEN", "CROSS_ASSET_CONFIRMED"):
        evidence.append(
            EvidenceItem(f"Cross-market evidence is {cross_market_confirmation.label.replace('_', ' ').lower()}.", "HIGH")
        )

    if gold_state is not None and gold_state.get("silver_confirmation") == "CONFIRMING":
        evidence.append(EvidenceItem("XAGUSD confirms the XAUUSD move.", "MEDIUM"))

    volume = symbol_intel["volume"]
    if volume.price_volume_relationship == "CONFIRMING_EXPANSION":
        evidence.append(EvidenceItem("Price expansion is supported by elevated relative volume.", "LOW"))

    return evidence


def build_invalidation_conditions(
    direction_sign: int,
    symbol_intel: dict[str, Any],
    event_context: EventContext,
) -> list[str]:
    conditions: list[str] = []
    liquidity = symbol_intel["liquidity"]
    # LONG is invalidated by losing support below; SHORT is invalidated by
    # reclaiming resistance above -- not the other way around.
    opposing_zone = liquidity.nearest_below if direction_sign > 0 else liquidity.nearest_above
    if opposing_zone is not None:
        side = "below" if direction_sign > 0 else "above"
        conditions.append(
            f"A confirmed close {side} {opposing_zone.label} ({opposing_zone.price:.5g}) would invalidate this hypothesis."
        )

    conditions.append("A reversal in the USD composite's direction would remove cross-market support for this view.")

    if event_context.phase == "PRE_EVENT" and event_context.nearest_event is not None:
        conditions.append(
            f"An unexpected surprise from {event_context.nearest_event['title']} could immediately invalidate this view."
        )

    return conditions


def assess_event_risk_flags(event_context: EventContext) -> list[str]:
    flags: list[str] = []
    if event_context.phase == "AT_EVENT":
        flags.append("A high-impact event is releasing right now (T0) — outcome unknown.")
    elif event_context.phase == "PRE_EVENT" and event_context.proximity_bucket in ("T-5", "T-15"):
        title = event_context.nearest_event["title"] if event_context.nearest_event else "a high-impact event"
        flags.append(f"{title} is due within 15 minutes ({event_context.proximity_bucket}).")
    return flags


def assess_tradeability(
    symbol_intel_status: str,
    regime_label: str,
    confirming_evidence: list[EvidenceItem],
    contradicting_evidence: list[EvidenceItem],
    cross_market_confirmation_label: str,
    event_risk_flags: list[str],
    rr: Optional[float],
) -> tuple[str, list[str]]:
    if symbol_intel_status != "OK":
        return Tradeability.NOT_TRADEABLE.value, ["Insufficient/degraded market data."]

    if regime_label == "CHAOTIC":
        return Tradeability.NOT_TRADEABLE.value, ["Regime is CHAOTIC — no coherent structure to trade against."]

    if any("releasing right now" in f for f in event_risk_flags):
        return Tradeability.NOT_TRADEABLE.value, list(event_risk_flags)

    reasons: list[str] = list(event_risk_flags)

    if regime_label == "TRANSITION":
        reasons.append("Regime is in TRANSITION — structure just flipped and hasn't confirmed a new direction.")

    if cross_market_confirmation_label == "MIXED":
        reasons.append("Cross-market evidence is MIXED — some streams confirm, others contradict.")

    confirm_score = evidence_score(confirming_evidence)
    contradict_score = evidence_score(contradicting_evidence)
    if contradict_score > 0 and contradict_score >= confirm_score:
        reasons.append(
            f"Contradicting evidence (score {contradict_score}) outweighs or matches confirming evidence (score {confirm_score})."
        )

    if rr is not None and rr < 1.0:
        reasons.append(f"Risk/reward to TP1 is poor ({rr:.2f}:1).")

    if reasons:
        return Tradeability.WAIT.value, reasons

    return Tradeability.TRADEABLE.value, ["Evidence streams are aligned and no material contradiction or event risk was found."]


def build_hypothesis(
    symbol: str,
    direction: str,
    symbol_intel: dict[str, Any],
    cross_market_state: dict[str, Any],
    gold_state: Optional[dict[str, Any]],
    cross_market_confirmation: CrossMarketConfirmationResult,
    event_context: EventContext,
    regime_label: str,
    rr: Optional[float],
    rates_provider: Optional[RatesProvider] = None,
) -> MarketHypothesis:
    if symbol_intel.get("status") != "OK":
        return MarketHypothesis(
            direction=direction,
            confirming_evidence=[],
            contradicting_evidence=[],
            invalidation_conditions=[],
            tradeability=Tradeability.NOT_TRADEABLE.value,
            tradeability_reasons=["Insufficient/degraded market data."],
        )

    direction_sign = 1 if direction == "LONG" else -1

    confirming = gather_confirming_evidence(direction_sign, symbol_intel, cross_market_confirmation, gold_state)
    contradicting = detect_contradictions(
        symbol,
        direction_sign,
        symbol_intel,
        cross_market_state,
        gold_state,
        cross_market_confirmation,
        event_context,
        rates_provider,
    )
    invalidation = build_invalidation_conditions(direction_sign, symbol_intel, event_context)
    event_risk_flags = assess_event_risk_flags(event_context)

    tradeability, tradeability_reasons = assess_tradeability(
        symbol_intel.get("status", "UNAVAILABLE"),
        regime_label,
        confirming,
        contradicting,
        cross_market_confirmation.label,
        event_risk_flags,
        rr,
    )

    return MarketHypothesis(
        direction=direction,
        confirming_evidence=confirming,
        contradicting_evidence=contradicting,
        invalidation_conditions=invalidation,
        tradeability=tradeability,
        tradeability_reasons=tradeability_reasons,
    )
