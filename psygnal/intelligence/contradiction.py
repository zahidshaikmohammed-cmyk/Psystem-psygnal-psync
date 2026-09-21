"""Contradiction engine.

Actively looks for evidence AGAINST the standing directional hypothesis,
per the mission's explicit checklist: is USD actually moving against it,
is price accepting on the wrong side of structure, is silver failing to
confirm, is momentum stalling, is volume failing to support the move, has
a broken level been reclaimed against it, and did the market react
opposite to the theoretical macro interpretation. The engine must not
cling to its own hypothesis in the face of this evidence — see
`intelligence/reasoning.py::build_hypothesis`, which feeds these into the
tradeability decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from psygnal.forecasting.ensemble import cross_market_component
from psygnal.intelligence.cross_market_confirmation import CrossMarketConfirmationResult
from psygnal.macro.event_engine import EventContext
from psygnal.macro.providers import RatesProvider, UnavailableRatesProvider

_WEIGHT_SCORE = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}


@dataclass
class EvidenceItem:
    text: str
    weight: str  # "HIGH" | "MEDIUM" | "LOW" — information quality, not just a tally

    def as_dict(self) -> dict[str, str]:
        return {"text": self.text, "weight": self.weight}


def evidence_score(items: list[EvidenceItem]) -> int:
    """Weighted tally — per the mission's information-quality principle
    (Phase 16), two or three HIGH-weight items should not be outvoted by
    a pile of LOW-weight ones."""
    return sum(_WEIGHT_SCORE.get(i.weight, 1) for i in items)


def detect_contradictions(
    symbol: str,
    direction_sign: int,
    symbol_intel: dict[str, Any],
    cross_market_state: dict[str, Any],
    gold_state: Optional[dict[str, Any]],
    cross_market_confirmation: CrossMarketConfirmationResult,
    event_context: EventContext,
    rates_provider: Optional[RatesProvider] = None,
) -> list[EvidenceItem]:
    contradictions: list[EvidenceItem] = []
    rates_provider = rates_provider or UnavailableRatesProvider()

    # 1. Is USD actually moving against what this direction implies?
    usd_component = cross_market_component(symbol, cross_market_state, gold_state)
    if usd_component != 0 and (usd_component > 0) != (direction_sign > 0):
        usd_state = cross_market_state.get("usd_composite", {}).get("state", "UNAVAILABLE")
        contradictions.append(EvidenceItem(f"USD composite is {usd_state}, which does not support this direction.", "HIGH"))

    # 2. Are yields moving contrary? No live source is wired up (see
    # macro/providers.py::UnavailableRatesProvider) — say so explicitly
    # rather than silently skipping the check.
    yield_result = rates_provider.fetch_yield("US10Y")
    if yield_result.status == "UNAVAILABLE":
        contradictions.append(
            EvidenceItem("Yield reaction could not be checked (no RatesProvider configured) — macro confirmation is incomplete.", "LOW")
        )

    # 3. Is price accepting on the wrong side of structure?
    m5_structure = symbol_intel["structures"].get("M5")
    if m5_structure is not None:
        if direction_sign > 0 and m5_structure.trend_structure == "DOWNTREND":
            contradictions.append(EvidenceItem("M5 structure is DOWNTREND — price is not accepting higher.", "MEDIUM"))
        elif direction_sign < 0 and m5_structure.trend_structure == "UPTREND":
            contradictions.append(EvidenceItem("M5 structure is UPTREND — price is not accepting lower.", "MEDIUM"))

    # 4. Is silver confirming (Gold only)?
    if gold_state is not None and gold_state.get("silver_confirmation") == "DIVERGING":
        contradictions.append(EvidenceItem("XAGUSD is diverging from XAUUSD rather than confirming.", "MEDIUM"))

    # 5. Is momentum continuing in this direction? Is it losing participation?
    momentum = symbol_intel["momentum"]
    if (momentum.momentum_score > 0) != (direction_sign > 0) and abs(momentum.momentum_score) > 0.1:
        contradictions.append(EvidenceItem(f"Momentum is {momentum.label.replace('_', ' ').lower()}, opposing this direction.", "MEDIUM"))
    elif "DECELERATING" in momentum.label and (momentum.momentum_score > 0) == (direction_sign > 0):
        contradictions.append(EvidenceItem("Momentum is decelerating — the move may be losing participation.", "LOW"))

    # 6. Is volume failing to support the move?
    volume = symbol_intel["volume"]
    if volume.price_volume_relationship == "DIVERGING":
        contradictions.append(EvidenceItem("Price expansion is occurring on declining relative volume.", "MEDIUM"))

    # 7. Has a level been reclaimed AGAINST this direction?
    liquidity = symbol_intel["liquidity"]
    opposing_sweep_reclaimed = any(
        e.get("recent")
        and (
            (direction_sign > 0 and e["direction"] == "sweep_high")
            or (direction_sign < 0 and e["direction"] == "sweep_low")
        )
        for e in liquidity.sweep_events
    )
    if liquidity.reclaim and opposing_sweep_reclaimed:
        contradictions.append(EvidenceItem("A liquidity sweep against this direction has just reclaimed the level.", "HIGH"))

    # 8. Did the market react opposite to the theoretical macro interpretation?
    if cross_market_confirmation.macro_contradicted:
        for note in cross_market_confirmation.evidence.get("macro_evidence", []):
            if "CONTRADICTS" in note:
                contradictions.append(EvidenceItem(note, "HIGH"))

    return contradictions
