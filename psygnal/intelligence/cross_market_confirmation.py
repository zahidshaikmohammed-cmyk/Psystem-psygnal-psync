"""Cross-market confirmation/contradiction classifier.

Answers "what kind of evidence is actually behind the current move?" —
never assumes correlations are permanent; everything here is computed
fresh from the current snapshot's already-computed intelligence layers
(USD composite, Gold/Silver relationship, event reactions, news
reconciliation, liquidity/structure state), not a fixed correlation
table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from psygnal.macro.event_engine import EventContext
from psygnal.models import CrossMarketConfirmation
from psygnal.news.interpretation import NewsInterpretation


@dataclass
class CrossMarketConfirmationResult:
    label: str  # CrossMarketConfirmation value
    macro_confirmed: bool
    macro_contradicted: bool
    cross_asset_confirmed: bool
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "macro_confirmed": self.macro_confirmed,
            "macro_contradicted": self.macro_contradicted,
            "cross_asset_confirmed": self.cross_asset_confirmed,
            "evidence": self.evidence,
        }


def _usd_direction_sign(usd_state: dict[str, Any]) -> int:
    return {"STRENGTHENING": 1, "WEAKENING": -1}.get(usd_state.get("state"), 0)


def classify_cross_market_confirmation(
    symbol_direction_sign: int,
    cross_market_state: dict[str, Any],
    gold_state: Optional[dict[str, Any]],
    event_context: EventContext,
    news_interpretation: NewsInterpretation,
    shock_is_shock: bool,
    liquidity_sweep_recent: bool,
    breakout_or_bos_active: bool,
) -> CrossMarketConfirmationResult:
    evidence: dict[str, Any] = {}
    usd_state = cross_market_state.get("usd_composite", {})
    usd_dir = _usd_direction_sign(usd_state)
    evidence["usd_direction"] = usd_dir

    macro_confirmed = False
    macro_contradicted = False
    macro_evidence: list[str] = []
    for record in event_context.reacting_events:
        expected = record.transmission.expected_usd_direction
        if expected == 0 or usd_dir == 0:
            continue
        if expected == usd_dir:
            macro_confirmed = True
            macro_evidence.append(f"{record.event_title}: observed USD reaction matches the theoretical transmission")
        else:
            macro_contradicted = True
            macro_evidence.append(
                f"{record.event_title}: observed USD reaction CONTRADICTS the theoretical transmission "
                f"({record.transmission.rationale})"
            )
    evidence["macro_evidence"] = macro_evidence

    news_reconciliation = None
    if news_interpretation.expected_gold_lean != 0 and symbol_direction_sign != 0:
        news_reconciliation = "CONFIRMED" if news_interpretation.expected_gold_lean == symbol_direction_sign else "CONTRADICTED"
    evidence["news_reconciliation"] = news_reconciliation

    cross_asset_confirmed = False
    if gold_state is not None:
        silver_ok = gold_state.get("silver_confirmation") == "CONFIRMING"
        usd_ok = gold_state.get("usd_alignment") == "SUPPORTIVE"
        cross_asset_confirmed = silver_ok and usd_ok
        evidence["gold_silver_confirmation"] = gold_state.get("silver_confirmation")
        evidence["gold_usd_alignment"] = gold_state.get("usd_alignment")
    else:
        symbol_usd_leg_confirmed = usd_dir != 0 and symbol_direction_sign != 0
        cross_asset_confirmed = symbol_usd_leg_confirmed
        evidence["usd_leg_alignment_checked"] = symbol_usd_leg_confirmed

    if shock_is_shock and event_context.phase in ("AT_EVENT", "POST_EVENT") and event_context.reacting_events:
        label = CrossMarketConfirmation.NEWS_SHOCK
    elif macro_confirmed and macro_contradicted:
        label = CrossMarketConfirmation.MIXED
    elif macro_confirmed:
        label = CrossMarketConfirmation.MACRO_DRIVEN
    elif macro_contradicted:
        label = CrossMarketConfirmation.MIXED
    elif cross_asset_confirmed:
        label = CrossMarketConfirmation.CROSS_ASSET_CONFIRMED
    elif liquidity_sweep_recent and not breakout_or_bos_active:
        label = CrossMarketConfirmation.LIQUIDITY_DRIVEN
    elif breakout_or_bos_active:
        label = CrossMarketConfirmation.TECHNICAL
    else:
        label = CrossMarketConfirmation.UNCLEAR

    return CrossMarketConfirmationResult(
        label=label.value,
        macro_confirmed=macro_confirmed,
        macro_contradicted=macro_contradicted,
        cross_asset_confirmed=cross_asset_confirmed,
        evidence=evidence,
    )
