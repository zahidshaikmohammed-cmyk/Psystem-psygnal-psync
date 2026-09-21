"""Direct unit tests for the contradiction engine's 8-point checklist.

All symbol_intel/cross_market_state objects here are lightweight
SimpleNamespace/dict mocks built specifically to isolate one check at a
time -- not real market data.
"""

from __future__ import annotations

from types import SimpleNamespace

from psygnal.intelligence.contradiction import EvidenceItem, detect_contradictions, evidence_score
from psygnal.intelligence.cross_market_confirmation import CrossMarketConfirmationResult
from psygnal.macro.event_engine import EventContext

EMPTY_EVENT_CONTEXT = EventContext(phase="NONE", nearest_event=None, proximity_bucket=None)


def _neutral_cmc(macro_contradicted=False, evidence=None) -> CrossMarketConfirmationResult:
    return CrossMarketConfirmationResult(
        label="UNCLEAR", macro_confirmed=False, macro_contradicted=macro_contradicted,
        cross_asset_confirmed=False, evidence=evidence or {},
    )


def _base_intel(
    trend_structure="RANGE", momentum_score=0.0, momentum_label="NEUTRAL",
    price_volume_relationship="INCONCLUSIVE", sweep_events=None, reclaim=False,
):
    return {
        "structures": {"M5": SimpleNamespace(trend_structure=trend_structure)},
        "momentum": SimpleNamespace(momentum_score=momentum_score, label=momentum_label),
        "volume": SimpleNamespace(price_volume_relationship=price_volume_relationship),
        "liquidity": SimpleNamespace(sweep_events=sweep_events or [], reclaim=reclaim),
    }


def test_usd_composite_opposing_direction_is_flagged_high_weight():
    intel = _base_intel()
    cross_market_state = {"usd_composite": {"state": "STRENGTHENING"}}
    contradictions = detect_contradictions(
        "EURUSD", 1, intel, cross_market_state, None, _neutral_cmc(), EMPTY_EVENT_CONTEXT
    )
    matches = [c for c in contradictions if "USD composite" in c.text]
    assert matches
    assert matches[0].weight == "HIGH"


def test_missing_rates_provider_always_flags_low_weight_incomplete_check():
    intel = _base_intel()
    contradictions = detect_contradictions(
        "EURUSD", 1, intel, {"usd_composite": {"state": "NEUTRAL"}}, None, _neutral_cmc(), EMPTY_EVENT_CONTEXT
    )
    matches = [c for c in contradictions if "Yield reaction could not be checked" in c.text]
    assert matches
    assert matches[0].weight == "LOW"


def test_opposing_m5_structure_is_flagged():
    intel = _base_intel(trend_structure="DOWNTREND")
    contradictions = detect_contradictions(
        "EURUSD", 1, intel, {"usd_composite": {"state": "NEUTRAL"}}, None, _neutral_cmc(), EMPTY_EVENT_CONTEXT
    )
    assert any("DOWNTREND" in c.text for c in contradictions)


def test_silver_divergence_flagged_only_when_gold_state_present():
    intel = _base_intel()
    gold_state = {"silver_confirmation": "DIVERGING"}
    contradictions = detect_contradictions(
        "XAUUSD", 1, intel, {"usd_composite": {"state": "NEUTRAL"}}, gold_state, _neutral_cmc(), EMPTY_EVENT_CONTEXT
    )
    assert any("XAGUSD" in c.text for c in contradictions)

    contradictions_no_gold = detect_contradictions(
        "EURUSD", 1, intel, {"usd_composite": {"state": "NEUTRAL"}}, None, _neutral_cmc(), EMPTY_EVENT_CONTEXT
    )
    assert not any("XAGUSD" in c.text for c in contradictions_no_gold)


def test_opposing_momentum_is_flagged():
    intel = _base_intel(momentum_score=-0.5, momentum_label="ACCELERATING_BEARISH")
    contradictions = detect_contradictions(
        "EURUSD", 1, intel, {"usd_composite": {"state": "NEUTRAL"}}, None, _neutral_cmc(), EMPTY_EVENT_CONTEXT
    )
    assert any("opposing this direction" in c.text for c in contradictions)


def test_decelerating_same_direction_momentum_is_a_low_weight_flag():
    intel = _base_intel(momentum_score=0.3, momentum_label="DECELERATING_BULLISH")
    contradictions = detect_contradictions(
        "EURUSD", 1, intel, {"usd_composite": {"state": "NEUTRAL"}}, None, _neutral_cmc(), EMPTY_EVENT_CONTEXT
    )
    matches = [c for c in contradictions if "losing participation" in c.text]
    assert matches
    assert matches[0].weight == "LOW"


def test_diverging_volume_is_flagged():
    intel = _base_intel(price_volume_relationship="DIVERGING")
    contradictions = detect_contradictions(
        "EURUSD", 1, intel, {"usd_composite": {"state": "NEUTRAL"}}, None, _neutral_cmc(), EMPTY_EVENT_CONTEXT
    )
    assert any("declining relative volume" in c.text for c in contradictions)


def test_opposing_sweep_reclaim_is_flagged_high_weight():
    sweep_events = [{"recent": True, "direction": "sweep_high"}]
    intel = _base_intel(sweep_events=sweep_events, reclaim=True)
    contradictions = detect_contradictions(
        "EURUSD", 1, intel, {"usd_composite": {"state": "NEUTRAL"}}, None, _neutral_cmc(), EMPTY_EVENT_CONTEXT
    )
    matches = [c for c in contradictions if "just reclaimed the level" in c.text]
    assert matches
    assert matches[0].weight == "HIGH"


def test_same_direction_sweep_reclaim_is_not_flagged_as_contradiction():
    # A sweep_low reclaim does not contradict a LONG -- only an opposing
    # (sweep_high) reclaim does.
    sweep_events = [{"recent": True, "direction": "sweep_low"}]
    intel = _base_intel(sweep_events=sweep_events, reclaim=True)
    contradictions = detect_contradictions(
        "EURUSD", 1, intel, {"usd_composite": {"state": "NEUTRAL"}}, None, _neutral_cmc(), EMPTY_EVENT_CONTEXT
    )
    assert not any("just reclaimed the level" in c.text for c in contradictions)


def test_macro_contradiction_notes_are_surfaced_as_high_weight_evidence():
    cmc = _neutral_cmc(macro_contradicted=True, evidence={"macro_evidence": ["Actual CPI beat forecast but USD CONTRADICTS expected direction"]})
    intel = _base_intel()
    contradictions = detect_contradictions(
        "EURUSD", 1, intel, {"usd_composite": {"state": "NEUTRAL"}}, None, cmc, EMPTY_EVENT_CONTEXT
    )
    matches = [c for c in contradictions if "CONTRADICTS" in c.text]
    assert matches
    assert matches[0].weight == "HIGH"


def test_evidence_score_sums_weighted_values_not_raw_counts():
    items = [EvidenceItem("a", "HIGH"), EvidenceItem("b", "LOW"), EvidenceItem("c", "LOW")]
    assert evidence_score(items) == 3 + 1 + 1
