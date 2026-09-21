from __future__ import annotations

from datetime import datetime, timezone

from psygnal.intelligence.cross_market_confirmation import classify_cross_market_confirmation
from psygnal.macro.event_engine import EventContext, EventReactionRecord
from psygnal.macro.transmission import TransmissionHypothesis
from psygnal.news.interpretation import NewsInterpretation

EMPTY_EVENT_CONTEXT = EventContext(phase="NONE", nearest_event=None, proximity_bucket=None)
NEUTRAL_NEWS = NewsInterpretation([], 0, "UNAVAILABLE", "n/a")


def _reacting_event(expected_usd_direction: int) -> EventContext:
    record = EventReactionRecord(
        event_title="CPI m/m",
        category="INFLATION",
        time_utc=datetime(2025, 1, 1, tzinfo=timezone.utc),
        actual_provenance="VERIFIED",
        forecast_provenance="VERIFIED",
        surprise_normalized=1.5,
        surprise_provenance="DERIVED",
        transmission=TransmissionHypothesis(
            category="INFLATION", surprise_sign=1, expected_usd_direction=expected_usd_direction,
            rationale="test", confidence="THEORETICAL",
        ),
    )
    return EventContext(phase="POST_EVENT", nearest_event=None, proximity_bucket="T+15", reacting_events=[record])


def test_macro_driven_when_usd_reaction_confirms_theory():
    ctx = _reacting_event(expected_usd_direction=1)
    cross_market_state = {"usd_composite": {"state": "STRENGTHENING"}}
    result = classify_cross_market_confirmation(
        symbol_direction_sign=-1, cross_market_state=cross_market_state, gold_state=None,
        event_context=ctx, news_interpretation=NEUTRAL_NEWS, shock_is_shock=False,
        liquidity_sweep_recent=False, breakout_or_bos_active=False,
    )
    assert result.label == "MACRO_DRIVEN"
    assert result.macro_confirmed is True


def test_news_shock_when_event_reaction_and_shock_coincide():
    ctx = _reacting_event(expected_usd_direction=1)
    cross_market_state = {"usd_composite": {"state": "STRENGTHENING"}}
    result = classify_cross_market_confirmation(
        symbol_direction_sign=-1, cross_market_state=cross_market_state, gold_state=None,
        event_context=ctx, news_interpretation=NEUTRAL_NEWS, shock_is_shock=True,
        liquidity_sweep_recent=False, breakout_or_bos_active=False,
    )
    assert result.label == "NEWS_SHOCK"


def test_mixed_when_usd_reaction_contradicts_theory():
    ctx = _reacting_event(expected_usd_direction=1)
    cross_market_state = {"usd_composite": {"state": "WEAKENING"}}  # opposite of theoretical +1
    result = classify_cross_market_confirmation(
        symbol_direction_sign=1, cross_market_state=cross_market_state, gold_state=None,
        event_context=ctx, news_interpretation=NEUTRAL_NEWS, shock_is_shock=False,
        liquidity_sweep_recent=False, breakout_or_bos_active=False,
    )
    assert result.label == "MIXED"
    assert result.macro_contradicted is True


def test_cross_asset_confirmed_for_gold_when_silver_and_usd_align():
    gold_state = {"silver_confirmation": "CONFIRMING", "usd_alignment": "SUPPORTIVE"}
    result = classify_cross_market_confirmation(
        symbol_direction_sign=1, cross_market_state={"usd_composite": {"state": "NEUTRAL"}}, gold_state=gold_state,
        event_context=EMPTY_EVENT_CONTEXT, news_interpretation=NEUTRAL_NEWS, shock_is_shock=False,
        liquidity_sweep_recent=False, breakout_or_bos_active=False,
    )
    assert result.label == "CROSS_ASSET_CONFIRMED"
    assert result.cross_asset_confirmed is True


def test_liquidity_driven_when_sweep_without_other_confirmation():
    result = classify_cross_market_confirmation(
        symbol_direction_sign=1, cross_market_state={"usd_composite": {"state": "NEUTRAL"}}, gold_state=None,
        event_context=EMPTY_EVENT_CONTEXT, news_interpretation=NEUTRAL_NEWS, shock_is_shock=False,
        liquidity_sweep_recent=True, breakout_or_bos_active=False,
    )
    assert result.label == "LIQUIDITY_DRIVEN"


def test_technical_when_breakout_without_other_confirmation():
    result = classify_cross_market_confirmation(
        symbol_direction_sign=1, cross_market_state={"usd_composite": {"state": "NEUTRAL"}}, gold_state=None,
        event_context=EMPTY_EVENT_CONTEXT, news_interpretation=NEUTRAL_NEWS, shock_is_shock=False,
        liquidity_sweep_recent=False, breakout_or_bos_active=True,
    )
    assert result.label == "TECHNICAL"


def test_unclear_when_nothing_confirms():
    result = classify_cross_market_confirmation(
        symbol_direction_sign=0, cross_market_state={"usd_composite": {"state": "NEUTRAL"}}, gold_state=None,
        event_context=EMPTY_EVENT_CONTEXT, news_interpretation=NEUTRAL_NEWS, shock_is_shock=False,
        liquidity_sweep_recent=False, breakout_or_bos_active=False,
    )
    assert result.label == "UNCLEAR"
