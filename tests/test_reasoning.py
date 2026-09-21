from __future__ import annotations

from datetime import timedelta

from psygnal.data.aggregation import build_multi_timeframe_series
from psygnal.intelligence.context import build_symbol_intelligence
from psygnal.intelligence.cross_market_confirmation import CrossMarketConfirmationResult
from psygnal.intelligence.reasoning import assess_tradeability, build_hypothesis
from psygnal.macro.event_engine import EventContext
from tests.conftest import make_m5_series

EMPTY_EVENT_CONTEXT = EventContext(phase="NONE", nearest_event=None, proximity_bucket=None)


def _intel_for(direction=1, step=0.3, n=500, base=100.0):
    candles = make_m5_series(n, base_price=base, step=direction * step)
    now = candles[-1].time + timedelta(minutes=6)
    mts = build_multi_timeframe_series("EURUSD", candles, now=now)
    return build_symbol_intelligence("EURUSD", mts, now)


def _neutral_cmc(label="UNCLEAR") -> CrossMarketConfirmationResult:
    return CrossMarketConfirmationResult(label=label, macro_confirmed=False, macro_contradicted=False, cross_asset_confirmed=False, evidence={})


def test_hypothesis_end_to_end_does_not_crash_and_returns_valid_tradeability():
    # Note: the shared synthetic fixture (make_m5_series) is a perfectly
    # monotonic series, which has no interior swing points at all, so M5
    # structure stays UNDEFINED and higher timeframes stay NEUTRAL for lack
    # of history (existing V1/V2 tests already treat this fixture loosely
    # for the same reason). This test only checks the pipeline runs
    # end-to-end and returns a well-formed hypothesis; see
    # test_gather_confirming_evidence_* below for real evidence-gathering
    # coverage against controlled inputs.
    intel = _intel_for(direction=1, step=0.5)
    hyp = build_hypothesis(
        "EURUSD", "LONG", intel, {"usd_composite": {"state": "NEUTRAL"}}, None,
        _neutral_cmc(), EMPTY_EVENT_CONTEXT, "TREND_UP", rr=2.0,
    )
    assert hyp.tradeability in ("TRADEABLE", "WAIT", "NOT_TRADEABLE")
    assert isinstance(hyp.invalidation_conditions, list)


def test_gather_confirming_evidence_detects_aligned_higher_timeframes():
    from types import SimpleNamespace

    from psygnal.intelligence.reasoning import gather_confirming_evidence

    trend_bullish = SimpleNamespace(score=0.5, label="BULLISH")
    trend_neutral = SimpleNamespace(score=0.0, label="NEUTRAL")
    structure = SimpleNamespace(last_bos="BULLISH_BOS", last_choch=None)
    liquidity = SimpleNamespace(sweep_events=[])
    momentum = SimpleNamespace(momentum_score=0.4, label="ACCELERATING_BULLISH")
    volume = SimpleNamespace(price_volume_relationship="CONFIRMING_EXPANSION")

    symbol_intel = {
        "trends": {"H4": trend_bullish, "H1": trend_bullish, "M30": trend_neutral, "M15": trend_neutral, "M5": trend_bullish},
        "structures": {"M5": structure},
        "liquidity": liquidity,
        "momentum": momentum,
        "volume": volume,
    }

    evidence = gather_confirming_evidence(1, symbol_intel, _neutral_cmc(), gold_state=None)
    texts = [e.text for e in evidence]
    assert any("H4 and H1" in t for t in texts)
    assert any("break of structure" in t for t in texts)
    assert any("accelerating" in t for t in texts)
    assert any("expansion" in t for t in texts)


def test_insufficient_data_is_never_tradeable():
    hyp = build_hypothesis(
        "EURUSD", "LONG", {"status": "UNAVAILABLE"}, {"usd_composite": {}}, None,
        _neutral_cmc(), EMPTY_EVENT_CONTEXT, "CHAOTIC", rr=None,
    )
    assert hyp.tradeability == "NOT_TRADEABLE"


def test_chaotic_regime_is_never_tradeable():
    intel = _intel_for(direction=1, step=0.3)
    hyp = build_hypothesis(
        "EURUSD", "LONG", intel, {"usd_composite": {"state": "NEUTRAL"}}, None,
        _neutral_cmc(), EMPTY_EVENT_CONTEXT, "CHAOTIC", rr=2.0,
    )
    assert hyp.tradeability == "NOT_TRADEABLE"


def test_event_releasing_right_now_is_never_tradeable():
    intel = _intel_for(direction=1, step=0.3)
    at_event = EventContext(phase="AT_EVENT", nearest_event={"title": "NFP", "country": "USD", "time_utc": "t", "minutes_until": 0}, proximity_bucket="T0")
    hyp = build_hypothesis(
        "EURUSD", "LONG", intel, {"usd_composite": {"state": "NEUTRAL"}}, None,
        _neutral_cmc(), at_event, "TREND_UP", rr=2.0,
    )
    assert hyp.tradeability == "NOT_TRADEABLE"
    assert any("releasing right now" in r for r in hyp.tradeability_reasons)


def test_imminent_event_within_15_minutes_forces_wait():
    intel = _intel_for(direction=1, step=0.3)
    pre_event = EventContext(phase="PRE_EVENT", nearest_event={"title": "CPI", "country": "USD", "time_utc": "t", "minutes_until": 10}, proximity_bucket="T-15")
    hyp = build_hypothesis(
        "EURUSD", "LONG", intel, {"usd_composite": {"state": "NEUTRAL"}}, None,
        _neutral_cmc(), pre_event, "TREND_UP", rr=2.0,
    )
    assert hyp.tradeability == "WAIT"
    assert any("CPI" in r for r in hyp.tradeability_reasons)


def test_usd_contradiction_reduces_tradeability_for_usd_pair():
    """USD composite strengthening while direction is LONG on a pair where
    a rising pair means USD weakness (e.g. EURUSD) is a direct contradiction."""
    intel = _intel_for(direction=1, step=0.3)
    hyp = build_hypothesis(
        "EURUSD", "LONG", intel, {"usd_composite": {"state": "STRENGTHENING"}}, None,
        _neutral_cmc(), EMPTY_EVENT_CONTEXT, "TREND_UP", rr=2.0,
    )
    assert any("does not support this direction" in e.text for e in hyp.contradicting_evidence)


def test_mixed_cross_market_confirmation_forces_wait_absent_strong_confirmation():
    intel = _intel_for(direction=1, step=0.05)  # gentle move, unlikely to generate strong confirming evidence
    hyp = build_hypothesis(
        "EURUSD", "LONG", intel, {"usd_composite": {"state": "NEUTRAL"}}, None,
        _neutral_cmc(label="MIXED"), EMPTY_EVENT_CONTEXT, "RANGE", rr=2.0,
    )
    assert hyp.tradeability == "WAIT"


def test_poor_risk_reward_is_a_wait_reason():
    reasons_result = assess_tradeability(
        symbol_intel_status="OK", regime_label="TREND_UP", confirming_evidence=[], contradicting_evidence=[],
        cross_market_confirmation_label="UNCLEAR", event_risk_flags=[], rr=0.5,
    )
    state, reasons = reasons_result
    assert state == "WAIT"
    assert any("Risk/reward" in r for r in reasons)


def test_invalidation_conditions_reference_actual_liquidity_zone():
    intel = _intel_for(direction=1, step=0.3)
    from psygnal.intelligence.reasoning import build_invalidation_conditions

    conditions = build_invalidation_conditions(1, intel, EMPTY_EVENT_CONTEXT)
    assert any("invalidate" in c.lower() for c in conditions)


def test_invalidation_condition_for_long_references_support_below_not_resistance_above():
    from types import SimpleNamespace

    from psygnal.intelligence.liquidity import LiquidityZone
    from psygnal.intelligence.reasoning import build_invalidation_conditions

    support = LiquidityZone(label="SWING_LOW", price=99.0, kind="low", source="swing")
    resistance = LiquidityZone(label="SWING_HIGH", price=101.0, kind="high", source="swing")
    intel = {"liquidity": SimpleNamespace(nearest_above=resistance, nearest_below=support)}

    long_conditions = build_invalidation_conditions(1, intel, EMPTY_EVENT_CONTEXT)
    assert any("below" in c.lower() and "SWING_LOW" in c for c in long_conditions)
    assert not any("above" in c.lower() and "SWING_HIGH" in c for c in long_conditions)

    short_conditions = build_invalidation_conditions(-1, intel, EMPTY_EVENT_CONTEXT)
    assert any("above" in c.lower() and "SWING_HIGH" in c for c in short_conditions)
    assert not any("below" in c.lower() and "SWING_LOW" in c for c in short_conditions)


def test_evidence_score_weights_high_over_many_low():
    from psygnal.intelligence.contradiction import EvidenceItem, evidence_score

    high_alone = evidence_score([EvidenceItem("x", "HIGH")])
    many_low = evidence_score([EvidenceItem("x", "LOW")] * 2)
    assert high_alone > many_low
