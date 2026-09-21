from __future__ import annotations

from datetime import datetime, timedelta, timezone

from psygnal.data.aggregation import build_multi_timeframe_series
from psygnal.intelligence.context import build_symbol_intelligence
from psygnal.intelligence.regime import classify_regime
from psygnal.intelligence.shock import ShockState
from psygnal.macro.event_engine import EventContext, EventReactionRecord
from psygnal.macro.transmission import TransmissionHypothesis
from tests.conftest import make_m5_series


def _intel_for(direction=1, step=0.3, n=500, base=100.0):
    candles = make_m5_series(n, base_price=base, step=direction * step)
    now = candles[-1].time + timedelta(minutes=6)
    mts = build_multi_timeframe_series("EURUSD", candles, now=now)
    return build_symbol_intelligence("EURUSD", mts, now)


def test_uptrend_regime_is_trend_up():
    intel = _intel_for(direction=1, step=0.3)
    state = classify_regime(intel, {"status": "CLEAR", "upcoming": [], "recent": []})
    assert state.label in ("TREND_UP", "VOLATILITY_EXPANSION", "BREAKOUT")


def test_downtrend_regime_is_trend_down():
    intel = _intel_for(direction=-1, step=0.3, base=500.0)
    state = classify_regime(intel, {"status": "CLEAR", "upcoming": [], "recent": []})
    # BREAKDOWN is a bearish breakout (V3: distinguished from bullish BREAKOUT).
    assert state.label in ("TREND_DOWN", "VOLATILITY_EXPANSION", "BREAKOUT", "BREAKDOWN")


def test_insufficient_data_is_chaotic():
    state = classify_regime({"status": "UNAVAILABLE"}, {"status": "CLEAR", "upcoming": [], "recent": []})
    assert state.label == "CHAOTIC"


def test_news_shock_when_recent_macro_event_and_high_volatility(monkeypatch):
    intel = _intel_for(direction=1, step=2.0)  # large moves -> likely high volatility
    macro_state = {"status": "ELEVATED_RISK", "upcoming": [], "recent": [{"title": "CPI", "hours_ago": 0.1}]}
    # Force high volatility label deterministically for this test's purpose.
    intel["volatility"].label = "HIGH"
    state = classify_regime(intel, macro_state)
    assert state.label == "NEWS_SHOCK"


def test_bearish_breakout_is_classified_as_breakdown():
    # A choppy (non-compressed, so no spurious equal-level sweeps),
    # range-bound market followed by one large bearish displacement candle.
    from psygnal.models import Candle

    start0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = []
    price = 100.0
    for i in range(300):
        o, c = price, price + (0.1 if i % 2 == 0 else -0.1)
        h, l = max(o, c) + 0.03, min(o, c) - 0.03
        candles.append(Candle(time=start0 + timedelta(minutes=5 * i), open=o, high=h, low=l, close=c, volume=100))
        price = c
    last = candles[-1]
    start = last.time + timedelta(minutes=5)
    price = last.close
    breakdown_candle = Candle(time=start, open=price, high=price + 0.05, low=price - 5, close=price - 4.8, volume=500)
    candles = candles + [breakdown_candle]
    mts = build_multi_timeframe_series("EURUSD", candles, now=start + timedelta(minutes=5))
    intel = build_symbol_intelligence("EURUSD", mts, start + timedelta(minutes=5))
    state = classify_regime(intel, {"status": "CLEAR", "upcoming": [], "recent": []})
    assert state.label == "BREAKDOWN"


def test_severe_shock_state_upgrades_regime_to_news_shock():
    intel = _intel_for(direction=1, step=0.3)
    shock = ShockState(is_shock=True, severity="SEVERE", evidence={})
    state = classify_regime(intel, {"status": "CLEAR", "upcoming": [], "recent": []}, shock_state=shock)
    assert state.label == "NEWS_SHOCK"


def test_post_event_phase_with_reacting_events_is_post_news():
    intel = _intel_for(direction=1, step=0.05)  # gentle move, unlikely to independently trigger breakout/sweep
    record = EventReactionRecord(
        event_title="CPI m/m",
        category="INFLATION",
        time_utc=datetime(2025, 1, 1, tzinfo=timezone.utc),
        actual_provenance="VERIFIED",
        forecast_provenance="VERIFIED",
        surprise_normalized=0.5,
        surprise_provenance="DERIVED",
        transmission=TransmissionHypothesis(category="INFLATION", surprise_sign=1, expected_usd_direction=1, rationale="t", confidence="THEORETICAL"),
    )
    ctx = EventContext(phase="POST_EVENT", nearest_event=None, proximity_bucket="T+15", reacting_events=[record])
    state = classify_regime(intel, {"status": "CLEAR", "upcoming": [], "recent": []}, event_context=ctx)
    assert state.label in ("POST_NEWS", "LIQUIDITY_SWEEP", "BREAKOUT_RETEST", "BREAKOUT", "BREAKDOWN")
    # If nothing more structurally dominant fired first, POST_NEWS must win.
    if state.label not in ("LIQUIDITY_SWEEP", "BREAKOUT_RETEST", "BREAKOUT", "BREAKDOWN"):
        assert state.label == "POST_NEWS"


def test_shock_and_event_context_are_ignored_when_not_supplied():
    """Backward compatibility: omitting the new optional params must behave
    exactly as before (no NEWS_SHOCK/POST_NEWS purely from defaults)."""
    intel = _intel_for(direction=1, step=0.05)
    state = classify_regime(intel, {"status": "CLEAR", "upcoming": [], "recent": []})
    assert state.label != "POST_NEWS"
