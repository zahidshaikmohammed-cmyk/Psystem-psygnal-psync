from __future__ import annotations

from datetime import datetime, timedelta

from psygnal.data.aggregation import build_multi_timeframe_series
from psygnal.intelligence.context import build_symbol_intelligence
from psygnal.intelligence.regime import classify_regime
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
    assert state.label in ("TREND_DOWN", "VOLATILITY_EXPANSION", "BREAKOUT")


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
