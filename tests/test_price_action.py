from __future__ import annotations

from datetime import datetime, timedelta, timezone

from psygnal.intelligence.candle_features import compute_candle_features
from psygnal.intelligence.price_action import classify_price_action
from psygnal.models import Candle, candles_to_frame


def _trending_candles(n: int, direction: int = 1, step: float = 0.2, start_price: float = 100.0) -> list[Candle]:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = []
    price = start_price
    for i in range(n):
        o = price
        c = price + direction * step
        h = max(o, c) + 0.05
        l = min(o, c) - 0.05
        candles.append(Candle(time=start + timedelta(minutes=5 * i), open=o, high=h, low=l, close=c, volume=100))
        price = c
    return candles


def test_insufficient_data_label():
    candles = _trending_candles(3)
    feats = compute_candle_features(candles_to_frame(candles))
    state = classify_price_action(feats)
    assert state.label == "insufficient_data"


def test_strong_persistent_uptrend_is_continuation_or_impulse():
    candles = _trending_candles(60, direction=1, step=0.3)
    feats = compute_candle_features(candles_to_frame(candles))
    state = classify_price_action(feats)
    assert state.label in ("continuation", "impulse", "breakout")


def test_flat_choppy_market_is_consolidation():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = []
    price = 100.0
    for i in range(60):
        o = price
        c = price + (0.05 if i % 2 == 0 else -0.05)
        h = max(o, c) + 0.02
        l = min(o, c) - 0.02
        candles.append(Candle(time=start + timedelta(minutes=5 * i), open=o, high=h, low=l, close=c, volume=100))
        price = c
    feats = compute_candle_features(candles_to_frame(candles))
    state = classify_price_action(feats)
    assert state.label == "consolidation"


def test_breakout_detected_when_closing_beyond_recent_range():
    candles = _trending_candles(24, direction=1, step=0.02)  # tight range
    start = candles[-1].time + timedelta(minutes=5)
    price = candles[-1].close
    # one big breakout candle far beyond the recent range
    breakout = Candle(time=start, open=price, high=price + 5, low=price - 0.05, close=price + 4.8, volume=500)
    candles.append(breakout)
    feats = compute_candle_features(candles_to_frame(candles))
    state = classify_price_action(feats)
    assert state.label == "breakout"
    assert state.evidence["broke_up"] is True
