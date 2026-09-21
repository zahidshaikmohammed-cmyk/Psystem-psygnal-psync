from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from psygnal.intelligence.candle_features import compute_candle_features
from psygnal.intelligence.momentum import analyze_momentum
from psygnal.intelligence.structure import analyze_structure
from psygnal.intelligence.trend import analyze_trend
from psygnal.intelligence.volatility import analyze_volatility
from psygnal.intelligence.volume import analyze_volume
from psygnal.models import Candle, candles_to_frame


def _trending_candles(n: int, direction: int = 1, step: float = 0.3, start_price: float = 100.0) -> list[Candle]:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = []
    price = start_price
    for i in range(n):
        o = price
        c = price + direction * step
        h = max(o, c) + 0.05
        l = min(o, c) - 0.05
        candles.append(Candle(time=start + timedelta(minutes=5 * i), open=o, high=h, low=l, close=c, volume=100 + i))
        price = c
    return candles


def test_trend_strong_uptrend_scores_positive():
    candles = _trending_candles(260, direction=1, step=0.3)
    df = candles_to_frame(candles)
    structure = analyze_structure(df, "M5")
    state = analyze_trend(df, structure, "M5")
    assert state.score > 0
    assert state.label in ("BULLISH", "STRONG_BULLISH")


def test_trend_strong_downtrend_scores_negative():
    candles = _trending_candles(260, direction=-1, step=0.3, start_price=500.0)
    df = candles_to_frame(candles)
    structure = analyze_structure(df, "M5")
    state = analyze_trend(df, structure, "M5")
    assert state.score < 0
    assert state.label in ("BEARISH", "STRONG_BEARISH")


def test_trend_insufficient_data_is_neutral():
    candles = _trending_candles(10)
    df = candles_to_frame(candles)
    structure = analyze_structure(df, "M5")
    state = analyze_trend(df, structure, "M5")
    assert state.label == "NEUTRAL"


def test_momentum_uptrend_is_bullish():
    candles = _trending_candles(120, direction=1, step=0.4)
    df = candles_to_frame(candles)
    state = analyze_momentum(df)
    assert state.momentum_score > 0
    assert "BULLISH" in state.label


def test_momentum_downtrend_is_bearish():
    candles = _trending_candles(120, direction=-1, step=0.4, start_price=500.0)
    df = candles_to_frame(candles)
    state = analyze_momentum(df)
    assert state.momentum_score < 0
    assert "BEARISH" in state.label


def test_volatility_expansion_detected_after_range_widens():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = []
    price = 100.0
    for i in range(60):
        step = 0.05
        o, c = price, price + step
        h, l = max(o, c) + 0.02, min(o, c) - 0.02
        candles.append(Candle(time=start + timedelta(minutes=5 * i), open=o, high=h, low=l, close=c, volume=100))
        price = c
    for i in range(60, 80):
        step = 2.0
        o, c = price, price + step
        h, l = max(o, c) + 0.1, min(o, c) - 0.1
        candles.append(Candle(time=start + timedelta(minutes=5 * i), open=o, high=h, low=l, close=c, volume=100))
        price = c
    df = candles_to_frame(candles)
    state = analyze_volatility(df)
    assert state.label == "HIGH"


def test_volume_elevated_on_relative_volume_spike():
    candles = _trending_candles(40, direction=1, step=0.1)
    # spike the final candle's volume well above the rolling average
    start = candles[-1].time + timedelta(minutes=5)
    from psygnal.models import Candle as C

    price = candles[-1].close
    candles.append(C(time=start, open=price, high=price + 1, low=price - 0.1, close=price + 0.9, volume=10000))
    df = candles_to_frame(candles)
    feats = compute_candle_features(df)
    state = analyze_volume(feats)
    assert state.label == "ELEVATED"
    assert state.relative_volume > 1.5
