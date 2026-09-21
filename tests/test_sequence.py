from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from psygnal.intelligence.candle_features import compute_candle_features
from psygnal.intelligence.sequence import compute_all_window_stats, compute_window_stats
from psygnal.models import Candle, candles_to_frame


def _trending_candles(n: int, direction: int = 1, step: float = 0.2) -> list[Candle]:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = []
    price = 100.0
    for i in range(n):
        o = price
        c = price + direction * step
        h = max(o, c) + 0.05
        l = min(o, c) - 0.05
        candles.append(Candle(time=start + timedelta(minutes=5 * i), open=o, high=h, low=l, close=c, volume=100 + i))
        price = c
    return candles


def test_uptrend_has_high_persistence_and_positive_dominant_direction():
    candles = _trending_candles(60, direction=1)
    feats = compute_candle_features(candles_to_frame(candles))
    stats = compute_window_stats(feats, 24)
    assert stats is not None
    assert stats.dominant_direction == 1
    assert stats.trend_persistence == pytest.approx(1.0)
    assert stats.consecutive_direction >= 20


def test_downtrend_dominant_direction_negative():
    candles = _trending_candles(60, direction=-1)
    feats = compute_candle_features(candles_to_frame(candles))
    stats = compute_window_stats(feats, 24)
    assert stats.dominant_direction == -1


def test_window_stats_none_when_insufficient_history():
    candles = _trending_candles(5)
    feats = compute_candle_features(candles_to_frame(candles))
    stats = compute_window_stats(feats, 24)
    assert stats is None


def test_compute_all_window_stats_only_returns_satisfiable_windows():
    candles = _trending_candles(20)
    feats = compute_candle_features(candles_to_frame(candles))
    stats = compute_all_window_stats(feats, windows=(3, 6, 12, 24, 48))
    assert set(stats.keys()) == {3, 6, 12}


def test_acceleration_positive_when_move_speeds_up():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = []
    price = 100.0
    steps = [0.05] * 6 + [0.5] * 6  # accelerating uptrend
    for i, step in enumerate(steps):
        o = price
        c = price + step
        h = c + 0.02
        l = o - 0.02
        candles.append(Candle(time=start + timedelta(minutes=5 * i), open=o, high=h, low=l, close=c, volume=100))
        price = c
    feats = compute_candle_features(candles_to_frame(candles))
    stats = compute_window_stats(feats, 12)
    assert stats.acceleration > 0
