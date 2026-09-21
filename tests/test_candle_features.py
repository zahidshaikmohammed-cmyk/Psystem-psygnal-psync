from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from psygnal.intelligence.candle_features import compute_candle_features
from psygnal.models import Candle, candles_to_frame
from tests.conftest import make_m5_series


def test_basic_anatomy_fields_present():
    candles = make_m5_series(150)
    df = candles_to_frame(candles)
    feats = compute_candle_features(df)
    for col in ["return", "log_return", "body", "upper_wick", "lower_wick", "true_range", "close_location_value", "directional_strength", "behavior"]:
        assert col in feats.columns
    assert len(feats) == len(df)


def test_close_location_value_bounds():
    candles = make_m5_series(150)
    df = candles_to_frame(candles)
    feats = compute_candle_features(df)
    clv = feats["close_location_value"].dropna()
    assert (clv >= -1.0001).all()
    assert (clv <= 1.0001).all()


def test_inside_and_outside_candle_detection():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = [
        Candle(time=start, open=10, high=12, low=8, close=11, volume=100),
        Candle(time=start + timedelta(minutes=5), open=10.5, high=11.5, low=9.5, close=10.8, volume=100),  # inside
        Candle(time=start + timedelta(minutes=10), open=10, high=13, low=7, close=12, volume=100),  # outside
    ]
    df = candles_to_frame(candles)
    feats = compute_candle_features(df)
    assert bool(feats["is_inside"].iloc[1]) is True
    assert bool(feats["is_outside"].iloc[2]) is True


def test_strong_bullish_displacement_detected():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = []
    price = 100.0
    # small choppy candles to build a baseline range distribution
    for i in range(60):
        o = price
        c = price + (0.05 if i % 2 == 0 else -0.05)
        h = max(o, c) + 0.02
        l = min(o, c) - 0.02
        candles.append(Candle(time=start + timedelta(minutes=5 * i), open=o, high=h, low=l, close=c, volume=100))
        price = c
    # one big displacement candle
    o = price
    c = price + 5.0
    h = c + 0.05
    l = o - 0.05
    candles.append(Candle(time=start + timedelta(minutes=5 * 60), open=o, high=h, low=l, close=c, volume=500))

    df = candles_to_frame(candles)
    feats = compute_candle_features(df)
    assert feats["behavior"].iloc[-1] == "strong_bullish_displacement"


def test_zero_range_candle_does_not_crash():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = [
        Candle(time=start, open=10, high=10, low=10, close=10, volume=0),
        Candle(time=start + timedelta(minutes=5), open=10, high=10.5, low=9.5, close=10.2, volume=10),
    ]
    df = candles_to_frame(candles)
    feats = compute_candle_features(df)
    assert np.isnan(feats["body_pct"].iloc[0])
