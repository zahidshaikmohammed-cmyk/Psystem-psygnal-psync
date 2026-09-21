from __future__ import annotations

from datetime import datetime, timedelta, timezone

from psygnal.intelligence.candle_features import compute_candle_features
from psygnal.intelligence.shock import detect_shock
from psygnal.models import Candle, candles_to_frame
from tests.conftest import make_m5_series


def test_no_shock_in_calm_market():
    candles = make_m5_series(120, step=0.05)
    feats = compute_candle_features(candles_to_frame(candles))
    state = detect_shock(feats, relative_volume=1.0, atr_value=0.1)
    assert state.is_shock is False
    assert state.severity == "NONE"


def test_shock_detected_on_displacement_plus_volume_spike():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = []
    price = 100.0
    for i in range(60):
        o, c = price, price + (0.05 if i % 2 == 0 else -0.05)
        h, l = max(o, c) + 0.02, min(o, c) - 0.02
        candles.append(Candle(time=start + timedelta(minutes=5 * i), open=o, high=h, low=l, close=c, volume=100))
        price = c
    # A big displacement candle with a huge volume spike.
    o = price
    c = price + 5.0
    h, l = c + 0.1, o - 0.1
    candles.append(Candle(time=start + timedelta(minutes=5 * 60), open=o, high=h, low=l, close=c, volume=5000))

    feats = compute_candle_features(candles_to_frame(candles))
    state = detect_shock(feats, relative_volume=4.0, atr_value=0.2)
    assert state.is_shock is True
    assert state.severity in ("MODERATE", "SEVERE")
    assert state.evidence["abnormal_range"] is True
    assert state.evidence["abnormal_volume"] is True


def test_single_trigger_alone_is_not_a_shock():
    # Choppy/alternating series so consecutive-direction doesn't
    # incidentally fire alongside the one signal under test.
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = []
    price = 100.0
    for i in range(60):
        o, c = price, price + (0.05 if i % 2 == 0 else -0.05)
        h, l = max(o, c) + 0.02, min(o, c) - 0.02
        candles.append(Candle(time=start + timedelta(minutes=5 * i), open=o, high=h, low=l, close=c, volume=100))
        price = c
    feats = compute_candle_features(candles_to_frame(candles))
    # Only relative_volume is elevated; nothing structural changed.
    state = detect_shock(feats, relative_volume=3.0, atr_value=10.0)
    assert state.evidence["trigger_count"] <= 1
    assert state.is_shock is False


def test_consecutive_directional_candles_contribute_a_trigger():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = []
    price = 100.0
    for i in range(20):
        o, c = price, price + 0.3  # always up
        h, l = c + 0.02, o - 0.02
        candles.append(Candle(time=start + timedelta(minutes=5 * i), open=o, high=h, low=l, close=c, volume=100))
        price = c
    feats = compute_candle_features(candles_to_frame(candles))
    state = detect_shock(feats, relative_volume=1.0, atr_value=10.0)
    assert state.evidence["consecutive_directional"] is True


def test_empty_features_returns_no_shock_gracefully():
    import pandas as pd

    state = detect_shock(pd.DataFrame(), relative_volume=None, atr_value=None)
    assert state.is_shock is False
    assert state.severity == "NONE"
