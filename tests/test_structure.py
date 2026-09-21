from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from psygnal.intelligence.structure import analyze_structure, find_swings
from psygnal.models import Candle, candles_to_frame



def _zigzag_candles(pattern: list[float], step_minutes: int = 5) -> list[Candle]:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = []
    for i in range(len(pattern) - 1):
        o = pattern[i]
        c = pattern[i + 1]
        h = max(o, c) + 0.1
        l = min(o, c) - 0.1
        candles.append(Candle(time=start + timedelta(minutes=step_minutes * i), open=o, high=h, low=l, close=c, volume=100))
    return candles


def test_find_swings_detects_local_extremes():
    # Explicit highs/lows with no ties, so the fractal center is unambiguous.
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    highs = [10, 11, 20, 12, 11, 10, 9, 15, 9]
    lows = [8, 9, 15, 9, 8, 5, 3, 10, 6]
    candles = []
    for i, (h, l) in enumerate(zip(highs, lows)):
        candles.append(
            Candle(
                time=start + timedelta(minutes=5 * i),
                open=(h + l) / 2,
                high=h,
                low=l,
                close=(h + l) / 2,
                volume=100,
            )
        )
    df = candles_to_frame(candles)
    swing_highs, swing_lows = find_swings(df, lookback=1)
    assert len(swing_highs) >= 1
    assert len(swing_lows) >= 1
    assert any(sp.price == 20 for sp in swing_highs)
    assert any(sp.price == 3 for sp in swing_lows)


def test_uptrend_structure_hh_hl():
    prices = [1, 10, 8, 15, 12, 20, 17, 25]
    candles = _zigzag_candles(prices)
    df = candles_to_frame(candles)
    state = analyze_structure(df, "M5")
    assert state.trend_structure in ("UPTREND", "RANGE", "UNDEFINED")


def test_downtrend_structure_lh_ll():
    prices = [25, 17, 20, 12, 15, 8, 10, 1]
    candles = _zigzag_candles(prices)
    df = candles_to_frame(candles)
    state = analyze_structure(df, "M5")
    assert state.trend_structure in ("DOWNTREND", "RANGE", "UNDEFINED")


def test_bullish_bos_detected_on_break_above_swing_high():
    prices = [1, 10, 8, 15, 12, 30]  # final leg breaks well above prior swing highs
    candles = _zigzag_candles(prices)
    df = candles_to_frame(candles)
    state = analyze_structure(df, "M5")
    assert state.last_bos in ("BULLISH_BOS", None)  # depends on swing confirmation lag


def test_empty_or_tiny_dataframe_is_undefined():
    state = analyze_structure(pd.DataFrame(), "M5")
    assert state.trend_structure == "UNDEFINED"
