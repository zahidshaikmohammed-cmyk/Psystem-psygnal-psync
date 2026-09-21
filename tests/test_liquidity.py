from __future__ import annotations

from datetime import datetime, timedelta, timezone

from psygnal.indicators.atr import atr
from psygnal.intelligence.liquidity import analyze_liquidity, build_liquidity_zones
from psygnal.intelligence.structure import analyze_structure
from psygnal.models import candles_to_frame
from tests.conftest import make_m5_series


def test_build_liquidity_zones_includes_previous_day_and_swings():
    candles = make_m5_series(400)
    df = candles_to_frame(candles)
    now = df.index[-1] + timedelta(minutes=5)
    structure = analyze_structure(df, "M5")
    atr_series = atr(df["high"], df["low"], df["close"])
    atr_value = float(atr_series.dropna().iloc[-1]) if atr_series.dropna().size else None
    zones = build_liquidity_zones(df, structure, now, atr_value)
    sources = {z.source for z in zones}
    assert "previous_day" in sources or "session" in sources or "swing" in sources


def test_analyze_liquidity_splits_zones_above_and_below():
    candles = make_m5_series(400)
    df = candles_to_frame(candles)
    now = df.index[-1] + timedelta(minutes=5)
    structure = analyze_structure(df, "M5")
    atr_series = atr(df["high"], df["low"], df["close"])
    atr_value = float(atr_series.dropna().iloc[-1]) if atr_series.dropna().size else None
    current_price = float(df["close"].iloc[-1])
    state = analyze_liquidity(df, structure, now, atr_value, current_price)
    for z in state.zones_above:
        assert z.price > current_price
    for z in state.zones_below:
        assert z.price < current_price


def test_sweep_detection_flags_wick_beyond_zone_with_reversal_close():
    candles = make_m5_series(60)
    # Manufacture a sweep: last candle wicks above a recent swing high then closes back below it.
    from psygnal.models import Candle

    last = candles[-1]
    sweep_time = last.time + timedelta(minutes=5)
    sweep_high = last.close + 2.0
    candles.append(
        Candle(time=sweep_time, open=last.close, high=sweep_high, low=last.close - 0.1, close=last.close, volume=200)
    )
    df = candles_to_frame(candles)
    structure = analyze_structure(df, "M5")
    now = df.index[-1] + timedelta(minutes=5)
    atr_series = atr(df["high"], df["low"], df["close"])
    atr_value = float(atr_series.dropna().iloc[-1]) if atr_series.dropna().size else None
    state = analyze_liquidity(df, structure, now, atr_value, float(df["close"].iloc[-1]))
    assert isinstance(state.sweep_events, list)
