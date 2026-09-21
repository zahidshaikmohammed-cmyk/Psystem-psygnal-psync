from __future__ import annotations

from datetime import datetime, timedelta, timezone

from psygnal.data.parser import RawCandleRecord
from psygnal.data.validation import validate_symbol
from psygnal.models import DataStatus


def _rec(t, o, h, l, c, v=100.0):
    return RawCandleRecord(time=t, open=o, high=h, low=l, close=c, volume=v)


def test_valid_series_status_ok():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    records = [
        _rec(start + timedelta(minutes=5 * i), 1.1, 1.11, 1.09, 1.105)
        for i in range(150)
    ]
    now = records[-1].time + timedelta(minutes=1)
    candles, quality = validate_symbol("EURUSD", records, now=now, min_candles_required=100)
    assert quality.status == DataStatus.OK
    assert len(candles) == 150


def test_invalid_ohlc_relationship_dropped():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    records = [
        _rec(start, 1.1, 0.5, 1.09, 1.105),  # high < close, invalid
        _rec(start + timedelta(minutes=5), 1.1, 1.11, 1.09, 1.105),
    ]
    candles, quality = validate_symbol("EURUSD", records, now=start + timedelta(minutes=10))
    assert len(candles) == 1
    assert any("invalid OHLC" in i for i in quality.issues)


def test_non_positive_price_dropped():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    records = [
        _rec(start, -1.0, 1.11, 1.09, 1.105),
        _rec(start + timedelta(minutes=5), 1.1, 1.11, 1.09, 1.105),
    ]
    candles, quality = validate_symbol("EURUSD", records, now=start + timedelta(minutes=10))
    assert len(candles) == 1
    assert any("non-positive" in i for i in quality.issues)


def test_duplicates_collapsed():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    records = [
        _rec(start, 1.1, 1.11, 1.09, 1.10),
        _rec(start, 1.1, 1.12, 1.09, 1.11),  # duplicate timestamp, keep last
    ]
    candles, quality = validate_symbol("EURUSD", records, now=start + timedelta(minutes=10))
    assert len(candles) == 1
    assert candles[0].close == 1.11
    assert any("duplicate" in w for w in quality.warnings)


def test_empty_series_is_unavailable():
    candles, quality = validate_symbol("EURUSD", [], now=datetime.now(timezone.utc))
    assert quality.status == DataStatus.UNAVAILABLE
    assert candles == []


def test_insufficient_depth_is_degraded():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    records = [
        _rec(start + timedelta(minutes=5 * i), 1.1, 1.11, 1.09, 1.105) for i in range(5)
    ]
    now = records[-1].time + timedelta(minutes=1)
    candles, quality = validate_symbol("EURUSD", records, now=now, min_candles_required=100)
    assert quality.status == DataStatus.DEGRADED
    assert len(candles) == 5


def test_staleness_warning():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    records = [
        _rec(start + timedelta(minutes=5 * i), 1.1, 1.11, 1.09, 1.105) for i in range(150)
    ]
    now = records[-1].time + timedelta(hours=2)
    candles, quality = validate_symbol(
        "EURUSD", records, now=now, min_candles_required=100, max_staleness_minutes=20
    )
    assert any("stale" in w for w in quality.warnings)
