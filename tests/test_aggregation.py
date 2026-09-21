from __future__ import annotations

from datetime import datetime, timedelta, timezone

from psygnal.data.aggregation import (
    aggregate_timeframe,
    build_multi_timeframe_series,
    split_forming_m5,
)
from tests.conftest import make_m5_series


def test_split_forming_m5_excludes_open_bucket():
    candles = make_m5_series(10)
    last_time = candles[-1].time
    now = last_time + timedelta(minutes=2)  # bucket [last, last+5) still open
    completed, forming = split_forming_m5(candles, now=now)
    assert len(completed) == 9
    assert forming is not None
    assert forming.time == last_time


def test_split_forming_m5_includes_closed_bucket():
    candles = make_m5_series(10)
    now = candles[-1].time + timedelta(minutes=5, seconds=1)
    completed, forming = split_forming_m5(candles, now=now)
    assert len(completed) == 10
    assert forming is None


def test_aggregate_m15_is_mathematically_correct():
    candles = make_m5_series(12)  # exactly 4 M15 buckets
    completed, forming = aggregate_timeframe(candles, 3)
    assert forming is None
    assert len(completed) == 4
    first_group = candles[0:3]
    agg = completed[0]
    assert agg.open == first_group[0].open
    assert agg.close == first_group[-1].close
    assert agg.high == max(c.high for c in first_group)
    assert agg.low == min(c.low for c in first_group)
    assert agg.volume == sum(c.volume for c in first_group)


def test_aggregate_marks_incomplete_final_bucket_as_forming():
    candles = make_m5_series(10)  # 3 full M15 buckets would need 9; 10 leaves 1 partial
    completed, forming = aggregate_timeframe(candles, 3)
    assert len(completed) == 3
    assert forming is not None
    assert forming.completed is False


def test_h4_aggregation_needs_48_m5_candles():
    candles = make_m5_series(48)
    completed, forming = aggregate_timeframe(candles, 48)
    assert len(completed) == 1
    assert forming is None


def test_build_multi_timeframe_series_no_future_leakage():
    candles = make_m5_series(300)
    now = candles[-1].time + timedelta(minutes=6)
    mts = build_multi_timeframe_series("EURUSD", candles, now=now)
    for tf in ("M5", "M15", "M30", "H1", "H4"):
        frame = mts.get(tf)
        if frame.empty:
            continue
        latest_completed_time = frame.index[-1]
        assert latest_completed_time <= now
        if mts.forming[tf] is not None:
            assert mts.forming[tf].time >= latest_completed_time
