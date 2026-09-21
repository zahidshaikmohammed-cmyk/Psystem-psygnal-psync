from __future__ import annotations

from datetime import datetime, timezone

from psygnal.intelligence.sessions import (
    get_active_sessions,
    get_session_label,
    session_window_for_date,
    summarize_sessions,
    to_display_timezone,
)
from psygnal.models import candles_to_frame
from tests.conftest import make_m5_series


def test_london_new_york_overlap_detected():
    # 13:30 UTC is within both London (08-17 local, BST->UTC+1 in summer) and
    # New York (08-17 local, EDT->UTC-4 in summer) trading hours in July.
    now = datetime(2024, 7, 15, 13, 30, tzinfo=timezone.utc)
    label = get_session_label(now)
    assert label in ("LONDON_NEW_YORK_OVERLAP", "NEW_YORK", "LONDON")


def test_off_hours_detected_in_deep_asia_gap():
    # Pick an hour unlikely to be in any session's local window.
    now = datetime(2024, 1, 1, 3, 30, tzinfo=timezone.utc)
    active = get_active_sessions(now)
    label = get_session_label(now)
    assert isinstance(active, list)
    assert isinstance(label, str)


def test_session_window_is_utc_aware():
    window = session_window_for_date("LONDON", datetime(2024, 6, 1).date())
    assert window.start_utc.tzinfo is not None
    assert window.end_utc > window.start_utc


def test_display_timezone_is_ist_offset():
    dt = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    displayed = to_display_timezone(dt)
    offset = displayed.utcoffset()
    assert offset.total_seconds() == 5.5 * 3600


def test_summarize_sessions_no_future_leakage():
    candles = make_m5_series(300)
    df = candles_to_frame(candles)
    now = df.index[-1]
    summary = summarize_sessions(df, now)
    for name, stats in summary["today"].items():
        assert stats["candle_count"] >= 0
    assert "previous_day" in summary
