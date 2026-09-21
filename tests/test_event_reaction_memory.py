"""Tests for historical event-reaction memory.

All price/event fixtures here are SYNTHETIC, constructed in-test. They
validate the computation mechanism and its leakage/honesty guarantees —
they are not real market history and must never be treated as such.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from psygnal.forecasting.event_reaction_memory import (
    MIN_SAMPLES_FOR_STATISTIC,
    compute_event_reaction_statistics,
    query_reaction_statistic,
)
from psygnal.macro.calendar import MacroEvent
from psygnal.models import candles_to_frame
from tests.conftest import make_m5_series


def _flat_price_frame(n=2000, start=None):
    candles = make_m5_series(n, start=start, step=0.0)
    return candles_to_frame(candles)


def test_insufficient_sample_reported_when_too_few_events():
    df = _flat_price_frame()
    events = [
        MacroEvent(
            title="CPI m/m", country="USD", time_utc=df.index[100 + i * 200], impact="High", forecast="0.2%", previous="0.2%", actual="0.5%"
        )
        for i in range(3)  # far fewer than MIN_SAMPLES_FOR_STATISTIC
    ]
    stats = compute_event_reaction_statistics(df, events)
    assert len(stats) == 1
    assert stats[0].status == "INSUFFICIENT_HISTORICAL_SAMPLE"
    assert stats[0].n_samples == 3


def test_sufficient_sample_produces_ok_status_with_real_mean_reaction():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    n = 200 * 30
    candles = make_m5_series(n, start=start, step=0.0)
    # Manufacture a consistent +1.0 price jump exactly 5 minutes after each event.
    events = []
    for i in range(MIN_SAMPLES_FOR_STATISTIC + 5):
        event_index = 50 + i * 30
        event_time = candles[event_index].time
        events.append(
            MacroEvent(title="CPI m/m", country="USD", time_utc=event_time, impact="High", forecast="0.2%", previous="0.2%", actual="0.9%")
        )
        # bump every candle from event_index+1 onward by +1.0 to simulate a lasting reaction
        for j in range(event_index + 1, min(event_index + 15, n)):
            candles[j] = candles[j].__class__(
                time=candles[j].time, open=candles[j].open + 1.0, high=candles[j].high + 1.0,
                low=candles[j].low + 1.0, close=candles[j].close + 1.0, volume=candles[j].volume,
            )
    df = candles_to_frame(candles)
    stats = compute_event_reaction_statistics(df, events)
    assert len(stats) == 1
    assert stats[0].status == "OK"
    assert stats[0].n_samples >= MIN_SAMPLES_FOR_STATISTIC
    assert stats[0].mean_return_by_horizon[5] > 0


def test_events_without_actual_are_ignored_not_fabricated():
    df = _flat_price_frame()
    events = [
        MacroEvent(title="CPI m/m", country="USD", time_utc=df.index[100], impact="High", forecast="0.2%", previous="0.2%", actual=None)
    ]
    stats = compute_event_reaction_statistics(df, events)
    assert stats == []


def test_low_impact_events_are_excluded():
    df = _flat_price_frame()
    events = [
        MacroEvent(title="Some Minor Release", country="USD", time_utc=df.index[100], impact="Low", forecast="0.2%", previous="0.2%", actual="0.5%")
    ]
    stats = compute_event_reaction_statistics(df, events)
    assert stats == []


def test_query_reaction_statistic_returns_insufficient_when_not_found():
    result = query_reaction_statistic([], "INFLATION", "POSITIVE")
    assert result.status == "INSUFFICIENT_HISTORICAL_SAMPLE"
    assert result.n_samples == 0
