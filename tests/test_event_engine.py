"""Tests for the economic-event proximity/transmission engine.

All fixtures here are SYNTHETIC — constructed MacroEvent objects, not
data pulled from any live source. They exist to validate the state
machine and leakage guarantees, not to represent a real calendar.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from psygnal.macro.calendar import MacroEvent
from psygnal.macro.categories import MacroCategory, classify_category
from psygnal.macro.event_engine import build_event_context
from psygnal.macro.surprise import compute_normalized_surprise, compute_surprise
from psygnal.macro.transmission import build_transmission_hypothesis
from psygnal.models import DataProvenance, EventPhase


def _event(minutes_from_now: float, now: datetime, actual=None, forecast="0.3%", title="CPI m/m") -> MacroEvent:
    return MacroEvent(
        title=title,
        country="USD",
        time_utc=now + timedelta(minutes=minutes_from_now),
        impact="High",
        forecast=forecast,
        previous="0.2%",
        actual=actual,
    )


def test_classify_category_covers_major_types():
    assert classify_category("CPI y/y") == MacroCategory.INFLATION
    assert classify_category("FOMC Statement") == MacroCategory.MONETARY_POLICY
    assert classify_category("Jerome Powell Speaks") == MacroCategory.CENTRAL_BANK_SPEECH
    assert classify_category("Non-Farm Employment Change") == MacroCategory.EMPLOYMENT
    assert classify_category("ISM Manufacturing PMI") == MacroCategory.MANUFACTURING
    assert classify_category("Existing Home Sales") == MacroCategory.HOUSING
    assert classify_category("Some Totally Unknown Release") == MacroCategory.OTHER


def test_compute_surprise_none_when_missing_inputs():
    assert compute_surprise(None, 0.3) is None
    assert compute_surprise(0.5, None) is None
    assert compute_surprise(0.5, 0.3) == pytest.approx(0.2)


def test_normalized_surprise_scales_by_category():
    inflation_norm = compute_normalized_surprise(0.15, MacroCategory.INFLATION)
    employment_norm = compute_normalized_surprise(60.0, MacroCategory.EMPLOYMENT)
    assert inflation_norm == pytest.approx(1.0)
    assert employment_norm == pytest.approx(1.0)


def test_transmission_hypothesis_in_line_has_no_directional_prior():
    hyp = build_transmission_hypothesis("INFLATION", 0.05)
    assert hyp.surprise_sign == 0
    assert hyp.expected_usd_direction == 0


def test_transmission_hypothesis_hot_inflation_implies_usd_strength_as_theoretical_prior():
    hyp = build_transmission_hypothesis("INFLATION", 1.5)
    assert hyp.surprise_sign == 1
    assert hyp.expected_usd_direction == 1
    assert hyp.confidence == "THEORETICAL"
    assert "theoretical prior" in hyp.rationale.lower()


def test_transmission_hypothesis_central_bank_speech_has_no_reliable_sign():
    hyp = build_transmission_hypothesis("CENTRAL_BANK_SPEECH", 2.0)
    assert hyp.expected_usd_direction == 0


def test_event_phase_none_when_no_high_impact_events():
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    ctx = build_event_context([], now)
    assert ctx.phase == EventPhase.NONE.value


def test_event_phase_pre_event_at_various_buckets():
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    for minutes, expected_bucket in [(45, "T-60"), (25, "T-30"), (10, "T-15"), (3, None)]:
        event = _event(minutes, now)
        ctx = build_event_context([event], now)
        if expected_bucket is None:
            assert ctx.phase == EventPhase.AT_EVENT.value
        else:
            assert ctx.phase == EventPhase.PRE_EVENT.value
            assert ctx.proximity_bucket == expected_bucket


def test_event_phase_pre_event_never_includes_reacting_events():
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    event = _event(30, now)  # 30 minutes in the future, no actual yet
    ctx = build_event_context([event], now)
    assert ctx.phase == EventPhase.PRE_EVENT.value
    assert ctx.reacting_events == []


def test_event_phase_post_event_builds_reaction_when_actual_known():
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    event = _event(-15, now, actual="0.6%")  # released 15 minutes ago, actual now known
    ctx = build_event_context([event], now)
    assert ctx.phase == EventPhase.POST_EVENT.value
    assert ctx.proximity_bucket == "T+15"
    assert len(ctx.reacting_events) == 1
    record = ctx.reacting_events[0]
    assert record.actual_provenance == DataProvenance.VERIFIED.value
    assert record.surprise_provenance == DataProvenance.DERIVED.value
    assert record.transmission.expected_usd_direction == 1  # hotter CPI -> theoretical USD strength prior


def test_event_phase_post_event_no_reaction_when_actual_not_yet_reported():
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    event = _event(-15, now, actual=None)  # released but source hasn't posted the actual value yet
    ctx = build_event_context([event], now)
    assert ctx.phase == EventPhase.POST_EVENT.value
    assert ctx.reacting_events == []


def test_event_beyond_post_window_has_no_reaction_record():
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    event = _event(-90, now, actual="0.6%")  # released 90 minutes ago, beyond the 60-minute post window
    ctx = build_event_context([event], now)
    assert ctx.phase == EventPhase.NONE.value
    assert ctx.reacting_events == []
