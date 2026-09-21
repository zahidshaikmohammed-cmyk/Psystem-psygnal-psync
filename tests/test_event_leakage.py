"""Dedicated future-leakage tests for the economic-event engine.

Non-negotiable per the mission brief: an event's actual/surprise/
transmission may only ever be used once its own release time has passed,
and never merged back into a "before" analysis. All fixtures are
synthetic, constructed in-test — never real calendar data.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from psygnal.macro.calendar import MacroEvent
from psygnal.macro.event_engine import build_event_context, enrich_event
from psygnal.models import EventPhase


def _future_event_with_actual_already_set(now: datetime, minutes_ahead: float) -> MacroEvent:
    """Simulates a buggy/malicious upstream feed that already populated
    `actual` for an event that hasn't happened yet — the engine must
    still refuse to react to it."""
    return MacroEvent(
        title="NFP",
        country="USD",
        time_utc=now + timedelta(minutes=minutes_ahead),
        impact="High",
        forecast="180K",
        previous="150K",
        actual="250K",
    )


def test_future_event_never_produces_a_reaction_record_even_with_actual_set():
    now = datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc)
    for minutes_ahead in (1, 10, 30, 59, 90):
        event = _future_event_with_actual_already_set(now, minutes_ahead)
        ctx = build_event_context([event], now)
        assert ctx.reacting_events == [], f"leaked at +{minutes_ahead}m"


def test_event_exactly_at_release_time_with_no_actual_yet_does_not_react():
    now = datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc)
    event = MacroEvent(title="NFP", country="USD", time_utc=now, impact="High", forecast="180K", previous="150K", actual=None)
    ctx = build_event_context([event], now)
    assert ctx.phase == EventPhase.AT_EVENT.value
    assert ctx.reacting_events == []


def test_pre_event_context_at_multiple_time_steps_never_reveals_the_outcome():
    """Walk a single event across T-60 down to T-1 and confirm no step
    ever exposes actual/surprise/transmission — only proximity."""
    now = datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc)
    release_time = now + timedelta(minutes=60)
    event = MacroEvent(title="CPI m/m", country="USD", time_utc=release_time, impact="High", forecast="0.3%", previous="0.2%", actual="0.9%")

    for elapsed in range(0, 60, 5):
        step_now = now + timedelta(minutes=elapsed)
        ctx = build_event_context([event], step_now)
        assert ctx.reacting_events == [], f"leaked at now+{elapsed}m (release at +60m)"
        if ctx.nearest_event is not None:
            # The nearest-event summary may show timing, never the outcome.
            assert "actual" not in ctx.nearest_event
            assert "surprise" not in ctx.nearest_event


def test_post_event_walk_only_reveals_reaction_after_release():
    """Walk across the release boundary and confirm the reaction record
    appears only strictly after `time_utc`, never before or exactly at
    a moment the source hadn't posted `actual` for."""
    now = datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc)
    release_time = now
    event_before_release_known = MacroEvent(
        title="CPI m/m", country="USD", time_utc=release_time, impact="High", forecast="0.3%", previous="0.2%", actual=None
    )
    event_after_release_known = MacroEvent(
        title="CPI m/m", country="USD", time_utc=release_time, impact="High", forecast="0.3%", previous="0.2%", actual="0.6%"
    )

    # Before release: even if we (incorrectly) queried with an "actual" already
    # attached to the fixture, using the correct fixture (no actual yet) must
    # yield no reaction.
    minus_1 = build_event_context([event_before_release_known], now - timedelta(minutes=1))
    assert minus_1.reacting_events == []

    # After release with actual now known: reaction appears.
    plus_1 = build_event_context([event_after_release_known], now + timedelta(minutes=1))
    assert len(plus_1.reacting_events) == 1


def test_enrich_event_is_a_pure_function_of_the_event_itself():
    """enrich_event must not consult wall-clock time or any external
    state — it only transforms the event's own already-known fields."""
    now = datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc)
    event = MacroEvent(title="CPI m/m", country="USD", time_utc=now, impact="High", forecast="0.3%", previous="0.2%", actual="0.6%")
    record_a = enrich_event(event)
    record_b = enrich_event(event)
    assert record_a.as_dict() == record_b.as_dict()
