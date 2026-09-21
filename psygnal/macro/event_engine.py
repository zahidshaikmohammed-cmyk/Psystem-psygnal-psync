"""Economic-event proximity/phase state machine.

Classifies "now" relative to the nearest high-impact event into
PRE_EVENT (T-60..T-5), AT_EVENT (T-5..T0), or POST_EVENT (T0..T+60), and
— only for events that have already happened — builds a transmission
hypothesis from their actual-vs-forecast surprise.

LEAKAGE RULE (non-negotiable, see tests/test_event_leakage.py): an event
whose `time_utc` is in the future relative to `now_utc` may only ever
contribute its *proximity* (used for pre-event risk warnings) — never its
`actual` value, surprise, or transmission hypothesis, even if a caller
mistakenly attached one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from psygnal.macro.calendar import MacroEvent, parse_calendar_events
from psygnal.macro.categories import classify_category
from psygnal.macro.sources import fetch_forex_factory_calendar
from psygnal.macro.surprise import compute_normalized_surprise, compute_surprise
from psygnal.macro.transmission import TransmissionHypothesis, build_transmission_hypothesis
from psygnal.models import DataProvenance, EventPhase

PRE_EVENT_BUCKETS: tuple[int, ...] = (60, 30, 15, 5)
POST_EVENT_BUCKETS: tuple[int, ...] = (1, 5, 15, 30, 60)
AT_EVENT_WINDOW_MINUTES = 5.0


def _parse_numeric(raw: Optional[str]) -> Optional[float]:
    """Parses the calendar's free-form actual/forecast/previous strings
    ("3.1%", "180K", "-0.4") into a float. Returns None — never a
    fabricated 0.0 — for anything it can't confidently parse."""
    if raw is None:
        return None
    text = raw.strip().replace(",", "")
    if not text or text in ("-", "N/A", "n/a"):
        return None
    multiplier = 1.0
    if text.endswith("%"):
        text = text[:-1]
    elif text and text[-1] in "Kk":
        multiplier, text = 1_000.0, text[:-1]
    elif text and text[-1] in "Mm":
        multiplier, text = 1_000_000.0, text[:-1]
    elif text and text[-1] in "Bb":
        multiplier, text = 1_000_000_000.0, text[:-1]
    try:
        return float(text) * multiplier
    except ValueError:
        return None


@dataclass
class EventReactionRecord:
    event_title: str
    category: str
    time_utc: datetime
    actual_provenance: str
    forecast_provenance: str
    surprise_normalized: Optional[float]
    surprise_provenance: str
    transmission: TransmissionHypothesis

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_title": self.event_title,
            "category": self.category,
            "time_utc": self.time_utc.isoformat(),
            "actual_provenance": self.actual_provenance,
            "forecast_provenance": self.forecast_provenance,
            "surprise_normalized": self.surprise_normalized,
            "surprise_provenance": self.surprise_provenance,
            "transmission": self.transmission.as_dict(),
        }


@dataclass
class EventContext:
    phase: str  # EventPhase value
    nearest_event: Optional[dict[str, Any]]
    proximity_bucket: Optional[str]
    reacting_events: list[EventReactionRecord] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "nearest_event": self.nearest_event,
            "proximity_bucket": self.proximity_bucket,
            "reacting_events": [r.as_dict() for r in self.reacting_events],
        }


def enrich_event(event: MacroEvent) -> EventReactionRecord:
    """Builds the enriched reaction record for an event that has already
    happened. Caller is responsible for the leakage guarantee (only ever
    called for `event.time_utc <= now_utc`)."""
    category = classify_category(event.title).value
    actual_numeric = _parse_numeric(event.actual)
    forecast_numeric = _parse_numeric(event.forecast)

    actual_provenance = DataProvenance.VERIFIED.value if actual_numeric is not None else DataProvenance.UNAVAILABLE.value
    forecast_provenance = DataProvenance.VERIFIED.value if forecast_numeric is not None else DataProvenance.UNAVAILABLE.value

    surprise = compute_surprise(actual_numeric, forecast_numeric)
    surprise_normalized = compute_normalized_surprise(surprise, classify_category(event.title))
    surprise_provenance = DataProvenance.DERIVED.value if surprise_normalized is not None else DataProvenance.UNAVAILABLE.value

    transmission = build_transmission_hypothesis(category, surprise_normalized)

    return EventReactionRecord(
        event_title=event.title,
        category=category,
        time_utc=event.time_utc,
        actual_provenance=actual_provenance,
        forecast_provenance=forecast_provenance,
        surprise_normalized=surprise_normalized,
        surprise_provenance=surprise_provenance,
        transmission=transmission,
    )


def build_event_context(
    events: list[MacroEvent],
    now_utc: datetime,
    pre_event_buckets: tuple[int, ...] = PRE_EVENT_BUCKETS,
    post_event_buckets: tuple[int, ...] = POST_EVENT_BUCKETS,
) -> EventContext:
    high_impact = [e for e in events if e.time_utc is not None and e.is_high_impact()]
    if not high_impact:
        return EventContext(phase=EventPhase.NONE.value, nearest_event=None, proximity_bucket=None)

    nearest = min(high_impact, key=lambda e: abs((e.time_utc - now_utc).total_seconds()))
    delta_minutes = (nearest.time_utc - now_utc).total_seconds() / 60.0

    max_pre = max(pre_event_buckets)
    max_post = max(post_event_buckets)

    phase = EventPhase.NONE
    proximity_bucket: Optional[str] = None
    if AT_EVENT_WINDOW_MINUTES >= delta_minutes >= -AT_EVENT_WINDOW_MINUTES:
        phase = EventPhase.AT_EVENT
        proximity_bucket = "T0"
    elif AT_EVENT_WINDOW_MINUTES < delta_minutes <= max_pre:
        phase = EventPhase.PRE_EVENT
        bucket = min((b for b in pre_event_buckets if delta_minutes <= b), default=max_pre)
        proximity_bucket = f"T-{bucket}"
    elif -max_post <= delta_minutes < -AT_EVENT_WINDOW_MINUTES:
        phase = EventPhase.POST_EVENT
        elapsed = -delta_minutes
        bucket = min((b for b in post_event_buckets if elapsed <= b), default=max_post)
        proximity_bucket = f"T+{bucket}"

    reacting: list[EventReactionRecord] = []
    if phase in (EventPhase.AT_EVENT, EventPhase.POST_EVENT):
        for e in high_impact:
            if e.time_utc > now_utc:
                continue  # LEAKAGE GUARD: never react to an event that hasn't happened yet
            elapsed = (now_utc - e.time_utc).total_seconds() / 60.0
            if elapsed > max_post:
                continue
            if e.actual is None:
                continue  # release time passed but no actual value reported yet -> nothing to react to
            reacting.append(enrich_event(e))

    nearest_dict = {
        "title": nearest.title,
        "country": nearest.country,
        "time_utc": nearest.time_utc.isoformat(),
        "minutes_until": round(delta_minutes, 1),
    }
    return EventContext(phase=phase.value, nearest_event=nearest_dict, proximity_bucket=proximity_bucket, reacting_events=reacting)


def get_event_context(now_utc: datetime, enabled: bool = True) -> EventContext:
    """Live-mode convenience wrapper: fetch + parse the calendar (same free
    ForexFactory mirror `macro/calendar.py` uses) and build the event
    context for `now_utc`. Any fetch failure degrades to an empty,
    `EventPhase.NONE` context rather than raising — event risk is optional
    context, not a hard dependency."""
    if not enabled:
        return EventContext(phase=EventPhase.NONE.value, nearest_event=None, proximity_bucket=None)

    result = fetch_forex_factory_calendar()
    if not result.ok:
        return EventContext(phase=EventPhase.NONE.value, nearest_event=None, proximity_bucket=None)

    events = parse_calendar_events(result.data)
    return build_event_context(events, now_utc)
