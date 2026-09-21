"""Macro calendar parsing and event-risk contextualization.

Tracks the major event categories the constitution names (CPI, NFP, FOMC,
GDP, PPI, ISM, Jobless Claims, JOLTS, central-bank decisions, ...) and
reports whether the current moment sits near one of them. This is context
for the ensemble/score, never a hard gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from psygnal.data.parser import coerce_time
from psygnal.macro.sources import fetch_forex_factory_calendar

HIGH_IMPACT_KEYWORDS = (
    "cpi",
    "core cpi",
    "pce",
    "core pce",
    "non-farm",
    "nonfarm",
    "nfp",
    "unemployment",
    "employment change",
    "average hourly earnings",
    "fomc",
    "federal funds rate",
    "rate decision",
    "interest rate decision",
    "press conference",
    "gdp",
    "retail sales",
    "ism",
    "ppi",
    "jobless claims",
    "jolts",
    "sep",
    "dot plot",
)


@dataclass
class MacroEvent:
    title: str
    country: str
    time_utc: Optional[datetime]
    impact: str
    forecast: Optional[str]
    previous: Optional[str]

    def is_high_impact(self) -> bool:
        if self.impact.strip().lower() == "high":
            return True
        title_lower = self.title.lower()
        return any(kw in title_lower for kw in HIGH_IMPACT_KEYWORDS)


def parse_calendar_events(raw: Any) -> list[MacroEvent]:
    """Defensive parsing: the public FF mirror's exact field names have
    varied historically (title/Title, date/Date, impact casing, etc.), and
    this could not be reconfirmed live from this build environment (the
    endpoint is blocked by this sandbox's egress policy — see README).
    Unrecognized shapes simply yield an empty list rather than crashing or
    fabricating events."""
    if not isinstance(raw, list):
        return []

    events: list[MacroEvent] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        title = item.get("title") or item.get("Title") or item.get("event")
        if not title:
            continue
        country = item.get("country") or item.get("Country") or ""
        impact = str(item.get("impact") or item.get("Impact") or "")
        forecast = item.get("forecast") or item.get("Forecast")
        previous = item.get("previous") or item.get("Previous")
        raw_time = item.get("date") or item.get("Date") or item.get("timestamp")
        time_utc = coerce_time(raw_time)
        events.append(
            MacroEvent(
                title=str(title),
                country=str(country),
                time_utc=time_utc,
                impact=impact,
                forecast=str(forecast) if forecast is not None else None,
                previous=str(previous) if previous is not None else None,
            )
        )
    return events


def assess_macro_state(
    events: list[MacroEvent],
    now_utc: datetime,
    lookahead_hours: float = 6.0,
    lookback_hours: float = 3.0,
) -> dict[str, Any]:
    if not events:
        return {"status": "UNAVAILABLE", "upcoming": [], "recent": []}

    upcoming = []
    recent = []
    for e in events:
        if e.time_utc is None or not e.is_high_impact():
            continue
        delta_hours = (e.time_utc - now_utc).total_seconds() / 3600.0
        if 0 <= delta_hours <= lookahead_hours:
            upcoming.append({"title": e.title, "country": e.country, "hours_until": round(delta_hours, 2)})
        elif -lookback_hours <= delta_hours < 0:
            recent.append({"title": e.title, "country": e.country, "hours_ago": round(-delta_hours, 2)})

    status = "ELEVATED_RISK" if (upcoming or recent) else "CLEAR"
    return {"status": status, "upcoming": upcoming, "recent": recent}


def get_macro_state(now_utc: datetime, enabled: bool = True) -> dict[str, Any]:
    if not enabled:
        return {"status": "DISABLED", "upcoming": [], "recent": []}

    result = fetch_forex_factory_calendar()
    if not result.ok:
        return {"status": "UNAVAILABLE", "upcoming": [], "recent": [], "fetch_error": result.error}

    events = parse_calendar_events(result.data)
    state = assess_macro_state(events, now_utc)
    state["from_cache"] = result.from_cache
    return state
