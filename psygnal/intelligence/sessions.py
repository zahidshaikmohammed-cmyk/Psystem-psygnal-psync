"""Timezone-aware trading-session engine.

Internal calculations are always in UTC; `to_display_timezone()` converts
only at the reporting boundary (IST for the terminal report). Session
civil hours are converted from each session city's local timezone so
daylight-saving transitions are respected automatically via `zoneinfo`.

ASSUMPTION: the "previous trading day" boundary is taken as the UTC
calendar day (00:00-24:00 UTC). Many brokers instead roll the day at
17:00 New York time; the exact convention of the PSYGRID feed was not
knowable without a live sample, so this is documented rather than
silently guessed as broker truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Optional

import pandas as pd

from psygnal import config

SESSION_NAMES = ("ASIA", "LONDON", "NEW_YORK")


@dataclass
class SessionWindow:
    name: str
    start_utc: datetime
    end_utc: datetime


def to_display_timezone(dt: datetime) -> datetime:
    return dt.astimezone(config.DISPLAY_TIMEZONE)


def session_window_for_date(session_name: str, calendar_date: date) -> SessionWindow:
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(config.SESSION_CITY_TIMEZONES[session_name])
    start_hour, end_hour = config.SESSION_LOCAL_HOURS[session_name]
    local_start = datetime(calendar_date.year, calendar_date.month, calendar_date.day, start_hour, 0, tzinfo=tz)
    local_end = datetime(calendar_date.year, calendar_date.month, calendar_date.day, end_hour, 0, tzinfo=tz)
    return SessionWindow(
        name=session_name,
        start_utc=local_start.astimezone(config.UTC),
        end_utc=local_end.astimezone(config.UTC),
    )


def get_active_sessions(now_utc: datetime) -> list[str]:
    active: list[str] = []
    for session_name in SESSION_NAMES:
        for day_offset in (0, -1, 1):
            window = session_window_for_date(session_name, (now_utc + timedelta(days=day_offset)).date())
            if window.start_utc <= now_utc < window.end_utc:
                active.append(session_name)
                break
    return active


def get_session_label(now_utc: datetime) -> str:
    active = get_active_sessions(now_utc)
    if "LONDON" in active and "NEW_YORK" in active:
        return "LONDON_NEW_YORK_OVERLAP"
    if active:
        return active[0]
    return "OFF_HOURS"


def _window_slice(df: pd.DataFrame, start_utc: datetime, end_utc: datetime, now_utc: datetime) -> pd.DataFrame:
    clipped_end = min(end_utc, now_utc)
    if df.empty:
        return df
    mask = (df.index >= start_utc) & (df.index < clipped_end)
    return df.loc[mask]


def compute_session_stats(
    df_m5: pd.DataFrame, session_name: str, now_utc: datetime, day_offset: int = 0
) -> dict[str, Any]:
    window = session_window_for_date(session_name, (now_utc + timedelta(days=day_offset)).date())
    sliced = _window_slice(df_m5, window.start_utc, window.end_utc, now_utc)
    if sliced.empty:
        return {
            "session": session_name,
            "day_offset": day_offset,
            "high": None,
            "low": None,
            "range": None,
            "candle_count": 0,
            "is_complete": window.end_utc <= now_utc,
        }
    high = float(sliced["high"].max())
    low = float(sliced["low"].min())
    return {
        "session": session_name,
        "day_offset": day_offset,
        "high": high,
        "low": low,
        "range": high - low,
        "candle_count": len(sliced),
        "is_complete": window.end_utc <= now_utc,
    }


def compute_previous_day_levels(df_m5: pd.DataFrame, now_utc: datetime) -> dict[str, Optional[float]]:
    if df_m5.empty:
        return {"prev_day_high": None, "prev_day_low": None, "prev_day_close": None}
    today = now_utc.date()
    yesterday = today - timedelta(days=1)
    mask = (df_m5.index.date >= yesterday) & (df_m5.index.date < today)
    prev_day = df_m5.loc[mask]
    if prev_day.empty:
        return {"prev_day_high": None, "prev_day_low": None, "prev_day_close": None}
    return {
        "prev_day_high": float(prev_day["high"].max()),
        "prev_day_low": float(prev_day["low"].min()),
        "prev_day_close": float(prev_day["close"].iloc[-1]),
    }


def summarize_sessions(df_m5: pd.DataFrame, now_utc: datetime) -> dict[str, Any]:
    active = get_active_sessions(now_utc)
    label = get_session_label(now_utc)
    today_stats = {name: compute_session_stats(df_m5, name, now_utc, day_offset=0) for name in SESSION_NAMES}
    prior_stats = {name: compute_session_stats(df_m5, name, now_utc, day_offset=-1) for name in SESSION_NAMES}
    prev_day = compute_previous_day_levels(df_m5, now_utc)
    return {
        "active_sessions": active,
        "session_label": label,
        "today": today_stats,
        "prior": prior_stats,
        "previous_day": prev_day,
        "display_time": to_display_timezone(now_utc).isoformat(),
    }
