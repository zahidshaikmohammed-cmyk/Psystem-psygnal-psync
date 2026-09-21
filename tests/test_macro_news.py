from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from psygnal.data.http_cache import CachedFetchResult
from psygnal.macro import calendar as macro_calendar
from psygnal.macro.calendar import assess_macro_state, get_macro_state, parse_calendar_events
from psygnal.news import aggregator as news_aggregator
from psygnal.news.aggregator import get_news_state, summarize_articles


def test_parse_calendar_events_handles_known_fields():
    raw = [
        {"title": "CPI m/m", "country": "USD", "date": "2024-06-12T12:30:00Z", "impact": "High", "forecast": "0.3%", "previous": "0.3%"},
        {"title": "Some Minor Release", "country": "EUR", "date": "2024-06-12T09:00:00Z", "impact": "Low"},
    ]
    events = parse_calendar_events(raw)
    assert len(events) == 2
    assert events[0].is_high_impact() is True
    assert events[1].is_high_impact() is False


def test_parse_calendar_events_malformed_shape_returns_empty():
    assert parse_calendar_events({"not": "a list"}) == []
    assert parse_calendar_events(None) == []


def test_assess_macro_state_flags_upcoming_high_impact_event():
    now = datetime(2024, 6, 12, 10, 0, tzinfo=timezone.utc)
    events = parse_calendar_events(
        [{"title": "FOMC Rate Decision", "country": "USD", "date": "2024-06-12T14:00:00Z", "impact": "High"}]
    )
    state = assess_macro_state(events, now, lookahead_hours=6)
    assert state["status"] == "ELEVATED_RISK"
    assert len(state["upcoming"]) == 1


def test_assess_macro_state_no_events_is_unavailable():
    state = assess_macro_state([], datetime.now(timezone.utc))
    assert state["status"] == "UNAVAILABLE"


def test_get_macro_state_disabled():
    state = get_macro_state(datetime.now(timezone.utc), enabled=False)
    assert state["status"] == "DISABLED"


def test_get_macro_state_gracefully_handles_fetch_failure(monkeypatch):
    def fake_fetch():
        return CachedFetchResult(ok=False, data=None, from_cache=False, error="network blocked")

    monkeypatch.setattr(macro_calendar, "fetch_forex_factory_calendar", fake_fetch)
    state = get_macro_state(datetime.now(timezone.utc))
    assert state["status"] == "UNAVAILABLE"
    assert "fetch_error" in state


def test_summarize_articles_tags_themes():
    articles = [
        {"title": "Federal Reserve signals rate cut amid inflation slowdown"},
        {"title": "Gold prices rally as dollar weakens"},
        {"title": "Completely unrelated sports news"},
    ]
    summary = summarize_articles(articles)
    assert summary["article_count"] == 3
    assert "federal_reserve" in summary["dominant_themes"] or "inflation" in summary["dominant_themes"]
    assert "gold" in summary["dominant_themes"]


def test_get_news_state_disabled():
    state = get_news_state(enabled=False)
    assert state["status"] == "DISABLED"


def test_get_news_state_gracefully_handles_fetch_failure(monkeypatch):
    def fake_fetch(*args, **kwargs):
        return CachedFetchResult(ok=False, data=None, from_cache=False, error="network blocked")

    monkeypatch.setattr(news_aggregator, "fetch_market_news", fake_fetch)
    state = get_news_state()
    assert state["status"] == "UNAVAILABLE"
    assert "fetch_error" in state
