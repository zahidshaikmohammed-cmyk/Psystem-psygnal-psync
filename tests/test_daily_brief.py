from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from psygnal.memory.daily import load_daily_memory, update_daily_memory_for_symbol
from psygnal.reporting.daily_brief import format_daily_brief


def test_empty_memory_produces_honest_no_activity_brief(tmp_path: Path):
    now = datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc)
    memory = load_daily_memory(now, base_dir=tmp_path)
    brief = format_daily_brief(memory)
    assert "No symbol activity recorded yet today." in brief
    assert "No confirmed event results logged today." in brief


def test_brief_reflects_recorded_forecasts_and_transitions(tmp_path: Path):
    now = datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc)
    memory = load_daily_memory(now, base_dir=tmp_path)
    update_daily_memory_for_symbol(
        memory, "XAUUSD", now, "TREND_UP", {"time": now.isoformat(), "direction": "LONG", "regime": "TREND_UP"}
    )
    later = now.replace(minute=30)
    update_daily_memory_for_symbol(
        memory, "XAUUSD", later, "REVERSAL", {"time": later.isoformat(), "direction": "SHORT", "regime": "REVERSAL"}
    )
    brief = format_daily_brief(memory)
    assert "XAUUSD" in brief
    assert "SHORT / REVERSAL" in brief
    assert "TREND_UP -> REVERSAL" in brief
    assert "Forecasts logged today: 2" in brief


def test_brief_never_crashes_on_missing_optional_fields():
    memory = {"date": "2025-06-01", "timezone": "UTC", "symbols": {"XAUUSD": {}}, "event_results": []}
    brief = format_daily_brief(memory)
    assert "XAUUSD" in brief
    assert "No forecasts recorded yet today." in brief
