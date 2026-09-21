from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from psygnal.memory.daily import (
    daily_memory_path,
    last_regime_for_symbol,
    load_daily_memory,
    record_event_result,
    save_daily_memory,
    update_daily_memory_for_symbol,
)


def test_load_daily_memory_starts_fresh_when_file_missing(tmp_path: Path):
    now = datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc)
    memory = load_daily_memory(now, base_dir=tmp_path)
    assert memory["symbols"] == {}
    assert "date" in memory


def test_save_and_reload_round_trips(tmp_path: Path):
    now = datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc)
    memory = load_daily_memory(now, base_dir=tmp_path)
    update_daily_memory_for_symbol(memory, "XAUUSD", now, "TREND_UP", {"time": now.isoformat(), "direction": "LONG", "regime": "TREND_UP"})
    path = save_daily_memory(memory, now, base_dir=tmp_path)
    assert path.exists()

    reloaded = load_daily_memory(now, base_dir=tmp_path)
    assert "XAUUSD" in reloaded["symbols"]
    assert len(reloaded["symbols"]["XAUUSD"]["forecast_history"]) == 1


def test_corrupt_file_degrades_to_fresh_memory(tmp_path: Path):
    now = datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc)
    path = daily_memory_path(now, base_dir=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json")

    memory = load_daily_memory(now, base_dir=tmp_path)
    assert memory["symbols"] == {}


def test_regime_transition_recorded_only_on_change(tmp_path: Path):
    now = datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc)
    memory = load_daily_memory(now, base_dir=tmp_path)

    update_daily_memory_for_symbol(memory, "XAUUSD", now, "TREND_UP", {"time": now.isoformat(), "regime": "TREND_UP"})
    assert memory["symbols"]["XAUUSD"]["regime_transitions"] == []  # no previous regime yet

    later = now + timedelta(minutes=5)
    update_daily_memory_for_symbol(memory, "XAUUSD", later, "TREND_UP", {"time": later.isoformat(), "regime": "TREND_UP"})
    assert memory["symbols"]["XAUUSD"]["regime_transitions"] == []  # same regime, no transition

    even_later = later + timedelta(minutes=5)
    update_daily_memory_for_symbol(memory, "XAUUSD", even_later, "REVERSAL", {"time": even_later.isoformat(), "regime": "REVERSAL"})
    assert len(memory["symbols"]["XAUUSD"]["regime_transitions"]) == 1
    assert memory["symbols"]["XAUUSD"]["regime_transitions"][0]["from"] == "TREND_UP"
    assert memory["symbols"]["XAUUSD"]["regime_transitions"][0]["to"] == "REVERSAL"


def test_last_regime_for_symbol_reads_most_recent_forecast():
    memory = {"symbols": {"XAUUSD": {"forecast_history": [{"regime": "RANGE"}, {"regime": "TREND_UP"}]}}}
    assert last_regime_for_symbol(memory, "XAUUSD") == "TREND_UP"
    assert last_regime_for_symbol(memory, "EURUSD") is None


def test_event_results_are_deduplicated(tmp_path: Path):
    now = datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc)
    memory = load_daily_memory(now, base_dir=tmp_path)
    event = {"title": "CPI", "time": now.isoformat()}
    record_event_result(memory, event)
    record_event_result(memory, event)
    assert memory["event_results"] == [event]


def test_daily_memory_file_named_by_display_timezone_date(tmp_path: Path):
    # Just before UTC midnight, but well into the next IST calendar day
    # (IST is UTC+5:30) -- the file should be named for the IST date.
    now = datetime(2025, 6, 1, 23, 0, tzinfo=timezone.utc)
    path = daily_memory_path(now, base_dir=tmp_path)
    assert path.name == "2025-06-02.json"
