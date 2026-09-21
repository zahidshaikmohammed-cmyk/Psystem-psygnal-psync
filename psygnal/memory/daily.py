"""Daily market memory: a structured, machine-readable journal of what
happened during a trading day, so the next run can understand what came
before it — genuine cross-run state, not just a text log.

One file per trading day: `data/daily/YYYY-MM-DD.json`, keyed by the
display-timezone (IST) calendar date, matching the "what day is it
trading" framing used throughout the terminal report. Internally every
timestamp stored is UTC ISO-8601.

All operations are safe against a missing file (a fresh day starts a
fresh skeleton) and never raise on a corrupt file — a damaged journal
degrades to "start fresh" rather than crashing the engine.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from psygnal import config

MAX_FORECAST_HISTORY_PER_SYMBOL = 500
MAX_EVENT_LOG_LENGTH = 200


def _local_date_string(now_utc: datetime) -> str:
    return now_utc.astimezone(config.DISPLAY_TIMEZONE).date().isoformat()


def daily_memory_path(now_utc: datetime, base_dir: Path = config.DAILY_MEMORY_DIR) -> Path:
    return base_dir / f"{_local_date_string(now_utc)}.json"


def _fresh_memory(now_utc: datetime) -> dict[str, Any]:
    return {
        "date": _local_date_string(now_utc),
        "timezone": str(config.DISPLAY_TIMEZONE),
        "symbols": {},
        "major_events": [],
        "event_results": [],
    }


def load_daily_memory(now_utc: datetime, base_dir: Path = config.DAILY_MEMORY_DIR) -> dict[str, Any]:
    path = daily_memory_path(now_utc, base_dir)
    if not path.exists():
        return _fresh_memory(now_utc)
    try:
        with path.open("r") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "symbols" not in data:
            return _fresh_memory(now_utc)
        return data
    except (OSError, ValueError):
        return _fresh_memory(now_utc)


def save_daily_memory(memory: dict[str, Any], now_utc: datetime, base_dir: Path = config.DAILY_MEMORY_DIR) -> Path:
    path = daily_memory_path(now_utc, base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    with tmp_path.open("w") as f:
        json.dump(memory, f, indent=2, default=str)
    tmp_path.replace(path)
    return path


def _symbol_bucket(memory: dict[str, Any], symbol: str) -> dict[str, Any]:
    return memory["symbols"].setdefault(
        symbol,
        {
            "session_levels": {},
            "structural_events": [],
            "volatility_regime_timeline": [],
            "liquidity_sweeps": [],
            "regime_transitions": [],
            "forecast_history": [],
        },
    )


def append_forecast_record(memory: dict[str, Any], symbol: str, record: dict[str, Any]) -> dict[str, Any]:
    bucket = _symbol_bucket(memory, symbol)
    bucket["forecast_history"].append(record)
    if len(bucket["forecast_history"]) > MAX_FORECAST_HISTORY_PER_SYMBOL:
        bucket["forecast_history"] = bucket["forecast_history"][-MAX_FORECAST_HISTORY_PER_SYMBOL:]
    return memory


def record_regime_transition(memory: dict[str, Any], symbol: str, time_utc_iso: str, from_regime: Optional[str], to_regime: str) -> dict[str, Any]:
    bucket = _symbol_bucket(memory, symbol)
    if from_regime is not None and from_regime != to_regime:
        bucket["regime_transitions"].append({"time": time_utc_iso, "from": from_regime, "to": to_regime})
    return memory


def last_regime_for_symbol(memory: dict[str, Any], symbol: str) -> Optional[str]:
    bucket = memory.get("symbols", {}).get(symbol)
    if not bucket or not bucket.get("forecast_history"):
        return None
    return bucket["forecast_history"][-1].get("regime")


def record_liquidity_sweep(memory: dict[str, Any], symbol: str, sweep_event: dict[str, Any]) -> dict[str, Any]:
    bucket = _symbol_bucket(memory, symbol)
    if sweep_event not in bucket["liquidity_sweeps"]:
        bucket["liquidity_sweeps"].append(sweep_event)
    return memory


def update_session_levels(memory: dict[str, Any], symbol: str, session_summary: dict[str, Any]) -> dict[str, Any]:
    bucket = _symbol_bucket(memory, symbol)
    bucket["session_levels"] = session_summary
    return memory


def record_event_result(memory: dict[str, Any], event_record: dict[str, Any]) -> dict[str, Any]:
    if event_record not in memory["event_results"]:
        memory["event_results"].append(event_record)
        if len(memory["event_results"]) > MAX_EVENT_LOG_LENGTH:
            memory["event_results"] = memory["event_results"][-MAX_EVENT_LOG_LENGTH:]
    return memory


def update_daily_memory_for_symbol(
    memory: dict[str, Any],
    symbol: str,
    now_utc: datetime,
    regime_label: str,
    forecast_record: dict[str, Any],
    session_summary: Optional[dict[str, Any]] = None,
    recent_sweep_events: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """One call per symbol per run: appends the forecast record, detects
    and logs a regime transition against the previously stored regime,
    and refreshes session levels/sweep log. Returns the (mutated) memory
    so the caller can batch several symbols before a single save."""
    previous_regime = last_regime_for_symbol(memory, symbol)
    record_regime_transition(memory, symbol, now_utc.isoformat(), previous_regime, regime_label)
    append_forecast_record(memory, symbol, forecast_record)
    if session_summary is not None:
        update_session_levels(memory, symbol, session_summary)
    for sweep in recent_sweep_events or []:
        record_liquidity_sweep(memory, symbol, sweep)
    return memory
