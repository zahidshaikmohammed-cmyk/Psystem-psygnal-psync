"""Structured JSON output — for logging, dashboards, Telegram, Obsidian,
backtesting, and performance analysis."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from psygnal.models import FinalSignal


def build_json_output(
    signals: list[FinalSignal],
    unavailable_symbols: dict[str, dict[str, Any]],
    generated_at: datetime,
) -> dict[str, Any]:
    return {
        "generated_at": generated_at.isoformat(),
        "engine": "psygnal",
        "forecast_horizon": "~60 minutes (12 x M5)",
        "signals": [s.as_dict() for s in signals],
        "unavailable_symbols": unavailable_symbols,
    }


def write_json_output(data: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def default_output_path(base_dir: Path, generated_at: datetime) -> Path:
    stamp = generated_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return base_dir / f"psygnal_{stamp}.json"
