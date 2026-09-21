"""Daily market brief: a narrative summary built from the day's
structured `daily memory` journal (see `psygnal.memory.daily`), answering
"what has actually happened today, so far" rather than re-deriving it
from scratch each run.

Every fact here comes straight from the persisted memory record for
today — nothing is fabricated or inferred beyond what was actually
logged during a prior run. A symbol with no history yet today says so.
"""

from __future__ import annotations

from typing import Any

BAR = "=" * 60
THIN = "-" * 60


def _symbol_brief_lines(symbol: str, bucket: dict[str, Any]) -> list[str]:
    history = bucket.get("forecast_history", [])
    lines = [f"{symbol}:"]
    if not history:
        lines.append("  No forecasts recorded yet today.")
        return lines

    latest = history[-1]
    lines.append(f"  Latest: {latest.get('direction', 'N/A')} / {latest.get('regime', 'N/A')} at {latest.get('time', 'N/A')}")
    lines.append(f"  Forecasts logged today: {len(history)}")

    transitions = bucket.get("regime_transitions", [])
    if transitions:
        lines.append(f"  Regime transitions today ({len(transitions)}):")
        for t in transitions[-5:]:
            lines.append(f"    {t['time']}: {t['from']} -> {t['to']}")
    else:
        lines.append("  No regime transitions recorded today.")

    sweeps = bucket.get("liquidity_sweeps", [])
    if sweeps:
        lines.append(f"  Liquidity sweeps logged today: {len(sweeps)}")

    session_levels = bucket.get("session_levels") or {}
    if session_levels:
        lines.append(f"  Session levels: {session_levels}")

    return lines


def format_daily_brief(memory: dict[str, Any]) -> str:
    lines = [BAR, "DAILY MARKET BRIEF", BAR, ""]
    lines.append(f"Date: {memory.get('date', 'UNKNOWN')} ({memory.get('timezone', 'UNKNOWN')})")
    lines.append("")

    lines += [THIN, "SYMBOLS", THIN, ""]
    symbols = memory.get("symbols", {})
    if not symbols:
        lines.append("No symbol activity recorded yet today.")
    else:
        for symbol, bucket in symbols.items():
            lines += _symbol_brief_lines(symbol, bucket)
            lines.append("")

    lines += [THIN, "EVENT RESULTS TODAY", THIN, ""]
    event_results = memory.get("event_results", [])
    if event_results:
        for e in event_results:
            lines.append(f"  {e.get('time', 'N/A')}: {e.get('title', 'N/A')}")
    else:
        lines.append("No confirmed event results logged today.")

    lines += ["", BAR]
    return "\n".join(lines)
