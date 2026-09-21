"""Human-readable terminal report."""

from __future__ import annotations

from typing import Any

from psygnal.intelligence.sessions import to_display_timezone
from psygnal.models import FinalSignal

BAR = "=" * 60
THIN = "-" * 60


def _fmt(value: Any, precision: int = 5) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:,.{precision}f}"
    return str(value)


def format_header(now_utc) -> str:
    display_time = to_display_timezone(now_utc)
    lines = [
        BAR,
        "PSYGRID FOREX PSYGNAL",
        BAR,
        "",
        "TIME:",
        display_time.strftime("%Y-%m-%d %H:%M IST"),
        "",
        "FORECAST:",
        "NEXT ~60 MINUTES",
    ]
    return "\n".join(lines)


def format_data_unavailable(symbol: str, quality: dict[str, Any]) -> str:
    lines = [
        BAR,
        symbol,
        BAR,
        "",
        "STATUS: DATA UNAVAILABLE / INSUFFICIENT DATA",
        "",
        f"Candles available: {quality.get('candles_available')}",
        f"Candles required:  {quality.get('candles_required')}",
    ]
    for issue in quality.get("issues", []):
        lines.append(f"  - {issue}")
    return "\n".join(lines)


def format_symbol_report(signal: FinalSignal, symbol_intel: dict[str, Any]) -> str:
    precision = 3 if signal.current_price >= 100 else 5
    lines = [
        BAR,
        signal.symbol,
        BAR,
        "",
        "DIRECTION:",
        signal.direction,
        "",
        "LONG PROBABILITY:",
        f"{signal.probability_long * 100:.0f}%",
        "",
        "SHORT PROBABILITY:",
        f"{signal.probability_short * 100:.0f}%",
        "",
        "NEUTRAL:",
        f"{signal.probability_neutral * 100:.0f}%",
        "",
        "CONFIDENCE:",
        signal.confidence,
        "",
        "CONFIDENCE SCORE:",
        f"{signal.confidence_score:.0f}/100",
        "",
        "SIGNAL SCORE:",
        f"{signal.signal_score:.0f}/100",
        "",
        "REGIME:",
        signal.market_regime,
        "",
        "CURRENT PRICE:",
        _fmt(signal.current_price, precision),
        "",
        "ENTRY:",
        _fmt(signal.entry, precision),
        "",
        "ENTRY ZONE:",
        f"{_fmt(signal.entry_zone[0], precision)} — {_fmt(signal.entry_zone[1], precision)}",
        "",
        "STOP LOSS:",
        _fmt(signal.stop_loss, precision),
        "",
        "TP1:",
        _fmt(signal.tp1, precision),
        "",
        "TP2:",
        _fmt(signal.tp2, precision),
        "",
        "EXPECTED 60M RANGE:",
        f"{_fmt(signal.expected_60m_low, precision)} — {_fmt(signal.expected_60m_high, precision)}",
        "",
        "EXPECTED 60M RETURN:",
        f"{signal.expected_60m_return * 100:+.2f}%",
        "",
        "R:R:",
        f"1 : {signal.rr:.2f}" if signal.rr is not None else "N/A",
        "",
        THIN,
        "INTELLIGENCE",
        THIN,
        "",
        "PRICE ACTION:",
        symbol_intel["price_action"].label.replace("_", " ").title(),
        "",
        "STRUCTURE (M5):",
        symbol_intel["structures"]["M5"].trend_structure,
        "",
        "LIQUIDITY:",
        f"{len(symbol_intel['liquidity'].sweep_events)} recent sweep event(s); reclaim={symbol_intel['liquidity'].reclaim}",
        "",
        "VOLUME:",
        f"{symbol_intel['volume'].label} ({symbol_intel['volume'].price_volume_relationship})",
        "",
        "MOMENTUM:",
        symbol_intel["momentum"].label.replace("_", " ").title(),
        "",
        "VOLATILITY:",
        f"{symbol_intel['volatility'].label} / {symbol_intel['volatility'].regime}",
        "",
        "H4:",
        symbol_intel["trends"]["H4"].label,
        "",
        "H1:",
        symbol_intel["trends"]["H1"].label,
        "",
        "M30:",
        symbol_intel["trends"]["M30"].label,
        "",
        "M15:",
        symbol_intel["trends"]["M15"].label,
        "",
        "M5:",
        symbol_intel["trends"]["M5"].label,
        "",
        "USD COMPOSITE:",
        signal.usd_composite.get("state", "UNAVAILABLE"),
        "",
        "SESSION:",
        signal.session_state.get("session_label", "UNKNOWN"),
        "",
        "MACRO:",
        signal.macro_state.get("status", "UNAVAILABLE"),
        "",
        "NEWS:",
        signal.news_state.get("status", "UNAVAILABLE"),
        "",
        "HISTORICAL PATTERN MEMORY:",
        signal.historical_pattern_state.get("status", "INSUFFICIENT_DATA"),
        "",
        THIN,
        "WHY " + signal.direction,
        THIN,
        "",
    ]
    for i, reason in enumerate(signal.reasons, 1):
        lines.append(f"{i}. {reason}")
    lines += ["", THIN, "CONFLICTS", THIN, ""]
    if signal.conflicts:
        for i, conflict in enumerate(signal.conflicts, 1):
            lines.append(f"{i}. {conflict}")
    else:
        lines.append("None identified.")
    lines += ["", THIN, "WARNINGS", THIN, ""]
    if signal.warnings:
        for i, warning in enumerate(signal.warnings, 1):
            lines.append(f"{i}. {warning}")
    else:
        lines.append("None.")
    lines += ["", BAR]
    return "\n".join(lines)


def format_market_summary(signals: list[FinalSignal]) -> str:
    lines = [BAR, "MARKET SUMMARY", BAR, ""]
    for s in signals:
        lines.append(f"{s.symbol:<8} {s.direction:<6} {s.signal_score:>5.0f}/100")
    lines += [
        "",
        "This is informational output from a forecasting engine, not a",
        "guarantee of outcome. Trade decisions remain the user's own.",
        BAR,
    ]
    return "\n".join(lines)
