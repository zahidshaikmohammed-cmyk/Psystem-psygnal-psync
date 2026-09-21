"""Human-readable terminal report.

V3 note: this reads as a market-intelligence brief a discretionary trader
would recognize (market state -> macro -> cross-market -> structure ->
liquidity -> behavior -> hypothesis -> evidence for/against ->
invalidation -> tradeability -> forecast -> why), not a raw indicator
dump. Every section only states what the engine actually computed;
sections whose upstream data is UNAVAILABLE say so rather than being
silently omitted, per the project's no-fabrication rule.
"""

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


def _section(title: str, body_lines: list[str]) -> list[str]:
    return [THIN, title, THIN, ""] + (body_lines if body_lines else ["None."]) + [""]


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


def _market_state_lines(signal: FinalSignal, symbol_intel: dict[str, Any]) -> list[str]:
    shock = signal.shock_state or {}
    lines = [
        f"Regime: {signal.market_regime}",
        f"Price action: {symbol_intel['price_action'].label.replace('_', ' ').title()}",
        f"Volatility: {symbol_intel['volatility'].label} / {symbol_intel['volatility'].regime}",
    ]
    if shock.get("is_shock"):
        lines.append(f"MARKET SHOCK detected (severity: {shock.get('severity')}) -- {shock.get('evidence')}")
    else:
        lines.append("No market shock detected.")
    return lines


def _macro_lines(signal: FinalSignal) -> list[str]:
    macro_status = signal.macro_state.get("status", "UNAVAILABLE")
    lines = [f"Macro calendar status: {macro_status}"]
    event_risk = signal.event_risk or {}
    phase = event_risk.get("phase", "NONE")
    lines.append(f"Event phase: {phase}")
    nearest = event_risk.get("nearest_event")
    if nearest:
        lines.append(f"Nearest high-impact event: {nearest.get('title')} ({event_risk.get('proximity_bucket')})")
    reacting = event_risk.get("reacting_events") or []
    if reacting:
        for r in reacting:
            lines.append(
                f"  Reacting to: {r['event_title']} [{r['category']}] "
                f"surprise={_fmt(r.get('surprise_normalized'), 2)} ({r.get('surprise_provenance')})"
            )
    news_status = signal.news_state.get("status", "UNAVAILABLE")
    lines.append(f"News feed status: {news_status}")
    return lines


def _cross_market_lines(signal: FinalSignal) -> list[str]:
    cmc = signal.cross_market_confirmation or {}
    lines = [
        f"USD composite: {signal.usd_composite.get('state', 'UNAVAILABLE')}",
        f"Cross-market classification: {cmc.get('label', 'UNCLEAR')}",
    ]
    if cmc.get("macro_confirmed"):
        lines.append("  Macro transmission theory is CONFIRMED by observed cross-market reaction.")
    if cmc.get("macro_contradicted"):
        lines.append("  Macro transmission theory is CONTRADICTED by observed cross-market reaction.")
    if cmc.get("cross_asset_confirmed"):
        lines.append("  Cross-asset correlation (e.g. gold/silver) is confirming.")
    return lines


def _structure_lines(symbol_intel: dict[str, Any]) -> list[str]:
    lines = [f"Structure (M5): {symbol_intel['structures']['M5'].trend_structure}"]
    for tf in ("H4", "H1", "M30", "M15", "M5"):
        lines.append(f"  {tf}: {symbol_intel['trends'][tf].label}")
    return lines


def _liquidity_lines(symbol_intel: dict[str, Any]) -> list[str]:
    liquidity = symbol_intel["liquidity"]
    return [
        f"{len(liquidity.sweep_events)} recent sweep event(s); reclaim={liquidity.reclaim}",
        f"Breakout retest: {liquidity.breakout_retest}",
    ]


def _behavior_lines(symbol_intel: dict[str, Any]) -> list[str]:
    return [
        f"Momentum: {symbol_intel['momentum'].label.replace('_', ' ').title()}",
        f"Volume: {symbol_intel['volume'].label} ({symbol_intel['volume'].price_volume_relationship})",
    ]


def format_symbol_report(signal: FinalSignal, symbol_intel: dict[str, Any]) -> str:
    precision = 3 if signal.current_price >= 100 else 5
    det = signal.deterministic_forecast

    lines = [BAR, signal.symbol, BAR, ""]
    lines += _section("MARKET STATE", _market_state_lines(signal, symbol_intel))
    lines += _section("MACRO", _macro_lines(signal))
    lines += _section("CROSS-MARKET", _cross_market_lines(signal))
    lines += _section("STRUCTURE", _structure_lines(symbol_intel))
    lines += _section("LIQUIDITY", _liquidity_lines(symbol_intel))
    lines += _section("MARKET BEHAVIOR", _behavior_lines(symbol_intel))

    lines += _section(
        "CURRENT HYPOTHESIS",
        [
            f"Direction: {signal.direction}",
            f"Forecast engine: {signal.forecast_engine} (model status: {signal.model_status})",
            f"Confidence: {signal.confidence} ({signal.confidence_score:.0f}/100)",
            f"Signal score: {signal.signal_score:.0f}/100 (scoring_mode: {signal.scoring_mode})",
        ],
    )
    lines += _section("EVIDENCE FOR", list(signal.confirming_evidence))
    lines += _section("EVIDENCE AGAINST", list(signal.contradicting_evidence))
    lines += _section("INVALIDATION", list(signal.invalidation_conditions))
    lines += _section(
        "TRADEABILITY",
        [signal.tradeability] + [f"  - {r}" for r in signal.tradeability_reasons],
    )

    lines += _section(
        "60-MINUTE FORECAST",
        [
            f"Deterministic: LONG {det['long'] * 100:.0f}%  SHORT {det['short'] * 100:.0f}%  NEUTRAL {det['neutral'] * 100:.0f}%",
        ]
        + (
            [
                f"Model probability ({signal.model_probability.get('model_name')}, "
                f"{'calibrated' if signal.model_probability.get('calibrated') else 'UNCALIBRATED'}): "
                f"LONG {signal.model_probability['long'] * 100:.0f}%  "
                f"SHORT {signal.model_probability['short'] * 100:.0f}%  "
                f"NEUTRAL {signal.model_probability['neutral'] * 100:.0f}%"
            ]
            if signal.model_probability is not None
            else ["Model probability: N/A (no trained model for this symbol)"]
        )
        + [
            f"Operational (blended): LONG {signal.probability_long * 100:.0f}%  "
            f"SHORT {signal.probability_short * 100:.0f}%  NEUTRAL {signal.probability_neutral * 100:.0f}%",
            "",
            f"Current price: {_fmt(signal.current_price, precision)}",
            f"Entry: {_fmt(signal.entry, precision)}  "
            f"(zone {_fmt(signal.entry_zone[0], precision)} - {_fmt(signal.entry_zone[1], precision)})",
            f"Stop loss: {_fmt(signal.stop_loss, precision)}",
            f"TP1: {_fmt(signal.tp1, precision)}   TP2: {_fmt(signal.tp2, precision)}",
            f"Expected 60m range: {_fmt(signal.expected_60m_low, precision)} - "
            f"{_fmt(signal.expected_60m_high, precision)} [{signal.expected_range_methodology}]",
            f"Expected 60m return: {signal.expected_60m_return * 100:+.2f}%",
            f"R:R: {'1 : ' + format(signal.rr, '.2f') if signal.rr is not None else 'N/A'}",
        ],
    )

    lines += _section("WHY " + signal.direction, [f"{i}. {r}" for i, r in enumerate(signal.reasons, 1)])
    lines += _section(
        "CONFLICTS", [f"{i}. {c}" for i, c in enumerate(signal.conflicts, 1)] if signal.conflicts else []
    )
    lines += _section(
        "WARNINGS", [f"{i}. {w}" for i, w in enumerate(signal.warnings, 1)] if signal.warnings else []
    )
    lines += _section(
        "OTHER INTELLIGENCE",
        [
            f"Session: {signal.session_state.get('session_label', 'UNKNOWN')}",
            f"Historical pattern memory: {signal.historical_pattern_state.get('status', 'INSUFFICIENT_DATA')}",
        ],
    )
    lines += [BAR]
    return "\n".join(lines)


def format_market_summary(signals: list[FinalSignal]) -> str:
    lines = [BAR, "MARKET SUMMARY", BAR, ""]
    for s in signals:
        lines.append(
            f"{s.symbol:<8} {s.direction:<6} [{s.tradeability:<13}] "
            f"{s.signal_score:>5.0f}/100   [{s.forecast_engine}/{s.model_status}]"
        )
    lines += [
        "",
        "This is informational output from a forecasting engine, not a",
        "guarantee of outcome. Trade decisions remain the user's own.",
        BAR,
    ]
    return "\n".join(lines)
