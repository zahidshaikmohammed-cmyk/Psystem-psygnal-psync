"""Builds the human-readable WHY / CONFLICTS / WARNINGS lists.

Every reason/conflict is traced to a concrete computed value — nothing
here is templated boilerplate independent of the actual state.
"""

from __future__ import annotations

from typing import Any


def build_explanation(
    direction: str,
    symbol_intel: dict[str, Any],
    ensemble_result: dict[str, Any],
    regime_label: str,
    cross_market_state: dict[str, Any],
    macro_state: dict[str, Any],
    news_state: dict[str, Any],
    data_quality_issues: list[str],
    data_quality_warnings: list[str],
) -> tuple[list[str], list[str], list[str]]:
    sign = 1.0 if direction == "LONG" else -1.0
    reasons: list[str] = []
    conflicts: list[str] = []
    warnings: list[str] = list(data_quality_warnings)

    components = ensemble_result.get("deterministic_detail", {}).get("components", {})

    trend_h1 = symbol_intel["trends"].get("H1")
    trend_h4 = symbol_intel["trends"].get("H4")
    if trend_h1 is not None:
        if sign * trend_h1.score > 0.15:
            reasons.append(f"H1 trend is {trend_h1.label} (score {trend_h1.score:+.2f}), aligned with {direction}.")
        elif sign * trend_h1.score < -0.15:
            conflicts.append(f"H1 trend is {trend_h1.label} (score {trend_h1.score:+.2f}), against {direction}.")
    if trend_h4 is not None and sign * trend_h4.score > 0.15:
        reasons.append(f"H4 trend is {trend_h4.label}, providing higher-timeframe context supportive of {direction}.")
    elif trend_h4 is not None and sign * trend_h4.score < -0.15:
        conflicts.append(f"H4 trend is {trend_h4.label}, conflicting with the {direction} call.")

    momentum = symbol_intel["momentum"]
    if sign * momentum.momentum_score > 0.1:
        reasons.append(f"Momentum is {momentum.label.lower().replace('_', ' ')}, supporting {direction}.")
    elif sign * momentum.momentum_score < -0.1:
        conflicts.append(f"Momentum is {momentum.label.lower().replace('_', ' ')}, against {direction}.")
    if direction == "LONG" and momentum.divergence_bearish:
        conflicts.append("Bearish RSI/price divergence detected on recent swing highs.")
    if direction == "SHORT" and momentum.divergence_bullish:
        conflicts.append("Bullish RSI/price divergence detected on recent swing lows.")

    m5_structure = symbol_intel["structures"].get("M5")
    if m5_structure is not None:
        if (direction == "LONG" and m5_structure.last_bos == "BULLISH_BOS") or (
            direction == "SHORT" and m5_structure.last_bos == "BEARISH_BOS"
        ):
            reasons.append(f"M5 structure shows a {m5_structure.last_bos.replace('_', ' ').lower()}.")
        if (direction == "LONG" and m5_structure.last_choch == "BEARISH_CHOCH") or (
            direction == "SHORT" and m5_structure.last_choch == "BULLISH_CHOCH"
        ):
            conflicts.append(f"M5 shows a {m5_structure.last_choch.replace('_', ' ').lower()} against {direction}.")

    liquidity = symbol_intel["liquidity"]
    recent_sweeps = [e for e in liquidity.sweep_events if e.get("recent")]
    for e in recent_sweeps:
        favorable = (direction == "LONG" and e["direction"] == "sweep_low") or (
            direction == "SHORT" and e["direction"] == "sweep_high"
        )
        text = f"Liquidity sweep at {e['zone']} ({e['zone_price']:.5g}) with {'reversal' if favorable else 'continuation risk'}."
        (reasons if favorable else conflicts).append(text)

    price_action = symbol_intel["price_action"]
    reasons.append(f"Price action classified as {price_action.label.replace('_', ' ')}.")

    usd_composite = cross_market_state.get("usd_composite", {})
    cross_component = components.get("cross_market", 0.0)
    if sign * cross_component > 0.1:
        reasons.append(f"USD composite is {usd_composite.get('state', 'UNAVAILABLE').lower()}, supportive of {direction}.")
    elif sign * cross_component < -0.1:
        conflicts.append(f"USD composite is {usd_composite.get('state', 'UNAVAILABLE').lower()}, conflicting with {direction}.")

    volatility = symbol_intel["volatility"]
    if volatility.label == "LOW":
        warnings.append("Volatility is LOW — expected 60-minute movement may be limited.")
    if regime_label in ("CHAOTIC", "NEWS_SHOCK"):
        warnings.append(f"Market regime is {regime_label} — directional clarity is reduced.")

    if macro_state.get("status") == "ELEVATED_RISK":
        upcoming = macro_state.get("upcoming", [])
        if upcoming:
            warnings.append(f"High-impact macro event upcoming: {upcoming[0]['title']} in {upcoming[0]['hours_until']}h.")
        recent = macro_state.get("recent", [])
        if recent:
            warnings.append(f"High-impact macro event occurred recently: {recent[0]['title']} ({recent[0]['hours_ago']}h ago).")
    elif macro_state.get("status") == "UNAVAILABLE":
        warnings.append("Macro calendar data unavailable — proceeding with reduced context.")

    if news_state.get("status") == "UNAVAILABLE":
        warnings.append("News context unavailable — proceeding with reduced context.")

    if not reasons:
        reasons.append(f"Composite deterministic evidence weakly favors {direction}; no single dominant factor.")

    return reasons, conflicts, warnings
