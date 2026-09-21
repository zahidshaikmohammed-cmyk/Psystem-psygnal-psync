"""End-to-end orchestration: raw live payload -> FinalSignal per symbol.

This module contains no I/O side effects other than what's explicitly
passed in (`raw_payload`) — `__main__.py` is responsible for fetching the
live payload and printing/writing the result, which keeps this function
directly unit-testable with synthetic payloads.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from psygnal import config
from psygnal.data.aggregation import build_multi_timeframe_series
from psygnal.data.parser import parse_live_payload
from psygnal.data.validation import validate_symbol
from psygnal.forecasting.ensemble import combine_forecasts, compute_deterministic_probabilities
from psygnal.intelligence.context import attach_cross_market, build_symbol_intelligence
from psygnal.intelligence.cross_market import analyze_cross_market
from psygnal.intelligence.gold import analyze_gold
from psygnal.intelligence.regime import classify_regime
from psygnal.macro.calendar import get_macro_state
from psygnal.models import DataStatus, FinalSignal
from psygnal.news.aggregator import get_news_state
from psygnal.signal.confidence import compute_confidence
from psygnal.signal.direction import select_direction
from psygnal.signal.entry import compute_entry
from psygnal.signal.explain import build_explanation
from psygnal.signal.score import compute_rr, compute_signal_score
from psygnal.signal.stop import compute_stop_loss
from psygnal.signal.targets import compute_targets


def run_engine(
    raw_payload: Any,
    cfg: config.EngineConfig = config.EngineConfig(),
    now_utc: Optional[datetime] = None,
    baseline_models: Optional[dict[str, Any]] = None,
    pattern_memory_stores: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    now_utc = now_utc or datetime.now(timezone.utc)
    baseline_models = baseline_models or {}
    pattern_memory_stores = pattern_memory_stores or {}

    parse_outcome = parse_live_payload(raw_payload)
    if not parse_outcome.ok:
        return {
            "status": "SCHEMA_UNRECOGNIZED",
            "error": parse_outcome.error,
            "schema_dump": parse_outcome.schema_dump,
            "now_utc": now_utc,
        }

    ordered_symbols = list(dict.fromkeys(list(cfg.symbols) + list(parse_outcome.symbols.keys())))
    ordered_symbols = [s for s in ordered_symbols if s in parse_outcome.symbols]

    validated: dict[str, tuple[list, Any]] = {}
    for sym in ordered_symbols:
        candles, quality = validate_symbol(
            sym,
            parse_outcome.symbols[sym].records,
            now=now_utc,
            min_candles_required=cfg.min_candles_required,
            max_staleness_minutes=cfg.max_staleness_minutes,
        )
        validated[sym] = (candles, quality)

    intel_by_symbol: dict[str, dict[str, Any]] = {}
    unavailable: dict[str, dict[str, Any]] = {}

    for sym, (candles, quality) in validated.items():
        if quality.status == DataStatus.UNAVAILABLE:
            unavailable[sym] = quality.as_dict()
            continue
        mts = build_multi_timeframe_series(sym, candles, now=now_utc)
        intel_by_symbol[sym] = build_symbol_intelligence(sym, mts, now_utc)

    closes_by_symbol = {
        sym: intel["closes"]["M5"] for sym, intel in intel_by_symbol.items() if "M5" in intel.get("closes", {})
    }
    trends_h1_by_symbol = {
        sym: intel["trends"]["H1"] for sym, intel in intel_by_symbol.items() if intel["trends"].get("H1") is not None
    }

    macro_state = get_macro_state(now_utc, enabled=cfg.enable_macro)
    news_state = get_news_state(enabled=cfg.enable_news)

    results: list[tuple[FinalSignal, dict[str, Any]]] = []

    for sym, intel in intel_by_symbol.items():
        cross_market_state = analyze_cross_market(closes_by_symbol, trends_h1_by_symbol)
        gold_state = None
        if sym == config.GOLD_SYMBOL:
            xag_close = closes_by_symbol.get(config.SILVER_SYMBOL)
            gold_state = analyze_gold(
                closes_by_symbol[sym], xag_close, intel["trends"].get("H1", intel["trends"]["M5"]), cross_market_state["usd_composite"]
            )
        intel_full = attach_cross_market(intel, cross_market_state, gold_state)

        regime = classify_regime(intel_full, macro_state)

        deterministic = compute_deterministic_probabilities(sym, intel_full, cross_market_state, gold_state)

        baseline_probs = None
        model = baseline_models.get(sym)
        if model is not None and model.is_fitted:
            from psygnal.forecasting.features import build_feature_frame

            feature_row = build_feature_frame(intel_full["m5_features"][["open", "high", "low", "close", "volume"]]).iloc[-1]
            baseline_probs = model.predict_proba(feature_row)

        pattern_memory_result: dict[str, Any] = {"status": "INSUFFICIENT_DATA", "k_used": 0}
        store = pattern_memory_stores.get(sym)
        if store is not None:
            from psygnal.forecasting.features import build_feature_frame

            feature_row = build_feature_frame(intel_full["m5_features"][["open", "high", "low", "close", "volume"]]).iloc[-1]
            pattern_memory_result = store.query(feature_row)

        ensemble = combine_forecasts(
            deterministic,
            baseline_probs=baseline_probs,
            pattern_memory=pattern_memory_result if pattern_memory_result.get("status") == "OK" else None,
        )

        direction = select_direction(ensemble["probability_long"], ensemble["probability_short"])

        current_price = intel_full["current_price"]
        atr_value = intel_full["volatility"].atr_value

        entry_plan = compute_entry(direction, current_price, atr_value, intel_full["liquidity"])
        stop_plan = compute_stop_loss(direction, entry_plan.entry, atr_value, intel_full["structures"]["M5"])
        target_plan = compute_targets(
            direction, entry_plan.entry, atr_value, intel_full["liquidity"], intel_full["volatility"].expected_move_60m
        )
        rr = compute_rr(entry_plan.entry, stop_plan.stop_loss, target_plan.tp1)

        quality = validated[sym][1]
        score_result = compute_signal_score(
            direction,
            ensemble,
            pattern_memory_result,
            deterministic["components"],
            intel_full["volatility"].label,
            intel_full["volume"].label,
            intel_full["volume"].price_volume_relationship,
            intel_full["sessions"]["session_label"],
            macro_state.get("status", "UNAVAILABLE"),
            rr,
        )
        confidence_result = compute_confidence(
            direction,
            ensemble,
            pattern_memory_result,
            quality.status.value,
            regime.label,
            deterministic["components"]["cross_market"],
        )

        reasons, conflicts, warnings = build_explanation(
            direction,
            intel_full,
            ensemble,
            regime.label,
            cross_market_state,
            macro_state,
            news_state,
            quality.issues,
            quality.warnings,
        )

        expected_move = intel_full["volatility"].expected_move_60m or 0.0
        expected_return = (expected_move / current_price) if current_price else 0.0
        if direction == "SHORT":
            expected_return = -expected_return

        signal = FinalSignal(
            symbol=sym,
            timestamp=now_utc,
            direction=direction,
            probability_long=ensemble["probability_long"],
            probability_short=ensemble["probability_short"],
            probability_neutral=ensemble["probability_neutral"],
            confidence=confidence_result["confidence"],
            confidence_score=confidence_result["confidence_score"],
            signal_score=score_result["signal_score"],
            current_price=current_price,
            entry=entry_plan.entry,
            entry_zone=entry_plan.entry_zone,
            stop_loss=stop_plan.stop_loss,
            tp1=target_plan.tp1,
            tp2=target_plan.tp2,
            expected_60m_return=expected_return,
            expected_60m_high=current_price + expected_move,
            expected_60m_low=current_price - expected_move,
            expected_60m_range=2 * expected_move,
            rr=rr,
            market_regime=regime.label,
            trend_state={tf: t.label for tf, t in intel_full["trends"].items()},
            structure_state={tf: s.as_dict() for tf, s in intel_full["structures"].items()},
            liquidity_state=intel_full["liquidity"].as_dict(),
            momentum_state=intel_full["momentum"].as_dict(),
            volatility_state=intel_full["volatility"].as_dict(),
            volume_state=intel_full["volume"].as_dict(),
            session_state=intel_full["sessions"],
            usd_composite=cross_market_state["usd_composite"],
            cross_market_state=cross_market_state,
            macro_state=macro_state,
            news_state=news_state,
            historical_pattern_state=pattern_memory_result,
            reasons=reasons,
            conflicts=conflicts,
            warnings=warnings,
            data_quality=quality.as_dict(),
        )
        results.append((signal, intel_full))

    return {
        "status": "OK",
        "results": results,
        "unavailable": unavailable,
        "macro_state": macro_state,
        "news_state": news_state,
        "now_utc": now_utc,
    }
