from __future__ import annotations

from datetime import timedelta

import pytest

from psygnal.data.aggregation import build_multi_timeframe_series
from psygnal.forecasting.ensemble import combine_forecasts, compute_deterministic_probabilities
from psygnal.indicators.atr import atr
from psygnal.intelligence.context import build_symbol_intelligence
from psygnal.intelligence.regime import classify_regime
from psygnal.models import candles_to_frame
from psygnal.signal.confidence import compute_confidence
from psygnal.signal.direction import select_direction
from psygnal.signal.entry import compute_entry
from psygnal.signal.explain import build_explanation
from psygnal.signal.score import compute_rr, compute_signal_score
from psygnal.signal.stop import compute_stop_loss
from psygnal.signal.targets import compute_targets
from tests.conftest import make_m5_series


def _build_intel(direction=1, step=0.3, n=500, base=100.0):
    candles = make_m5_series(n, base_price=base, step=direction * step)
    now = candles[-1].time + timedelta(minutes=6)
    mts = build_multi_timeframe_series("EURUSD", candles, now=now)
    intel = build_symbol_intelligence("EURUSD", mts, now)
    return intel, now


def test_direction_selection_always_picks_higher_probability():
    assert select_direction(0.54, 0.46) == "LONG"
    assert select_direction(0.46, 0.54) == "SHORT"
    assert select_direction(0.5, 0.5) == "LONG"  # documented tie-break


def test_entry_stop_targets_pipeline_for_uptrend():
    intel, now = _build_intel(direction=1, step=0.3)
    current_price = intel["current_price"]
    m5 = intel["structures"]["M5"]

    atr_series = atr(
        candles_to_frame(make_m5_series(500, step=0.3))["high"],
        candles_to_frame(make_m5_series(500, step=0.3))["low"],
        candles_to_frame(make_m5_series(500, step=0.3))["close"],
    )
    atr_value = intel["volatility"].atr_value

    entry_plan = compute_entry("LONG", current_price, atr_value, intel["liquidity"])
    assert entry_plan.entry_zone[0] < entry_plan.entry_zone[1]

    stop_plan = compute_stop_loss("LONG", entry_plan.entry, atr_value, m5)
    assert stop_plan.stop_loss < entry_plan.entry

    target_plan = compute_targets("LONG", entry_plan.entry, atr_value, intel["liquidity"], intel["volatility"].expected_move_60m)
    assert target_plan.tp1 > entry_plan.entry

    rr = compute_rr(entry_plan.entry, stop_plan.stop_loss, target_plan.tp1)
    assert rr is not None and rr > 0


def test_entry_stop_targets_pipeline_for_downtrend():
    intel, now = _build_intel(direction=-1, step=0.3, base=500.0)
    current_price = intel["current_price"]
    m5 = intel["structures"]["M5"]
    atr_value = intel["volatility"].atr_value

    entry_plan = compute_entry("SHORT", current_price, atr_value, intel["liquidity"])
    stop_plan = compute_stop_loss("SHORT", entry_plan.entry, atr_value, m5)
    assert stop_plan.stop_loss > entry_plan.entry

    target_plan = compute_targets("SHORT", entry_plan.entry, atr_value, intel["liquidity"], intel["volatility"].expected_move_60m)
    assert target_plan.tp1 < entry_plan.entry


def test_signal_score_in_valid_range():
    intel, now = _build_intel(direction=1, step=0.3)
    cross_market_state = {"usd_composite": {"state": "NEUTRAL"}}
    deterministic = compute_deterministic_probabilities("EURUSD", intel, cross_market_state)
    ensemble = combine_forecasts(deterministic)
    direction = select_direction(ensemble["probability_long"], ensemble["probability_short"])

    result = compute_signal_score(
        direction,
        ensemble,
        None,
        deterministic["components"],
        intel["volatility"].label,
        intel["volume"].label,
        intel["volume"].price_volume_relationship,
        intel["sessions"]["session_label"],
        "CLEAR",
        rr=2.0,
    )
    assert 0 <= result["signal_score"] <= 100


def test_confidence_in_valid_range_and_label_consistent():
    intel, now = _build_intel(direction=1, step=0.3)
    cross_market_state = {"usd_composite": {"state": "NEUTRAL"}}
    deterministic = compute_deterministic_probabilities("EURUSD", intel, cross_market_state)
    ensemble = combine_forecasts(deterministic)
    direction = select_direction(ensemble["probability_long"], ensemble["probability_short"])

    result = compute_confidence(direction, ensemble, None, "OK", "TREND_UP", deterministic["components"]["cross_market"])
    assert 0 <= result["confidence_score"] <= 100
    assert result["confidence"] in ("HIGH", "MEDIUM", "LOW")


def test_explanation_produces_reasons_for_strong_uptrend():
    intel, now = _build_intel(direction=1, step=0.5)
    cross_market_state = {"usd_composite": {"state": "NEUTRAL"}}
    deterministic = compute_deterministic_probabilities("EURUSD", intel, cross_market_state)
    ensemble = combine_forecasts(deterministic)
    direction = select_direction(ensemble["probability_long"], ensemble["probability_short"])
    regime = classify_regime(intel, {"status": "CLEAR", "upcoming": [], "recent": []})

    reasons, conflicts, warnings = build_explanation(
        direction,
        intel,
        ensemble,
        regime.label,
        cross_market_state,
        {"status": "CLEAR", "upcoming": [], "recent": []},
        {"status": "OK", "dominant_themes": []},
        [],
        [],
    )
    assert len(reasons) >= 1
    assert isinstance(conflicts, list)
    assert isinstance(warnings, list)
