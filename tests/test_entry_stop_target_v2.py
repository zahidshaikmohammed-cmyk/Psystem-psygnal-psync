from __future__ import annotations

from dataclasses import dataclass

import pytest

from psygnal.intelligence.liquidity import LiquidityState, LiquidityZone
from psygnal.intelligence.structure import StructureState
from psygnal.signal.entry import compute_entry
from psygnal.signal.stop import compute_stop_loss
from psygnal.signal.targets import compute_targets


@dataclass
class _FakeVolatility:
    label: str
    regime: str


@dataclass
class _FakePriceAction:
    label: str


def _empty_liquidity() -> LiquidityState:
    return LiquidityState(
        zones_above=[], zones_below=[], nearest_above=None, nearest_below=None,
        sweep_events=[], reclaim=False, failed_breakout=False, breakout_retest=False, continuation_after_breakout=False,
    )


def test_entry_offset_scales_with_volatility_regime():
    liquidity = _empty_liquidity()
    entry_low_vol = compute_entry(
        "LONG", 100.0, atr_value=1.0, liquidity_state=liquidity,
        volatility_state=_FakeVolatility("LOW", "STABLE"), price_action_state=_FakePriceAction("continuation"),
    )
    entry_high_vol = compute_entry(
        "LONG", 100.0, atr_value=1.0, liquidity_state=liquidity,
        volatility_state=_FakeVolatility("HIGH", "EXPANSION"), price_action_state=_FakePriceAction("continuation"),
    )
    offset_low = 100.0 - entry_low_vol.entry
    offset_high = 100.0 - entry_high_vol.entry
    assert offset_high > offset_low > 0


def test_entry_offset_shallow_for_impulse_deep_for_reversal():
    liquidity = _empty_liquidity()
    vol = _FakeVolatility("NORMAL", "STABLE")
    entry_impulse = compute_entry("LONG", 100.0, atr_value=1.0, liquidity_state=liquidity, volatility_state=vol, price_action_state=_FakePriceAction("impulse"))
    entry_reversal = compute_entry("LONG", 100.0, atr_value=1.0, liquidity_state=liquidity, volatility_state=vol, price_action_state=_FakePriceAction("reversal"))
    assert (100.0 - entry_impulse.entry) < (100.0 - entry_reversal.entry)


def test_entry_zero_offset_when_already_pulled_back():
    liquidity = _empty_liquidity()
    vol = _FakeVolatility("NORMAL", "STABLE")
    entry = compute_entry("LONG", 100.0, atr_value=1.0, liquidity_state=liquidity, volatility_state=vol, price_action_state=_FakePriceAction("pullback"))
    assert entry.entry == pytest.approx(100.0)


def test_entry_pullback_capped_by_expected_move():
    liquidity = _empty_liquidity()
    vol = _FakeVolatility("HIGH", "EXPANSION")
    # A tiny expected_move_60m should force a much smaller offset than the
    # volatility-implied one.
    entry = compute_entry(
        "LONG", 100.0, atr_value=5.0, liquidity_state=liquidity,
        volatility_state=vol, price_action_state=_FakePriceAction("reversal"), expected_move_60m=0.5,
    )
    assert (100.0 - entry.entry) <= 0.4 * 0.5 + 1e-9


def test_entry_v1_backward_compatible_without_new_params():
    liquidity = _empty_liquidity()
    entry = compute_entry("LONG", 100.0, atr_value=1.0, liquidity_state=liquidity)
    assert entry.entry == pytest.approx(100.0 - 0.25)


def _structure_with_swing_low(price: float) -> StructureState:
    from psygnal.intelligence.structure import SwingPoint

    return StructureState(
        timeframe="M5", trend_structure="RANGE",
        swing_highs=[], swing_lows=[SwingPoint(time=None, price=price, kind="low", index=0)],
        last_bos=None, last_choch=None, structural_reclaim=False, structural_failure=False,
        range_high=None, range_low=None,
    )


def test_stop_buffer_scales_with_volatility_regime():
    structure = _structure_with_swing_low(95.0)
    stop_compression = compute_stop_loss("LONG", 100.0, 1.0, structure, volatility_state=_FakeVolatility("LOW", "COMPRESSION"))
    stop_expansion = compute_stop_loss("LONG", 100.0, 1.0, structure, volatility_state=_FakeVolatility("HIGH", "EXPANSION"))
    assert stop_expansion.stop_distance > stop_compression.stop_distance


def test_stop_extends_to_liquidity_sweep_extreme():
    structure = _structure_with_swing_low(95.0)
    liquidity = LiquidityState(
        zones_above=[], zones_below=[], nearest_above=None, nearest_below=None,
        sweep_events=[{"zone": "SWING_LOW", "zone_price": 92.0, "direction": "sweep_low", "time": "t", "recent": True}],
        reclaim=True, failed_breakout=False, breakout_retest=False, continuation_after_breakout=False,
    )
    stop_with_sweep = compute_stop_loss("LONG", 100.0, 1.0, structure, liquidity_state=liquidity)
    stop_without_sweep = compute_stop_loss("LONG", 100.0, 1.0, structure)
    assert stop_with_sweep.stop_loss < stop_without_sweep.stop_loss
    assert "sweep" in stop_with_sweep.invalidation_reason.lower()


def test_stop_v1_backward_compatible_without_new_params():
    structure = _structure_with_swing_low(95.0)
    stop = compute_stop_loss("LONG", 100.0, 1.0, structure)
    assert stop.stop_loss == pytest.approx(95.0 - 0.2)


def test_target_skips_liquidity_zone_too_close_for_horizon():
    liquidity = LiquidityState(
        zones_above=[LiquidityZone("TOO_CLOSE", 100.05, "high", "swing"), LiquidityZone("USABLE", 105.0, "high", "swing")],
        zones_below=[], nearest_above=None, nearest_below=None,
        sweep_events=[], reclaim=False, failed_breakout=False, breakout_retest=False, continuation_after_breakout=False,
    )
    target = compute_targets("LONG", 100.0, atr_value=1.0, liquidity_state=liquidity, expected_move_60m=2.0)
    assert target.tp1 == pytest.approx(105.0)


def test_target_uses_historical_mfe_atr_when_available():
    liquidity = LiquidityState(
        zones_above=[LiquidityZone("ONLY_ONE", 105.0, "high", "swing")],
        zones_below=[], nearest_above=None, nearest_below=None,
        sweep_events=[], reclaim=False, failed_breakout=False, breakout_retest=False, continuation_after_breakout=False,
    )
    target = compute_targets("LONG", 100.0, atr_value=1.0, liquidity_state=liquidity, expected_move_60m=2.0, historical_mfe_atr=3.0)
    assert target.tp2 == pytest.approx(103.0)
    assert "historical" in target.tp2_reason.lower()


def test_target_v1_backward_compatible_without_new_params():
    liquidity = _empty_liquidity()
    target = compute_targets("LONG", 100.0, atr_value=1.0, liquidity_state=liquidity, expected_move_60m=None)
    assert target.tp1 == pytest.approx(101.0)
    assert target.tp2 == pytest.approx(102.0)
