"""Entry engine: connects the forecast direction to a structurally-anchored
entry price, rather than blindly returning the current market price.

V2: the pullback offset is now state-dependent (volatility regime, price
action, and the 60-minute expected move all scale it) instead of a fixed
0.25 ATR — see `_pullback_offset_atr_multiple`. Passing `volatility_state`/
`price_action_state`/`expected_move_60m` is optional so existing call
sites keep working; omitting them falls back to the V1 fixed-offset
behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from psygnal.intelligence.liquidity import LiquidityState

MAX_PULLBACK_ATR_MULTIPLE = 1.5
DEFAULT_PULLBACK_ATR_MULTIPLE = 0.25
ENTRY_ZONE_HALF_WIDTH_ATR_MULTIPLE = 0.15

# Volatility-regime scaling for both the default pullback offset and how
# far we're willing to reach for a liquidity-zone anchor: a quiet market
# retraces less and a liquidity zone far away is less reachable in ~60
# minutes than in an expanding/high-volatility one.
_VOLATILITY_LABEL_SCALE = {"LOW": 0.6, "NORMAL": 1.0, "HIGH": 1.4}

# Price-action-conditioned pullback fraction (of ATR): momentum-style
# setups (impulse/breakout) justify chasing with a shallow pullback since
# waiting for a deep retracement risks missing the move; an already-
# retracing setup justifies entering at the current price rather than
# waiting for more; range-bound/consolidating conditions need very little
# offset since price oscillates around the same level anyway.
_PRICE_ACTION_PULLBACK_FRACTION = {
    "impulse": 0.15,
    "breakout": 0.15,
    "pullback": 0.0,
    "continuation": 0.25,
    "consolidation": 0.10,
    "reversal": 0.35,
    "failed_breakout": 0.35,
    "exhaustion": 0.35,
    "rejection": 0.20,
}

# An entry pullback deeper than this fraction of the 60-minute expected
# move would leave little room left to reach TP1 within the forecast
# horizon, so it's capped.
_MAX_PULLBACK_FRACTION_OF_EXPECTED_MOVE = 0.4


@dataclass
class EntryPlan:
    entry: float
    entry_zone: tuple[float, float]
    anchor_reason: str

    def as_dict(self) -> dict[str, Any]:
        return {"entry": self.entry, "entry_zone": list(self.entry_zone), "anchor_reason": self.anchor_reason}


def _pullback_offset_atr_multiple(
    volatility_state: Optional[Any], price_action_state: Optional[Any]
) -> tuple[float, float]:
    """Returns (offset_multiple, max_reach_multiple) in ATR units."""
    if volatility_state is None and price_action_state is None:
        return DEFAULT_PULLBACK_ATR_MULTIPLE, MAX_PULLBACK_ATR_MULTIPLE

    vol_scale = _VOLATILITY_LABEL_SCALE.get(getattr(volatility_state, "label", None), 1.0)
    base_fraction = _PRICE_ACTION_PULLBACK_FRACTION.get(
        getattr(price_action_state, "label", None), DEFAULT_PULLBACK_ATR_MULTIPLE
    )
    offset = base_fraction * vol_scale
    max_reach = MAX_PULLBACK_ATR_MULTIPLE * vol_scale
    return offset, max_reach


def compute_entry(
    direction: str,
    current_price: float,
    atr_value: Optional[float],
    liquidity_state: LiquidityState,
    volatility_state: Optional[Any] = None,
    price_action_state: Optional[Any] = None,
    expected_move_60m: Optional[float] = None,
) -> EntryPlan:
    atr_value = atr_value if atr_value and atr_value > 0 else current_price * 0.001

    offset_multiple, max_reach_multiple = _pullback_offset_atr_multiple(volatility_state, price_action_state)
    offset = offset_multiple * atr_value

    if expected_move_60m and offset > _MAX_PULLBACK_FRACTION_OF_EXPECTED_MOVE * expected_move_60m:
        offset = _MAX_PULLBACK_FRACTION_OF_EXPECTED_MOVE * expected_move_60m

    pa_label = getattr(price_action_state, "label", None)
    reason_state = f" (price action: {pa_label}, volatility: {getattr(volatility_state, 'label', 'unknown')})" if (
        volatility_state is not None or price_action_state is not None
    ) else ""

    if direction == "LONG":
        nearest = liquidity_state.nearest_below
        if nearest is not None and (current_price - nearest.price) <= max_reach_multiple * atr_value:
            entry = nearest.price
            reason = f"pullback to {nearest.label} liquidity zone{reason_state}"
        else:
            entry = current_price - offset
            reason = f"state-dependent ATR pullback ({offset_multiple:.2f}x ATR, no reachable liquidity zone){reason_state}"
    else:
        nearest = liquidity_state.nearest_above
        if nearest is not None and (nearest.price - current_price) <= max_reach_multiple * atr_value:
            entry = nearest.price
            reason = f"pullback to {nearest.label} liquidity zone{reason_state}"
        else:
            entry = current_price + offset
            reason = f"state-dependent ATR pullback ({offset_multiple:.2f}x ATR, no reachable liquidity zone){reason_state}"

    half_width = ENTRY_ZONE_HALF_WIDTH_ATR_MULTIPLE * atr_value
    zone = (entry - half_width, entry + half_width)
    return EntryPlan(entry=entry, entry_zone=zone, anchor_reason=reason)
