"""Entry engine: connects the forecast direction to a structurally-anchored
entry price, rather than blindly returning the current market price."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from psygnal.intelligence.liquidity import LiquidityState

MAX_PULLBACK_ATR_MULTIPLE = 1.5
DEFAULT_PULLBACK_ATR_MULTIPLE = 0.25
ENTRY_ZONE_HALF_WIDTH_ATR_MULTIPLE = 0.15


@dataclass
class EntryPlan:
    entry: float
    entry_zone: tuple[float, float]
    anchor_reason: str

    def as_dict(self) -> dict[str, Any]:
        return {"entry": self.entry, "entry_zone": list(self.entry_zone), "anchor_reason": self.anchor_reason}


def compute_entry(
    direction: str,
    current_price: float,
    atr_value: Optional[float],
    liquidity_state: LiquidityState,
) -> EntryPlan:
    atr_value = atr_value if atr_value and atr_value > 0 else current_price * 0.001

    if direction == "LONG":
        nearest = liquidity_state.nearest_below
        if nearest is not None and (current_price - nearest.price) <= MAX_PULLBACK_ATR_MULTIPLE * atr_value:
            entry = nearest.price
            reason = f"pullback to {nearest.label} liquidity zone"
        else:
            entry = current_price - DEFAULT_PULLBACK_ATR_MULTIPLE * atr_value
            reason = "modest ATR-based pullback from current price (no nearby liquidity zone)"
    else:
        nearest = liquidity_state.nearest_above
        if nearest is not None and (nearest.price - current_price) <= MAX_PULLBACK_ATR_MULTIPLE * atr_value:
            entry = nearest.price
            reason = f"pullback to {nearest.label} liquidity zone"
        else:
            entry = current_price + DEFAULT_PULLBACK_ATR_MULTIPLE * atr_value
            reason = "modest ATR-based pullback from current price (no nearby liquidity zone)"

    half_width = ENTRY_ZONE_HALF_WIDTH_ATR_MULTIPLE * atr_value
    zone = (entry - half_width, entry + half_width)
    return EntryPlan(entry=entry, entry_zone=zone, anchor_reason=reason)
