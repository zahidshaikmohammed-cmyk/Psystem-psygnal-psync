"""Stop-loss engine: structural invalidation, ATR-buffered."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from psygnal.intelligence.structure import StructureState

STRUCTURAL_BUFFER_ATR_MULTIPLE = 0.2
FALLBACK_STOP_ATR_MULTIPLE = 1.5


@dataclass
class StopPlan:
    stop_loss: float
    stop_distance: float
    invalidation_level: float
    invalidation_reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "stop_loss": self.stop_loss,
            "stop_distance": self.stop_distance,
            "invalidation_level": self.invalidation_level,
            "invalidation_reason": self.invalidation_reason,
        }


def compute_stop_loss(
    direction: str,
    entry: float,
    atr_value: Optional[float],
    structure_m5: StructureState,
) -> StopPlan:
    atr_value = atr_value if atr_value and atr_value > 0 else entry * 0.001
    buffer = STRUCTURAL_BUFFER_ATR_MULTIPLE * atr_value

    if direction == "LONG":
        structural_lows = [sp.price for sp in structure_m5.swing_lows if sp.price < entry]
        if structural_lows:
            level = max(structural_lows)
            stop_loss = level - buffer
            reason = "below nearest confirmed swing low"
        else:
            stop_loss = entry - FALLBACK_STOP_ATR_MULTIPLE * atr_value
            level = stop_loss
            reason = "no confirmed swing low available; ATR-based fallback stop"
        stop_loss = min(stop_loss, entry - buffer)
    else:
        structural_highs = [sp.price for sp in structure_m5.swing_highs if sp.price > entry]
        if structural_highs:
            level = min(structural_highs)
            stop_loss = level + buffer
            reason = "above nearest confirmed swing high"
        else:
            stop_loss = entry + FALLBACK_STOP_ATR_MULTIPLE * atr_value
            level = stop_loss
            reason = "no confirmed swing high available; ATR-based fallback stop"
        stop_loss = max(stop_loss, entry + buffer)

    return StopPlan(
        stop_loss=stop_loss,
        stop_distance=abs(entry - stop_loss),
        invalidation_level=level,
        invalidation_reason=reason,
    )
