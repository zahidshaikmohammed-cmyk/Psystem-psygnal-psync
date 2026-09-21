"""Stop-loss engine: structural invalidation, ATR-buffered.

V2: the buffer and ATR fallback multiple now scale with the volatility
regime (a compressed market needs a tighter stop to stay meaningful; an
expanding one needs a wider one to avoid noise stopouts), and a recent
liquidity sweep beyond the nearest swing extends the invalidation level to
the actual proven wick extreme rather than the swing point alone — since
price has already demonstrated it can reach that far. `volatility_state`
and `liquidity_state` are optional so existing call sites keep working
with the V1 fixed-buffer behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from psygnal.intelligence.structure import StructureState

STRUCTURAL_BUFFER_ATR_MULTIPLE = 0.2
FALLBACK_STOP_ATR_MULTIPLE = 1.5

# EXPANSION needs more room (avoid getting stopped out by noise on a
# widening range); COMPRESSION can afford a tighter stop since the
# invalidation thesis breaks with a smaller adverse move.
_VOLATILITY_REGIME_SCALE = {"EXPANSION": 1.5, "COMPRESSION": 0.75, "STABLE": 1.0}


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


def _sweep_extreme(liquidity_state: Optional[Any], direction: str) -> Optional[float]:
    """The furthest price actually reached by a recent liquidity sweep in
    the direction that would invalidate this trade — a stronger
    invalidation anchor than the swing point alone, since it's a proven
    wick rather than an inferred pivot."""
    if liquidity_state is None:
        return None
    recent = [e for e in liquidity_state.sweep_events if e.get("recent")]
    if direction == "LONG":
        lows = [e["zone_price"] for e in recent if e["direction"] == "sweep_low"]
        return min(lows) if lows else None
    highs = [e["zone_price"] for e in recent if e["direction"] == "sweep_high"]
    return max(highs) if highs else None


def compute_stop_loss(
    direction: str,
    entry: float,
    atr_value: Optional[float],
    structure_m5: StructureState,
    volatility_state: Optional[Any] = None,
    liquidity_state: Optional[Any] = None,
) -> StopPlan:
    atr_value = atr_value if atr_value and atr_value > 0 else entry * 0.001
    regime_scale = _VOLATILITY_REGIME_SCALE.get(getattr(volatility_state, "regime", None), 1.0)
    buffer = STRUCTURAL_BUFFER_ATR_MULTIPLE * regime_scale * atr_value
    fallback_multiple = FALLBACK_STOP_ATR_MULTIPLE * regime_scale

    sweep_extreme = _sweep_extreme(liquidity_state, direction)

    if direction == "LONG":
        structural_lows = [sp.price for sp in structure_m5.swing_lows if sp.price < entry]
        nearest_swing = max(structural_lows) if structural_lows else None
        sweep_candidate = sweep_extreme if (sweep_extreme is not None and sweep_extreme < entry) else None

        if nearest_swing is not None:
            level = nearest_swing
            reason = "below nearest confirmed swing low"
            # A sweep that dipped BELOW the nearest swing proves that swing
            # alone isn't a reliable invalidation — extend to the wick.
            if sweep_candidate is not None and sweep_candidate < nearest_swing:
                level = sweep_candidate
                reason = "below the extreme of a recent liquidity sweep (proven wick, not just a swing pivot)"
            stop_loss = level - buffer
        elif sweep_candidate is not None:
            level = sweep_candidate
            stop_loss = level - buffer
            reason = "below the extreme of a recent liquidity sweep (no confirmed swing low available)"
        else:
            stop_loss = entry - fallback_multiple * atr_value
            level = stop_loss
            reason = "no confirmed swing low available; ATR-based fallback stop"
        stop_loss = min(stop_loss, entry - buffer)
    else:
        structural_highs = [sp.price for sp in structure_m5.swing_highs if sp.price > entry]
        nearest_swing = min(structural_highs) if structural_highs else None
        sweep_candidate = sweep_extreme if (sweep_extreme is not None and sweep_extreme > entry) else None

        if nearest_swing is not None:
            level = nearest_swing
            reason = "above nearest confirmed swing high"
            if sweep_candidate is not None and sweep_candidate > nearest_swing:
                level = sweep_candidate
                reason = "above the extreme of a recent liquidity sweep (proven wick, not just a swing pivot)"
            stop_loss = level + buffer
        elif sweep_candidate is not None:
            level = sweep_candidate
            stop_loss = level + buffer
            reason = "above the extreme of a recent liquidity sweep (no confirmed swing high available)"
        else:
            stop_loss = entry + fallback_multiple * atr_value
            level = stop_loss
            reason = "no confirmed swing high available; ATR-based fallback stop"
        stop_loss = max(stop_loss, entry + buffer)

    if volatility_state is not None:
        reason += f" (volatility regime: {getattr(volatility_state, 'regime', 'unknown')}, buffer x{regime_scale:.2f})"

    return StopPlan(
        stop_loss=stop_loss,
        stop_distance=abs(entry - stop_loss),
        invalidation_level=level,
        invalidation_reason=reason,
    )
