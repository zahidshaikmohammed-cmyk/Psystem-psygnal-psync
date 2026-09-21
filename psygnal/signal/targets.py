"""Target engine: TP1/TP2 from liquidity, structure, and expected volatility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from psygnal.intelligence.liquidity import LiquidityState

TP1_FALLBACK_ATR_MULTIPLE = 1.0
TP2_FALLBACK_ATR_MULTIPLE = 2.0


@dataclass
class TargetPlan:
    tp1: float
    tp2: Optional[float]
    tp1_reason: str
    tp2_reason: str

    def as_dict(self) -> dict[str, Any]:
        return {"tp1": self.tp1, "tp2": self.tp2, "tp1_reason": self.tp1_reason, "tp2_reason": self.tp2_reason}


def compute_targets(
    direction: str,
    entry: float,
    atr_value: Optional[float],
    liquidity_state: LiquidityState,
    expected_move_60m: Optional[float],
) -> TargetPlan:
    atr_value = atr_value if atr_value and atr_value > 0 else entry * 0.001
    zones = liquidity_state.zones_above if direction == "LONG" else liquidity_state.zones_below
    deduped: list = []
    seen_prices: set[float] = set()
    for z in sorted(zones, key=lambda z: abs(z.price - entry)):
        rounded = round(z.price, 6)
        if rounded in seen_prices:
            continue
        seen_prices.add(rounded)
        deduped.append(z)
    zones_sorted = deduped

    if len(zones_sorted) >= 1:
        tp1 = zones_sorted[0].price
        tp1_reason = f"nearest liquidity zone: {zones_sorted[0].label}"
    else:
        tp1 = entry + TP1_FALLBACK_ATR_MULTIPLE * atr_value if direction == "LONG" else entry - TP1_FALLBACK_ATR_MULTIPLE * atr_value
        tp1_reason = "no liquidity zone available; ATR-based fallback target"

    if len(zones_sorted) >= 2:
        tp2 = zones_sorted[1].price
        tp2_reason = f"next liquidity zone: {zones_sorted[1].label}"
    elif expected_move_60m:
        tp2 = entry + expected_move_60m if direction == "LONG" else entry - expected_move_60m
        tp2_reason = "expected 60-minute volatility-based range"
    else:
        tp2 = entry + TP2_FALLBACK_ATR_MULTIPLE * atr_value if direction == "LONG" else entry - TP2_FALLBACK_ATR_MULTIPLE * atr_value
        tp2_reason = "no liquidity zone or volatility estimate available; ATR-based fallback target"

    return TargetPlan(tp1=tp1, tp2=tp2, tp1_reason=tp1_reason, tp2_reason=tp2_reason)
