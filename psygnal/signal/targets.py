"""Target engine: TP1/TP2 from liquidity, structure, and expected volatility.

V2: a nearest liquidity zone that's too close for a ~60-minute horizon (a
tiny fraction of the expected move) is skipped in favor of the next
usable one or an ATR/expected-move-based target; fallback multiples scale
with the volatility regime; and when a trained model's out-of-sample
historical MFE/MAE statistics are available (`historical_mfe_atr`/
`historical_mae_atr`, in ATR units — see forecasting/train_pipeline.py),
they inform TP2 as a genuinely data-derived alternative to a flat ATR
multiple. All of this is optional and defaults to the V1 behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from psygnal.intelligence.liquidity import LiquidityState

TP1_FALLBACK_ATR_MULTIPLE = 1.0
TP2_FALLBACK_ATR_MULTIPLE = 2.0

_VOLATILITY_REGIME_SCALE = {"EXPANSION": 1.4, "COMPRESSION": 0.7, "STABLE": 1.0}

# A liquidity zone closer than this fraction of the expected 60-minute
# move is too close to serve as a meaningful target for this horizon.
_MIN_ZONE_DISTANCE_FRACTION_OF_EXPECTED_MOVE = 0.15


@dataclass
class TargetPlan:
    tp1: float
    tp2: Optional[float]
    tp1_reason: str
    tp2_reason: str

    def as_dict(self) -> dict[str, Any]:
        return {"tp1": self.tp1, "tp2": self.tp2, "tp1_reason": self.tp1_reason, "tp2_reason": self.tp2_reason}


def _dedupe_sorted_by_distance(zones: list, entry: float) -> list:
    deduped: list = []
    seen_prices: set[float] = set()
    for z in sorted(zones, key=lambda z: abs(z.price - entry)):
        rounded = round(z.price, 6)
        if rounded in seen_prices:
            continue
        seen_prices.add(rounded)
        deduped.append(z)
    return deduped


def compute_targets(
    direction: str,
    entry: float,
    atr_value: Optional[float],
    liquidity_state: LiquidityState,
    expected_move_60m: Optional[float],
    volatility_state: Optional[Any] = None,
    historical_mfe_atr: Optional[float] = None,
) -> TargetPlan:
    atr_value = atr_value if atr_value and atr_value > 0 else entry * 0.001
    regime_scale = _VOLATILITY_REGIME_SCALE.get(getattr(volatility_state, "regime", None), 1.0)

    zones = liquidity_state.zones_above if direction == "LONG" else liquidity_state.zones_below
    zones_sorted = _dedupe_sorted_by_distance(zones, entry)

    min_distance = (
        _MIN_ZONE_DISTANCE_FRACTION_OF_EXPECTED_MOVE * expected_move_60m if expected_move_60m else 0.0
    )
    usable_zones = [z for z in zones_sorted if abs(z.price - entry) >= min_distance]

    if usable_zones:
        tp1 = usable_zones[0].price
        tp1_reason = f"nearest liquidity zone at a meaningful distance for this horizon: {usable_zones[0].label}"
    else:
        tp1_fallback = TP1_FALLBACK_ATR_MULTIPLE * regime_scale * atr_value
        tp1 = entry + tp1_fallback if direction == "LONG" else entry - tp1_fallback
        reason_bits = "no liquidity zone available" if not zones_sorted else "nearest zone(s) too close for a 60-minute horizon"
        tp1_reason = f"{reason_bits}; ATR-based fallback target (volatility-scaled)"

    if len(usable_zones) >= 2:
        tp2 = usable_zones[1].price
        tp2_reason = f"next liquidity zone: {usable_zones[1].label}"
    elif historical_mfe_atr:
        historical_move = historical_mfe_atr * atr_value
        tp2 = entry + historical_move if direction == "LONG" else entry - historical_move
        tp2_reason = f"historical out-of-sample median favorable excursion ({historical_mfe_atr:.2f}x ATR)"
    elif expected_move_60m:
        tp2 = entry + expected_move_60m if direction == "LONG" else entry - expected_move_60m
        tp2_reason = "expected 60-minute volatility-based range"
    else:
        tp2_fallback = TP2_FALLBACK_ATR_MULTIPLE * regime_scale * atr_value
        tp2 = entry + tp2_fallback if direction == "LONG" else entry - tp2_fallback
        tp2_reason = "no liquidity zone or volatility estimate available; ATR-based fallback target"

    return TargetPlan(tp1=tp1, tp2=tp2, tp1_reason=tp1_reason, tp2_reason=tp2_reason)
