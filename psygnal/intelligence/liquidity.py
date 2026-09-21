"""Liquidity-zone mapping and sweep/reclaim/breakout-retest detection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

from psygnal import config
from psygnal.intelligence.sessions import summarize_sessions
from psygnal.intelligence.structure import StructureState


@dataclass
class LiquidityZone:
    label: str
    price: float
    kind: str  # "high" or "low"
    source: str


@dataclass
class LiquidityState:
    zones_above: list[LiquidityZone]
    zones_below: list[LiquidityZone]
    nearest_above: Optional[LiquidityZone]
    nearest_below: Optional[LiquidityZone]
    sweep_events: list[dict[str, Any]]
    reclaim: bool
    failed_breakout: bool
    breakout_retest: bool
    continuation_after_breakout: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "zones_above": [z.__dict__ for z in self.zones_above],
            "zones_below": [z.__dict__ for z in self.zones_below],
            "nearest_above": self.nearest_above.__dict__ if self.nearest_above else None,
            "nearest_below": self.nearest_below.__dict__ if self.nearest_below else None,
            "sweep_events": self.sweep_events,
            "reclaim": self.reclaim,
            "failed_breakout": self.failed_breakout,
            "breakout_retest": self.breakout_retest,
            "continuation_after_breakout": self.continuation_after_breakout,
        }


def _cluster_equal_levels(
    prices: list[float], tolerance: float, kind: str, label_prefix: str
) -> list[LiquidityZone]:
    if not prices:
        return []
    sorted_prices = sorted(prices)
    clusters: list[list[float]] = []
    current = [sorted_prices[0]]
    for p in sorted_prices[1:]:
        if abs(p - current[-1]) <= tolerance:
            current.append(p)
        else:
            clusters.append(current)
            current = [p]
    clusters.append(current)

    zones = []
    for cluster in clusters:
        if len(cluster) >= 2:
            avg = float(np.mean(cluster))
            zones.append(LiquidityZone(label=f"{label_prefix}", price=avg, kind=kind, source="equal_levels"))
    return zones


def build_liquidity_zones(
    df_m5: pd.DataFrame,
    structure_m5: StructureState,
    now_utc: datetime,
    atr_value: Optional[float],
) -> list[LiquidityZone]:
    zones: list[LiquidityZone] = []

    sessions = summarize_sessions(df_m5, now_utc)
    prev_day = sessions["previous_day"]
    if prev_day["prev_day_high"] is not None:
        zones.append(LiquidityZone("PREVIOUS_DAY_HIGH", prev_day["prev_day_high"], "high", "previous_day"))
        zones.append(LiquidityZone("PREVIOUS_DAY_LOW", prev_day["prev_day_low"], "low", "previous_day"))
        zones.append(LiquidityZone("PREVIOUS_DAY_CLOSE", prev_day["prev_day_close"], "close", "previous_day"))

    for name in ("ASIA", "LONDON", "NEW_YORK"):
        stats = sessions["prior"][name]
        if stats["high"] is not None:
            zones.append(LiquidityZone(f"{name}_HIGH", stats["high"], "high", "session"))
            zones.append(LiquidityZone(f"{name}_LOW", stats["low"], "low", "session"))
        today_stats = sessions["today"][name]
        if today_stats["high"] is not None and today_stats["candle_count"] > 0:
            zones.append(LiquidityZone(f"{name}_HIGH_TODAY", today_stats["high"], "high", "session"))
            zones.append(LiquidityZone(f"{name}_LOW_TODAY", today_stats["low"], "low", "session"))

    for sp in structure_m5.swing_highs[-8:]:
        zones.append(LiquidityZone("SWING_HIGH", sp.price, "high", "swing"))
    for sp in structure_m5.swing_lows[-8:]:
        zones.append(LiquidityZone("SWING_LOW", sp.price, "low", "swing"))

    if structure_m5.range_high is not None:
        zones.append(LiquidityZone("RANGE_HIGH", structure_m5.range_high, "high", "range_extreme"))
        zones.append(LiquidityZone("RANGE_LOW", structure_m5.range_low, "low", "range_extreme"))

    tolerance = (atr_value or 0.0) * config.EQUAL_LEVEL_TOLERANCE_ATR_MULTIPLE
    if tolerance > 0:
        swing_high_prices = [sp.price for sp in structure_m5.swing_highs[-12:]]
        swing_low_prices = [sp.price for sp in structure_m5.swing_lows[-12:]]
        zones.extend(_cluster_equal_levels(swing_high_prices, tolerance, "high", "EQUAL_HIGHS"))
        zones.extend(_cluster_equal_levels(swing_low_prices, tolerance, "low", "EQUAL_LOWS"))

    return zones


def _detect_sweep_events(df_m5: pd.DataFrame, zones: list[LiquidityZone], lookback: int = 12) -> list[dict[str, Any]]:
    if df_m5.empty:
        return []
    recent = df_m5.iloc[-lookback:]
    events: list[dict[str, Any]] = []
    for zone in zones:
        if zone.kind == "high":
            swept = (recent["high"] > zone.price) & (recent["close"] < zone.price)
        elif zone.kind == "low":
            swept = (recent["low"] < zone.price) & (recent["close"] > zone.price)
        else:
            continue
        if swept.any():
            last_idx = swept[swept].index[-1]
            events.append(
                {
                    "zone": zone.label,
                    "zone_price": zone.price,
                    "direction": "sweep_high" if zone.kind == "high" else "sweep_low",
                    "time": last_idx.isoformat() if hasattr(last_idx, "isoformat") else str(last_idx),
                    "recent": bool(last_idx == recent.index[-1]),
                }
            )
    return events


def analyze_liquidity(
    df_m5: pd.DataFrame,
    structure_m5: StructureState,
    now_utc: datetime,
    atr_value: Optional[float],
    current_price: float,
) -> LiquidityState:
    zones = build_liquidity_zones(df_m5, structure_m5, now_utc, atr_value)

    zones_above = sorted([z for z in zones if z.price > current_price], key=lambda z: z.price)
    zones_below = sorted([z for z in zones if z.price < current_price], key=lambda z: z.price, reverse=True)

    nearest_above = zones_above[0] if zones_above else None
    nearest_below = zones_below[0] if zones_below else None

    sweep_events = _detect_sweep_events(df_m5, zones)
    recent_sweep = any(e["recent"] for e in sweep_events)

    reclaim = structure_m5.structural_reclaim
    failed_breakout = structure_m5.structural_failure

    breakout_retest = False
    continuation_after_breakout = False
    if structure_m5.last_bos and len(df_m5) >= 8:
        broken_level = (
            structure_m5.swing_highs[-1].price
            if structure_m5.last_bos == "BULLISH_BOS" and structure_m5.swing_highs
            else (structure_m5.swing_lows[-1].price if structure_m5.swing_lows else None)
        )
        if broken_level is not None:
            recent = df_m5.iloc[-8:]
            tol = (atr_value or 0.0) * 0.5
            retested = (recent["low"] <= broken_level + tol).any() if structure_m5.last_bos == "BULLISH_BOS" else (recent["high"] >= broken_level - tol).any()
            held = (
                float(df_m5["close"].iloc[-1]) > broken_level
                if structure_m5.last_bos == "BULLISH_BOS"
                else float(df_m5["close"].iloc[-1]) < broken_level
            )
            if retested and held:
                breakout_retest = True
            if held and not retested:
                continuation_after_breakout = True

    return LiquidityState(
        zones_above=zones_above[:6],
        zones_below=zones_below[:6],
        nearest_above=nearest_above,
        nearest_below=nearest_below,
        sweep_events=sweep_events,
        reclaim=reclaim or recent_sweep,
        failed_breakout=failed_breakout,
        breakout_retest=breakout_retest,
        continuation_after_breakout=continuation_after_breakout,
    )
