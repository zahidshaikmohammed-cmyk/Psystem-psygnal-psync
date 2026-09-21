"""Market-structure engine: swings, HH/HL/LH/LL, BOS/CHOCH.

Run once per timeframe (M5/M15/M30/H1/H4). Higher timeframes give context;
M5 mostly assists with timing, per the constitution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from psygnal import config


@dataclass
class SwingPoint:
    time: Any
    price: float
    kind: str  # "high" or "low"
    index: int


@dataclass
class StructureState:
    timeframe: str
    trend_structure: str  # "UPTREND" | "DOWNTREND" | "RANGE" | "UNDEFINED"
    swing_highs: list[SwingPoint]
    swing_lows: list[SwingPoint]
    last_bos: Optional[str]  # "BULLISH_BOS" | "BEARISH_BOS" | None
    last_choch: Optional[str]  # "BULLISH_CHOCH" | "BEARISH_CHOCH" | None
    structural_reclaim: bool
    structural_failure: bool
    range_high: Optional[float]
    range_low: Optional[float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "timeframe": self.timeframe,
            "trend_structure": self.trend_structure,
            "swing_highs": [(s.price) for s in self.swing_highs[-5:]],
            "swing_lows": [(s.price) for s in self.swing_lows[-5:]],
            "last_bos": self.last_bos,
            "last_choch": self.last_choch,
            "structural_reclaim": self.structural_reclaim,
            "structural_failure": self.structural_failure,
            "range_high": self.range_high,
            "range_low": self.range_low,
        }


def find_swings(df: pd.DataFrame, lookback: int = config.SWING_LOOKBACK) -> tuple[list[SwingPoint], list[SwingPoint]]:
    """Fractal swing detection. A swing point at index i is only confirmed
    once `lookback` bars after it are known — this is intentional: it means
    the most recent `lookback` bars can never produce a confirmed swing,
    which prevents any hint of future information leaking backward."""
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    n = len(df)
    swing_highs: list[SwingPoint] = []
    swing_lows: list[SwingPoint] = []

    for i in range(lookback, n - lookback):
        window_h = highs[i - lookback : i + lookback + 1]
        window_l = lows[i - lookback : i + lookback + 1]
        if highs[i] == window_h.max() and np.argmax(window_h) == lookback:
            swing_highs.append(SwingPoint(time=df.index[i], price=float(highs[i]), kind="high", index=i))
        if lows[i] == window_l.min() and np.argmin(window_l) == lookback:
            swing_lows.append(SwingPoint(time=df.index[i], price=float(lows[i]), kind="low", index=i))

    return swing_highs, swing_lows


def classify_trend_structure(swing_highs: list[SwingPoint], swing_lows: list[SwingPoint]) -> str:
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "UNDEFINED"
    hh = swing_highs[-1].price > swing_highs[-2].price
    hl = swing_lows[-1].price > swing_lows[-2].price
    lh = swing_highs[-1].price < swing_highs[-2].price
    ll = swing_lows[-1].price < swing_lows[-2].price

    if hh and hl:
        return "UPTREND"
    if lh and ll:
        return "DOWNTREND"
    return "RANGE"


def _detect_bos_choch(
    df: pd.DataFrame,
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
    trend_structure: str,
) -> tuple[Optional[str], Optional[str]]:
    if df.empty or (not swing_highs and not swing_lows):
        return None, None

    last_close = float(df["close"].iloc[-1])
    last_swing_high = swing_highs[-1].price if swing_highs else None
    last_swing_low = swing_lows[-1].price if swing_lows else None

    bos = None
    choch = None

    if last_swing_high is not None and last_close > last_swing_high:
        if trend_structure in ("UPTREND", "RANGE", "UNDEFINED"):
            bos = "BULLISH_BOS"
        elif trend_structure == "DOWNTREND":
            choch = "BULLISH_CHOCH"

    if last_swing_low is not None and last_close < last_swing_low:
        if trend_structure in ("DOWNTREND", "RANGE", "UNDEFINED"):
            bos = "BEARISH_BOS"
        elif trend_structure == "UPTREND":
            choch = "BEARISH_CHOCH"

    return bos, choch


def _detect_reclaim_and_failure(
    df: pd.DataFrame, swing_highs: list[SwingPoint], swing_lows: list[SwingPoint]
) -> tuple[bool, bool]:
    """A structural failure: price broke a swing level then closed back
    beyond it within a short lookahead window (i.e. the break didn't hold).
    A structural reclaim: price lost a level then reclaimed it (bullish
    context) — treated symmetrically here as "the level flipped back"."""
    if len(df) < 6:
        return False, False

    recent = df.iloc[-6:]
    reclaim = False
    failure = False

    if swing_lows:
        level = swing_lows[-1].price
        broke_below = (recent["close"].iloc[:-1] < level).any()
        back_above = recent["close"].iloc[-1] > level
        if broke_below and back_above:
            reclaim = True

    if swing_highs:
        level = swing_highs[-1].price
        broke_above = (recent["close"].iloc[:-1] > level).any()
        back_below = recent["close"].iloc[-1] < level
        if broke_above and back_below:
            failure = True

    return reclaim, failure


def analyze_structure(df: pd.DataFrame, timeframe: str) -> StructureState:
    if df.empty or len(df) < 2 * config.SWING_LOOKBACK + 3:
        return StructureState(
            timeframe=timeframe,
            trend_structure="UNDEFINED",
            swing_highs=[],
            swing_lows=[],
            last_bos=None,
            last_choch=None,
            structural_reclaim=False,
            structural_failure=False,
            range_high=None,
            range_low=None,
        )

    swing_highs, swing_lows = find_swings(df)
    trend_structure = classify_trend_structure(swing_highs, swing_lows)
    bos, choch = _detect_bos_choch(df, swing_highs, swing_lows, trend_structure)
    reclaim, failure = _detect_reclaim_and_failure(df, swing_highs, swing_lows)

    lookback_range = df.iloc[-48:] if len(df) >= 48 else df
    range_high = float(lookback_range["high"].max())
    range_low = float(lookback_range["low"].min())

    return StructureState(
        timeframe=timeframe,
        trend_structure=trend_structure,
        swing_highs=swing_highs,
        swing_lows=swing_lows,
        last_bos=bos,
        last_choch=choch,
        structural_reclaim=reclaim,
        structural_failure=failure,
        range_high=range_high,
        range_low=range_low,
    )


def analyze_multi_timeframe_structure(
    frames: dict[str, pd.DataFrame]
) -> dict[str, StructureState]:
    return {tf: analyze_structure(df, tf) for tf, df in frames.items()}
