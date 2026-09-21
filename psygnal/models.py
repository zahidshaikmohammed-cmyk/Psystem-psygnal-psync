"""Shared typed data structures used across the Psygnal pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Raw market data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candle:
    """A single OHLCV candle. `time` is always UTC and tz-aware."""

    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    completed: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "time": self.time.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "completed": self.completed,
        }


def candles_to_frame(candles: list[Candle]) -> pd.DataFrame:
    """Convert a list of Candle into a sorted, indexed DataFrame."""
    if not candles:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume", "completed"]
        )
    df = pd.DataFrame(
        {
            "time": [c.time for c in candles],
            "open": [c.open for c in candles],
            "high": [c.high for c in candles],
            "low": [c.low for c in candles],
            "close": [c.close for c in candles],
            "volume": [c.volume for c in candles],
            "completed": [c.completed for c in candles],
        }
    )
    df = df.sort_values("time").drop_duplicates(subset="time", keep="last")
    df = df.set_index("time")
    return df


def frame_to_candles(df: pd.DataFrame) -> list[Candle]:
    out: list[Candle] = []
    for ts, row in df.iterrows():
        out.append(
            Candle(
                time=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume", 0.0)),
                completed=bool(row.get("completed", True)),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------


class DataStatus(str, Enum):
    OK = "OK"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass
class DataQuality:
    status: DataStatus
    symbol: str
    candles_available: int
    candles_required: int
    freshness_seconds: Optional[float]
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


# ---------------------------------------------------------------------------
# Multi-timeframe container
# ---------------------------------------------------------------------------


@dataclass
class MultiTimeframeSeries:
    """Completed-candle-only OHLCV frames for each timeframe, plus the
    currently-forming (incomplete) candle kept separately so it can never
    leak into structural/statistical analysis by accident."""

    symbol: str
    frames: dict[str, pd.DataFrame]  # timeframe -> completed candles
    forming: dict[str, Optional[Candle]]  # timeframe -> forming candle or None

    def get(self, timeframe: str) -> pd.DataFrame:
        return self.frames.get(timeframe, pd.DataFrame())


# ---------------------------------------------------------------------------
# Enums for qualitative states
# ---------------------------------------------------------------------------


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class TrendLabel(str, Enum):
    STRONG_BULLISH = "STRONG_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"
    STRONG_BEARISH = "STRONG_BEARISH"


class RegimeLabel(str, Enum):
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE = "RANGE"
    BREAKOUT = "BREAKOUT"
    BREAKOUT_RETEST = "BREAKOUT_RETEST"
    LIQUIDITY_SWEEP = "LIQUIDITY_SWEEP"
    REVERSAL = "REVERSAL"
    VOLATILITY_EXPANSION = "VOLATILITY_EXPANSION"
    VOLATILITY_CONTRACTION = "VOLATILITY_CONTRACTION"
    NEWS_SHOCK = "NEWS_SHOCK"
    CHAOTIC = "CHAOTIC"


# ---------------------------------------------------------------------------
# Final signal object
# ---------------------------------------------------------------------------


@dataclass
class FinalSignal:
    symbol: str
    timestamp: datetime

    direction: str
    probability_long: float
    probability_short: float
    probability_neutral: float

    # --- V2: three distinct concepts, never mixed (see signal/README notes) ---
    # A. SIGNAL SCORE (quality/confluence) — `signal_score` below.
    # B. MODEL PROBABILITY — only populated once a genuine trained model
    #    contributed; `probability_long/short/neutral` above remain the
    #    *operational* blended forecast used for direction/entry/etc, which
    #    may be purely deterministic. `deterministic_forecast` and
    #    `model_probability` disaggregate the two so neither is mistaken
    #    for a historically validated statistic when it isn't one.
    # C. CONFIDENCE — how strongly available information supports the
    #    forecast; `confidence`/`confidence_score` below.
    forecast_engine: str  # "DETERMINISTIC" | "TRAINED_ML" | "ENSEMBLE"
    model_status: str  # "TRAINED" | "UNTRAINED"
    deterministic_forecast: dict[str, float]
    model_probability: Optional[dict[str, Any]]
    calibration_state: dict[str, Any]

    confidence: str
    confidence_score: float

    signal_score: float
    scoring_mode: str  # "EXPERT_WEIGHTED" | "HISTORICALLY_CALIBRATED"

    current_price: float
    entry: float
    entry_zone: tuple[float, float]

    stop_loss: float
    tp1: float
    tp2: Optional[float]

    expected_60m_return: float
    expected_60m_high: float
    expected_60m_low: float
    expected_60m_range: float
    expected_range_methodology: str  # "SYMMETRIC_ATR_V1" | "HISTORICAL_MFE_MAE"

    rr: Optional[float]

    market_regime: str

    trend_state: dict[str, str]
    structure_state: dict[str, Any]
    liquidity_state: dict[str, Any]
    momentum_state: dict[str, Any]
    volatility_state: dict[str, Any]
    volume_state: dict[str, Any]
    session_state: dict[str, Any]

    usd_composite: dict[str, Any]
    cross_market_state: dict[str, Any]

    macro_state: dict[str, Any]
    news_state: dict[str, Any]

    historical_pattern_state: dict[str, Any]

    reasons: list[str]
    conflicts: list[str]
    warnings: list[str]

    data_quality: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d
