"""Sequence intelligence: what has just happened, across rolling windows.

This is core intelligence per the constitution — the engine must reason
about *sequences* of candles, not just the latest bar or a single
indicator snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from psygnal import config


@dataclass
class WindowSequenceStats:
    window: int
    consecutive_direction: int
    dominant_direction: int  # +1, -1, 0
    trend_persistence: float  # fraction of candles agreeing with dominant direction
    return_sum: float
    return_mean: float
    return_std: float
    range_mean: float
    volume_trend: float  # >0 rising, <0 falling (relative slope)
    volatility_trend: float
    body_trend: float
    acceleration: float  # second-half mean return - first-half mean return
    reversal_attempt: bool
    failed_continuation: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "window": self.window,
            "consecutive_direction": self.consecutive_direction,
            "dominant_direction": self.dominant_direction,
            "trend_persistence": round(self.trend_persistence, 4),
            "return_sum": self.return_sum,
            "return_mean": self.return_mean,
            "return_std": self.return_std,
            "range_mean": self.range_mean,
            "volume_trend": round(self.volume_trend, 4),
            "volatility_trend": round(self.volatility_trend, 4),
            "body_trend": round(self.body_trend, 4),
            "acceleration": self.acceleration,
            "reversal_attempt": self.reversal_attempt,
            "failed_continuation": self.failed_continuation,
        }


def _safe_mean(arr: np.ndarray) -> float:
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return 0.0
    return float(valid.mean())


def _relative_slope(values: np.ndarray) -> float:
    """Slope of a simple linear regression against index, normalized by the
    series mean so it is comparable across instruments/scales."""
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    mean_v = np.nanmean(values)
    if not np.isfinite(mean_v) or mean_v == 0:
        return 0.0
    slope = np.polyfit(x, values, 1)[0]
    return float(slope / abs(mean_v))


def compute_window_stats(features: pd.DataFrame, window: int) -> WindowSequenceStats | None:
    if len(features) < window:
        return None
    w = features.iloc[-window:]

    direction = w["direction"].to_numpy()
    returns = w["return"].to_numpy()
    ranges = w["true_range"].to_numpy()
    volumes = w["volume"].to_numpy()
    bodies = w["body"].to_numpy()

    last_dir = direction[-1] if direction[-1] != 0 else (direction[direction != 0][-1] if (direction != 0).any() else 0)
    consecutive = 0
    for d in direction[::-1]:
        if d == 0:
            continue
        if consecutive == 0 or d == last_dir:
            consecutive += 1
            last_dir = d
        else:
            break

    pos = int((direction > 0).sum())
    neg = int((direction < 0).sum())
    dominant = 1 if pos > neg else (-1 if neg > pos else 0)
    persistence = max(pos, neg) / max(1, (pos + neg))

    half = window // 2
    first_half = returns[:half] if half > 0 else returns[:1]
    second_half = returns[half:] if half > 0 else returns[1:]
    accel = _safe_mean(second_half) - _safe_mean(first_half) if half > 0 else 0.0

    reversal_attempt = False
    failed_continuation = False
    if window >= 3:
        prior_dir = 1 if (direction[:-1] > 0).sum() > (direction[:-1] < 0).sum() else -1
        if direction[-1] != 0 and direction[-1] != prior_dir and w["body_pct"].iloc[-1] >= 0.5:
            reversal_attempt = True
        if direction[-1] == prior_dir and w["body_pct"].iloc[-1] < 0.25:
            failed_continuation = True

    return WindowSequenceStats(
        window=window,
        consecutive_direction=consecutive,
        dominant_direction=dominant,
        trend_persistence=float(persistence),
        return_sum=float(np.nansum(returns)),
        return_mean=float(np.nanmean(returns)),
        return_std=float(np.nanstd(returns)),
        range_mean=float(np.nanmean(ranges)),
        volume_trend=_relative_slope(volumes),
        volatility_trend=_relative_slope(ranges),
        body_trend=_relative_slope(bodies),
        acceleration=accel,
        reversal_attempt=reversal_attempt,
        failed_continuation=failed_continuation,
    )


def compute_all_window_stats(
    features: pd.DataFrame, windows: tuple[int, ...] = config.SEQUENCE_WINDOWS
) -> dict[int, WindowSequenceStats]:
    result: dict[int, WindowSequenceStats] = {}
    for w in windows:
        stats = compute_window_stats(features, w)
        if stats is not None:
            result[w] = stats
    return result


def detect_compression_to_expansion(features: pd.DataFrame, short: int = 6, long: int = 24) -> bool:
    if len(features) < long:
        return False
    recent = features["range_percentile"].iloc[-short:].dropna()
    older = features["range_percentile"].iloc[-long:-short].dropna()
    if recent.empty or older.empty:
        return False
    return bool(recent.mean() >= 0.6 and older.mean() <= 0.35)


def detect_expansion_to_exhaustion(features: pd.DataFrame, short: int = 6, long: int = 24) -> bool:
    if len(features) < long:
        return False
    older = features.iloc[-long:-short]
    recent = features.iloc[-short:]
    if older.empty or recent.empty:
        return False
    older_expansion = older["range_percentile"].mean() >= 0.55
    recent_fading = recent["body_pct"].mean() <= 0.35 and recent["directional_strength"].mean() <= 0.5
    return bool(older_expansion and recent_fading)


def summarize_sequences(features: pd.DataFrame) -> dict[str, Any]:
    window_stats = compute_all_window_stats(features)
    return {
        "windows": {w: s.as_dict() for w, s in window_stats.items()},
        "compression_to_expansion": detect_compression_to_expansion(features),
        "expansion_to_exhaustion": detect_expansion_to_exhaustion(features),
    }
