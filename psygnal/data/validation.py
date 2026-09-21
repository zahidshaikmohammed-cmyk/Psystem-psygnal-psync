"""Data-integrity validation for parsed M5 candle series.

Nothing in this module fabricates data. Individual malformed candles are
dropped (and counted); if the remaining series is insufficient the caller
gets `DataStatus.UNAVAILABLE` and must report DATA UNAVAILABLE rather than
proceed with analysis.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from psygnal import config
from psygnal.data.parser import RawCandleRecord
from psygnal.models import Candle, DataQuality, DataStatus

M5_SPACING = timedelta(minutes=5)


def validate_symbol(
    symbol: str,
    records: list[RawCandleRecord],
    *,
    now: datetime | None = None,
    min_candles_required: int = config.MIN_M5_CANDLES_REQUIRED,
    max_staleness_minutes: int = config.MAX_DATA_STALENESS_MINUTES,
) -> tuple[list[Candle], DataQuality]:
    now = now or datetime.now(timezone.utc)
    issues: list[str] = []
    warnings: list[str] = []

    # Sort and de-duplicate by timestamp (keep last occurrence).
    by_time: dict[datetime, RawCandleRecord] = {}
    duplicate_count = 0
    for rec in records:
        if rec.time in by_time:
            duplicate_count += 1
        by_time[rec.time] = rec
    if duplicate_count:
        warnings.append(f"{duplicate_count} duplicate timestamps collapsed")

    ordered = sorted(by_time.items(), key=lambda kv: kv[0])

    clean_candles: list[Candle] = []
    ohlc_violations = 0
    non_positive_price = 0
    negative_volume = 0

    for ts, rec in ordered:
        if rec.open <= 0 or rec.high <= 0 or rec.low <= 0 or rec.close <= 0:
            non_positive_price += 1
            continue
        if rec.volume is not None and rec.volume < 0:
            negative_volume += 1
            continue
        max_oc = max(rec.open, rec.close)
        min_oc = min(rec.open, rec.close)
        if rec.high < max_oc or rec.low > min_oc or rec.high < rec.low:
            ohlc_violations += 1
            continue
        clean_candles.append(
            Candle(
                time=ts,
                open=rec.open,
                high=rec.high,
                low=rec.low,
                close=rec.close,
                volume=rec.volume or 0.0,
                completed=True,
            )
        )

    if non_positive_price:
        issues.append(f"{non_positive_price} candles dropped: non-positive OHLC price")
    if negative_volume:
        issues.append(f"{negative_volume} candles dropped: negative volume")
    if ohlc_violations:
        issues.append(f"{ohlc_violations} candles dropped: invalid OHLC relationship")

    # Continuity: count gaps larger than one M5 step (informational only —
    # markets close on weekends/holidays, so gaps are expected, not fatal).
    gap_count = 0
    largest_gap = timedelta(0)
    for (t1, _), (t2, _) in zip(ordered, ordered[1:]):
        gap = t2 - t1
        if gap > M5_SPACING:
            gap_count += 1
            largest_gap = max(largest_gap, gap)
    if gap_count:
        warnings.append(
            f"{gap_count} gap(s) in candle continuity, largest {largest_gap}"
        )

    freshness_seconds: float | None = None
    if clean_candles:
        freshness_seconds = (now - clean_candles[-1].time).total_seconds()
        if freshness_seconds > max_staleness_minutes * 60:
            warnings.append(
                f"latest candle is stale: {freshness_seconds / 60:.1f} min old "
                f"(threshold {max_staleness_minutes} min) — market may be closed"
            )

    candles_available = len(clean_candles)
    status = DataStatus.OK
    if candles_available == 0:
        status = DataStatus.UNAVAILABLE
        issues.append("no valid candles remained after validation")
    elif candles_available < min_candles_required:
        status = DataStatus.DEGRADED
        issues.append(
            f"only {candles_available} valid candles "
            f"(need {min_candles_required} for full analysis)"
        )
    elif issues:
        status = DataStatus.DEGRADED

    quality = DataQuality(
        status=status,
        symbol=symbol,
        candles_available=candles_available,
        candles_required=min_candles_required,
        freshness_seconds=freshness_seconds,
        issues=issues,
        warnings=warnings,
    )
    return clean_candles, quality
