"""Mathematically-correct M5 -> M15/M30/H1/H4 aggregation.

Only *completed* higher-timeframe candles are exposed for structural
analysis. The currently-forming candle of every timeframe (including M5
itself, if the freshest bar's 5-minute bucket has not yet closed) is kept
separate so it can never leak future information into historical analysis.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from psygnal import config
from psygnal.models import Candle, MultiTimeframeSeries, candles_to_frame

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def split_forming_m5(
    candles: list[Candle], now: datetime | None = None
) -> tuple[list[Candle], Candle | None]:
    """Separate the trailing M5 candle if its 5-minute bucket has not closed."""
    if not candles:
        return [], None
    now = now or datetime.now(timezone.utc)
    last = candles[-1]
    bucket_end = last.time + timedelta(minutes=5)
    if now < bucket_end:
        return candles[:-1], last
    return candles, None


def _bucket_start(t: datetime, minutes: int) -> datetime:
    total_minutes = int((t - _EPOCH).total_seconds() // 60)
    bucket_index = total_minutes // minutes
    return _EPOCH + timedelta(minutes=bucket_index * minutes)


def aggregate_timeframe(
    m5_candles: list[Candle], multiple: int
) -> tuple[list[Candle], Candle | None]:
    """Aggregate completed M5 candles into a higher timeframe.

    Returns (completed_candles, forming_candle_or_None). The final bucket is
    treated as "forming" (excluded from completed history) whenever it does
    not contain the full `multiple` count of M5 candles, since that can mean
    either "still in progress" or "not yet fully reported" — in both cases
    it must not be used as a completed structural candle.
    """
    if multiple <= 1:
        return list(m5_candles), None
    if not m5_candles:
        return [], None

    minutes = 5 * multiple
    buckets: dict[datetime, list[Candle]] = {}
    order: list[datetime] = []
    for c in m5_candles:
        bstart = _bucket_start(c.time, minutes)
        if bstart not in buckets:
            buckets[bstart] = []
            order.append(bstart)
        buckets[bstart].append(c)

    completed: list[Candle] = []
    forming: Candle | None = None
    last_bucket_key = order[-1]

    for key in order:
        members = sorted(buckets[key], key=lambda c: c.time)
        agg = Candle(
            time=key,
            open=members[0].open,
            high=max(m.high for m in members),
            low=min(m.low for m in members),
            close=members[-1].close,
            volume=sum(m.volume for m in members),
            completed=True,
        )
        is_full = len(members) == multiple
        if key == last_bucket_key and not is_full:
            forming = Candle(
                time=agg.time,
                open=agg.open,
                high=agg.high,
                low=agg.low,
                close=agg.close,
                volume=agg.volume,
                completed=False,
            )
        else:
            completed.append(agg)

    return completed, forming


def build_multi_timeframe_series(
    symbol: str,
    m5_candles: list[Candle],
    now: datetime | None = None,
) -> MultiTimeframeSeries:
    now = now or datetime.now(timezone.utc)
    completed_m5, forming_m5 = split_forming_m5(m5_candles, now)

    frames: dict[str, object] = {}
    forming: dict[str, Candle | None] = {}

    frames["M5"] = candles_to_frame(completed_m5)
    forming["M5"] = forming_m5

    for tf, multiple in config.TIMEFRAME_M5_MULTIPLES.items():
        if tf == "M5":
            continue
        comp, form = aggregate_timeframe(completed_m5, multiple)
        frames[tf] = candles_to_frame(comp)
        forming[tf] = form

    return MultiTimeframeSeries(symbol=symbol, frames=frames, forming=forming)
