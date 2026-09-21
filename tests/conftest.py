from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from psygnal.models import Candle


def make_m5_series(
    n: int,
    start: datetime | None = None,
    base_price: float = 100.0,
    step: float = 0.1,
    volume: float = 1000.0,
) -> list[Candle]:
    """Deterministic synthetic M5 series: a gentle uptrend with small wicks."""
    start = start or datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    candles: list[Candle] = []
    price = base_price
    for i in range(n):
        t = start + timedelta(minutes=5 * i)
        o = price
        c = price + step
        h = max(o, c) + step * 0.3
        l = min(o, c) - step * 0.3
        candles.append(Candle(time=t, open=o, high=h, low=l, close=c, volume=volume + i))
        price = c
    return candles


@pytest.fixture
def m5_candles() -> list[Candle]:
    return make_m5_series(200)
