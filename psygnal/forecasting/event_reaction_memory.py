"""Historical event-reaction memory.

Given genuine local historical M5 OHLCV plus a local historical
economic-events file (both supplied by the user — this repository ships
neither), computes empirical reaction statistics: "under conditions
similar to today, this category of event, surprising in this direction,
has historically produced these kinds of reactions over the next 5/15/
30/60 minutes."

Reports `INSUFFICIENT_HISTORICAL_SAMPLE` — never a fabricated statistic —
whenever fewer than `MIN_SAMPLES_FOR_STATISTIC` historical instances of a
given (category, surprise-direction) pair exist.

LEAKAGE NOTE: for each historical event, this uses only price bars at or
after the event's own release time, and only the event's own already-
settled actual/forecast values — nothing from any other, later event, and
nothing from before the event beyond the price level at release time
itself. One known limitation: the free calendar source does not expose a
separate "revision timestamp" for previous-value revisions, so a
revision that happened before the release this reaction is computed from
could theoretically have been incorporated already — this can only be
fixed by a source that publishes point-in-time vintages, which is out of
scope here and is disclosed rather than silently assumed away.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from psygnal.macro.calendar import MacroEvent
from psygnal.macro.event_engine import enrich_event

MIN_SAMPLES_FOR_STATISTIC = 20
REACTION_HORIZONS_MINUTES: tuple[int, ...] = (5, 15, 30, 60)


@dataclass
class EventReactionStatistic:
    category: str
    surprise_direction: str  # "POSITIVE" | "NEGATIVE"
    n_samples: int
    mean_return_by_horizon: dict[int, Optional[float]]
    status: str  # "OK" | "INSUFFICIENT_HISTORICAL_SAMPLE"

    def as_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "surprise_direction": self.surprise_direction,
            "n_samples": self.n_samples,
            "mean_return_by_horizon": self.mean_return_by_horizon,
            "status": self.status,
        }


def _price_at_or_after(df_m5: pd.DataFrame, target_time: pd.Timestamp) -> Optional[float]:
    if df_m5.empty:
        return None
    pos = df_m5.index.searchsorted(target_time)
    if pos >= len(df_m5):
        return None
    return float(df_m5["close"].iloc[pos])


def compute_event_reaction_statistics(
    df_m5: pd.DataFrame,
    historical_events: list[MacroEvent],
    horizons_minutes: tuple[int, ...] = REACTION_HORIZONS_MINUTES,
    min_samples: int = MIN_SAMPLES_FOR_STATISTIC,
) -> list[EventReactionStatistic]:
    groups: dict[tuple[str, str], list[dict[int, float]]] = defaultdict(list)

    for event in historical_events:
        if event.time_utc is None or event.actual is None or not event.is_high_impact():
            continue
        record = enrich_event(event)
        if record.surprise_normalized is None:
            continue
        direction = "POSITIVE" if record.surprise_normalized > 0 else "NEGATIVE"

        price_at_event = _price_at_or_after(df_m5, event.time_utc)
        if price_at_event is None:
            continue

        returns: dict[int, float] = {}
        for horizon in horizons_minutes:
            target_time = event.time_utc + pd.Timedelta(minutes=horizon)
            price_after = _price_at_or_after(df_m5, target_time)
            if price_after is not None:
                returns[horizon] = (price_after - price_at_event) / price_at_event

        if returns:
            groups[(record.category, direction)].append(returns)

    results: list[EventReactionStatistic] = []
    for (category, direction), samples in groups.items():
        n = len(samples)
        mean_by_horizon: dict[int, Optional[float]] = {}
        for horizon in horizons_minutes:
            values = [s[horizon] for s in samples if horizon in s]
            mean_by_horizon[horizon] = float(np.mean(values)) if len(values) >= min_samples else None
        status = "OK" if n >= min_samples else "INSUFFICIENT_HISTORICAL_SAMPLE"
        results.append(
            EventReactionStatistic(
                category=category, surprise_direction=direction, n_samples=n, mean_return_by_horizon=mean_by_horizon, status=status
            )
        )

    return results


def query_reaction_statistic(
    statistics: list[EventReactionStatistic], category: str, surprise_direction: str
) -> EventReactionStatistic:
    for s in statistics:
        if s.category == category and s.surprise_direction == surprise_direction:
            return s
    return EventReactionStatistic(
        category=category, surprise_direction=surprise_direction, n_samples=0, mean_return_by_horizon={}, status="INSUFFICIENT_HISTORICAL_SAMPLE"
    )
