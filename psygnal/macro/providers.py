"""Provider-agnostic adapter interfaces for external data.

Each provider has exactly one job and one honest failure mode: return a
`ProviderResult(status="UNAVAILABLE")`, never fabricate. Swapping a real
implementation in later means writing a new class that satisfies the same
interface — the intelligence engine (`intelligence/`, `forecasting/`)
never needs to change.

Concrete implementations wired up in this build:
    - `ForexFactoryCalendarProvider` (CalendarProvider) -- see macro/sources.py
    - `GdeltNewsProvider` (NewsProvider) -- see news/gdelt.py

Interfaces defined but NOT wired to any live source in this build (both
degrade to UNAVAILABLE — see the class docstrings for what a real
implementation would need):
    - `MacroDataProvider` (verified statistical-agency release series)
    - `RatesProvider` (sovereign yield levels, for the yield-reaction
      contradiction check in intelligence/contradiction.py)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

from psygnal.macro.sources import fetch_forex_factory_calendar
from psygnal.news.gdelt import fetch_market_news


@dataclass
class ProviderResult:
    status: str  # "OK" | "UNAVAILABLE"
    data: Any = None
    error: Optional[str] = None


@runtime_checkable
class CalendarProvider(Protocol):
    """Supplies economic-calendar events: title/time/impact/forecast/
    previous/actual, in whatever raw shape `macro/calendar.py::
    parse_calendar_events` expects."""

    def fetch_events(self) -> ProviderResult: ...


@runtime_checkable
class NewsProvider(Protocol):
    def fetch_articles(self) -> ProviderResult: ...


@runtime_checkable
class MacroDataProvider(Protocol):
    """Supplies VERIFIED macro time series directly from a statistical
    agency (e.g. BLS/BEA/Federal Reserve release history) — actual and
    revised historical values, NOT forward-looking consensus forecasts
    (those are CalendarProvider territory, since consensus estimates are
    aggregator/survey data, not primary-source data)."""

    def fetch_series(self, series_id: str) -> ProviderResult: ...


@runtime_checkable
class RatesProvider(Protocol):
    """Supplies sovereign-yield levels (e.g. US 10Y/2Y) for the
    yield-reaction cross-market check."""

    def fetch_yield(self, tenor: str) -> ProviderResult: ...


class ForexFactoryCalendarProvider:
    """Concrete CalendarProvider backed by the free ForexFactory weekly
    calendar mirror already used by macro/calendar.py."""

    def fetch_events(self) -> ProviderResult:
        result = fetch_forex_factory_calendar()
        if not result.ok:
            return ProviderResult(status="UNAVAILABLE", error=result.error)
        return ProviderResult(status="OK", data=result.data)


class GdeltNewsProvider:
    """Concrete NewsProvider backed by the free GDELT DOC 2.0 API already
    used by news/gdelt.py."""

    def fetch_articles(self) -> ProviderResult:
        result = fetch_market_news()
        if not result.ok:
            return ProviderResult(status="UNAVAILABLE", error=result.error)
        return ProviderResult(status="OK", data=result.data)


class UnavailableMacroDataProvider:
    """Default MacroDataProvider: no free, network-reachable,
    no-API-key statistical-agency release feed was wired up in this
    build (BLS/BEA/FRED all require either an API key or were
    unreachable from the build/test environment — see README). Plug in
    a real implementation (e.g. an authenticated FRED client) by
    swapping this class out; nothing else needs to change."""

    def fetch_series(self, series_id: str) -> ProviderResult:
        return ProviderResult(status="UNAVAILABLE", error="no MacroDataProvider implementation configured")


class UnavailableRatesProvider:
    """Default RatesProvider: no yield-data source is wired up. Every
    cross-market check that would use yields degrades to an honest
    'yield reaction: UNAVAILABLE' rather than crashing or fabricating a
    level (see intelligence/contradiction.py)."""

    def fetch_yield(self, tenor: str) -> ProviderResult:
        return ProviderResult(status="UNAVAILABLE", error="no RatesProvider implementation configured")
