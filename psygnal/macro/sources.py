"""Free, public macro-calendar source.

Uses the widely-mirrored public ForexFactory weekly calendar JSON feed.
No API key. Wrapped in `fetch_json_with_cache`, so failures degrade to
MACRO_STATUS = UNAVAILABLE rather than raising or fabricating events.
"""

from __future__ import annotations

from psygnal import config
from psygnal.data.http_cache import CachedFetchResult, fetch_json_with_cache


def fetch_forex_factory_calendar() -> CachedFetchResult:
    return fetch_json_with_cache(
        config.FOREX_FACTORY_WEEKLY_CALENDAR_URL,
        cache_key="ff_calendar_thisweek",
    )
