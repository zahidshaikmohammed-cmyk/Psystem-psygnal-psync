"""Free GDELT DOC 2.0 API client. No API key required."""

from __future__ import annotations

from typing import Any

from psygnal import config
from psygnal.data.http_cache import CachedFetchResult, fetch_json_with_cache

DEFAULT_QUERY = (
    '("US dollar" OR "Federal Reserve" OR inflation OR "interest rates" OR '
    '"Treasury yields" OR gold OR "central bank" OR employment OR oil)'
)


def fetch_market_news(
    query: str = DEFAULT_QUERY, timespan: str = "6h", max_records: int = 75
) -> CachedFetchResult:
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "timespan": timespan,
        "maxrecords": str(max_records),
        "sort": "datedesc",
    }
    return fetch_json_with_cache(
        config.GDELT_DOC_API_URL,
        cache_key="gdelt_market_news",
        params=params,
    )


def extract_articles(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, dict):
        return []
    articles = raw.get("articles")
    if not isinstance(articles, list):
        return []
    return [a for a in articles if isinstance(a, dict)]
