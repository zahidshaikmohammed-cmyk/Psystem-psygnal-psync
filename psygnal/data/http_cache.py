"""Small shared helper for free/public secondary data sources (macro, news).

Every secondary source goes through here so the "timeout + retry limit +
graceful failure + caching, don't hammer public endpoints" requirement is
enforced in exactly one place rather than re-implemented per adapter.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import requests

from psygnal import config


@dataclass
class CachedFetchResult:
    ok: bool
    data: Optional[Any]
    from_cache: bool
    error: Optional[str]


def _cache_path(cache_key: str) -> Path:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_key = "".join(c if c.isalnum() else "_" for c in cache_key)
    return config.CACHE_DIR / f"{safe_key}.json"


def _read_cache(cache_key: str, ttl_seconds: int) -> Optional[Any]:
    path = _cache_path(cache_key)
    if not path.exists():
        return None
    age = time.time() - path.stat().st_mtime
    if age > ttl_seconds:
        return None
    try:
        with path.open("r") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_cache(cache_key: str, data: Any) -> None:
    path = _cache_path(cache_key)
    try:
        with path.open("w") as f:
            json.dump(data, f)
    except OSError:
        pass


def fetch_json_with_cache(
    url: str,
    cache_key: str,
    *,
    params: Optional[dict[str, Any]] = None,
    ttl_seconds: int = config.MACRO_NEWS_CACHE_TTL_SECONDS,
    timeout_seconds: int = config.MACRO_NEWS_FETCH_TIMEOUT_SECONDS,
    retries: int = config.MACRO_NEWS_FETCH_RETRIES,
) -> CachedFetchResult:
    cached = _read_cache(cache_key, ttl_seconds)
    if cached is not None:
        return CachedFetchResult(ok=True, data=cached, from_cache=True, error=None)

    last_error: Optional[str] = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout_seconds)
            if response.status_code != 200:
                last_error = f"HTTP {response.status_code}"
            else:
                try:
                    data = response.json()
                except ValueError as exc:
                    last_error = f"invalid JSON: {exc}"
                else:
                    _write_cache(cache_key, data)
                    return CachedFetchResult(ok=True, data=data, from_cache=False, error=None)
        except requests.exceptions.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            time.sleep(1.0 * attempt)

    # Graceful degradation: serve stale cache rather than nothing, if any exists.
    stale = _read_cache(cache_key, ttl_seconds=10**9)
    if stale is not None:
        return CachedFetchResult(ok=True, data=stale, from_cache=True, error=f"live fetch failed ({last_error}), served stale cache")

    return CachedFetchResult(ok=False, data=None, from_cache=False, error=last_error or "unknown fetch failure")
