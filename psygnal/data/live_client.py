"""HTTP client for the primary live M5 endpoint.

Only ever talks to `config.LIVE_M5_ENDPOINT` (or an override explicitly
passed in). No other market-data endpoint is invented or guessed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import requests

from psygnal import config


@dataclass
class FetchResult:
    ok: bool
    status_code: Optional[int]
    raw: Optional[Any]
    error: Optional[str]
    latency_seconds: float
    attempts: int
    url: str


def fetch_live_m5(
    url: str = config.LIVE_M5_ENDPOINT,
    timeout_seconds: int = config.LIVE_FETCH_TIMEOUT_SECONDS,
    retries: int = config.LIVE_FETCH_RETRIES,
    backoff_seconds: float = config.LIVE_FETCH_BACKOFF_SECONDS,
) -> FetchResult:
    """Fetch the live M5 JSON payload with bounded retries.

    Never fabricates a response: on total failure `ok=False` and `raw=None`.
    """
    last_error: Optional[str] = None
    last_status: Optional[int] = None
    start = time.monotonic()
    attempts = 0

    for attempt in range(1, retries + 1):
        attempts = attempt
        try:
            response = requests.get(url, timeout=timeout_seconds)
            last_status = response.status_code
            if response.status_code != 200:
                last_error = f"HTTP {response.status_code}"
            else:
                try:
                    payload = response.json()
                except ValueError as exc:
                    last_error = f"invalid JSON: {exc}"
                else:
                    return FetchResult(
                        ok=True,
                        status_code=response.status_code,
                        raw=payload,
                        error=None,
                        latency_seconds=time.monotonic() - start,
                        attempts=attempts,
                        url=url,
                    )
        except requests.exceptions.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        if attempt < retries:
            time.sleep(backoff_seconds * attempt)

    return FetchResult(
        ok=False,
        status_code=last_status,
        raw=None,
        error=last_error or "unknown fetch failure",
        latency_seconds=time.monotonic() - start,
        attempts=attempts,
        url=url,
    )
