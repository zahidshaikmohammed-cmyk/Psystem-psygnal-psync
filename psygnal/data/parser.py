"""Adaptive parser for the live M5 endpoint payload.

SCHEMA STATUS
-------------
`config.LIVE_M5_ENDPOINT` (http://140.245.226.102:8080/public/m5-live.json)
was NOT reachable from the sandboxed environment this engine has been
built and iterated in (plain HTTP, non-443 port, raw IP — blocked by the
build sandbox's egress policy, confirmed directly on more than one
occasion, not assumed). The actual schema below was supplied by the user
after inspecting the live endpoint directly and is handled by a dedicated
adapter, `_adapter_oracle_realmarketapi` — but it still could not be
independently verified from this sandbox, so the multi-adapter,
never-fabricate fallback strategy below remains in place for any other
shape (or a future breaking change to this one):

    {"schema_version": "1.0", "service": "pysgrid-forex",
     "provider": "realmarketapi", "timeframe": "M5", "status": "ok",
     "symbols": {
         "XAUUSD": {
             "symbol": "XAUUSD", "market_state": "open", "status": "ok",
             "m5_valid": true,
             "candles_5m": [{"timestamp": ..., "open": ..., "high": ...,
                              "low": ..., "close": ..., "volume": ...,
                              "bid": null, "ask": null}, ...]
         }, ...
     }}

For anything else:

1. Tries a small set of documented, structurally-reasonable adapters
   against whatever JSON the endpoint actually returns.
2. Refuses to guess when nothing matches — it reports a clear parsing
   failure with a schema dump, never invented candles.
3. Ships `describe_schema()` / `python -m psygnal --inspect-schema` so the
   first person who *can* reach the endpoint (i.e. the end user, from a
   network that isn't sandboxed) gets a precise structural report they
   can use to add a new adapter in minutes if none of the shipped ones
   fit.

This is the single most important calibration point in the whole engine.
If live runs report "SCHEMA_UNRECOGNIZED", inspect the dumped structure
and extend `_ADAPTERS` below — do not patch around it by fabricating data
elsewhere in the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

_TIME_KEYS = ("time", "timestamp", "ts", "t", "date", "datetime", "candle_time")
_OPEN_KEYS = ("open", "o", "Open", "open_price")
_HIGH_KEYS = ("high", "h", "High", "high_price")
_LOW_KEYS = ("low", "l", "Low", "low_price")
_CLOSE_KEYS = ("close", "c", "Close", "close_price")
_VOLUME_KEYS = ("volume", "vol", "v", "tick_volume", "tickVolume", "Volume")

_SYMBOL_CONTAINER_KEYS = (
    "symbols",
    "data",
    "instruments",
    "pairs",
    "quotes",
    "markets",
    "universe",
)
_CANDLE_LIST_KEYS = ("candles", "bars", "ohlc", "m5", "M5", "rates", "history", "data")

# The PSYGRID RealMarketAPI feed's own M5 candle-list key, confirmed live at
# http://140.245.226.102:8080/public/m5-live.json (provider="realmarketapi",
# service="pysgrid-forex"). Kept separate from _CANDLE_LIST_KEYS so the
# dedicated adapter below (_adapter_oracle_realmarketapi) is the one that
# claims this exact shape and can apply its status/m5_valid gating —
# rather than the generic symbol_keyed_container adapter picking it up
# without that gating.
_ORACLE_CANDLE_LIST_KEY = "candles_5m"


@dataclass
class RawCandleRecord:
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class ParsedSymbolData:
    symbol: str
    records: list[RawCandleRecord]
    malformed_count: int = 0
    malformed_examples: list[str] = field(default_factory=list)


@dataclass
class ParseOutcome:
    ok: bool
    adapter_used: Optional[str]
    symbols: dict[str, ParsedSymbolData]
    server_status: Optional[str]
    server_timestamp: Optional[datetime]
    error: Optional[str] = None
    schema_dump: Optional[str] = None


def _coerce_time(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        # Heuristic: values above ~10^12 are milliseconds since epoch.
        seconds = value / 1000.0 if value > 10**12 else float(value)
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            return _coerce_time(int(text))
        try:
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    return None


def coerce_time(value: Any) -> Optional[datetime]:
    """Public wrapper around the timestamp-coercion heuristic, reused by the
    macro/news adapters so timestamp parsing logic lives in exactly one
    place."""
    return _coerce_time(value)


def _first_present(d: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_candle_dict(rec: dict[str, Any]) -> Optional[RawCandleRecord]:
    t = _coerce_time(_first_present(rec, _TIME_KEYS))
    o = _coerce_float(_first_present(rec, _OPEN_KEYS))
    h = _coerce_float(_first_present(rec, _HIGH_KEYS))
    l = _coerce_float(_first_present(rec, _LOW_KEYS))
    c = _coerce_float(_first_present(rec, _CLOSE_KEYS))
    v = _coerce_float(_first_present(rec, _VOLUME_KEYS))
    if t is None or None in (o, h, l, c):
        return None
    return RawCandleRecord(time=t, open=o, high=h, low=l, close=c, volume=v or 0.0)


def _parse_candle_sequence(rec: list[Any]) -> Optional[RawCandleRecord]:
    # Common array form: [time, open, high, low, close, volume]
    if len(rec) < 5:
        return None
    t = _coerce_time(rec[0])
    o, h, l, c = (_coerce_float(x) for x in rec[1:5])
    v = _coerce_float(rec[5]) if len(rec) > 5 else 0.0
    if t is None or None in (o, h, l, c):
        return None
    return RawCandleRecord(time=t, open=o, high=h, low=l, close=c, volume=v or 0.0)


def _extract_candle_list(value: Any) -> Optional[list[Any]]:
    """Given a symbol's payload, find the actual list of candle records."""
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        found = _first_present(value, _CANDLE_LIST_KEYS)
        if isinstance(found, list):
            return found
    return None


def _looks_like_symbol_name(key: str) -> bool:
    if not isinstance(key, str):
        return False
    k = key.upper()
    if len(k) < 3 or len(k) > 10:
        return False
    return k.replace("_", "").replace("/", "").isalnum()


def _build_parsed_symbol(symbol: str, raw_list: list[Any]) -> Optional[ParsedSymbolData]:
    records: list[RawCandleRecord] = []
    malformed = 0
    examples: list[str] = []
    for item in raw_list:
        parsed = None
        if isinstance(item, dict):
            parsed = _parse_candle_dict(item)
        elif isinstance(item, (list, tuple)):
            parsed = _parse_candle_sequence(list(item))
        if parsed is None:
            malformed += 1
            if len(examples) < 3:
                examples.append(repr(item)[:200])
            continue
        records.append(parsed)
    if not records:
        return None
    return ParsedSymbolData(
        symbol=symbol, records=records, malformed_count=malformed, malformed_examples=examples
    )


# ---------------------------------------------------------------------------
# Adapters — each takes the raw decoded JSON and returns a dict of
# symbol -> ParsedSymbolData, or None if the shape doesn't match at all.
# ---------------------------------------------------------------------------


def _adapter_oracle_realmarketapi(raw: Any) -> Optional[dict[str, ParsedSymbolData]]:
    """Dedicated adapter for the confirmed live PSYGRID RealMarketAPI M5
    feed schema:

        {"schema_version": "1.0", "service": "pysgrid-forex",
         "provider": "realmarketapi", "timeframe": "M5", "status": "ok",
         "symbols": {
             "XAUUSD": {
                 "symbol": "XAUUSD", "market_state": "open", "status": "ok",
                 "m5_valid": true,
                 "candles_5m": [{"timestamp": ..., "open": ..., "high": ...,
                                  "low": ..., "close": ..., "volume": ...,
                                  "bid": null, "ask": null}, ...],
             }, ...
         }}

    A per-symbol `status` other than "ok", or `m5_valid: false`, means the
    provider itself is telling us that symbol's M5 history isn't
    trustworthy right now — that symbol is skipped here (never parsed
    anyway), not silently accepted. `bid`/`ask` aren't part of the
    canonical OHLCV candle model and are ignored.

    Returns `{}` (not `None`) when the shape is recognized but every
    symbol ended up excluded (absent status/m5_valid, or truly empty
    candle lists) — `parse_live_payload` treats that as a successful,
    if empty, parse rather than falling through to SCHEMA_UNRECOGNIZED
    for a payload it actually understood.
    """
    if not isinstance(raw, dict):
        return None
    symbols = raw.get("symbols")
    if not isinstance(symbols, dict):
        return None

    recognized = False
    out: dict[str, ParsedSymbolData] = {}
    for container_key, payload in symbols.items():
        if not isinstance(payload, dict) or _ORACLE_CANDLE_LIST_KEY not in payload:
            continue
        recognized = True

        symbol_status = payload.get("status")
        if symbol_status is not None and symbol_status != "ok":
            continue
        if payload.get("m5_valid") is False:
            continue

        candle_list = payload.get(_ORACLE_CANDLE_LIST_KEY)
        if not isinstance(candle_list, list):
            continue

        symbol_name = payload.get("symbol") or container_key
        parsed = _build_parsed_symbol(symbol_name, candle_list)
        if parsed is None:
            continue
        # Preserve chronological ordering explicitly at the source, even
        # though downstream validation also sorts/dedupes defensively.
        parsed.records.sort(key=lambda r: r.time)
        out[str(symbol_name).upper()] = parsed

    if not recognized:
        return None
    return out


def _adapter_symbol_keyed_container(raw: Any) -> Optional[dict[str, ParsedSymbolData]]:
    """{"symbols": {"EURUSD": {"candles": [...]}, ...}} or similar container key."""
    if not isinstance(raw, dict):
        return None
    container = _first_present(raw, _SYMBOL_CONTAINER_KEYS)
    if not isinstance(container, dict):
        return None
    out: dict[str, ParsedSymbolData] = {}
    for sym, payload in container.items():
        candle_list = _extract_candle_list(payload)
        if candle_list is None:
            continue
        parsed = _build_parsed_symbol(sym, candle_list)
        if parsed:
            out[sym.upper()] = parsed
    return out or None


def _adapter_symbol_keyed_list_container(raw: Any) -> Optional[dict[str, ParsedSymbolData]]:
    """{"data": [{"symbol": "EURUSD", "candles": [...]}, ...]}"""
    if not isinstance(raw, dict):
        return None
    container = _first_present(raw, _SYMBOL_CONTAINER_KEYS)
    if not isinstance(container, list):
        return None
    out: dict[str, ParsedSymbolData] = {}
    for entry in container:
        if not isinstance(entry, dict):
            continue
        sym = entry.get("symbol") or entry.get("Symbol") or entry.get("instrument") or entry.get("pair")
        if not sym:
            continue
        candle_list = _extract_candle_list(entry)
        if candle_list is None:
            continue
        parsed = _build_parsed_symbol(sym, candle_list)
        if parsed:
            out[sym.upper()] = parsed
    return out or None


def _adapter_top_level_symbol_map(raw: Any) -> Optional[dict[str, ParsedSymbolData]]:
    """{"EURUSD": [...candles...], "XAUUSD": {"candles": [...]}, ...} directly
    at the top level, alongside optional metadata keys."""
    if not isinstance(raw, dict):
        return None
    out: dict[str, ParsedSymbolData] = {}
    for key, value in raw.items():
        if not _looks_like_symbol_name(key):
            continue
        candle_list = _extract_candle_list(value)
        if candle_list is None:
            continue
        parsed = _build_parsed_symbol(key, candle_list)
        if parsed:
            out[key.upper()] = parsed
    return out or None


def _adapter_top_level_list_of_symbols(raw: Any) -> Optional[dict[str, ParsedSymbolData]]:
    """[{"symbol": "EURUSD", "candles": [...]}, ...] as the bare top-level array."""
    if not isinstance(raw, list):
        return None
    out: dict[str, ParsedSymbolData] = {}
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        sym = entry.get("symbol") or entry.get("Symbol") or entry.get("instrument") or entry.get("pair")
        if not sym:
            continue
        candle_list = _extract_candle_list(entry)
        if candle_list is None:
            continue
        parsed = _build_parsed_symbol(sym, candle_list)
        if parsed:
            out[sym.upper()] = parsed
    return out or None


_ADAPTERS: list[tuple[str, Callable[[Any], Optional[dict[str, ParsedSymbolData]]]]] = [
    ("oracle_realmarketapi_v1", _adapter_oracle_realmarketapi),
    ("symbol_keyed_container", _adapter_symbol_keyed_container),
    ("symbol_keyed_list_container", _adapter_symbol_keyed_list_container),
    ("top_level_symbol_map", _adapter_top_level_symbol_map),
    ("top_level_list_of_symbols", _adapter_top_level_list_of_symbols),
]


def _extract_server_status(raw: Any) -> Optional[str]:
    if isinstance(raw, dict):
        for k in ("status", "server_status", "connection_status", "state"):
            if k in raw:
                return str(raw[k])
    return None


def _extract_server_timestamp(raw: Any) -> Optional[datetime]:
    if isinstance(raw, dict):
        for k in ("timestamp", "server_time", "generated_at", "updated_at", "time"):
            if k in raw:
                dt = _coerce_time(raw[k])
                if dt:
                    return dt
    return None


def describe_schema(raw: Any, max_depth: int = 3, max_items: int = 5) -> str:
    """Produce a compact, human-readable structural dump of arbitrary JSON,
    used both in error reports and via `--inspect-schema`."""

    def _describe(value: Any, depth: int) -> Any:
        if depth > max_depth:
            return f"<{type(value).__name__}>"
        if isinstance(value, dict):
            keys = list(value.keys())[:max_items]
            return {
                "__type__": "object",
                "__keys__": list(value.keys())[:20],
                **{k: _describe(value[k], depth + 1) for k in keys},
            }
        if isinstance(value, list):
            return {
                "__type__": "array",
                "__length__": len(value),
                "__sample__": [_describe(v, depth + 1) for v in value[:max_items]],
            }
        if isinstance(value, str):
            return f"str: {value[:60]!r}"
        return f"{type(value).__name__}: {value!r}"

    import json as _json

    return _json.dumps(_describe(raw, 0), indent=2, default=str)


def parse_live_payload(raw: Any) -> ParseOutcome:
    if raw is None:
        return ParseOutcome(
            ok=False,
            adapter_used=None,
            symbols={},
            server_status=None,
            server_timestamp=None,
            error="empty payload",
        )

    for name, adapter in _ADAPTERS:
        try:
            result = adapter(raw)
        except Exception:  # defensive: a malformed shape must not crash parsing
            result = None
        # `is not None` (not a truthiness check) so an adapter can signal
        # "I recognized this shape but zero symbols survived" via `{}`
        # rather than being treated the same as "shape not recognized".
        # Existing adapters never return `{}` themselves (they use
        # `out or None`), so this is a no-op for them.
        if result is not None:
            return ParseOutcome(
                ok=True,
                adapter_used=name,
                symbols=result,
                server_status=_extract_server_status(raw),
                server_timestamp=_extract_server_timestamp(raw),
            )

    return ParseOutcome(
        ok=False,
        adapter_used=None,
        symbols={},
        server_status=_extract_server_status(raw),
        server_timestamp=_extract_server_timestamp(raw),
        error="SCHEMA_UNRECOGNIZED: no adapter matched the payload shape",
        schema_dump=describe_schema(raw),
    )
