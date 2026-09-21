"""Regression tests for the confirmed live PSYGRID RealMarketAPI M5 schema
(http://140.245.226.102:8080/public/m5-live.json), using the exact
payload shape reported live: top-level `symbols` dict, each entry
carrying its own `status`/`m5_valid`/`market_state` plus a `candles_5m`
list with `timestamp`/`open`/`high`/`low`/`close`/`volume`/`bid`/`ask`.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from psygnal.data.parser import parse_live_payload
from psygnal.data.validation import validate_symbol
from psygnal.main import run_engine
from psygnal import config


def _candle(ts: str, o: float, h: float, l: float, c: float, v: float) -> dict:
    return {"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": v, "bid": None, "ask": None}


def _oracle_payload(n_candles: int = 13, extra_symbols: dict | None = None) -> dict:
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    candles = []
    price = 4350.0
    for i in range(n_candles):
        o = price
        c = price + 0.5
        h, l = max(o, c) + 0.1, min(o, c) - 0.1
        candles.append(_candle((start + timedelta(minutes=5 * i)).isoformat(), o, h, l, c, 756.0))
        price = c

    symbols = {
        "XAUUSD": {
            "symbol": "XAUUSD",
            "market_state": "open",
            "status": "ok",
            "last_candle_timestamp": candles[-1]["timestamp"],
            "updated_at": candles[-1]["timestamp"],
            "websocket_connected": True,
            "reconnect_count": 0,
            "gap_recoveries": 0,
            "rejected_count": 0,
            "candles_5m": candles,
            "m5_valid": True,
        }
    }
    if extra_symbols:
        symbols.update(extra_symbols)

    return {
        "schema_version": "1.0",
        "service": "pysgrid-forex",
        "provider": "realmarketapi",
        "timeframe": "M5",
        "generated_at": candles[-1]["timestamp"],
        "status": "ok",
        "universe_size": 1 + len(extra_symbols or {}),
        "symbols": symbols,
    }


def test_oracle_adapter_is_selected_and_parses_candles_5m():
    payload = _oracle_payload(n_candles=13)
    outcome = parse_live_payload(payload)
    assert outcome.ok
    assert outcome.adapter_used == "oracle_realmarketapi_v1"
    assert "XAUUSD" in outcome.symbols
    records = outcome.symbols["XAUUSD"].records
    assert len(records) == 13
    assert records[0].open == 4350.0
    assert records[0].volume == 756.0


def test_oracle_adapter_preserves_chronological_ordering_even_if_source_is_reversed():
    payload = _oracle_payload(n_candles=6)
    payload["symbols"]["XAUUSD"]["candles_5m"].reverse()
    outcome = parse_live_payload(payload)
    records = outcome.symbols["XAUUSD"].records
    times = [r.time for r in records]
    assert times == sorted(times)


def test_oracle_adapter_rejects_malformed_candle_without_fabricating():
    payload = _oracle_payload(n_candles=5)
    payload["symbols"]["XAUUSD"]["candles_5m"].append(
        {"timestamp": "not-a-real-timestamp", "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0}
    )
    payload["symbols"]["XAUUSD"]["candles_5m"].append(
        {"timestamp": "2025-01-01T05:00:00+00:00", "open": "NaN-ish", "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0}
    )
    outcome = parse_live_payload(payload)
    parsed = outcome.symbols["XAUUSD"]
    assert len(parsed.records) == 5  # the two malformed appended candles are dropped
    assert parsed.malformed_count == 2


def test_oracle_adapter_skips_symbol_flagged_m5_invalid():
    extra = {
        "EURUSD": {
            "symbol": "EURUSD",
            "market_state": "open",
            "status": "ok",
            "m5_valid": False,
            "candles_5m": [_candle("2025-01-01T00:00:00Z", 1.1, 1.11, 1.09, 1.105, 100.0)],
        }
    }
    payload = _oracle_payload(n_candles=5, extra_symbols=extra)
    outcome = parse_live_payload(payload)
    assert outcome.ok
    assert "XAUUSD" in outcome.symbols
    assert "EURUSD" not in outcome.symbols


def test_oracle_adapter_skips_symbol_with_non_ok_status():
    extra = {
        "GBPUSD": {
            "symbol": "GBPUSD",
            "market_state": "closed",
            "status": "degraded",
            "m5_valid": True,
            "candles_5m": [_candle("2025-01-01T00:00:00Z", 1.27, 1.271, 1.269, 1.2705, 10.0)],
        }
    }
    payload = _oracle_payload(n_candles=5, extra_symbols=extra)
    outcome = parse_live_payload(payload)
    assert outcome.ok
    assert "XAUUSD" in outcome.symbols
    assert "GBPUSD" not in outcome.symbols


def test_oracle_adapter_recognized_but_all_symbols_excluded_is_not_schema_unrecognized():
    payload = _oracle_payload(n_candles=5)
    payload["symbols"]["XAUUSD"]["m5_valid"] = False
    outcome = parse_live_payload(payload)
    assert outcome.ok  # recognized shape, just zero usable symbols -> not SCHEMA_UNRECOGNIZED
    assert outcome.adapter_used == "oracle_realmarketapi_v1"
    assert outcome.symbols == {}


def test_oracle_schema_with_insufficient_candles_reports_degraded_not_fabricated():
    """The live endpoint currently exposes ~13 M5 candles per symbol —
    below MIN_M5_CANDLES_REQUIRED (120). Confirm the pipeline reports this
    honestly (DEGRADED, not enough for full analysis) rather than padding
    the series with invented candles."""
    payload = _oracle_payload(n_candles=13)
    outcome = parse_live_payload(payload)
    now = datetime.fromisoformat(payload["symbols"]["XAUUSD"]["candles_5m"][-1]["timestamp"]) + timedelta(minutes=1)
    candles, quality = validate_symbol("XAUUSD", outcome.symbols["XAUUSD"].records, now=now)
    assert len(candles) == 13
    assert quality.candles_available == 13
    assert quality.candles_required == config.MIN_M5_CANDLES_REQUIRED
    assert quality.status.value in ("DEGRADED", "UNAVAILABLE")


def test_run_engine_end_to_end_with_real_oracle_schema_reports_insufficient_data():
    """13 candles is below the 120-candle minimum. Per the engine's
    existing DEGRADED-continues design, a signal is still produced (never
    blocked outright — zero valid candles is the only case that goes to
    `unavailable`), but it must honestly flag the shortfall via
    `data_quality`, not silently proceed as if 120+ candles existed."""
    payload = _oracle_payload(n_candles=13)
    now = datetime.fromisoformat(payload["symbols"]["XAUUSD"]["candles_5m"][-1]["timestamp"]) + timedelta(minutes=1)
    cfg = config.EngineConfig(symbols=("XAUUSD",), enable_macro=False, enable_news=False)
    outcome = run_engine(payload, cfg=cfg, now_utc=now)
    assert outcome["status"] == "OK"
    assert "XAUUSD" not in outcome["unavailable"]
    signal, _ = outcome["results"][0]
    assert signal.data_quality["status"] == "DEGRADED"
    assert signal.data_quality["candles_available"] == 13
    assert signal.data_quality["candles_required"] == config.MIN_M5_CANDLES_REQUIRED
    assert any("13" in issue and "120" in issue for issue in signal.data_quality["issues"])


def test_run_engine_reports_missing_requested_symbol_as_unavailable_not_silently_dropped():
    """A symbol requested via cfg.symbols that the provider excluded
    (non-ok status, m5_valid false, or simply absent) must appear in
    `unavailable`, never vanish from the report entirely."""
    extra = {
        "EURUSD": {
            "symbol": "EURUSD",
            "status": "ok",
            "m5_valid": False,
            "candles_5m": [_candle("2025-01-01T00:00:00Z", 1.1, 1.11, 1.09, 1.105, 100.0)],
        }
    }
    payload = _oracle_payload(n_candles=200, extra_symbols=extra)
    now = datetime.fromisoformat(payload["symbols"]["XAUUSD"]["candles_5m"][-1]["timestamp"]) + timedelta(minutes=1)
    cfg = config.EngineConfig(symbols=("XAUUSD", "EURUSD", "GBPUSD"), enable_macro=False, enable_news=False)
    outcome = run_engine(payload, cfg=cfg, now_utc=now)
    assert outcome["status"] == "OK"
    result_symbols = [s.symbol for s, _ in outcome["results"]]
    assert "XAUUSD" in result_symbols
    assert "EURUSD" in outcome["unavailable"]
    assert "GBPUSD" in outcome["unavailable"]  # never present in the payload at all
