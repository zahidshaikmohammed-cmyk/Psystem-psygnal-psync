from __future__ import annotations

from datetime import timedelta

import pytest

from psygnal import config
from psygnal.main import run_engine
from psygnal.models import Candle
from psygnal.reporting.json import build_json_output
from psygnal.reporting.terminal import format_market_summary, format_symbol_report
from tests.conftest import make_m5_series


def _raw_payload_for(symbols_candles: dict[str, list[Candle]]) -> dict:
    return {
        "status": "ok",
        "symbols": {
            sym: {
                "candles": [
                    {
                        "time": c.time.isoformat(),
                        "open": c.open,
                        "high": c.high,
                        "low": c.low,
                        "close": c.close,
                        "volume": c.volume,
                    }
                    for c in candles
                ]
            }
            for sym, candles in symbols_candles.items()
        },
    }


def _random_walk_candles(n=500, seed=1, base_price=1.10):
    import numpy as np
    from datetime import datetime, timezone

    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = []
    price = base_price
    for i in range(n):
        step = rng.normal(0, base_price * 0.001)
        o, c = price, price + step
        h, l = max(o, c) + abs(rng.normal(0, base_price * 0.0002)), min(o, c) - abs(rng.normal(0, base_price * 0.0002))
        candles.append(Candle(time=start + timedelta(minutes=5 * i), open=o, high=h, low=l, close=c, volume=100 + abs(rng.normal(0, 20))))
        price = c
    return candles


def test_run_engine_end_to_end_produces_signals():
    eurusd = _random_walk_candles(500, seed=1, base_price=1.10)
    gbpusd = _random_walk_candles(500, seed=2, base_price=1.27)
    usdjpy = _random_walk_candles(500, seed=3, base_price=150.0)
    payload = _raw_payload_for({"EURUSD": eurusd, "GBPUSD": gbpusd, "USDJPY": usdjpy})

    now = eurusd[-1].time + timedelta(minutes=6)
    cfg = config.EngineConfig(symbols=("EURUSD", "GBPUSD", "USDJPY"), enable_macro=False, enable_news=False)
    outcome = run_engine(payload, cfg=cfg, now_utc=now)

    assert outcome["status"] == "OK"
    assert len(outcome["results"]) == 3
    for signal, intel in outcome["results"]:
        assert signal.direction in ("LONG", "SHORT")
        assert signal.entry is not None
        assert signal.stop_loss is not None
        assert signal.tp1 is not None
        assert 0 <= signal.signal_score <= 100
        assert 0 <= signal.confidence_score <= 100
        assert signal.reasons


def test_run_engine_reports_unavailable_symbol_separately():
    good = _random_walk_candles(500, seed=1, base_price=1.10)
    bad = _random_walk_candles(5, seed=2, base_price=1.27)
    for c in bad:  # force every candle to fail OHLC validation -> zero valid candles
        object.__setattr__(c, "close", -1.0)
    payload = _raw_payload_for({"EURUSD": good, "GBPUSD": bad})

    now = good[-1].time + timedelta(minutes=6)
    cfg = config.EngineConfig(symbols=("EURUSD", "GBPUSD"), enable_macro=False, enable_news=False)
    outcome = run_engine(payload, cfg=cfg, now_utc=now)

    assert outcome["status"] == "OK"
    result_symbols = [s.symbol for s, _ in outcome["results"]]
    assert "EURUSD" in result_symbols
    assert "GBPUSD" in outcome["unavailable"]


def test_run_engine_schema_unrecognized_payload():
    cfg = config.EngineConfig(enable_macro=False, enable_news=False)
    outcome = run_engine({"totally": "unrecognized"}, cfg=cfg)
    assert outcome["status"] == "SCHEMA_UNRECOGNIZED"
    assert outcome["schema_dump"] is not None


def test_terminal_and_json_rendering_do_not_crash():
    eurusd = _random_walk_candles(500, seed=1, base_price=1.10)
    payload = _raw_payload_for({"EURUSD": eurusd})
    now = eurusd[-1].time + timedelta(minutes=6)
    cfg = config.EngineConfig(symbols=("EURUSD",), enable_macro=False, enable_news=False)
    outcome = run_engine(payload, cfg=cfg, now_utc=now)

    signal, intel = outcome["results"][0]
    report_text = format_symbol_report(signal, intel)
    assert signal.symbol in report_text
    assert signal.direction in report_text

    summary_text = format_market_summary([signal])
    assert signal.symbol in summary_text

    json_output = build_json_output([signal], outcome["unavailable"], now)
    assert json_output["signals"][0]["symbol"] == "EURUSD"
