from __future__ import annotations

from psygnal.data.parser import describe_schema, parse_live_payload


def test_parses_symbol_keyed_container():
    raw = {
        "status": "ok",
        "timestamp": "2024-01-01T00:00:00Z",
        "symbols": {
            "EURUSD": {
                "candles": [
                    {"time": "2024-01-01T00:00:00Z", "open": 1.1, "high": 1.11, "low": 1.09, "close": 1.105, "volume": 100},
                    {"time": "2024-01-01T00:05:00Z", "open": 1.105, "high": 1.115, "low": 1.10, "close": 1.11, "volume": 120},
                ]
            }
        },
    }
    outcome = parse_live_payload(raw)
    assert outcome.ok
    assert outcome.adapter_used == "symbol_keyed_container"
    assert "EURUSD" in outcome.symbols
    assert len(outcome.symbols["EURUSD"].records) == 2


def test_parses_top_level_symbol_map_with_array_candles():
    raw = {
        "XAUUSD": [
            [1704067200, 2000.0, 2005.0, 1998.0, 2002.0, 500],
            [1704067500, 2002.0, 2006.0, 2001.0, 2004.0, 480],
        ]
    }
    outcome = parse_live_payload(raw)
    assert outcome.ok
    assert "XAUUSD" in outcome.symbols
    assert len(outcome.symbols["XAUUSD"].records) == 2
    assert outcome.symbols["XAUUSD"].records[0].open == 2000.0


def test_parses_list_of_symbol_objects():
    raw = [
        {
            "symbol": "GBPUSD",
            "bars": [
                {"t": 1704067200000, "o": 1.27, "h": 1.271, "l": 1.269, "c": 1.2705, "v": 10},
            ],
        }
    ]
    outcome = parse_live_payload(raw)
    assert outcome.ok
    assert "GBPUSD" in outcome.symbols


def test_unrecognized_schema_reports_dump_not_fabrication():
    raw = {"weird": {"nested": {"thing": 42}}}
    outcome = parse_live_payload(raw)
    assert not outcome.ok
    assert outcome.symbols == {}
    assert outcome.schema_dump is not None
    assert "SCHEMA_UNRECOGNIZED" in outcome.error


def test_malformed_records_are_dropped_not_fabricated():
    raw = {
        "symbols": {
            "EURUSD": {
                "candles": [
                    {"time": "2024-01-01T00:00:00Z", "open": 1.1, "high": 1.11, "low": 1.09, "close": 1.105, "volume": 100},
                    {"time": "not-a-time", "open": 1.1, "high": 1.11, "low": 1.09, "close": 1.105},
                    {"open": 1.1, "high": 1.11, "low": 1.09, "close": 1.105},
                ]
            }
        }
    }
    outcome = parse_live_payload(raw)
    assert outcome.ok
    parsed = outcome.symbols["EURUSD"]
    assert len(parsed.records) == 1
    assert parsed.malformed_count == 2


def test_describe_schema_is_json_serializable_string():
    dump = describe_schema({"a": [1, 2, {"b": "c"}]})
    assert isinstance(dump, str)
    assert "array" in dump
