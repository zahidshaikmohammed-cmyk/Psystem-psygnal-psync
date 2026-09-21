from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from psygnal.data.aggregation import build_multi_timeframe_series
from psygnal.intelligence.context import attach_cross_market, build_symbol_intelligence
from psygnal.intelligence.cross_market import analyze_cross_market
from psygnal.intelligence.gold import analyze_gold, compute_rolling_correlation
from psygnal.intelligence.structure import analyze_structure
from psygnal.intelligence.trend import analyze_trend
from psygnal.intelligence.usd_composite import compute_usd_composite_index, summarize_usd_composite
from psygnal.models import candles_to_frame
from tests.conftest import make_m5_series


def _close_series(n=300, direction=1, step=0.05, base=100.0):
    candles = make_m5_series(n, base_price=base, step=direction * step)
    return candles_to_frame(candles)["close"]


def test_usd_composite_reflects_dollar_weakness_when_eur_gbp_rise():
    # EURUSD and GBPUSD rising (USD weak leg = -1), USDJPY flat/rising slowly.
    eurusd = _close_series(direction=1, step=0.001, base=1.10)
    gbpusd = _close_series(direction=1, step=0.001, base=1.27)
    usdjpy = _close_series(direction=0.1, step=0.001, base=150.0)
    closes = {"EURUSD": eurusd, "GBPUSD": gbpusd, "USDJPY": usdjpy}
    index = compute_usd_composite_index(closes)
    assert not index.empty
    summary = summarize_usd_composite(index, list(closes.keys()))
    assert summary["state"] == "WEAKENING"


def test_usd_composite_unavailable_with_no_legs():
    index = compute_usd_composite_index({})
    summary = summarize_usd_composite(index, [])
    assert summary["state"] == "UNAVAILABLE"


def test_cross_market_risk_sentiment_from_gbpjpy_trend():
    gbpjpy_close = _close_series(direction=1, step=0.05, base=190.0)
    df = candles_to_frame(make_m5_series(300, base_price=190.0, step=0.05))
    structure = analyze_structure(df, "M5")
    trend = analyze_trend(df, structure, "M5")
    result = analyze_cross_market({}, {"GBPJPY": trend})
    assert result["risk_sentiment"] in ("RISK_ON_LEANING", "NEUTRAL", "RISK_OFF_LEANING")


def test_gold_analysis_reports_silver_confirmation():
    xau_candles = make_m5_series(300, base_price=2000.0, step=0.5)
    xag_candles = make_m5_series(300, base_price=24.0, step=0.006)
    xau_close = candles_to_frame(xau_candles)["close"]
    xag_close = candles_to_frame(xag_candles)["close"]
    structure = analyze_structure(candles_to_frame(xau_candles), "M5")
    trend = analyze_trend(candles_to_frame(xau_candles), structure, "M5")
    usd_state = {"state": "WEAKENING"}
    result = analyze_gold(xau_close, xag_close, trend, usd_state)
    assert result["silver_confirmation"] in ("CONFIRMING", "DIVERGING", "UNAVAILABLE")
    assert "usd_alignment" in result


def test_gold_analysis_handles_missing_silver():
    xau_candles = make_m5_series(300, base_price=2000.0, step=0.5)
    xau_close = candles_to_frame(xau_candles)["close"]
    structure = analyze_structure(candles_to_frame(xau_candles), "M5")
    trend = analyze_trend(candles_to_frame(xau_candles), structure, "M5")
    result = analyze_gold(xau_close, None, trend, {"state": "NEUTRAL"})
    assert result["silver_confirmation"] == "UNAVAILABLE"


def test_build_symbol_intelligence_end_to_end():
    candles = make_m5_series(400, base_price=100.0, step=0.05)
    now = candles[-1].time + timedelta(minutes=6)
    mts = build_multi_timeframe_series("EURUSD", candles, now=now)
    intel = build_symbol_intelligence("EURUSD", mts, now)
    assert intel["status"] == "OK"
    for key in ("structures", "trends", "liquidity", "momentum", "volatility", "volume", "price_action", "sequences", "sessions"):
        assert key in intel


def test_attach_cross_market_merges_without_mutating_original():
    intel = {"symbol": "XAUUSD", "status": "OK"}
    merged = attach_cross_market(intel, {"usd_composite": {}}, gold_state={"xau_trend": "BULLISH"})
    assert "cross_market" not in intel
    assert merged["cross_market"] == {"usd_composite": {}}
    assert merged["gold"]["xau_trend"] == "BULLISH"
