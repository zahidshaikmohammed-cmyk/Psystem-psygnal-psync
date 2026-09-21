"""Per-symbol intelligence aggregator.

Two-phase build:
  1. `build_symbol_intelligence` — everything computable from a single
     symbol's own multi-timeframe series (structure, liquidity, trend,
     momentum, volatility, volume, price action, sequences, sessions).
  2. `attach_cross_market` — merges in cross-symbol context (USD
     composite, Gold specialization) once every symbol's phase-1 result is
     available, since those genuinely need other symbols' data.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import pandas as pd

from psygnal import config
from psygnal.intelligence.candle_features import compute_candle_features
from psygnal.intelligence.liquidity import analyze_liquidity
from psygnal.intelligence.momentum import analyze_momentum
from psygnal.intelligence.price_action import classify_price_action
from psygnal.intelligence.sequence import summarize_sequences
from psygnal.intelligence.sessions import summarize_sessions
from psygnal.intelligence.structure import analyze_structure
from psygnal.intelligence.trend import analyze_trend
from psygnal.intelligence.volatility import analyze_volatility
from psygnal.intelligence.volume import analyze_volume
from psygnal.indicators.atr import atr
from psygnal.models import MultiTimeframeSeries


def build_symbol_intelligence(
    symbol: str, mts: MultiTimeframeSeries, now_utc: datetime
) -> dict[str, Any]:
    m5 = mts.get("M5")
    if m5.empty:
        return {"symbol": symbol, "status": "UNAVAILABLE"}

    m5_features = compute_candle_features(m5)
    current_price = float(m5["close"].iloc[-1])

    structures = {tf: analyze_structure(mts.get(tf), tf) for tf in config.TIMEFRAME_M5_MULTIPLES}
    trends = {tf: analyze_trend(mts.get(tf), structures[tf], tf) for tf in config.TIMEFRAME_M5_MULTIPLES}

    atr_series = atr(m5["high"], m5["low"], m5["close"], config.ATR_PERIOD)
    atr_value = float(atr_series.dropna().iloc[-1]) if atr_series.dropna().size else None

    liquidity = analyze_liquidity(m5, structures["M5"], now_utc, atr_value, current_price)
    momentum = analyze_momentum(m5, structures["M5"].swing_highs, structures["M5"].swing_lows)
    volatility = analyze_volatility(m5)
    volume = analyze_volume(m5_features)
    price_action = classify_price_action(m5_features)
    sequences = summarize_sequences(m5_features)
    sessions = summarize_sessions(m5, now_utc)

    return {
        "symbol": symbol,
        "status": "OK",
        "current_price": current_price,
        "forming_m5": mts.forming.get("M5"),
        "structures": structures,
        "trends": trends,
        "liquidity": liquidity,
        "momentum": momentum,
        "volatility": volatility,
        "volume": volume,
        "price_action": price_action,
        "sequences": sequences,
        "sessions": sessions,
        "m5_features": m5_features,
        "closes": {tf: mts.get(tf)["close"] for tf in config.TIMEFRAME_M5_MULTIPLES if not mts.get(tf).empty},
    }


def attach_cross_market(
    symbol_intel: dict[str, Any],
    cross_market_state: dict[str, Any],
    gold_state: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    symbol_intel = dict(symbol_intel)
    symbol_intel["cross_market"] = cross_market_state
    if gold_state is not None:
        symbol_intel["gold"] = gold_state
    return symbol_intel
