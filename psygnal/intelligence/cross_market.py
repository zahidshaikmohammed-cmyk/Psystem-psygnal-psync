"""Cross-market intelligence: USD composite, risk barometer, oil/CAD linkage.

These relationships are inputs to the forecast, never deterministic rules.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from psygnal import config
from psygnal.intelligence.trend import TrendState
from psygnal.intelligence.usd_composite import compute_usd_composite_index, summarize_usd_composite


def analyze_cross_market(
    closes_by_symbol: dict[str, pd.Series],
    trends_by_symbol: dict[str, TrendState],
) -> dict[str, Any]:
    legs_used = [s for s in config.USD_COMPOSITE_LEGS if s in closes_by_symbol]
    composite_index = compute_usd_composite_index(closes_by_symbol)
    usd_state = summarize_usd_composite(composite_index, legs_used)

    risk_trend = trends_by_symbol.get(config.RISK_BAROMETER_SYMBOL)
    risk_sentiment = "NEUTRAL"
    if risk_trend is not None:
        if risk_trend.score > 0.2:
            risk_sentiment = "RISK_ON_LEANING"
        elif risk_trend.score < -0.2:
            risk_sentiment = "RISK_OFF_LEANING"

    oil_trend = trends_by_symbol.get(config.OIL_SYMBOL)
    usdcad_trend = trends_by_symbol.get("USDCAD")
    oil_usdcad_relationship: Optional[str] = None
    if oil_trend is not None and usdcad_trend is not None:
        aligned = (oil_trend.score > 0 and usdcad_trend.score < 0) or (oil_trend.score < 0 and usdcad_trend.score > 0)
        oil_usdcad_relationship = "ALIGNED" if aligned else "DIVERGING"

    return {
        "usd_composite": usd_state,
        "risk_barometer_symbol": config.RISK_BAROMETER_SYMBOL,
        "risk_sentiment": risk_sentiment,
        "oil_usdcad_relationship": oil_usdcad_relationship,
    }
