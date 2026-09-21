"""Specialized Gold (XAUUSD) intelligence.

Never reduces Gold to a single deterministic rule such as
"USD weak => BUY GOLD"; instead reports the confirming/diverging evidence
so the ensemble can weigh it in context.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from psygnal.intelligence.trend import TrendState


def compute_rolling_correlation(a: pd.Series, b: pd.Series, window: int = 48) -> Optional[float]:
    aligned = pd.concat({"a": a, "b": b}, axis=1, join="inner").dropna()
    if len(aligned) < min(window, 10):
        return None
    recent = aligned.iloc[-window:]
    corr = recent["a"].corr(recent["b"])
    return float(corr) if corr is not None and corr == corr else None


def analyze_gold(
    xau_close: pd.Series,
    xag_close: Optional[pd.Series],
    xau_trend: TrendState,
    usd_composite_state: dict[str, Any],
) -> dict[str, Any]:
    silver_correlation = None
    silver_confirmation = "UNAVAILABLE"
    if xag_close is not None and not xag_close.empty:
        silver_correlation = compute_rolling_correlation(xau_close, xag_close)
        if silver_correlation is not None:
            xag_returns = xag_close.pct_change().dropna()
            xau_returns = xau_close.pct_change().dropna()
            if len(xag_returns) and len(xau_returns):
                same_direction = (xau_returns.iloc[-1] > 0) == (xag_returns.iloc[-1] > 0)
                silver_confirmation = "CONFIRMING" if (silver_correlation > 0.3 and same_direction) else "DIVERGING"

    usd_backdrop = usd_composite_state.get("state", "UNAVAILABLE")
    usd_alignment = "NEUTRAL"
    if usd_backdrop == "WEAKENING" and xau_trend.score > 0:
        usd_alignment = "SUPPORTIVE"
    elif usd_backdrop == "STRENGTHENING" and xau_trend.score < 0:
        usd_alignment = "SUPPORTIVE"
    elif usd_backdrop in ("STRENGTHENING", "WEAKENING"):
        usd_alignment = "CONFLICTING"

    return {
        "xau_trend": xau_trend.label,
        "usd_backdrop": usd_backdrop,
        "usd_alignment": usd_alignment,
        "silver_correlation": silver_correlation,
        "silver_confirmation": silver_confirmation,
    }
