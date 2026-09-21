"""Internal USD strength composite.

Explicitly NOT the official ICE Dollar Index (DXY) — it is a simple,
transparent, equal-weighted synthetic built only from the pairs present in
the current PSYGRID universe. Labeled everywhere as
`PSYGRID_USD_COMPOSITE — INTERNAL COMPOSITE, NOT OFFICIAL DXY`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from psygnal import config

COMPOSITE_LABEL = "PSYGRID_USD_COMPOSITE (INTERNAL COMPOSITE — NOT OFFICIAL DXY)"


def compute_usd_composite_index(
    closes_by_symbol: dict[str, pd.Series],
    legs: dict[str, int] = config.USD_COMPOSITE_LEGS,
) -> pd.Series:
    available = {sym: sign for sym, sign in legs.items() if sym in closes_by_symbol and not closes_by_symbol[sym].empty}
    if not available:
        return pd.Series(dtype=float)

    aligned = pd.concat({sym: closes_by_symbol[sym] for sym in available}, axis=1, join="inner").sort_index()
    if aligned.empty or len(aligned) < 2:
        return pd.Series(dtype=float)

    log_returns = np.log(aligned / aligned.shift(1))
    signs = pd.Series(available)
    signed_returns = log_returns.mul(signs, axis=1)
    composite_log_return = signed_returns.mean(axis=1).fillna(0.0)
    composite_index = 100.0 * np.exp(composite_log_return.cumsum())
    composite_index.name = "usd_composite"
    return composite_index


def summarize_usd_composite(composite_index: pd.Series, legs_used: list[str]) -> dict[str, Any]:
    if composite_index.empty or len(composite_index) < 6:
        return {
            "label": COMPOSITE_LABEL,
            "state": "UNAVAILABLE",
            "value": None,
            "change_1h_pct": None,
            "legs_used": legs_used,
        }

    values = composite_index.to_numpy()
    n = len(values)
    window = min(20, n)
    x = np.arange(window)
    slope = float(np.polyfit(x, values[-window:], 1)[0])
    mean_v = float(np.mean(values[-window:])) or 1e-9
    rel_slope = slope / mean_v

    if rel_slope > 0.0001:
        state = "STRENGTHENING"
    elif rel_slope < -0.0001:
        state = "WEAKENING"
    else:
        state = "NEUTRAL"

    hr_window = min(config.FORECAST_HORIZON_M5_CANDLES, n - 1)
    change_1h_pct = float((values[-1] / values[-1 - hr_window] - 1) * 100) if hr_window > 0 else None

    return {
        "label": COMPOSITE_LABEL,
        "state": state,
        "value": float(values[-1]),
        "change_1h_pct": change_1h_pct,
        "legs_used": legs_used,
    }
