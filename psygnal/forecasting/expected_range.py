"""Expected 60-minute price range.

V1 methodology (`SYMMETRIC_ATR_V1`): current price +/- ATR*sqrt(horizon).
Always available, always honestly labeled as a volatility-based estimate,
never a validated distribution.

V2 methodology (`HISTORICAL_MFE_MAE`): once a trained model's held-out
test-slice evaluation has genuine direction-conditioned favorable/adverse
excursion statistics (`mean_mfe_atr`/`mean_mae_atr` — see
forecasting/backtest.py::evaluate_predictions), the range becomes
asymmetric: how far price has historically run in this instrument's
favor vs against, for trades taken in this direction, rather than
assuming symmetry.
"""

from __future__ import annotations

from typing import Any, Optional


def compute_expected_range(
    direction: str,
    current_price: float,
    atr_value: Optional[float],
    symmetric_expected_move: Optional[float],
    evaluation_metrics: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    mfe_atr = (evaluation_metrics or {}).get("mean_mfe_atr")
    mae_atr = (evaluation_metrics or {}).get("mean_mae_atr")

    if atr_value and mfe_atr is not None and mae_atr is not None:
        favorable_move = mfe_atr * atr_value
        adverse_move = mae_atr * atr_value
        if direction == "LONG":
            high = current_price + favorable_move
            low = current_price - adverse_move
        else:
            high = current_price + adverse_move
            low = current_price - favorable_move
        return {
            "expected_60m_high": high,
            "expected_60m_low": low,
            "expected_60m_range": high - low,
            "methodology": "HISTORICAL_MFE_MAE",
        }

    move = symmetric_expected_move or 0.0
    return {
        "expected_60m_high": current_price + move,
        "expected_60m_low": current_price - move,
        "expected_60m_range": 2 * move,
        "methodology": "SYMMETRIC_ATR_V1",
    }
