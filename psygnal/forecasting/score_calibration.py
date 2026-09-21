"""Mechanism to (eventually) learn signal-score component weights from
historical outcomes, instead of the fixed V1 "expert" weights.

HONESTY NOTE — this is a documented approximation, not a refit of the
exact live scoring function. `signal/score.py`'s sub-scores are built from
rich per-symbol intelligence objects (TrendState, MomentumState, ...)
which are expensive to reconstruct row-by-row across an entire historical
series. Instead, this module learns relative importances from the
closest available proxies already computed causally in
`forecasting/features.py`, and only reallocates the weight mass of the
score categories that have a reasonable proxy — every other category
keeps its V1 expert weight untouched. Treat the result as directional
guidance ("historically, trend-like evidence mattered more than volume-
like evidence for this instrument"), not a precision instrument.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from psygnal.signal.score import SCORE_WEIGHTS

MIN_SAMPLES_FOR_SCORE_CALIBRATION = 300

# Score category -> feature column(s) in forecasting/features.py treated as
# its closest causal proxy. Categories absent here keep their V1 weight.
PROXY_MAP: dict[str, tuple[str, ...]] = {
    "trend_alignment": ("structure_component", "ema_alignment"),
    "momentum_alignment": ("momentum_component",),
    "volatility_suitability": ("atr_pct_of_price",),
    "volume_confirmation": ("relative_volume",),
}


def calibrate_score_weights(X: pd.DataFrame, y: pd.Series) -> Optional[dict[str, Any]]:
    """Returns None when there isn't enough data to calibrate anything;
    otherwise a dict with `weights` (full 13-key, sums to 1.0),
    `learned_categories`, and `n_samples`."""
    required_cols = {col for cols in PROXY_MAP.values() for col in cols}
    missing = required_cols - set(X.columns)
    if missing:
        return None

    proxy_frame = X[list(required_cols)].copy()
    proxy_frame["structure_or_ema"] = proxy_frame[["structure_component", "ema_alignment"]].mean(axis=1)

    direction_proxy = np.sign(proxy_frame["structure_or_ema"].fillna(0.0))
    fallback = np.sign(proxy_frame["momentum_component"].fillna(0.0))
    direction_proxy = direction_proxy.where(direction_proxy != 0, fallback)

    valid_direction = direction_proxy != 0
    valid_features = proxy_frame[list(required_cols)].notna().all(axis=1)
    mask = valid_direction & valid_features
    if mask.sum() < MIN_SAMPLES_FOR_SCORE_CALIBRATION:
        return None

    y_masked = y[mask].to_numpy()
    direction_masked = direction_proxy[mask].to_numpy()
    favorable = (
        ((direction_masked > 0) & (y_masked == "UP")) | ((direction_masked < 0) & (y_masked == "DOWN"))
    ).astype(int)

    if len(set(favorable)) < 2:
        return None

    feature_cols = ["structure_component", "ema_alignment", "momentum_component", "atr_pct_of_price", "relative_volume"]
    feature_matrix = proxy_frame.loc[mask, feature_cols].to_numpy(dtype=float)
    mean = feature_matrix.mean(axis=0)
    std = feature_matrix.std(axis=0)
    std[std == 0] = 1.0
    standardized = (feature_matrix - mean) / std

    model = LogisticRegression(max_iter=1000)
    try:
        model.fit(standardized, favorable)
    except ValueError:
        return None

    coefs = np.abs(model.coef_[0])
    # Map the 5 raw feature coefficients back onto the 4 score categories
    # (trend_alignment draws on 2 features -> average their |coef|).
    category_importance = {
        "trend_alignment": float(np.mean([coefs[0], coefs[1]])),
        "momentum_alignment": float(coefs[2]),
        "volatility_suitability": float(coefs[3]),
        "volume_confirmation": float(coefs[4]),
    }
    total_importance = sum(category_importance.values())
    if total_importance <= 0:
        return None

    learnable_mass = sum(SCORE_WEIGHTS[cat] for cat in PROXY_MAP)
    weights = dict(SCORE_WEIGHTS)
    for cat, importance in category_importance.items():
        weights[cat] = learnable_mass * (importance / total_importance)

    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    return {
        "weights": weights,
        "learned_categories": list(PROXY_MAP.keys()),
        "n_samples": int(mask.sum()),
    }
