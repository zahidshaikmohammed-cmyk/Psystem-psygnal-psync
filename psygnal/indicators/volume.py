"""Volume-derived features.

The provider's "volume" field for FX/CFD instruments is typically tick
count (number of price updates), not centralized exchange traded volume.
This module never claims otherwise — see
`intelligence/volume.py::describe_volume_semantics`.
"""

from __future__ import annotations

import pandas as pd

from psygnal import config
from psygnal.indicators.common import sma


def volume_features(
    volume: pd.Series, period: int = config.VOLUME_MA_PERIOD
) -> pd.DataFrame:
    volume_ma = sma(volume, period)
    relative_volume = volume / volume_ma.replace(0.0, float("nan"))
    volume_percentile = volume.rolling(window=period, min_periods=period).rank(pct=True)
    volume_roc = 100 * (volume - volume.shift(1)) / volume.shift(1).replace(0.0, float("nan"))
    return pd.DataFrame(
        {
            "volume_ma": volume_ma,
            "relative_volume": relative_volume,
            "volume_percentile": volume_percentile,
            "volume_acceleration": volume_roc,
        }
    )
