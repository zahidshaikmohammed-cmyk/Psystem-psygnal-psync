from __future__ import annotations

import pandas as pd

from psygnal import config
from psygnal.indicators.common import rolling_std, sma


def bollinger_bands(
    close: pd.Series,
    period: int = config.BOLLINGER_PERIOD,
    std_multiplier: float = config.BOLLINGER_STD_MULTIPLIER,
) -> pd.DataFrame:
    middle = sma(close, period)
    std = rolling_std(close, period, ddof=0)
    upper = middle + std_multiplier * std
    lower = middle - std_multiplier * std
    bandwidth = (upper - lower) / middle
    percent_b = (close - lower) / (upper - lower)
    return pd.DataFrame(
        {
            "bb_middle": middle,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_bandwidth": bandwidth,
            "bb_percent_b": percent_b,
        }
    )
