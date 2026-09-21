from __future__ import annotations

import pandas as pd

from psygnal import config
from psygnal.indicators.common import sma


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = config.STOCHASTIC_K_PERIOD,
    d_period: int = config.STOCHASTIC_D_PERIOD,
) -> pd.DataFrame:
    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()
    denom = (highest_high - lowest_low).replace(0.0, float("nan"))
    percent_k = 100 * (close - lowest_low) / denom
    percent_d = sma(percent_k, d_period)
    return pd.DataFrame({"stoch_k": percent_k, "stoch_d": percent_d})
