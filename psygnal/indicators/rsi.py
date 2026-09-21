"""Wilder-style RSI (Relative Strength Index)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from psygnal import config
from psygnal.indicators.common import wilder_smooth


def rsi(close: pd.Series, period: int = config.RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = wilder_smooth(gain, period)
    avg_loss = wilder_smooth(loss, period)

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    result = 100 - (100 / (1 + rs))
    # When average loss is exactly zero (and gain > 0) RSI is defined as 100.
    result = result.where(~((avg_loss == 0.0) & (avg_gain > 0.0)), 100.0)
    # When both are zero (flat market) RSI is conventionally 50.
    result = result.where(~((avg_loss == 0.0) & (avg_gain == 0.0)), 50.0)
    return result
