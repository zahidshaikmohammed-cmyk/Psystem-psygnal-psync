from __future__ import annotations

import pandas as pd

from psygnal import config
from psygnal.indicators.common import wilder_smooth


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    a = high - low
    b = (high - prev_close).abs()
    c = (low - prev_close).abs()
    tr = pd.concat([a, b, c], axis=1).max(axis=1)
    tr.iloc[0] = a.iloc[0]  # no previous close for the first bar
    return tr


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = config.ATR_PERIOD
) -> pd.Series:
    tr = true_range(high, low, close)
    return wilder_smooth(tr, period)
