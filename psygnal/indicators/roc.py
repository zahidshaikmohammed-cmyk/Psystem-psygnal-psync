from __future__ import annotations

import pandas as pd

from psygnal import config


def roc(close: pd.Series, period: int = config.ROC_PERIOD) -> pd.Series:
    shifted = close.shift(period)
    return 100 * (close - shifted) / shifted
