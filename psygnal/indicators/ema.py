from __future__ import annotations

import pandas as pd

from psygnal.indicators.common import ema
from psygnal import config


def compute_emas(close: pd.Series, periods: tuple[int, ...] = config.EMA_PERIODS) -> pd.DataFrame:
    return pd.DataFrame({f"ema_{p}": ema(close, p) for p in periods})
