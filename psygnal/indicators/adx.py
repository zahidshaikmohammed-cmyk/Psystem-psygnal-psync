"""Average Directional Index (Wilder)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from psygnal import config
from psygnal.indicators.atr import true_range
from psygnal.indicators.common import wilder_smooth


def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = config.ADX_PERIOD
) -> pd.DataFrame:
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index
    )

    tr = true_range(high, low, close)

    smoothed_tr = wilder_smooth(tr, period)
    smoothed_plus_dm = wilder_smooth(plus_dm, period)
    smoothed_minus_dm = wilder_smooth(minus_dm, period)

    plus_di = 100 * (smoothed_plus_dm / smoothed_tr)
    minus_di = 100 * (smoothed_minus_dm / smoothed_tr)

    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum

    adx_line = wilder_smooth(dx, period)

    return pd.DataFrame(
        {"plus_di": plus_di, "minus_di": minus_di, "dx": dx, "adx": adx_line}
    )
