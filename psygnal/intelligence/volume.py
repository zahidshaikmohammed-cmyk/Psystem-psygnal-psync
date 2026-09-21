"""Volume intelligence: interprets the raw volume features (indicators/volume.py)
in the context of concurrent price behaviour.

IMPORTANT semantics disclaimer: FX/CFD "volume" reported by most retail
feeds (including, presumptively, this one — unconfirmed since the live
endpoint could not be inspected) is broker tick count, not centralized
exchange traded volume. This module never claims otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from psygnal import config
from psygnal.indicators.volume import volume_features


VOLUME_SEMANTICS_NOTE = (
    "Volume figures are provider tick-activity counts, not centralized "
    "exchange traded volume. FX/CFD markets have no single order book, so "
    "this is a proxy for market activity intensity only."
)


@dataclass
class VolumeState:
    label: str  # ELEVATED | AVERAGE | LOW
    relative_volume: float | None
    volume_percentile: float | None
    price_volume_relationship: str  # CONFIRMING_EXPANSION | DIVERGING | INCONCLUSIVE
    semantics_note: str = VOLUME_SEMANTICS_NOTE

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "relative_volume": self.relative_volume,
            "volume_percentile": self.volume_percentile,
            "price_volume_relationship": self.price_volume_relationship,
            "semantics_note": self.semantics_note,
        }


def analyze_volume(candle_features: pd.DataFrame) -> VolumeState:
    if candle_features.empty or len(candle_features) < config.VOLUME_MA_PERIOD + 2:
        return VolumeState("AVERAGE", None, None, "INCONCLUSIVE")

    vf = volume_features(candle_features["volume"], config.VOLUME_MA_PERIOD)
    rel_vol = vf["relative_volume"].iloc[-1]
    vol_pctl = vf["volume_percentile"].iloc[-1]

    label = "AVERAGE"
    if not np.isnan(rel_vol):
        if rel_vol >= 1.5:
            label = "ELEVATED"
        elif rel_vol <= 0.6:
            label = "LOW"

    range_pctl = candle_features["range_percentile"].iloc[-1]
    relationship = "INCONCLUSIVE"
    if not (np.isnan(rel_vol) or (isinstance(range_pctl, float) and np.isnan(range_pctl))):
        expanding_price = bool(range_pctl >= 0.6)
        expanding_volume = bool(rel_vol >= 1.2)
        fading_volume = bool(rel_vol <= 0.8)
        if expanding_price and expanding_volume:
            relationship = "CONFIRMING_EXPANSION"
        elif expanding_price and fading_volume:
            relationship = "DIVERGING"

    return VolumeState(
        label=label,
        relative_volume=float(rel_vol) if not np.isnan(rel_vol) else None,
        volume_percentile=float(vol_pctl) if not np.isnan(vol_pctl) else None,
        price_volume_relationship=relationship,
    )
