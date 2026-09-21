"""Probability calibration.

Falls back to an identity mapping when there isn't enough held-out data to
fit a calibration curve — an uncalibrated-but-honest probability beats a
calibration fit on too few points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression

MIN_SAMPLES_FOR_CALIBRATION = 200


@dataclass
class Calibrator:
    isotonic: Optional[IsotonicRegression]

    @classmethod
    def fit(cls, predicted_probs: np.ndarray, actual_outcomes: np.ndarray) -> "Calibrator":
        if len(predicted_probs) < MIN_SAMPLES_FOR_CALIBRATION:
            return cls(isotonic=None)
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(predicted_probs, actual_outcomes)
        return cls(isotonic=model)

    def apply(self, prob: float) -> float:
        if self.isotonic is None:
            return prob
        return float(self.isotonic.predict([prob])[0])

    @property
    def is_calibrated(self) -> bool:
        return self.isotonic is not None
