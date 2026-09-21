"""Probability calibration.

Falls back to an identity mapping when there isn't enough held-out data to
fit a calibration curve — an uncalibrated-but-honest probability beats a
calibration fit on too few points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression

MIN_SAMPLES_FOR_CALIBRATION = 200
MIN_SAMPLES_PER_BIN_FOR_RELIABILITY = 20


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


def compute_calibration_curve(
    probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10
) -> list[dict[str, Any]]:
    """Bucket predictions into `n_bins` equal-width probability bins and
    compare the mean predicted probability against the actual empirical
    frequency in each bin — the standard reliability-diagram computation
    behind "do 70% predictions behave like 70% events"."""
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    curve: list[dict[str, Any]] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        count = int(mask.sum())
        curve.append(
            {
                "bin_range": (float(lo), float(hi)),
                "count": count,
                "predicted_mean": float(probs[mask].mean()) if count else None,
                "actual_frequency": float(outcomes[mask].mean()) if count else None,
            }
        )
    return curve


def summarize_calibration_curve(
    curve: list[dict[str, Any]], min_samples_per_bin: int = MIN_SAMPLES_PER_BIN_FOR_RELIABILITY
) -> dict[str, Any]:
    """Reduce a calibration curve to a pass/fail-style summary. Bins with
    too few samples are excluded from the deviation calculation rather than
    silently counted as well-calibrated."""
    usable = [b for b in curve if b["count"] >= min_samples_per_bin]
    if not usable:
        return {
            "status": "INSUFFICIENT_DATA",
            "reason": f"no calibration bin reached the minimum of {min_samples_per_bin} samples",
            "max_deviation": None,
            "usable_bins": 0,
            "total_bins": len(curve),
        }

    deviations = [abs(b["predicted_mean"] - b["actual_frequency"]) for b in usable]
    max_deviation = float(max(deviations))
    # A well-behaved calibration keeps predicted vs actual within ~10
    # percentage points in every bin that has enough samples to judge.
    status = "WELL_CALIBRATED" if max_deviation <= 0.10 else "POORLY_CALIBRATED"
    return {
        "status": status,
        "max_deviation": max_deviation,
        "usable_bins": len(usable),
        "total_bins": len(curve),
    }
