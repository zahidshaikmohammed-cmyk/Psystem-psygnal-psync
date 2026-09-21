from __future__ import annotations

import numpy as np
import pytest

from psygnal.forecasting.calibration import (
    Calibrator,
    compute_calibration_curve,
    summarize_calibration_curve,
)


def test_calibration_curve_bins_predictions_correctly():
    probs = np.array([0.05, 0.15, 0.25, 0.85, 0.95])
    outcomes = np.array([0, 0, 1, 1, 1])
    curve = compute_calibration_curve(probs, outcomes, n_bins=10)
    assert len(curve) == 10
    # first bin [0.0, 0.1) should contain the 0.05 prediction
    assert curve[0]["count"] == 1
    assert curve[0]["predicted_mean"] == pytest.approx(0.05)


def test_well_calibrated_curve_reports_well_calibrated():
    rng = np.random.default_rng(0)
    probs = rng.uniform(0, 1, 2000)
    outcomes = (rng.uniform(0, 1, 2000) < probs).astype(int)
    curve = compute_calibration_curve(probs, outcomes, n_bins=10)
    summary = summarize_calibration_curve(curve, min_samples_per_bin=20)
    assert summary["status"] == "WELL_CALIBRATED"
    assert summary["usable_bins"] > 0


def test_poorly_calibrated_curve_detected():
    rng = np.random.default_rng(0)
    probs = rng.uniform(0, 1, 2000)
    # Actual outcome is always the OPPOSITE of what's predicted -> badly miscalibrated.
    outcomes = (probs < 0.5).astype(int)
    curve = compute_calibration_curve(probs, outcomes, n_bins=10)
    summary = summarize_calibration_curve(curve, min_samples_per_bin=20)
    assert summary["status"] == "POORLY_CALIBRATED"


def test_insufficient_data_when_bins_too_sparse():
    probs = np.array([0.1, 0.9])
    outcomes = np.array([0, 1])
    curve = compute_calibration_curve(probs, outcomes, n_bins=10)
    summary = summarize_calibration_curve(curve, min_samples_per_bin=20)
    assert summary["status"] == "INSUFFICIENT_DATA"


def test_calibrator_apply_matches_isotonic_when_fitted():
    rng = np.random.default_rng(1)
    probs = rng.uniform(0, 1, 300)
    outcomes = (rng.uniform(0, 1, 300) < probs).astype(int)
    calib = Calibrator.fit(probs, outcomes)
    assert calib.is_calibrated
    adjusted = calib.apply(0.5)
    assert 0.0 <= adjusted <= 1.0
