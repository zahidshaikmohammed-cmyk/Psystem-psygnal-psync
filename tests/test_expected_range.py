from __future__ import annotations

import pytest

from psygnal.forecasting.expected_range import compute_expected_range


def test_symmetric_fallback_when_no_evaluation_metrics():
    result = compute_expected_range("LONG", 100.0, atr_value=2.0, symmetric_expected_move=5.0, evaluation_metrics=None)
    assert result["methodology"] == "SYMMETRIC_ATR_V1"
    assert result["expected_60m_high"] == pytest.approx(105.0)
    assert result["expected_60m_low"] == pytest.approx(95.0)
    assert result["expected_60m_range"] == pytest.approx(10.0)


def test_symmetric_fallback_when_metrics_incomplete():
    result = compute_expected_range(
        "LONG", 100.0, atr_value=2.0, symmetric_expected_move=5.0, evaluation_metrics={"mean_mfe_atr": 1.5}
    )
    assert result["methodology"] == "SYMMETRIC_ATR_V1"


def test_asymmetric_long_uses_mfe_for_upside_mae_for_downside():
    metrics = {"mean_mfe_atr": 2.0, "mean_mae_atr": 0.5}
    result = compute_expected_range("LONG", 100.0, atr_value=3.0, symmetric_expected_move=5.0, evaluation_metrics=metrics)
    assert result["methodology"] == "HISTORICAL_MFE_MAE"
    assert result["expected_60m_high"] == pytest.approx(100.0 + 2.0 * 3.0)
    assert result["expected_60m_low"] == pytest.approx(100.0 - 0.5 * 3.0)


def test_asymmetric_short_flips_favorable_direction():
    metrics = {"mean_mfe_atr": 2.0, "mean_mae_atr": 0.5}
    result = compute_expected_range("SHORT", 100.0, atr_value=3.0, symmetric_expected_move=5.0, evaluation_metrics=metrics)
    assert result["methodology"] == "HISTORICAL_MFE_MAE"
    # For a SHORT, the favorable move is DOWN and adverse is UP.
    assert result["expected_60m_low"] == pytest.approx(100.0 - 2.0 * 3.0)
    assert result["expected_60m_high"] == pytest.approx(100.0 + 0.5 * 3.0)


def test_no_atr_falls_back_to_symmetric():
    metrics = {"mean_mfe_atr": 2.0, "mean_mae_atr": 0.5}
    result = compute_expected_range("LONG", 100.0, atr_value=None, symmetric_expected_move=4.0, evaluation_metrics=metrics)
    assert result["methodology"] == "SYMMETRIC_ATR_V1"
    assert result["expected_60m_high"] == pytest.approx(104.0)
