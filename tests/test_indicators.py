from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from psygnal.indicators.adx import adx
from psygnal.indicators.atr import atr, true_range
from psygnal.indicators.bollinger import bollinger_bands
from psygnal.indicators.common import ema, wilder_smooth
from psygnal.indicators.macd import macd
from psygnal.indicators.roc import roc
from psygnal.indicators.rsi import rsi
from psygnal.indicators.stochastic import stochastic
from psygnal.indicators.volume import volume_features


def test_ema_matches_hand_computed_reference():
    # alpha = 2/(3+1) = 0.5
    # seed = 1 -> 1.5 -> 2.25 (unmasked, index==period-1=2) -> 3.125 -> 4.0625
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ema(series, period=3)
    assert result.iloc[:2].isna().all()
    assert result.iloc[2] == pytest.approx(2.25)
    assert result.iloc[3] == pytest.approx(3.125)
    assert result.iloc[4] == pytest.approx(4.0625)


def test_ema_handles_leading_nan_series():
    series = pd.Series([np.nan, np.nan, 1.0, 2.0, 3.0, 4.0])
    result = ema(series, period=3)
    assert result.iloc[:4].isna().all()
    assert result.iloc[4] == pytest.approx(2.25)


def test_wilder_smooth_seed_is_simple_average():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = wilder_smooth(series, period=3)
    # seed at index 2 = mean(1,2,3) = 2
    assert result.iloc[2] == pytest.approx(2.0)
    # next = (2*2 + 4)/3 = 8/3
    assert result.iloc[3] == pytest.approx(8 / 3)
    # next = ((8/3)*2 + 5)/3
    assert result.iloc[4] == pytest.approx(((8 / 3) * 2 + 5) / 3)


def test_rsi_all_gains_is_100():
    close = pd.Series([float(i) for i in range(1, 20)])  # strictly increasing
    result = rsi(close, period=14)
    assert result.iloc[-1] == pytest.approx(100.0)


def test_rsi_all_losses_is_0():
    close = pd.Series([float(i) for i in range(20, 1, -1)])  # strictly decreasing
    result = rsi(close, period=14)
    assert result.iloc[-1] == pytest.approx(0.0)


def test_rsi_flat_market_is_50():
    close = pd.Series([100.0] * 20)
    result = rsi(close, period=14)
    assert result.iloc[-1] == pytest.approx(50.0)


def test_true_range_first_bar_is_high_minus_low():
    high = pd.Series([10.0, 12.0])
    low = pd.Series([8.0, 9.0])
    close = pd.Series([9.0, 11.0])
    tr = true_range(high, low, close)
    assert tr.iloc[0] == pytest.approx(2.0)
    # second bar: max(12-9, |12-9|, |9-9|) = max(3,3,0) = 3
    assert tr.iloc[1] == pytest.approx(3.0)


def test_atr_reference_value():
    high = pd.Series([10.0, 12.0, 13.0, 11.0])
    low = pd.Series([8.0, 9.0, 10.0, 9.0])
    close = pd.Series([9.0, 11.0, 12.0, 10.0])
    result = atr(high, low, close, period=3)
    tr = true_range(high, low, close)
    expected_seed = tr.iloc[:3].mean()
    assert result.iloc[2] == pytest.approx(expected_seed)


def test_macd_line_equals_fast_minus_slow_ema():
    close = pd.Series(np.linspace(1, 50, 60))
    result = macd(close, fast=5, slow=10, signal=3)
    fast = ema(close, 5)
    slow = ema(close, 10)
    pd.testing.assert_series_equal(
        result["macd"], (fast - slow), check_names=False
    )
    diff = (result["histogram"] - (result["macd"] - result["signal"])).dropna()
    assert (diff.abs() < 1e-9).all()


def test_bollinger_bands_reference():
    close = pd.Series([10.0, 10.0, 10.0, 10.0, 10.0, 20.0])
    result = bollinger_bands(close, period=5, std_multiplier=2.0)
    window = close.iloc[1:6]
    expected_mid = window.mean()
    expected_std = window.std(ddof=0)
    assert result["bb_middle"].iloc[5] == pytest.approx(expected_mid)
    assert result["bb_upper"].iloc[5] == pytest.approx(expected_mid + 2 * expected_std)
    assert result["bb_lower"].iloc[5] == pytest.approx(expected_mid - 2 * expected_std)


def test_stochastic_bounds():
    high = pd.Series(np.linspace(10, 20, 30))
    low = pd.Series(np.linspace(8, 18, 30))
    close = pd.Series(np.linspace(9, 19, 30))
    result = stochastic(high, low, close, k_period=14, d_period=3)
    valid = result.dropna()
    assert (valid["stoch_k"] >= -1e-9).all()
    assert (valid["stoch_k"] <= 100 + 1e-9).all()


def test_roc_reference_value():
    close = pd.Series([100.0, 110.0, 121.0])
    result = roc(close, period=1)
    assert result.iloc[1] == pytest.approx(10.0)
    assert result.iloc[2] == pytest.approx(10.0)


def test_adx_output_is_bounded():
    n = 60
    high = pd.Series(np.linspace(10, 40, n) + np.sin(np.linspace(0, 10, n)))
    low = high - 1.0
    close = (high + low) / 2
    result = adx(high, low, close, period=14)
    valid = result["adx"].dropna()
    assert (valid >= 0).all()
    assert (valid <= 100 + 1e-6).all()


def test_volume_features_relative_volume_reference():
    volume = pd.Series([100.0] * 5 + [300.0])
    result = volume_features(volume, period=5)
    ma = result["volume_ma"].iloc[5]
    # rolling window of 5 ending at index 5 covers indices [1..5]:
    # [100, 100, 100, 100, 300] -> mean 140
    assert ma == pytest.approx(140.0)
    assert result["relative_volume"].iloc[5] == pytest.approx(300.0 / 140.0)
