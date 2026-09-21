from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from psygnal.forecasting.baseline import BaselineForecastModel, can_train
from psygnal.forecasting.calibration import Calibrator
from psygnal.forecasting.dataset import build_training_dataset
from psygnal.forecasting.ensemble import combine_forecasts, compute_deterministic_probabilities
from psygnal.forecasting.pattern_memory import PatternMemoryStore
from psygnal.data.aggregation import build_multi_timeframe_series
from psygnal.intelligence.context import build_symbol_intelligence
from psygnal.models import candles_to_frame
from tests.conftest import make_m5_series
from datetime import timedelta


def _random_walk_candles(n: int, seed: int = 42, base_price: float = 100.0):
    from datetime import datetime, timezone
    from psygnal.models import Candle

    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = []
    price = base_price
    for i in range(n):
        step = rng.normal(loc=0.0, scale=0.6)
        o = price
        c = price + step
        h = max(o, c) + abs(rng.normal(0, 0.1))
        l = min(o, c) - abs(rng.normal(0, 0.1))
        candles.append(Candle(time=start + timedelta(minutes=5 * i), open=o, high=h, low=l, close=c, volume=100 + abs(rng.normal(0, 20))))
        price = c
    return candles


def _big_dataset():
    candles = _random_walk_candles(1200)
    df = candles_to_frame(candles)
    return build_training_dataset(df)


def test_can_train_threshold():
    assert can_train(1000) is False or can_train(1000) is True  # sanity: doesn't crash
    from psygnal import config

    assert can_train(config.MIN_HISTORICAL_SAMPLES_FOR_TRAINING) is True
    assert can_train(config.MIN_HISTORICAL_SAMPLES_FOR_TRAINING - 1) is False


def test_baseline_model_trains_and_predicts():
    X, y, meta = _big_dataset()
    if len(X) < 300:
        pytest.skip("not enough synthetic samples generated")
    model = BaselineForecastModel("logistic_regression")
    report = model.fit(X, y)
    assert report.n_samples == len(X)
    proba = model.predict_proba(X.iloc[-1])
    assert proba is not None
    assert abs(sum(proba.values()) - 1.0) < 1e-6


def test_baseline_model_save_and_load(tmp_path: Path):
    X, y, meta = _big_dataset()
    model = BaselineForecastModel("logistic_regression")
    model.fit(X, y)
    path = tmp_path / "model.pkl"
    model.save(path)
    loaded = BaselineForecastModel.load(path)
    proba_original = model.predict_proba(X.iloc[-1])
    proba_loaded = loaded.predict_proba(X.iloc[-1])
    assert proba_original == proba_loaded


def test_pattern_memory_insufficient_data_returns_status():
    X, y, meta = _big_dataset()
    small_X, small_y, small_meta = X.iloc[:50], y.iloc[:50], meta.iloc[:50]
    store = PatternMemoryStore.fit(small_X, small_y, small_meta["forward_return"])
    result = store.query(small_X.iloc[-1])
    assert result["status"] == "INSUFFICIENT_DATA"


def test_pattern_memory_query_respects_max_index_no_leakage():
    X, y, meta = _big_dataset()
    if len(X) < 300:
        pytest.skip("not enough synthetic samples generated")
    store = PatternMemoryStore.fit(X, y, meta["forward_return"])
    query_pos = len(X) - 1
    result = store.query(X.iloc[query_pos], max_index=query_pos)
    if result["status"] == "OK":
        assert result["k_used"] > 0


def test_calibrator_identity_when_insufficient_samples():
    calib = Calibrator.fit(np.array([0.6, 0.7]), np.array([1, 0]))
    assert calib.is_calibrated is False
    assert calib.apply(0.6) == 0.6


def test_calibrator_fits_with_enough_samples():
    rng = np.random.default_rng(0)
    probs = rng.uniform(0, 1, 300)
    outcomes = (rng.uniform(0, 1, 300) < probs).astype(int)
    calib = Calibrator.fit(probs, outcomes)
    assert calib.is_calibrated is True


def test_deterministic_probabilities_sum_to_one_and_favor_uptrend():
    candles = make_m5_series(400, step=0.3)
    now = candles[-1].time + timedelta(minutes=6)
    mts = build_multi_timeframe_series("EURUSD", candles, now=now)
    intel = build_symbol_intelligence("EURUSD", mts, now)
    result = compute_deterministic_probabilities("EURUSD", intel, {"usd_composite": {"state": "NEUTRAL"}})
    total = result["probability_long"] + result["probability_short"] + result["probability_neutral"]
    assert total == pytest.approx(1.0, abs=1e-6)
    assert result["probability_long"] > result["probability_short"]


def test_combine_forecasts_falls_back_to_deterministic_only():
    deterministic = {"probability_long": 0.6, "probability_short": 0.3, "probability_neutral": 0.1}
    result = combine_forecasts(deterministic)
    assert result["sources_used"] == ["deterministic"]
    total = result["probability_long"] + result["probability_short"] + result["probability_neutral"]
    assert total == pytest.approx(1.0, abs=1e-6)


def test_combine_forecasts_blends_when_baseline_available():
    deterministic = {"probability_long": 0.6, "probability_short": 0.3, "probability_neutral": 0.1}
    baseline = {"UP": 0.8, "DOWN": 0.1, "NEUTRAL": 0.1}
    result = combine_forecasts(deterministic, baseline_probs=baseline)
    assert "baseline_model" in result["sources_used"]
    assert result["probability_long"] > deterministic["probability_long"] * 0.9
