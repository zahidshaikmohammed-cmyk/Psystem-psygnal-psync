from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from psygnal import config
from psygnal.forecasting.baseline import BaselineForecastModel
from psygnal.forecasting.ensemble import determine_forecast_engine_label
from psygnal.forecasting.pattern_memory import PatternMemoryStore
from psygnal.main import run_engine
from psygnal.models import Candle
from tests.conftest import make_m5_series


def test_determine_forecast_engine_label_deterministic_only():
    assert determine_forecast_engine_label(["deterministic"]) == "DETERMINISTIC"


def test_determine_forecast_engine_label_trained_ml():
    assert determine_forecast_engine_label(["deterministic", "baseline_model"]) == "TRAINED_ML"


def test_determine_forecast_engine_label_ensemble_with_pattern_memory():
    assert determine_forecast_engine_label(["deterministic", "pattern_memory"]) == "ENSEMBLE"
    assert determine_forecast_engine_label(["deterministic", "baseline_model", "pattern_memory"]) == "ENSEMBLE"


def _random_walk_candles(n=1400, seed=21, base_price=1.10):
    from datetime import datetime, timezone

    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = []
    price = base_price
    for i in range(n):
        step = rng.normal(0.0, base_price * 0.0006)
        o, c = price, price + step
        h, l = max(o, c) + abs(rng.normal(0, base_price * 0.0002)), min(o, c) - abs(rng.normal(0, base_price * 0.0002))
        candles.append(Candle(time=start + timedelta(minutes=5 * i), open=o, high=h, low=l, close=c, volume=100 + abs(rng.normal(0, 20))))
        price = c
    return candles


def _raw_payload_for(symbol: str, candles: list[Candle]) -> dict:
    return {
        "status": "ok",
        "symbols": {
            symbol: {
                "candles": [
                    {"time": c.time.isoformat(), "open": c.open, "high": c.high, "low": c.low, "close": c.close, "volume": c.volume}
                    for c in candles
                ]
            }
        },
    }


def _fit_toy_model(X_cols: list[str]) -> BaselineForecastModel:
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame(rng.normal(size=(n, len(X_cols))), columns=X_cols)
    y = pd.Series(rng.choice(["UP", "DOWN", "NEUTRAL"], size=n))
    model = BaselineForecastModel("logistic_regression")
    model.fit(X, y)
    return model


def test_run_engine_reports_deterministic_when_no_models_available():
    candles = _random_walk_candles(600, seed=1)
    payload = _raw_payload_for("EURUSD", candles)
    now = candles[-1].time + timedelta(minutes=6)
    cfg = config.EngineConfig(symbols=("EURUSD",), enable_macro=False, enable_news=False)
    outcome = run_engine(payload, cfg=cfg, now_utc=now)

    signal, _ = outcome["results"][0]
    assert signal.forecast_engine == "DETERMINISTIC"
    assert signal.model_status == "UNTRAINED"
    assert signal.model_probability is None
    assert signal.calibration_state == {"status": "NOT_APPLICABLE"}
    assert any("no trained model" in w.lower() or "deterministic" in w.lower() for w in signal.warnings)


def test_run_engine_reports_trained_ml_when_baseline_model_injected():
    from psygnal.forecasting.features import build_feature_frame

    candles = _random_walk_candles(600, seed=2)
    payload = _raw_payload_for("EURUSD", candles)
    now = candles[-1].time + timedelta(minutes=6)
    cfg = config.EngineConfig(symbols=("EURUSD",), enable_macro=False, enable_news=False)

    dummy_frame = build_feature_frame(pd.DataFrame({"open": [1.0] * 260, "high": [1.0] * 260, "low": [1.0] * 260, "close": [1.0] * 260, "volume": [1.0] * 260}))
    model = _fit_toy_model(list(dummy_frame.columns))

    outcome = run_engine(payload, cfg=cfg, now_utc=now, baseline_models={"EURUSD": model})
    signal, _ = outcome["results"][0]

    assert signal.model_status == "TRAINED"
    assert signal.forecast_engine == "TRAINED_ML"
    assert signal.model_probability is not None
    assert set(signal.model_probability.keys()) >= {"long", "short", "neutral", "model_name", "calibrated"}
    assert pytest.approx(signal.model_probability["long"] + signal.model_probability["short"] + signal.model_probability["neutral"], abs=1e-6) == 1.0


def test_run_engine_reports_ensemble_when_pattern_memory_also_available():
    from psygnal.forecasting.features import build_feature_frame

    candles = _random_walk_candles(600, seed=4)
    payload = _raw_payload_for("EURUSD", candles)
    now = candles[-1].time + timedelta(minutes=6)
    cfg = config.EngineConfig(symbols=("EURUSD",), enable_macro=False, enable_news=False)

    dummy_frame = build_feature_frame(pd.DataFrame({"open": [1.0] * 260, "high": [1.0] * 260, "low": [1.0] * 260, "close": [1.0] * 260, "volume": [1.0] * 260}))
    model = _fit_toy_model(list(dummy_frame.columns))

    rng = np.random.default_rng(0)
    n = 250
    X = pd.DataFrame(rng.normal(size=(n, len(dummy_frame.columns))), columns=list(dummy_frame.columns))
    y = pd.Series(rng.choice(["UP", "DOWN", "NEUTRAL"], size=n))
    forward_return = pd.Series(rng.normal(size=n))
    store = PatternMemoryStore.fit(X, y, forward_return)

    outcome = run_engine(
        payload,
        cfg=cfg,
        now_utc=now,
        baseline_models={"EURUSD": model},
        pattern_memory_stores={"EURUSD": store},
    )
    signal, _ = outcome["results"][0]
    assert signal.forecast_engine == "ENSEMBLE"
    assert signal.model_status == "TRAINED"
