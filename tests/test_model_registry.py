from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from psygnal.forecasting.baseline import BaselineForecastModel
from psygnal.forecasting.pattern_memory import PatternMemoryStore
from psygnal.forecasting.registry import (
    load_available_models,
    load_score_weights,
    model_artifact_path,
    pattern_memory_artifact_path,
    score_weights_artifact_path,
)


def _toy_dataset(n=60, n_features=3, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, n_features)), columns=[f"f{i}" for i in range(n_features)])
    y = pd.Series(rng.choice(["UP", "DOWN", "NEUTRAL"], size=n))
    forward_return = pd.Series(rng.normal(size=n))
    return X, y, forward_return


def test_load_available_models_when_directory_missing_never_crashes(tmp_path: Path):
    missing_dir = tmp_path / "does_not_exist"
    baseline_models, pattern_memory_stores, meta = load_available_models(("XAUUSD", "EURUSD"), model_dir=missing_dir)
    assert baseline_models == {}
    assert pattern_memory_stores == {}
    assert meta["XAUUSD"]["model_status"] == "UNTRAINED"
    assert meta["EURUSD"]["model_status"] == "UNTRAINED"


def test_load_available_models_loads_trained_artifact(tmp_path: Path):
    X, y, _ = _toy_dataset()
    model = BaselineForecastModel("logistic_regression")
    model.fit(X, y)
    model.save(model_artifact_path("XAUUSD", tmp_path))

    baseline_models, _, meta = load_available_models(("XAUUSD", "EURUSD"), model_dir=tmp_path)
    assert "XAUUSD" in baseline_models
    assert baseline_models["XAUUSD"].is_fitted
    assert meta["XAUUSD"]["model_status"] == "TRAINED"
    assert meta["EURUSD"]["model_status"] == "UNTRAINED"


def test_load_available_models_handles_corrupt_file_gracefully(tmp_path: Path):
    path = model_artifact_path("XAUUSD", tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a real pickle")

    baseline_models, _, meta = load_available_models(("XAUUSD",), model_dir=tmp_path)
    assert "XAUUSD" not in baseline_models
    assert meta["XAUUSD"]["model_status"] == "UNTRAINED"
    assert "error" in meta["XAUUSD"]


def test_load_available_models_loads_pattern_memory(tmp_path: Path):
    X, y, forward_return = _toy_dataset(n=250)
    store = PatternMemoryStore.fit(X, y, forward_return)
    store.save(pattern_memory_artifact_path("XAUUSD", tmp_path))

    _, pattern_memory_stores, _ = load_available_models(("XAUUSD",), model_dir=tmp_path)
    assert "XAUUSD" in pattern_memory_stores


def test_load_score_weights_missing_returns_none(tmp_path: Path):
    assert load_score_weights("XAUUSD", model_dir=tmp_path) is None


def test_load_score_weights_reads_valid_artifact(tmp_path: Path):
    import json

    path = score_weights_artifact_path("XAUUSD", tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"weights": {"forecast_strength": 1.0}, "learned_categories": [], "n_samples": 500}
    with path.open("w") as f:
        json.dump(payload, f)

    loaded = load_score_weights("XAUUSD", model_dir=tmp_path)
    assert loaded == payload


def test_load_score_weights_corrupt_json_returns_none(tmp_path: Path):
    path = score_weights_artifact_path("XAUUSD", tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json")

    assert load_score_weights("XAUUSD", model_dir=tmp_path) is None
