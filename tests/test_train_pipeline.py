from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from psygnal.forecasting.registry import (
    model_artifact_path,
    pattern_memory_artifact_path,
    score_weights_artifact_path,
)
from psygnal.forecasting.train_pipeline import run_training_pipeline
from psygnal.models import Candle, candles_to_frame


def _random_walk_candles(n=2500, seed=11, base_price=100.0):
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = []
    price = base_price
    for i in range(n):
        step = rng.normal(0.0, 0.6)
        o, c = price, price + step
        h, l = max(o, c) + abs(rng.normal(0, 0.1)), min(o, c) - abs(rng.normal(0, 0.1))
        candles.append(Candle(time=start + timedelta(minutes=5 * i), open=o, high=h, low=l, close=c, volume=100 + abs(rng.normal(0, 20))))
        price = c
    return candles


def test_training_pipeline_end_to_end(tmp_path: Path):
    df = candles_to_frame(_random_walk_candles(2500, seed=3))
    report = run_training_pipeline(df, "TESTSYM", model_dir=tmp_path)

    assert report["status"] == "OK"
    assert report["chosen_model"] in ("logistic_regression", "random_forest", "hist_gradient_boosting")
    assert model_artifact_path("TESTSYM", tmp_path).exists()
    assert pattern_memory_artifact_path("TESTSYM", tmp_path).exists()
    assert set(report["test_metrics"].keys()) >= {"balanced_accuracy", "precision_macro", "recall_macro", "f1_macro", "brier_score_up"}
    assert report["confusion_matrix"]["labels"] == ["DOWN", "NEUTRAL", "UP"]


def test_training_pipeline_never_shuffles_and_splits_chronologically(tmp_path: Path):
    df = candles_to_frame(_random_walk_candles(2500, seed=4))
    report = run_training_pipeline(df, "TESTSYM", model_dir=tmp_path)
    assert report["status"] == "OK"

    train_start, train_end = report["train_index_range"]
    calib_start, calib_end = report["calib_index_range"]
    test_start, test_end = report["test_index_range"]

    # Strict chronological, non-overlapping ordering: train entirely before
    # calib entirely before test.
    assert train_start < train_end <= calib_start
    assert calib_start < calib_end <= test_start
    assert test_start < test_end


def test_training_pipeline_insufficient_data_reports_gracefully(tmp_path: Path):
    df = candles_to_frame(_random_walk_candles(200, seed=5))
    report = run_training_pipeline(df, "TESTSYM", model_dir=tmp_path)
    assert report["status"] == "INSUFFICIENT_DATA"
    assert "required" in report


def test_training_pipeline_selects_by_out_of_sample_not_in_sample_metric(tmp_path: Path, monkeypatch):
    """The selection line must read the CALIBRATION-slice metric, not
    anything computed on the training slice itself. We assert this
    structurally: forcing every candidate's calibration-slice
    balanced_accuracy to a distinct, known value and checking the pipeline
    picks the one we marked as best — proving selection reads
    `candidate_results[name]["balanced_accuracy"]` from the out-of-sample
    evaluation, not some other source."""
    import psygnal.forecasting.train_pipeline as tp

    df = candles_to_frame(_random_walk_candles(2500, seed=6))

    original_evaluate = tp.evaluate_predictions
    call_count = {"n": 0}
    forced_scores = {0: 0.10, 1: 0.99, 2: 0.20}  # candidate index -> forced balanced_accuracy

    def fake_evaluate(y_true, y_pred, probas, meta):
        result = original_evaluate(y_true, y_pred, probas, meta)
        idx = call_count["n"]
        call_count["n"] += 1
        if idx in forced_scores:
            result = dict(result)
            result["balanced_accuracy"] = forced_scores[idx]
        return result

    monkeypatch.setattr(tp, "evaluate_predictions", fake_evaluate)
    report = run_training_pipeline(
        df,
        "TESTSYM",
        model_dir=tmp_path,
        candidate_models=("logistic_regression", "random_forest", "hist_gradient_boosting"),
    )
    assert report["status"] == "OK"
    # Candidate index 1 (random_forest, second in the tuple) was forced to
    # the highest calibration-slice score -> must be selected.
    assert report["chosen_model"] == "random_forest"


def test_training_pipeline_score_weights_can_be_disabled(tmp_path: Path):
    df = candles_to_frame(_random_walk_candles(2500, seed=7))
    report = run_training_pipeline(df, "TESTSYM", model_dir=tmp_path, calibrate_score=False)
    assert report["status"] == "OK"
    assert report["score_weights"] is None
    assert not score_weights_artifact_path("TESTSYM", tmp_path).exists()


def test_training_pipeline_rejects_bad_fractions():
    import pandas as pd

    with pytest.raises(ValueError):
        run_training_pipeline(pd.DataFrame(), "TESTSYM", train_fraction=0.7, calib_fraction=0.5)
