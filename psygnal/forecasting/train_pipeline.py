"""Historical model training pipeline (PSYGNAL V2).

    historical M5
        -> causal feature generation      (forecasting/features.py — the
                                             EXACT function LIVE mode uses)
        -> 12-candle future labels        (forecasting/labels.py)
        -> chronological train/calibration/test split — NEVER shuffled
        -> candidate model comparison, selected by OUT-OF-SAMPLE
           (calibration-slice) balanced accuracy, never in-sample accuracy
        -> probability calibration on the calibration slice
        -> walk-forward validation on a fully held-out test slice
        -> pattern-memory + (optional) score-weight artifacts
        -> data/models/<SYMBOL>_*.pkl / .json

No stage of this pipeline duplicates feature-engineering logic: it calls
`forecasting.dataset.build_training_dataset`, which in turn calls
`forecasting.features.build_feature_frame` — the identical function
`intelligence/context.py` would need to reconstruct at inference time
(the live path builds features implicitly via the intelligence stack;
`psygnal.main.run_engine` reuses `build_feature_frame` directly when
feeding a trained model, see `_build_model_feature_row`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from psygnal import config
from psygnal.forecasting.backtest import evaluate_predictions
from psygnal.forecasting.baseline import CLASSES, BaselineForecastModel
from psygnal.forecasting.calibration import compute_calibration_curve, summarize_calibration_curve
from psygnal.forecasting.dataset import build_training_dataset
from psygnal.forecasting.pattern_memory import PatternMemoryStore
from psygnal.forecasting.registry import (
    model_artifact_path,
    pattern_memory_artifact_path,
    score_weights_artifact_path,
)
from psygnal.forecasting.score_calibration import calibrate_score_weights

DEFAULT_CANDIDATE_MODELS: tuple[str, ...] = ("logistic_regression", "random_forest", "hist_gradient_boosting")
MIN_CALIB_SAMPLES = 30
MIN_TEST_SAMPLES = 30


def _predict_all(
    model: BaselineForecastModel, X: pd.DataFrame, calibrated: bool
) -> tuple[list[str], list[dict[str, float]]]:
    preds: list[str] = []
    probas: list[dict[str, float]] = []
    predict_fn = model.predict_proba if calibrated else model.raw_predict_proba
    for i in range(len(X)):
        proba = predict_fn(X.iloc[i])
        if proba is None:
            proba = {"UP": 0.0, "DOWN": 0.0, "NEUTRAL": 1.0}
        probas.append(proba)
        preds.append(max(proba, key=proba.get))
    return preds, probas


def run_training_pipeline(
    df_m5: pd.DataFrame,
    symbol: str,
    train_fraction: float = 0.6,
    calib_fraction: float = 0.2,
    horizon: int = config.FORECAST_HORIZON_M5_CANDLES,
    threshold_atr_mult: float = 0.5,
    candidate_models: tuple[str, ...] = DEFAULT_CANDIDATE_MODELS,
    model_dir: Path = config.MODEL_DIR,
    calibrate_score: bool = True,
    save_artifacts: bool = True,
) -> dict[str, Any]:
    if not (0 < train_fraction < 1) or not (0 < calib_fraction < 1) or train_fraction + calib_fraction >= 1:
        raise ValueError("train_fraction and calib_fraction must each be in (0, 1) and sum to < 1")

    X, y, meta = build_training_dataset(df_m5, horizon=horizon, threshold_atr_mult=threshold_atr_mult)
    n = len(X)

    # Time-series data is never shuffled: X/y/meta come straight out of a
    # chronologically-sorted DataFrame, and the split below simply slices
    # it — no train_test_split(shuffle=True) anywhere in this pipeline.
    if n > 0:
        assert X.index.is_monotonic_increasing, "training features must remain in chronological order"

    train_end = int(n * train_fraction)
    calib_end = int(n * (train_fraction + calib_fraction))
    n_train, n_calib, n_test = train_end, calib_end - train_end, n - calib_end

    if n_train < config.MIN_HISTORICAL_SAMPLES_FOR_TRAINING or n_calib < MIN_CALIB_SAMPLES or n_test < MIN_TEST_SAMPLES:
        return {
            "status": "INSUFFICIENT_DATA",
            "symbol": symbol,
            "n_samples": n,
            "n_train": n_train,
            "n_calib": n_calib,
            "n_test": n_test,
            "required": {
                "min_train": config.MIN_HISTORICAL_SAMPLES_FOR_TRAINING,
                "min_calib": MIN_CALIB_SAMPLES,
                "min_test": MIN_TEST_SAMPLES,
            },
        }

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_calib, y_calib = X.iloc[train_end:calib_end], y.iloc[train_end:calib_end]
    meta_calib = meta.iloc[train_end:calib_end]
    X_test, y_test = X.iloc[calib_end:], y.iloc[calib_end:]
    meta_test = meta.iloc[calib_end:]

    # --- candidate comparison: fit on TRAIN only, score on CALIB (out-of-sample) ---
    candidate_results: dict[str, Any] = {}
    fitted_candidates: dict[str, BaselineForecastModel] = {}
    for name in candidate_models:
        model = BaselineForecastModel(name)
        model.fit(X_train, y_train)
        preds, probas = _predict_all(model, X_calib, calibrated=False)
        candidate_results[name] = evaluate_predictions(y_calib.to_numpy(), preds, probas, meta_calib)
        fitted_candidates[name] = model

    # Selection is by OUT-OF-SAMPLE (calibration-slice) balanced accuracy —
    # never by in-sample training accuracy.
    best_name = max(candidate_results, key=lambda k: candidate_results[k]["balanced_accuracy"])
    chosen_model = fitted_candidates[best_name]

    # --- calibrate on the calibration slice (never seen during .fit) ---
    calibration_status = chosen_model.calibrate(X_calib, y_calib)

    # --- final evaluation on the fully held-out test slice ---
    preds_test, probas_test = _predict_all(chosen_model, X_test, calibrated=True)
    test_metrics = evaluate_predictions(y_test.to_numpy(), preds_test, probas_test, meta_test)
    confusion = confusion_matrix(y_test.to_numpy(), np.array(preds_test), labels=list(CLASSES)).tolist()

    up_probs_test = np.array([p["UP"] for p in probas_test])
    up_true_test = (y_test.to_numpy() == "UP").astype(int)
    test_reliability = summarize_calibration_curve(compute_calibration_curve(up_probs_test, up_true_test))

    chosen_model.evaluation_metrics = test_metrics
    chosen_model.training_metadata = {
        "symbol": symbol,
        "model_name": best_name,
        "n_train": n_train,
        "n_calib": n_calib,
        "n_test": n_test,
        "train_index_range": (str(X_train.index.min()), str(X_train.index.max())),
        "calib_index_range": (str(X_calib.index.min()), str(X_calib.index.max())),
        "test_index_range": (str(X_test.index.min()), str(X_test.index.max())),
        "candidate_comparison": candidate_results,
        "confusion_matrix": {"labels": list(CLASSES), "matrix": confusion},
        "test_reliability": test_reliability,
        "horizon": horizon,
        "threshold_atr_mult": threshold_atr_mult,
    }

    # --- pattern memory: fit on train+calib, test slice stays untouched ---
    X_fit = pd.concat([X_train, X_calib])
    y_fit = pd.concat([y_train, y_calib])
    forward_return_fit = meta.iloc[:calib_end]["forward_return"]
    pattern_store = PatternMemoryStore.fit(X_fit, y_fit, forward_return_fit)

    score_weights_result = calibrate_score_weights(X_fit, y_fit) if calibrate_score else None

    artifact_paths: dict[str, str] = {}
    if save_artifacts:
        model_path = model_artifact_path(symbol, model_dir)
        chosen_model.save(model_path)
        artifact_paths["model"] = str(model_path)

        pm_path = pattern_memory_artifact_path(symbol, model_dir)
        pattern_store.save(pm_path)
        artifact_paths["pattern_memory"] = str(pm_path)

        if score_weights_result is not None:
            sw_path = score_weights_artifact_path(symbol, model_dir)
            sw_path.parent.mkdir(parents=True, exist_ok=True)
            with sw_path.open("w") as f:
                json.dump(score_weights_result, f, indent=2)
            artifact_paths["score_weights"] = str(sw_path)

    return {
        "status": "OK",
        "symbol": symbol,
        "chosen_model": best_name,
        "n_train": n_train,
        "n_calib": n_calib,
        "n_test": n_test,
        "candidate_comparison": candidate_results,
        "calibration_status": calibration_status,
        "test_metrics": test_metrics,
        "confusion_matrix": {"labels": list(CLASSES), "matrix": confusion},
        "test_reliability": test_reliability,
        "score_weights": score_weights_result,
        "artifact_paths": artifact_paths,
        "train_index_range": (X_train.index.min(), X_train.index.max()),
        "calib_index_range": (X_calib.index.min(), X_calib.index.max()),
        "test_index_range": (X_test.index.min(), X_test.index.max()),
    }
