"""Interpretable statistical/ML baseline forecasting models.

HistGradientBoostingClassifier is used as the default because it handles
NaN features natively (useful for warm-up periods) and is a strong,
fast, well-validated baseline; Logistic Regression and Random Forest are
also available for comparison. None of this is assumed superior to the
deterministic intelligence fallback a priori — `forecasting/ensemble.py`
only uses a trained model's output when it exists and meets the minimum
sample-size bar.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import balanced_accuracy_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from psygnal import config
from psygnal.forecasting.calibration import (
    MIN_SAMPLES_FOR_CALIBRATION,
    Calibrator,
    compute_calibration_curve,
    summarize_calibration_curve,
)

MODEL_REGISTRY = {
    # Standardized: our causal features span wildly different scales (RSI
    # deviation in [-1,1] vs ADX in [0,100]), which otherwise stalls
    # lbfgs convergence well before max_iter — a numerics artifact, not a
    # modeling choice, so it's fixed at the pipeline level rather than by
    # just raising max_iter and living with the warning.
    "logistic_regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "random_forest": lambda: RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
    "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(max_depth=6, random_state=42),
}

CLASSES = ("DOWN", "NEUTRAL", "UP")


@dataclass
class TrainingReport:
    model_name: str
    n_samples: int
    n_features: int
    cv_balanced_accuracy: list[float]
    cv_brier_score: list[float]
    feature_columns: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "cv_balanced_accuracy": self.cv_balanced_accuracy,
            "cv_brier_score": self.cv_brier_score,
            "feature_columns": self.feature_columns,
        }


class BaselineForecastModel:
    def __init__(self, model_name: str = "hist_gradient_boosting"):
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"unknown model_name {model_name!r}; choices: {list(MODEL_REGISTRY)}")
        self.model_name = model_name
        self.model = MODEL_REGISTRY[model_name]()
        self.feature_columns: Optional[list[str]] = None
        self.is_fitted = False
        # Populated by `calibrate()` — a per-class Platt/isotonic mapping
        # from raw model probability to empirical outcome frequency.
        self.calibrators: dict[str, Calibrator] = {}
        self.calibration_status: dict[str, Any] = {"status": "UNCALIBRATED"}
        # Populated by the training pipeline (forecasting/train_pipeline.py)
        # with the FINAL held-out test-slice metrics — never the in-sample
        # or model-selection-slice numbers. Used to report honest
        # out-of-sample performance and (when available) to derive an
        # asymmetric expected-move estimate from real historical MFE/MAE.
        self.evaluation_metrics: Optional[dict[str, Any]] = None
        self.training_metadata: Optional[dict[str, Any]] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> TrainingReport:
        self.feature_columns = list(X.columns)
        X_arr = X.to_numpy(dtype=float)
        y_arr = y.to_numpy()

        n_splits = min(5, max(2, len(X) // 200))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        balanced_accs: list[float] = []
        briers: list[float] = []

        for train_idx, test_idx in tscv.split(X_arr):
            fold_model = MODEL_REGISTRY[self.model_name]()
            try:
                fold_model.fit(X_arr[train_idx], y_arr[train_idx])
                preds = fold_model.predict(X_arr[test_idx])
                balanced_accs.append(float(balanced_accuracy_score(y_arr[test_idx], preds)))
                proba = fold_model.predict_proba(X_arr[test_idx])
                class_order = list(fold_model.classes_)
                if "UP" in class_order:
                    up_idx = class_order.index("UP")
                    binary_true = (y_arr[test_idx] == "UP").astype(int)
                    briers.append(float(brier_score_loss(binary_true, proba[:, up_idx])))
            except ValueError:
                continue

        self.model.fit(X_arr, y_arr)
        self.is_fitted = True

        return TrainingReport(
            model_name=self.model_name,
            n_samples=len(X),
            n_features=X.shape[1],
            cv_balanced_accuracy=balanced_accs,
            cv_brier_score=briers,
            feature_columns=self.feature_columns,
        )

    def raw_predict_proba(self, x_row: pd.Series) -> Optional[dict[str, float]]:
        """Uncalibrated model output — used internally by `calibrate()` so
        calibration is always fit against the same raw scale it will later
        adjust at inference time."""
        if not self.is_fitted or self.feature_columns is None:
            return None
        x = x_row.reindex(self.feature_columns).to_numpy(dtype=float).reshape(1, -1)
        if np.isnan(x).all():
            return None
        proba = self.model.predict_proba(x)[0]
        class_order = list(self.model.classes_)
        result = {cls: 0.0 for cls in CLASSES}
        for cls, p in zip(class_order, proba):
            result[cls] = float(p)
        return result

    def predict_proba(self, x_row: pd.Series) -> Optional[dict[str, float]]:
        result = self.raw_predict_proba(x_row)
        if result is None:
            return None
        if not self.calibrators:
            return result

        calibrated_up = self.calibrators["UP"].apply(result["UP"]) if "UP" in self.calibrators else result["UP"]
        calibrated_down = self.calibrators["DOWN"].apply(result["DOWN"]) if "DOWN" in self.calibrators else result["DOWN"]
        calibrated_up = float(np.clip(calibrated_up, 0.0, 1.0))
        calibrated_down = float(np.clip(calibrated_down, 0.0, 1.0 - calibrated_up if calibrated_up < 1.0 else 0.0))
        calibrated_neutral = max(0.0, 1.0 - calibrated_up - calibrated_down)

        total = calibrated_up + calibrated_down + calibrated_neutral
        if total <= 0:
            return result
        return {"UP": calibrated_up / total, "DOWN": calibrated_down / total, "NEUTRAL": calibrated_neutral / total}

    def calibrate(self, X_calib: pd.DataFrame, y_calib: pd.Series) -> dict[str, Any]:
        """Fit per-class calibrators on a slice the model was NOT trained
        on. Falls back to leaving `calibrators` empty (uncalibrated) when
        there isn't enough data — see `Calibrator.fit`."""
        if not self.is_fitted:
            self.calibration_status = {"status": "UNCALIBRATED", "reason": "model not fitted"}
            return self.calibration_status

        raw_up: list[float] = []
        raw_down: list[float] = []
        matched_labels: list[str] = []
        y_arr = y_calib.to_numpy()
        for i in range(len(X_calib)):
            proba = self.raw_predict_proba(X_calib.iloc[i])
            if proba is None:
                continue
            raw_up.append(proba["UP"])
            raw_down.append(proba["DOWN"])
            matched_labels.append(y_arr[i])

        n = len(raw_up)
        if n < MIN_SAMPLES_FOR_CALIBRATION:
            self.calibrators = {}
            self.calibration_status = {
                "status": "INSUFFICIENT_DATA",
                "reason": f"only {n} calibration samples available (need >= {MIN_SAMPLES_FOR_CALIBRATION})",
                "n_samples": n,
            }
            return self.calibration_status

        matched_labels_arr = np.array(matched_labels)
        up_true = (matched_labels_arr == "UP").astype(int)
        down_true = (matched_labels_arr == "DOWN").astype(int)
        up_calibrator = Calibrator.fit(np.array(raw_up), up_true)
        down_calibrator = Calibrator.fit(np.array(raw_down), down_true)
        self.calibrators = {"UP": up_calibrator, "DOWN": down_calibrator}

        # Reliability check: after calibration, do predicted probabilities
        # actually behave like the frequencies they claim?
        calibrated_up = np.array([up_calibrator.apply(p) for p in raw_up])
        curve = compute_calibration_curve(calibrated_up, up_true)
        reliability = summarize_calibration_curve(curve)

        self.calibration_status = {
            "status": "CALIBRATED" if (up_calibrator.is_calibrated or down_calibrator.is_calibrated) else "INSUFFICIENT_DATA",
            "n_samples": n,
            "up_calibrated": up_calibrator.is_calibrated,
            "down_calibrated": down_calibrator.is_calibrated,
            "reliability": reliability,
        }
        return self.calibration_status

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "feature_columns": self.feature_columns,
                    "model_name": self.model_name,
                    "calibrators": self.calibrators,
                    "calibration_status": self.calibration_status,
                    "evaluation_metrics": self.evaluation_metrics,
                    "training_metadata": self.training_metadata,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> "BaselineForecastModel":
        with path.open("rb") as f:
            payload = pickle.load(f)
        instance = cls(model_name=payload["model_name"])
        instance.model = payload["model"]
        instance.feature_columns = payload["feature_columns"]
        instance.is_fitted = True
        instance.calibrators = payload.get("calibrators", {})
        instance.calibration_status = payload.get("calibration_status", {"status": "UNCALIBRATED"})
        instance.evaluation_metrics = payload.get("evaluation_metrics")
        instance.training_metadata = payload.get("training_metadata")
        return instance


def can_train(n_samples: int) -> bool:
    return n_samples >= config.MIN_HISTORICAL_SAMPLES_FOR_TRAINING
