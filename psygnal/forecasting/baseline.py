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

from psygnal import config

MODEL_REGISTRY = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000),
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

    def predict_proba(self, x_row: pd.Series) -> Optional[dict[str, float]]:
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

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"model": self.model, "feature_columns": self.feature_columns, "model_name": self.model_name}, f)

    @classmethod
    def load(cls, path: Path) -> "BaselineForecastModel":
        with path.open("rb") as f:
            payload = pickle.load(f)
        instance = cls(model_name=payload["model_name"])
        instance.model = payload["model"]
        instance.feature_columns = payload["feature_columns"]
        instance.is_fitted = True
        return instance


def can_train(n_samples: int) -> bool:
    return n_samples >= config.MIN_HISTORICAL_SAMPLES_FOR_TRAINING
