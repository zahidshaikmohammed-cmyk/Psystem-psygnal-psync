"""Walk-forward backtesting: train on an early slice, evaluate strictly on
a later, disjoint slice. This is the module-level leakage guarantee for
*model evaluation* (as opposed to feature computation, guaranteed instead
by `forecasting/features.py` and tested in `tests/test_leakage.py`):
the training slice `X.iloc[:split]` and test slice `X.iloc[split:]` never
overlap, and the model never sees a test-slice row until after it is
fully fit.

Metrics reported: directional accuracy, balanced accuracy, precision/
recall/F1, ROC-AUC (one-vs-rest), Brier score, average R-multiple, max
drawdown of the resulting R-multiple equity curve, and signal frequency.
None of this is a substitute for genuine historical data — with the
synthetic/small samples this framework will typically see before real
history is loaded, these numbers describe the mechanism working, not
trading edge.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from psygnal import config
from psygnal.forecasting.baseline import BaselineForecastModel
from psygnal.forecasting.dataset import build_training_dataset


def _max_drawdown(equity_curve: np.ndarray) -> float:
    if len(equity_curve) == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = equity_curve - running_max
    return float(drawdown.min())


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: list[str],
    probas: list[dict[str, float]],
    meta_test: pd.DataFrame,
) -> dict[str, Any]:
    y_pred_arr = np.array(y_pred)

    metrics: dict[str, Any] = {}
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred_arr))
    metrics["precision_macro"] = float(precision_score(y_true, y_pred_arr, average="macro", zero_division=0))
    metrics["recall_macro"] = float(recall_score(y_true, y_pred_arr, average="macro", zero_division=0))
    metrics["f1_macro"] = float(f1_score(y_true, y_pred_arr, average="macro", zero_division=0))

    directional_mask = y_true != "NEUTRAL"
    if directional_mask.any():
        metrics["directional_accuracy"] = float((y_pred_arr[directional_mask] == y_true[directional_mask]).mean())
    else:
        metrics["directional_accuracy"] = None

    up_probs = np.array([p.get("UP", 0.0) for p in probas])
    binary_up_true = (y_true == "UP").astype(int)
    if len(set(binary_up_true)) > 1:
        metrics["roc_auc_up_vs_rest"] = float(roc_auc_score(binary_up_true, up_probs))
    else:
        metrics["roc_auc_up_vs_rest"] = None
    metrics["brier_score_up"] = float(brier_score_loss(binary_up_true, up_probs))

    forward_move_atr = meta_test["forward_move_atr"].to_numpy()
    trade_mask = y_pred_arr != "NEUTRAL"
    direction_sign = np.where(y_pred_arr == "UP", 1.0, np.where(y_pred_arr == "DOWN", -1.0, 0.0))
    trade_r = direction_sign[trade_mask] * forward_move_atr[trade_mask]
    trade_r = trade_r[~np.isnan(trade_r)]

    metrics["signal_frequency"] = float(trade_mask.mean())
    metrics["average_r"] = float(np.mean(trade_r)) if len(trade_r) else None
    metrics["max_drawdown_r"] = _max_drawdown(np.cumsum(trade_r)) if len(trade_r) else None
    metrics["n_trades"] = int(len(trade_r))

    metrics["mean_mfe"] = float(np.nanmean(meta_test["mfe"])) if "mfe" in meta_test else None
    metrics["mean_mae"] = float(np.nanmean(meta_test["mae"])) if "mae" in meta_test else None

    # Direction-conditioned, ATR-normalized favorable/adverse excursion —
    # "if a trade were taken in the direction this model actually called,
    # how far did price move for and against it, historically?" Unlike
    # `mean_mfe`/`mean_mae` above (conditioned on the *realized label*),
    # this is conditioned on the *prediction*, which is what a live signal
    # can actually act on, and is what feeds the asymmetric expected-range
    # estimate and target sizing (see forecasting/expected_range.py,
    # signal/targets.py).
    if {"forward_high", "forward_low", "atr_at_t", "close_at_t"}.issubset(meta_test.columns):
        forward_high = meta_test["forward_high"].to_numpy()
        forward_low = meta_test["forward_low"].to_numpy()
        atr_at_t = meta_test["atr_at_t"].to_numpy()
        close_at_t = meta_test["close_at_t"].to_numpy()
        safe_atr = np.where(atr_at_t == 0, np.nan, atr_at_t)

        mfe_atr = np.full(len(y_pred_arr), np.nan)
        mae_atr = np.full(len(y_pred_arr), np.nan)
        up_mask = trade_mask & (y_pred_arr == "UP")
        down_mask = trade_mask & (y_pred_arr == "DOWN")

        mfe_atr[up_mask] = (forward_high[up_mask] - close_at_t[up_mask]) / safe_atr[up_mask]
        mae_atr[up_mask] = (close_at_t[up_mask] - forward_low[up_mask]) / safe_atr[up_mask]
        mfe_atr[down_mask] = (close_at_t[down_mask] - forward_low[down_mask]) / safe_atr[down_mask]
        mae_atr[down_mask] = (forward_high[down_mask] - close_at_t[down_mask]) / safe_atr[down_mask]

        valid_mfe = mfe_atr[~np.isnan(mfe_atr)]
        valid_mae = mae_atr[~np.isnan(mae_atr)]
        metrics["mean_mfe_atr"] = float(np.mean(valid_mfe)) if len(valid_mfe) else None
        metrics["mean_mae_atr"] = float(np.mean(valid_mae)) if len(valid_mae) else None
    else:
        metrics["mean_mfe_atr"] = None
        metrics["mean_mae_atr"] = None

    return metrics


def run_walk_forward_backtest(
    df_m5: pd.DataFrame,
    train_fraction: float = 0.7,
    model_name: str = "hist_gradient_boosting",
    horizon: int = config.FORECAST_HORIZON_M5_CANDLES,
    threshold_atr_mult: float = 0.5,
) -> dict[str, Any]:
    X, y, meta = build_training_dataset(df_m5, horizon=horizon, threshold_atr_mult=threshold_atr_mult)
    n = len(X)

    split = int(n * train_fraction)
    n_test = n - split
    if split < config.MIN_HISTORICAL_SAMPLES_FOR_TRAINING or n_test < 30:
        return {"status": "INSUFFICIENT_DATA", "n_samples": n, "n_train": split, "n_test": n_test}

    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_test, y_test = X.iloc[split:], y.iloc[split:]
    meta_test = meta.iloc[split:]

    model = BaselineForecastModel(model_name)
    training_report = model.fit(X_train, y_train)

    preds: list[str] = []
    probas: list[dict[str, float]] = []
    for i in range(len(X_test)):
        proba = model.predict_proba(X_test.iloc[i])
        probas.append(proba)
        preds.append(max(proba, key=proba.get))

    metrics = evaluate_predictions(y_test.to_numpy(), preds, probas, meta_test)

    return {
        "status": "OK",
        "n_train": split,
        "n_test": n_test,
        "training_report": training_report.as_dict(),
        "metrics": metrics,
        "train_index_range": (X_train.index.min(), X_train.index.max()),
        "test_index_range": (X_test.index.min(), X_test.index.max()),
    }
