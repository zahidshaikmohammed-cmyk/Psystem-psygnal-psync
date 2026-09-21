from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from psygnal.forecasting.dataset import build_training_dataset
from psygnal.forecasting.features import build_feature_frame
from psygnal.forecasting.labels import build_forward_labels
from psygnal.indicators.atr import atr
from psygnal.models import candles_to_frame
from tests.conftest import make_m5_series


def test_build_feature_frame_has_no_nan_at_the_end_of_a_long_series():
    candles = make_m5_series(400, step=0.2)
    df = candles_to_frame(candles)
    features = build_feature_frame(df)
    last_row = features.iloc[-1]
    assert not last_row.isna().any(), last_row[last_row.isna()]


def test_build_feature_frame_early_rows_are_nan_not_fabricated():
    candles = make_m5_series(400, step=0.2)
    df = candles_to_frame(candles)
    features = build_feature_frame(df)
    assert features.iloc[0].isna().any()


def test_labels_up_for_strong_uptrend():
    candles = make_m5_series(200, step=0.5)
    df = candles_to_frame(candles)
    atr_series = atr(df["high"], df["low"], df["close"])
    labels = build_forward_labels(df, atr_series, horizon=12, threshold_atr_mult=0.5)
    mid_labels = labels["label"].iloc[50:150]
    assert (mid_labels == "UP").mean() > 0.8


def test_labels_are_none_within_horizon_of_the_end():
    candles = make_m5_series(100, step=0.2)
    df = candles_to_frame(candles)
    atr_series = atr(df["high"], df["low"], df["close"])
    labels = build_forward_labels(df, atr_series, horizon=12)
    assert labels["label"].iloc[-12:].isna().all() or (labels["label"].iloc[-12:] == None).all()


def test_build_training_dataset_drops_incomplete_rows():
    candles = make_m5_series(300, step=0.3)
    df = candles_to_frame(candles)
    X, y, meta = build_training_dataset(df)
    assert len(X) == len(y) == len(meta)
    assert len(X) < len(df)
    assert not X.isna().any().any()
    assert y.isin(["UP", "DOWN", "NEUTRAL"]).all()


def test_feature_frame_is_leakage_free_under_truncation():
    """Truncating the input series at T must not change the feature row
    computed for T — this is the core leakage guarantee."""
    candles = make_m5_series(300, step=0.15)
    df_full = candles_to_frame(candles)
    T = 250

    features_full = build_feature_frame(df_full)
    features_truncated = build_feature_frame(df_full.iloc[: T + 1])

    row_full = features_full.iloc[T]
    row_truncated = features_truncated.iloc[-1]

    for col in features_full.columns:
        a, b = row_full[col], row_truncated[col]
        if pd.isna(a) and pd.isna(b):
            continue
        assert a == pytest.approx(b), f"leakage detected in column {col}: {a} != {b}"
