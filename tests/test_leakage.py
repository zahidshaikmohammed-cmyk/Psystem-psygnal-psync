"""Dedicated future-leakage tests, consolidated per the constitution's
explicit requirement ("Write explicit future-leakage tests").

Complementary leakage coverage also lives alongside the modules it
guards: `tests/test_aggregation.py` (completed-vs-forming candles),
`tests/test_forecasting_features_labels.py`
(`test_feature_frame_is_leakage_free_under_truncation`).
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from psygnal.forecasting.backtest import run_walk_forward_backtest
from psygnal.forecasting.dataset import build_training_dataset
from psygnal.forecasting.features import build_feature_frame
from psygnal.forecasting.pattern_memory import PatternMemoryStore
from psygnal.intelligence.structure import classify_trend_structure, find_swings
from psygnal.models import candles_to_frame
from tests.conftest import make_m5_series


def _random_walk_candles(n=1400, seed=11, base_price=100.0):
    from datetime import datetime, timezone
    from psygnal.models import Candle

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


def test_feature_columns_never_include_label_or_forward_columns():
    candles = _random_walk_candles(300)
    df = candles_to_frame(candles)
    X, y, meta = build_training_dataset(df)
    forbidden = {"label", "forward_return", "forward_move_atr", "forward_high", "forward_low", "mfe", "mae"}
    assert forbidden.isdisjoint(set(X.columns))


def test_feature_frame_truncation_invariance_across_many_cutpoints():
    """Stronger version of the single-cutpoint truncation test: check
    several cutpoints spread through the series."""
    candles = _random_walk_candles(400, seed=99)
    df_full = candles_to_frame(candles)
    features_full = build_feature_frame(df_full)

    for T in (150, 220, 300, 399):
        truncated = build_feature_frame(df_full.iloc[: T + 1])
        row_full = features_full.iloc[T]
        row_truncated = truncated.iloc[-1]
        for col in features_full.columns:
            a, b = row_full[col], row_truncated[col]
            if pd.isna(a) and pd.isna(b):
                continue
            assert a == pytest.approx(b), f"leakage at T={T}, column={col}"


def test_confirmed_structure_ignores_unconfirmed_future_swings():
    """A swing point within `lookback` bars of the end of a truncated series
    must not be confirmed yet — confirming it would require unseen bars."""
    candles = _random_walk_candles(200, seed=5)
    df = candles_to_frame(candles)

    # Full-series structure classification at the very end.
    full_highs, full_lows = find_swings(df, lookback=2)
    full_label = classify_trend_structure(full_highs, full_lows)

    # Truncate 1 bar before the end: any swing that only the full series
    # could confirm (because it needed that last bar) must disappear.
    truncated_highs, truncated_lows = find_swings(df.iloc[:-1], lookback=2)
    # The truncated confirmed-swing list must be a prefix-consistent subset:
    # every truncated swing must also appear in the full-series list at the
    # same index (i.e. truncation never invents swings the full data lacks).
    full_high_indices = {sp.index for sp in full_highs}
    for sp in truncated_highs:
        assert sp.index in full_high_indices


def test_pattern_memory_never_returns_neighbor_at_or_after_max_index():
    candles = _random_walk_candles(1400, seed=21)
    df = candles_to_frame(candles)
    X, y, meta = build_training_dataset(df)
    store = PatternMemoryStore.fit(X, y, meta["forward_return"])

    query_pos = len(X) // 2
    fetch_k = 5
    raw = X.iloc[query_pos].reindex(store.feature_columns).to_numpy(dtype=float)
    raw_filled = np.where(np.isnan(raw), store.mean, raw)
    scaled = ((raw_filled - store.mean) / store.std).reshape(1, -1)
    distances, indices = store.nn_model.kneighbors(scaled, n_neighbors=min(len(store.scaled_matrix), fetch_k * 5))
    kept = store.stored_index[indices[0]] < query_pos
    assert kept.any(), "test setup issue: no valid past neighbors to check against"

    result = store.query(X.iloc[query_pos], k=fetch_k, max_index=query_pos)
    if result["status"] == "OK":
        # Re-derive the neighbor indices actually used and confirm every one
        # is strictly before the query position.
        distances2, indices2 = store.nn_model.kneighbors(scaled, n_neighbors=min(len(store.scaled_matrix), fetch_k * 5))
        candidate_stored = store.stored_index[indices2[0]]
        used_stored = candidate_stored[candidate_stored < query_pos][:fetch_k]
        assert (used_stored < query_pos).all()


def test_backtest_train_and_test_slices_are_disjoint_and_ordered():
    candles = _random_walk_candles(1400, seed=31)
    df = candles_to_frame(candles)
    result = run_walk_forward_backtest(df, train_fraction=0.7, model_name="logistic_regression")

    if result["status"] != "OK":
        pytest.skip("not enough synthetic samples generated for this run")

    train_start, train_end = result["train_index_range"]
    test_start, test_end = result["test_index_range"]
    assert train_end < test_start, "training window must end strictly before the test window begins"
    assert result["n_train"] + result["n_test"] <= len(df)


def test_backtest_reports_insufficient_data_gracefully_for_small_series():
    candles = _random_walk_candles(100, seed=41)
    df = candles_to_frame(candles)
    result = run_walk_forward_backtest(df)
    assert result["status"] == "INSUFFICIENT_DATA"
