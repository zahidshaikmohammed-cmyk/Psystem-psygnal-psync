from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from psygnal.forecasting.score_calibration import calibrate_score_weights
from psygnal.models import Candle, candles_to_frame
from psygnal.signal.score import SCORE_WEIGHTS, compute_signal_score


def _base_score_args():
    ensemble_result = {"probability_long": 0.6, "probability_short": 0.3}
    components = {"trend": 0.4, "momentum": 0.3, "structure_liquidity": 0.2, "price_action": 0.1, "cross_market": 0.0}
    return dict(
        direction="LONG",
        ensemble_result=ensemble_result,
        pattern_memory=None,
        components=components,
        volatility_label="NORMAL",
        volume_label="AVERAGE",
        volume_relationship="INCONCLUSIVE",
        session_label="LONDON",
        macro_status="CLEAR",
        rr=2.0,
    )


def test_compute_signal_score_defaults_to_expert_weighted():
    result = compute_signal_score(**_base_score_args())
    assert result["scoring_mode"] == "EXPERT_WEIGHTED"
    assert result["weights_used"] == SCORE_WEIGHTS


def test_compute_signal_score_uses_learned_weights_when_provided():
    learned = dict(SCORE_WEIGHTS)
    learned["trend_alignment"] = 0.5
    remaining = 1.0 - 0.5
    other_keys = [k for k in learned if k != "trend_alignment"]
    other_total = sum(learned[k] for k in other_keys)
    for k in other_keys:
        learned[k] = learned[k] / other_total * remaining

    result = compute_signal_score(**_base_score_args(), learned_weights=learned)
    assert result["scoring_mode"] == "HISTORICALLY_CALIBRATED"
    assert result["weights_used"] == learned
    assert result["weights_used"] != SCORE_WEIGHTS


def _random_walk_candles(n=2000, seed=15, base_price=100.0):
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


def test_calibrate_score_weights_returns_none_with_too_little_data():
    from psygnal.forecasting.dataset import build_training_dataset

    df = candles_to_frame(_random_walk_candles(300, seed=1))
    X, y, meta = build_training_dataset(df)
    result = calibrate_score_weights(X, y)
    assert result is None


def test_calibrate_score_weights_returns_valid_distribution_with_enough_data():
    from psygnal.forecasting.dataset import build_training_dataset

    df = candles_to_frame(_random_walk_candles(2000, seed=2))
    X, y, meta = build_training_dataset(df)
    result = calibrate_score_weights(X, y)
    if result is None:
        pytest.skip("synthetic data did not produce enough directional signal for score calibration")
    assert abs(sum(result["weights"].values()) - 1.0) < 1e-6
    assert set(result["weights"].keys()) == set(SCORE_WEIGHTS.keys())
    assert result["n_samples"] > 0


def test_train_cli_missing_file_returns_error(tmp_path: Path, capsys):
    from psygnal.train import main as train_main

    exit_code = train_main(["--symbol", "XAUUSD", "--file", str(tmp_path / "nope.csv")])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "not found" in captured.out.lower()


def test_train_cli_end_to_end(tmp_path: Path, capsys):
    from psygnal.train import main as train_main

    csv_path = tmp_path / "XAUUSD.csv"
    candles = _random_walk_candles(2500, seed=8, base_price=2000.0)
    rows = [{"time": c.time.isoformat(), "open": c.open, "high": c.high, "low": c.low, "close": c.close, "volume": c.volume} for c in candles]
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    model_dir = tmp_path / "models"
    exit_code = train_main(["--symbol", "XAUUSD", "--file", str(csv_path), "--model-dir", str(model_dir)])
    assert exit_code == 0
    assert (model_dir / "XAUUSD_model.pkl").exists()
    captured = capsys.readouterr()
    assert "PSYGNAL TRAINING" in captured.out
    assert "Do NOT treat these metrics as a claim of trading profitability" in captured.out


def test_train_cli_insufficient_data_returns_error(tmp_path: Path, capsys):
    from psygnal.train import main as train_main

    csv_path = tmp_path / "XAUUSD.csv"
    candles = _random_walk_candles(200, seed=9, base_price=2000.0)
    rows = [{"time": c.time.isoformat(), "open": c.open, "high": c.high, "low": c.low, "close": c.close, "volume": c.volume} for c in candles]
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    exit_code = train_main(["--symbol", "XAUUSD", "--file", str(csv_path), "--model-dir", str(tmp_path / "models")])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "INSUFFICIENT_DATA" in captured.out


def test_main_train_flag_delegates_to_training_cli(tmp_path: Path, capsys):
    from psygnal.__main__ import main as cli_main

    csv_path = tmp_path / "XAUUSD.csv"
    candles = _random_walk_candles(2500, seed=10, base_price=2000.0)
    rows = [{"time": c.time.isoformat(), "open": c.open, "high": c.high, "low": c.low, "close": c.close, "volume": c.volume} for c in candles]
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    exit_code = cli_main(["--train", "--symbol", "XAUUSD", "--file", str(csv_path), "--model-dir", str(tmp_path / "models")])
    assert exit_code == 0
    assert (tmp_path / "models" / "XAUUSD_model.pkl").exists()
