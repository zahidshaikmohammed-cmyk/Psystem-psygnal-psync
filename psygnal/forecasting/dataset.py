"""Historical dataset assembly for training/backtesting.

Loads local CSV/JSON/Parquet files (never fabricates history), builds the
same causal feature frame used in LIVE mode, and joins it with
forward-looking labels — dropping only the necessarily-incomplete rows
(warm-up at the start, unfinished horizon at the end).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from psygnal import config
from psygnal.indicators.atr import atr
from psygnal.forecasting.features import build_feature_frame
from psygnal.forecasting.labels import build_forward_labels
from psygnal.models import Candle


def list_available_historical_files(historical_dir: Path = config.HISTORICAL_DIR) -> list[Path]:
    if not historical_dir.exists():
        return []
    exts = (".csv", ".json", ".parquet")
    return sorted(p for p in historical_dir.iterdir() if p.suffix.lower() in exts)


def load_historical_candles(path: Path) -> list[Candle]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".json":
        df = pd.read_json(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"unsupported historical file type: {suffix}")

    df.columns = [c.lower() for c in df.columns]
    required = {"time", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"historical file {path} missing required columns: {missing}")
    if "volume" not in df.columns:
        df["volume"] = 0.0

    df["time"] = pd.to_datetime(df["time"], utc=True)
    candles = [
        Candle(
            time=row["time"].to_pydatetime(),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )
        for _, row in df.iterrows()
    ]
    return candles


def build_training_dataset(
    df_m5: pd.DataFrame,
    horizon: int = config.FORECAST_HORIZON_M5_CANDLES,
    threshold_atr_mult: float = 0.5,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Returns (X, y, label_meta) with rows restricted to those that have
    both complete causal features and a complete forward label."""
    features = build_feature_frame(df_m5)
    atr_series = atr(df_m5["high"], df_m5["low"], df_m5["close"], config.ATR_PERIOD)
    labels = build_forward_labels(df_m5, atr_series, horizon=horizon, threshold_atr_mult=threshold_atr_mult)

    combined = features.join(labels[["label"]], how="inner")
    valid_mask = combined.notna().all(axis=1)
    valid_index = combined.index[valid_mask]

    X = features.loc[valid_index]
    y = labels.loc[valid_index, "label"]
    meta = labels.loc[valid_index]
    return X, y, meta
