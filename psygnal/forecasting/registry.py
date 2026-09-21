"""Automatic discovery/loading of trained model artifacts from
`data/models/`.

Used by both the live CLI (`psygnal/__main__.py`) and tests. Loading must
NEVER crash the engine: a missing directory, a missing per-symbol file, or
a corrupt/incompatible pickle all degrade to "no model for this symbol"
(UNTRAINED) rather than raising.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from psygnal import config
from psygnal.forecasting.baseline import BaselineForecastModel
from psygnal.forecasting.pattern_memory import PatternMemoryStore


def model_artifact_path(symbol: str, model_dir: Path = config.MODEL_DIR) -> Path:
    return model_dir / f"{symbol}_model.pkl"


def pattern_memory_artifact_path(symbol: str, model_dir: Path = config.MODEL_DIR) -> Path:
    return model_dir / f"{symbol}_pattern_memory.pkl"


def score_weights_artifact_path(symbol: str, model_dir: Path = config.MODEL_DIR) -> Path:
    return model_dir / f"{symbol}_score_weights.json"


def load_available_models(
    symbols: tuple[str, ...], model_dir: Path = config.MODEL_DIR
) -> tuple[dict[str, BaselineForecastModel], dict[str, PatternMemoryStore], dict[str, dict[str, Any]]]:
    """Returns (baseline_models, pattern_memory_stores, model_meta).

    `model_meta[symbol]` always has a `"model_status"` key
    (`"TRAINED"` or `"UNTRAINED"`), plus an `"error"` key when a model file
    existed but failed to load.
    """
    baseline_models: dict[str, BaselineForecastModel] = {}
    pattern_memory_stores: dict[str, PatternMemoryStore] = {}
    model_meta: dict[str, dict[str, Any]] = {}

    for sym in symbols:
        meta: dict[str, Any] = {"model_status": "UNTRAINED"}

        model_path = model_artifact_path(sym, model_dir)
        if model_path.exists():
            try:
                model = BaselineForecastModel.load(model_path)
                if model.is_fitted:
                    baseline_models[sym] = model
                    meta["model_status"] = "TRAINED"
                    meta["model_name"] = model.model_name
                    meta["calibration_status"] = model.calibration_status
                    meta["training_metadata"] = model.training_metadata
                else:
                    meta["error"] = "model artifact loaded but is not fitted"
            except Exception as exc:  # never let a corrupt artifact crash the CLI
                meta["error"] = f"failed to load model artifact: {type(exc).__name__}: {exc}"

        pm_path = pattern_memory_artifact_path(sym, model_dir)
        if pm_path.exists():
            try:
                pattern_memory_stores[sym] = PatternMemoryStore.load(pm_path)
            except Exception as exc:
                meta.setdefault("pattern_memory_error", f"failed to load pattern memory artifact: {type(exc).__name__}: {exc}")

        model_meta[sym] = meta

    return baseline_models, pattern_memory_stores, model_meta


def load_score_weights(symbol: str, model_dir: Path = config.MODEL_DIR) -> Optional[dict[str, Any]]:
    """Best-effort load of a calibrated signal-score weight artifact.
    Returns None (never raises) if absent or corrupt — the caller falls
    back to the V1 expert weights."""
    path = score_weights_artifact_path(symbol, model_dir)
    if not path.exists():
        return None
    try:
        with path.open("r") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None
