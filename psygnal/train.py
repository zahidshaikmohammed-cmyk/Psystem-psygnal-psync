"""Training CLI: `python -m psygnal.train`.

Trains a forecasting model for one symbol from genuine local historical
M5 OHLCV data (CSV/JSON/Parquet — see `forecasting/dataset.py`), using
the exact same feature-engineering path as live mode, and writes trained
artifacts to `data/models/` where `python -m psygnal` will automatically
pick them up on the next run.

Example:
    python -m psygnal.train --symbol XAUUSD
    python -m psygnal.train --symbol XAUUSD --file data/historical/XAUUSD.csv
    python -m psygnal.train --symbol XAUUSD --models logistic_regression,random_forest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from psygnal import config
from psygnal.forecasting.dataset import load_historical_candles
from psygnal.forecasting.train_pipeline import DEFAULT_CANDIDATE_MODELS, run_training_pipeline
from psygnal.models import candles_to_frame


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="psygnal.train", description="Train a PSYGNAL forecasting model from historical M5 data")
    parser.add_argument("--symbol", type=str, required=True, help="Symbol to train, e.g. XAUUSD")
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Historical M5 OHLCV file (.csv/.json/.parquet). Defaults to data/historical/<SYMBOL>.csv",
    )
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--calib-fraction", type=float, default=0.2)
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_CANDIDATE_MODELS),
        help="Comma-separated candidate models to compare: logistic_regression,random_forest,hist_gradient_boosting",
    )
    parser.add_argument("--no-score-calibration", action="store_true", help="Skip the (optional) signal-score weight calibration")
    parser.add_argument("--model-dir", type=str, default=str(config.MODEL_DIR), help="Where to write trained artifacts")
    return parser


def _print_report(report: dict) -> None:
    print("=" * 60)
    print(f"PSYGNAL TRAINING — {report.get('symbol')}")
    print("=" * 60)

    if report["status"] == "INSUFFICIENT_DATA":
        print("\nSTATUS: INSUFFICIENT_DATA")
        print(f"  samples available: {report['n_samples']}")
        print(f"  train/calib/test split would be: {report['n_train']}/{report['n_calib']}/{report['n_test']}")
        req = report.get("required", {})
        print(f"  minimums required: train>={req.get('min_train')}, calib>={req.get('min_calib')}, test>={req.get('min_test')}")
        print("\nSupply more historical M5 candles and try again.")
        return

    print(f"\nSamples: train={report['n_train']}  calib={report['n_calib']}  test={report['n_test']}")
    print("Chronological ranges (no shuffling):")
    print(f"  train: {report['train_index_range'][0]} -> {report['train_index_range'][1]}")
    print(f"  calib: {report['calib_index_range'][0]} -> {report['calib_index_range'][1]}")
    print(f"  test:  {report['test_index_range'][0]} -> {report['test_index_range'][1]}")

    print("\nCandidate comparison (out-of-sample, calibration slice):")
    for name, metrics in report["candidate_comparison"].items():
        marker = " <-- selected" if name == report["chosen_model"] else ""
        print(
            f"  {name:<24} balanced_acc={metrics['balanced_accuracy']:.3f}  "
            f"f1_macro={metrics['f1_macro']:.3f}  brier_up={metrics['brier_score_up']:.3f}{marker}"
        )

    print(f"\nCalibration: {report['calibration_status'].get('status')}")
    reliability = report.get("test_reliability", {})
    print(f"Held-out reliability check: {reliability.get('status')} (max deviation: {reliability.get('max_deviation')})")

    print("\nFinal held-out TEST metrics (never used for selection or calibration):")
    for key, value in report["test_metrics"].items():
        print(f"  {key}: {value}")

    print(f"\nConfusion matrix (labels={report['confusion_matrix']['labels']}):")
    for row in report["confusion_matrix"]["matrix"]:
        print(f"  {row}")

    if report.get("score_weights"):
        print(f"\nSignal-score weights calibrated from {report['score_weights']['n_samples']} samples")
        print(f"  learned categories: {report['score_weights']['learned_categories']}")
    else:
        print("\nSignal-score weights: not calibrated (insufficient data or skipped) — V1 expert weights remain in effect")

    print("\nArtifacts written:")
    for kind, path in report["artifact_paths"].items():
        print(f"  {kind}: {path}")

    print(f"\n{'=' * 60}")
    print("Do NOT treat these metrics as a claim of trading profitability.")
    print("They describe out-of-sample statistical performance on this")
    print("dataset only, evaluated exactly once on data none of the")
    print("training/selection/calibration steps ever touched.")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    symbol = args.symbol.strip().upper()

    file_path = Path(args.file) if args.file else config.HISTORICAL_DIR / f"{symbol}.csv"
    if not file_path.exists():
        print(f"Historical file not found: {file_path}")
        print("Expected columns: time, open, high, low, close[, volume]")
        return 1

    try:
        candles = load_historical_candles(file_path)
    except (ValueError, OSError) as exc:
        print(f"Failed to load historical file {file_path}: {exc}")
        return 1

    df = candles_to_frame(candles)
    if df.empty:
        print(f"No valid candles parsed from {file_path}")
        return 1

    candidate_models = tuple(m.strip() for m in args.models.split(",") if m.strip())

    report = run_training_pipeline(
        df,
        symbol,
        train_fraction=args.train_fraction,
        calib_fraction=args.calib_fraction,
        candidate_models=candidate_models,
        model_dir=Path(args.model_dir),
        calibrate_score=not args.no_score_calibration,
    )
    _print_report(report)
    return 0 if report["status"] == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
