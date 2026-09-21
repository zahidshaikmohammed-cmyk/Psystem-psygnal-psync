"""CLI entry point: `python -m psygnal`.

Fetches the live M5 feed, runs the full engine, and prints a terminal
report plus writes structured JSON. Any I/O (network fetch, stdout,
filesystem) lives here — `psygnal.main.run_engine` itself is pure.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from psygnal import config
from psygnal.data.live_client import fetch_live_m5
from psygnal.data.parser import describe_schema
from psygnal.forecasting.registry import load_available_models
from psygnal.main import run_engine
from psygnal.reporting.json import build_json_output, default_output_path, write_json_output
from psygnal.reporting.terminal import (
    format_data_unavailable,
    format_header,
    format_market_summary,
    format_symbol_report,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="psygnal", description="PSYGRID Forex Psygnal Psync engine")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated symbol override, e.g. XAUUSD,EURUSD")
    parser.add_argument("--endpoint", type=str, default=config.LIVE_M5_ENDPOINT, help="Override the live M5 endpoint URL")
    parser.add_argument("--inspect-schema", action="store_true", help="Fetch the live payload and print its structure, then exit")
    parser.add_argument("--no-macro", action="store_true", help="Disable the macro-calendar adapter")
    parser.add_argument("--no-news", action="store_true", help="Disable the news adapter")
    parser.add_argument("--no-models", action="store_true", help="Ignore any trained models under data/models/ and force deterministic mode")
    parser.add_argument("--model-dir", type=str, default=str(config.MODEL_DIR), help="Directory to load trained models from")
    parser.add_argument("--json-only", action="store_true", help="Print JSON to stdout instead of the terminal report")
    parser.add_argument("--output-dir", type=str, default=str(config.CACHE_DIR.parent / "output"), help="Directory for the JSON output file")
    parser.add_argument("--train", action="store_true", help="Train a model instead of running live (see: python -m psygnal.train --help)")
    return parser


def _run_train(argv: list[str] | None) -> int:
    from psygnal.train import main as train_main

    # Strip --train itself out before delegating the remaining args.
    forwarded = [a for a in (argv if argv is not None else sys.argv[1:]) if a != "--train"]
    return train_main(forwarded)


def main(argv: list[str] | None = None) -> int:
    if (argv is not None and "--train" in argv) or (argv is None and "--train" in sys.argv[1:]):
        return _run_train(argv)

    args = build_arg_parser().parse_args(argv)

    fetch_result = fetch_live_m5(url=args.endpoint)
    if not fetch_result.ok:
        print("=" * 60)
        print("DATA UNAVAILABLE / INSUFFICIENT DATA")
        print("=" * 60)
        print(f"\nCould not reach the live endpoint: {args.endpoint}")
        print(f"Error: {fetch_result.error}")
        print("\nNo analysis can be produced without primary market data.")
        return 1

    if args.inspect_schema:
        print(describe_schema(fetch_result.raw))
        return 0

    symbols = tuple(s.strip().upper() for s in args.symbols.split(",")) if args.symbols else config.DEFAULT_SYMBOL_UNIVERSE
    cfg = config.EngineConfig(
        symbols=symbols,
        enable_macro=not args.no_macro,
        enable_news=not args.no_news,
        model_dir=Path(args.model_dir),
        enable_models=not args.no_models,
    )

    baseline_models: dict = {}
    pattern_memory_stores: dict = {}
    if not args.no_models:
        # Never crashes: a missing data/models/ directory, a missing
        # per-symbol artifact, or a corrupt pickle all just mean that
        # symbol runs in deterministic mode.
        baseline_models, pattern_memory_stores, _model_meta = load_available_models(symbols, model_dir=Path(args.model_dir))

    now_utc = datetime.now(timezone.utc)
    outcome = run_engine(
        fetch_result.raw,
        cfg=cfg,
        now_utc=now_utc,
        baseline_models=baseline_models,
        pattern_memory_stores=pattern_memory_stores,
    )

    if outcome["status"] == "SCHEMA_UNRECOGNIZED":
        print("=" * 60)
        print("SCHEMA_UNRECOGNIZED")
        print("=" * 60)
        print(f"\n{outcome['error']}\n")
        print("The live endpoint returned a JSON shape no shipped parser")
        print("adapter recognizes. Structural dump of the response:\n")
        print(outcome["schema_dump"])
        print("\nSee psygnal/data/parser.py for how to add a new adapter.")
        return 2

    signals = [s for s, _ in outcome["results"]]
    intel_map = {s.symbol: intel for s, intel in outcome["results"]}

    output = build_json_output(signals, outcome["unavailable"], now_utc)

    output_dir = Path(args.output_dir)
    json_path = write_json_output(output, default_output_path(output_dir, now_utc))

    if args.json_only:
        import json

        print(json.dumps(output, indent=2, default=str))
        return 0

    print(format_header(now_utc))
    for sym, quality in outcome["unavailable"].items():
        print()
        print(format_data_unavailable(sym, quality))

    for signal in signals:
        print()
        print(format_symbol_report(signal, intel_map[signal.symbol]))

    print()
    print(format_market_summary(signals))
    print(f"\nJSON output written to: {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
