# PSYGRID Forex — Psygnal Psync

A Python forex/metals/oil market-intelligence engine. It is **M5-native**
(no M1 requirement, no fabricated data), builds every technical indicator
and higher timeframe internally, and reasons across structure, liquidity,
volume, volatility, momentum, trend, sessions, cross-market relationships,
macro context, and news context to forecast the most probable direction
over the **next ~60 minutes**, with entry/stop/targets, a signal score, and
a confidence rating — always explained.

This is a forecasting/intelligence system, not a rule-based indicator bot.
No `EMA9 > EMA20 → BUY` logic exists anywhere in this codebase.

## Quick start

```bash
git clone https://github.com/zahidshaikmohammed-cmyk/Psystem-psygnal-psync.git
cd Psystem-psygnal-psync
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python -m psygnal
```

That's it. The engine fetches the live M5 feed, runs the full pipeline,
prints a terminal report per instrument plus a market summary, and writes
a structured JSON file to `output/` (override with `--output-dir`).

Requires Python 3.12+.

### CLI options

```
python -m psygnal --symbols XAUUSD,EURUSD,GBPUSD   # restrict the universe
python -m psygnal --inspect-schema                  # dump the live payload's structure and exit
python -m psygnal --no-macro --no-news              # skip the optional secondary sources
python -m psygnal --no-models                       # ignore data/models/, force deterministic mode
python -m psygnal --model-dir ./my-models           # load trained models from a non-default directory
python -m psygnal --json-only                       # print JSON to stdout instead of the report
python -m psygnal --output-dir ./my-logs
```

If `data/models/` contains a trained artifact for a symbol (see
**Training a model** below), `python -m psygnal` picks it up
automatically — no flag needed. A missing directory, a missing
per-symbol file, or a corrupted pickle all degrade to deterministic mode
for that symbol rather than crashing the run.

## ⚠️ Important: live schema calibration

`psygnal/data/parser.py` talks to the primary endpoint
(`http://140.245.226.102:8080/public/m5-live.json`). **This endpoint was
not reachable from the sandboxed environment this engine was built in**
(plain HTTP to a non-standard port on a raw IP is blocked by that
sandbox's egress policy — confirmed directly, not assumed) — so the exact
JSON schema it returns was never inspected while writing this parser.

The parser is built defensively for exactly this situation: it tries a
handful of structurally-reasonable adapters (symbol-keyed containers,
list-of-symbol-objects, top-level symbol maps, array-form OHLCV rows,
several common field-name spellings for time/OHLCV) against whatever the
endpoint actually returns, and **never fabricates data** if none of them
match — instead it reports `SCHEMA_UNRECOGNIZED` with a structural dump.

If you see that on first run:

```bash
python -m psygnal --inspect-schema
```

This prints a compact structural dump of the real payload (keys, nesting,
array lengths, sample values) without attempting analysis. Compare it
against the adapters in `psygnal/data/parser.py::_ADAPTERS` and add a new
one if needed — each adapter is a short, independent function, so this is
usually a small, mechanical change. Please do not work around a parsing
failure by patching fabricated data in anywhere else in the pipeline;
`DATA UNAVAILABLE` is the correct, honest output when the primary feed
genuinely can't be parsed or is insufficient.

The same caveat applies, to a lesser extent, to the optional macro
calendar (public ForexFactory JSON mirror) and news (GDELT) adapters in
`psygnal/macro/` and `psygnal/news/` — both are free/public/keyless
sources with no official schema guarantee, and both were also unreachable
from the build sandbox (its egress policy blocks essentially all external
hosts outside a small allowlist, including plain HTTP entirely). They are
optional and degrade to `MACRO_STATUS=UNAVAILABLE` /
`NEWS_STATUS=UNAVAILABLE` on any failure — the engine still produces a
complete signal without them, just with reduced context.

## Architecture

```
psygnal/
├── __main__.py          CLI entry point (all I/O lives here); --train delegates to train.py
├── train.py              training CLI entry point: python -m psygnal.train
├── main.py               pure orchestration: raw payload -> FinalSignal[]
├── config.py              tunable parameters, universe, endpoints (no secrets)
├── models.py              shared dataclasses (Candle, FinalSignal, ...)
│
├── data/                  live client, adaptive parser, validation, M5->MTF aggregation, HTTP cache
├── indicators/             EMA/RSI/MACD/ATR/ADX/Bollinger/Stochastic/ROC/volume — all implemented from formulas
├── intelligence/           candle anatomy, sequences, price action, structure, liquidity,
│                           sessions, trend, momentum, volatility, volume, cross-market,
│                           USD composite, Gold specialization, regime, context aggregator
├── macro/                  free public economic calendar adapter
├── news/                   free GDELT news adapter
├── forecasting/            leakage-free features/labels, historical dataset builder,
│                           baseline ML models + calibration, pattern-similarity memory,
│                           ensemble, walk-forward backtest, train_pipeline, score
│                           calibration, model registry (auto-load from data/models/)
├── signal/                 direction, entry, stop, targets, score, confidence, explanation
└── reporting/               terminal report + JSON output

tests/                      167 tests: formulas vs hand-computed references, behavioural
                            classification, leakage guarantees, end-to-end integration,
                            training pipeline / calibration / model registry (V2)
```

## What "intelligence" means here

Every indicator (EMA/RSI/MACD/ATR/ADX/Bollinger/Stochastic/ROC/volume
stats) is implemented directly from its standard formula in
`psygnal/indicators/` — nothing calls out to TradingView, a broker, or an
external indicator API. But indicators are never used as standalone
triggers. They feed into:

- **Sequence intelligence** — rolling 3/6/12/24/48/96-candle analysis of
  what has *just happened* (persistence, acceleration, reversal attempts,
  compression→expansion, expansion→exhaustion).
- **Structure** — swing highs/lows (fractal, confirmation-lagged so no
  future bar can retroactively create a swing), HH/HL/LH/LL trend
  structure, break of structure / change of character, per timeframe
  (M5/M15/M30/H1/H4, aggregated correctly from real M5 candles — no M1
  required or fabricated).
- **Liquidity** — previous-day/session highs & lows, equal highs/lows,
  swing extremes, and sweep/reclaim/failed-breakout/breakout-retest
  detection.
- **Cross-market** — an internal `PSYGRID_USD_COMPOSITE` (explicitly
  labeled as **not** the official DXY) built from the pairs actually
  present in the universe, a risk-sentiment read from GBPJPY, an
  oil/USDCAD relationship check, and Gold-specific synthesis against
  XAGUSD correlation and the USD backdrop.
- **Regime classification** and a **forecast ensemble** that combines a
  deterministic, always-available intelligence-based estimate with an
  optional trained ML baseline and historical pattern-similarity memory
  once enough real historical data exists to support them (see below).

Correlated indicators are deliberately pre-aggregated (e.g. RSI/MACD/ROC
into one momentum score; the four EMAs into one alignment score) before
they enter any weighted combination, so near-duplicate signals don't get
counted as independent confirmations.

## Two forecast modes: DETERMINISTIC vs TRAINED_ML / ENSEMBLE

Every signal reports three things that are easy to conflate but are kept
strictly separate everywhere in the code, the terminal report, and the
JSON output:

- **`signal_score`** — a transparent, weighted quality/confluence score
  (`psygnal/signal/score.py`). Never a probability.
- **`model_probability`** — a genuine statistical estimate from a
  *trained* model, present only when one exists (`model_status ==
  "TRAINED"`) and calibrated on data it never trained on. `None`
  otherwise — never backfilled with a rule-based guess dressed up as a
  probability.
- **`confidence`** / **`confidence_score`** — how strongly the available
  evidence supports the forecast (probability separation, model
  agreement, data completeness, regime clarity). Not an outcome
  guarantee.

`deterministic_forecast` (always present) and `probability_long/short`
(the *operational*, blended forecast actually used for direction/entry)
are reported alongside so a purely rule-based call is never mistaken for
a validated one. `forecast_engine` names which regime produced the
operational forecast:

| `forecast_engine` | Means |
|---|---|
| `DETERMINISTIC` | No trained model for this symbol; rule-based intelligence composite only. |
| `TRAINED_ML` | A trained baseline model contributed. |
| `ENSEMBLE` | The trained model **and** historical pattern-memory both contributed. |

Until you train a model (see below), every symbol runs in
`DETERMINISTIC` / `model_status: UNTRAINED` mode — that is the honest,
correct state for a fresh clone with no historical data, and the engine
says so explicitly in its `warnings`.

## Training a model (PSYGNAL V2)

No historical data ships with this repository — none was fabricated.
Supply genuine historical M5 OHLCV data:

```
data/historical/XAUUSD.csv   (or .json / .parquet)
```

Expected columns: `time, open, high, low, close[, volume]`. Then:

```bash
python -m psygnal.train --symbol XAUUSD
# or equivalently:
python -m psygnal --train --symbol XAUUSD --file data/historical/XAUUSD.csv
```

This runs the full pipeline — **using the exact same causal
feature-engineering function (`forecasting/features.py`) live mode
uses**, so training and inference can never silently diverge:

```
historical M5
    -> causal features (forecasting/features.py — same function as live)
    -> 12-candle future labels, ATR-normalized (forecasting/labels.py)
    -> chronological train / calibration / test split — NEVER shuffled
    -> Logistic Regression vs Random Forest vs HistGradientBoosting,
       selected by OUT-OF-SAMPLE (calibration-slice) balanced accuracy —
       never in-sample training accuracy
    -> probability calibration (isotonic) on the calibration slice
    -> final walk-forward evaluation on a fully held-out test slice
       (balanced accuracy, precision/recall/F1, ROC-AUC, Brier score,
       confusion matrix, reliability curve, R-multiple stats)
    -> data/models/<SYMBOL>_model.pkl
    -> data/models/<SYMBOL>_pattern_memory.pkl
    -> data/models/<SYMBOL>_score_weights.json (optional — see below)
```

`python -m psygnal` then picks these artifacts up automatically on the
next run. The training CLI prints the candidate comparison table, the
selected model, calibration status, the held-out test metrics, and the
confusion matrix — and ends with an explicit reminder that these numbers
describe out-of-sample statistical performance on your dataset, not a
claim of trading profitability.

The (optional) `*_score_weights.json` artifact lets `signal_score`
reallocate part of its weight mass toward whichever evidence categories
historically correlated with favorable outcomes for that symbol
(`scoring_mode: HISTORICALLY_CALIBRATED` instead of the default
`EXPERT_WEIGHTED`) — see `forecasting/score_calibration.py` for exactly
which categories this covers and its documented limitations.

`forecasting/backtest.py::run_walk_forward_backtest` remains available
directly for ad-hoc single-split evaluation without writing artifacts.

## Testing

```bash
pip install -r requirements.txt
pytest -q
```

167 tests cover: adaptive parsing against multiple synthetic payload
shapes, data-integrity validation, M5→M15/M30/H1/H4 aggregation
correctness, every indicator formula against hand-computed reference
values, candle/sequence/price-action classification, structure/liquidity/
session/trend/momentum/volatility/volume engines, cross-market/USD
composite/Gold intelligence, macro/news adapters (mocked HTTP layer),
regime classification, leakage-free feature engineering (explicit
truncation-invariance tests), forecasting models, walk-forward
backtesting, the full signal engine, an end-to-end integration test that
runs the entire pipeline against a synthetic payload and renders both the
terminal report and JSON output, plus (V2) the training pipeline
(chronological no-shuffle split, out-of-sample model selection,
calibration, artifact save/load), the model registry (trained/untrained/
corrupt-artifact fallback, never crashes), forecast-engine mode reporting
(DETERMINISTIC/TRAINED_ML/ENSEMBLE), the training and `--train` CLIs, and
the volatility/price-action-aware entry/stop/target upgrades.

`tests/test_leakage.py` is the dedicated future-leakage suite the
constitution calls for; further leakage checks live next to the code they
guard (e.g. `tests/test_aggregation.py` for the completed-vs-forming
candle boundary).

## Honesty guarantees baked into the design

- `DATA UNAVAILABLE` is reported — never fabricated — when the primary
  M5 feed can't be reached, parsed, or doesn't meet the minimum candle
  depth (`config.MIN_M5_CANDLES_REQUIRED`).
- `signal_score` is a transparent, weighted **quality** composite
  (`psygnal/signal/score.py`), not a win probability.
- `confidence` reflects how much the engine trusts its own read of the
  market (probability separation, model agreement, data completeness,
  regime clarity), not a guarantee of outcome.
- There is **no hard tradeability veto**. Weak edges, poor R:R, or
  conflicting evidence lower the score/confidence — they never withhold a
  LONG/SHORT call once there is sufficient primary data.
- Every signal reports its `conflicts` and `warnings` alongside its
  `reasons`, and every reason is traced to a concrete computed value
  (`psygnal/signal/explain.py`), not templated boilerplate.
- `model_probability` is `None` — not a fabricated number — whenever
  `model_status == "UNTRAINED"`. A rule-based `deterministic_forecast` is
  never relabeled as a "model probability" anywhere in the codebase.
- Training model-selection and calibration always read from a slice the
  model was never fit on; final reported metrics come from a third slice
  neither selection nor calibration ever touched
  (`forecasting/train_pipeline.py`).
