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
python -m psygnal --json-only                       # print JSON to stdout instead of the report
python -m psygnal --output-dir ./my-logs
```

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
├── __main__.py          CLI entry point (all I/O lives here)
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
│                           baseline ML models, pattern-similarity memory, ensemble
├── signal/                 direction, entry, stop, targets, score, confidence, explanation
└── reporting/               terminal report + JSON output

tests/                      119 tests: formulas vs hand-computed references, behavioural
                            classification, leakage guarantees, end-to-end integration
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

## Historical data, training, and backtesting

No historical data ships with this repository — none was fabricated. The
forecasting layer runs a fully honest **deterministic mode** until real
history is supplied:

```
data/historical/your_file.csv   (or .json / .parquet)
```

Expected columns: `time, open, high, low, close[, volume]`. Then, from
Python:

```python
from psygnal.forecasting.dataset import load_historical_candles, build_training_dataset
from psygnal.forecasting.backtest import run_walk_forward_backtest
from psygnal.models import candles_to_frame

candles = load_historical_candles(Path("data/historical/your_file.csv"))
df = candles_to_frame(candles)
report = run_walk_forward_backtest(df)   # trains on an early slice, evaluates on a later disjoint slice
```

`forecasting/baseline.py` offers Logistic Regression, Random Forest, and
HistGradientBoosting; `forecasting/pattern_memory.py` does leakage-safe
k-NN lookup against historical states; `forecasting/ensemble.py` blends
whichever of these are actually available with the deterministic estimate
— it never blocks on a missing model, and it never claims a learned
probability before a model has been trained and validated
out-of-sample. `MIN_HISTORICAL_SAMPLES_FOR_TRAINING` (`config.py`) gates
when the baseline model is even attempted.

## Testing

```bash
pip install -r requirements.txt
pytest -q
```

119 tests cover: adaptive parsing against multiple synthetic payload
shapes, data-integrity validation, M5→M15/M30/H1/H4 aggregation
correctness, every indicator formula against hand-computed reference
values, candle/sequence/price-action classification, structure/liquidity/
session/trend/momentum/volatility/volume engines, cross-market/USD
composite/Gold intelligence, macro/news adapters (mocked HTTP layer),
regime classification, leakage-free feature engineering (explicit
truncation-invariance tests), forecasting models, walk-forward
backtesting, the full signal engine, and an end-to-end integration test
that runs the entire pipeline against a synthetic payload and renders
both the terminal report and JSON output.

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
