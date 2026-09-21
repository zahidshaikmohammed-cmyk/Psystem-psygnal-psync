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

**V3** adds a human-like reasoning layer on top of the V1/V2 technical
engine: economic-event awareness (proximity, surprise, macro-transmission
theory vs. observed reaction), news-theme interpretation, cross-market
confirmation/contradiction classification, market-shock detection, an
expanded regime taxonomy (`BREAKDOWN`/`POST_NEWS`/`TRANSITION` alongside
the V1 regimes), a hypothesis/evidence/invalidation reasoning layer, and a
first-class **`NO_TRADE`-capable `tradeability`** decision — see
[Human-like reasoning (V3)](#human-like-reasoning-v3) below, including why
this deliberately supersedes V1's original "always call LONG/SHORT, never
withhold" rule.

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
python -m psygnal --daily-brief                      # also print today's daily market brief (data/daily/)
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
├── models.py              shared dataclasses/enums (Candle, FinalSignal, Tradeability, ...)
│
├── data/                  live client, adaptive parser, validation, M5->MTF aggregation, HTTP cache
├── indicators/             EMA/RSI/MACD/ATR/ADX/Bollinger/Stochastic/ROC/volume — all implemented from formulas
├── intelligence/           candle anatomy, sequences, price action, structure, liquidity,
│                           sessions, trend, momentum, volatility, volume, cross-market,
│                           USD composite, Gold specialization, regime, context aggregator,
│                           (V3) shock detection, cross-market confirmation, contradiction
│                           engine, human-like reasoning/hypothesis/tradeability
├── macro/                  (V3) economic calendar adapter, event categories, surprise
│                           calculation, macro-transmission hypothesis, event proximity
│                           state machine, provider interfaces (CalendarProvider/
│                           MacroDataProvider/RatesProvider)
├── news/                   free GDELT news adapter + (V3) theme interpretation/reconciliation
├── memory/                 (V3) daily market memory — data/daily/YYYY-MM-DD.json journal
├── forecasting/            leakage-free features/labels, historical dataset builder,
│                           baseline ML models + calibration, pattern-similarity memory,
│                           ensemble, walk-forward backtest, train_pipeline, score
│                           calibration, model registry (auto-load from data/models/),
│                           (V3) historical event-reaction memory
├── signal/                 direction, entry, stop, targets, score, confidence, explanation
└── reporting/               narrative terminal report, JSON output, (V3) daily brief

tests/                      255 tests: formulas vs hand-computed references, behavioural
                            classification, leakage guarantees (including dedicated
                            economic-event leakage tests), end-to-end integration,
                            training pipeline / calibration / model registry (V2), and
                            (V3) event engine, news interpretation, shock detection,
                            cross-market confirmation, contradiction engine, reasoning/
                            tradeability, daily memory, historical event-reaction memory
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

255 tests cover: adaptive parsing against multiple synthetic payload
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

(V3) further coverage: the economic-event engine (category classification,
surprise calculation, proximity buckets, macro-transmission hypothesis),
a **dedicated event-leakage suite** (`tests/test_event_leakage.py`) proving
an event's `actual` value — even a maliciously pre-populated future one —
never leaks into a context built before its release time; news-theme
interpretation and reconciliation; market-shock detection (requiring
≥2 independent triggers, never a single wick); cross-market
confirmation/contradiction classification; the contradiction engine's full
8-point checklist tested in isolation
(`tests/test_contradiction.py`); the expanded regime taxonomy
(`BREAKDOWN`/`POST_NEWS`/`TRANSITION`); the human-like reasoning layer
(hypothesis construction, tradeability cascade, invalidation-condition
direction correctness — LONG invalidated by losing support, SHORT by
losing resistance, never the reverse); daily market memory (round-trip,
corrupt-file recovery, regime-transition logging, display-timezone file
naming); and historical event-reaction memory (honest
`INSUFFICIENT_HISTORICAL_SAMPLE` reporting below the minimum sample size).

`tests/test_leakage.py` is the dedicated future-leakage suite the
constitution calls for; further leakage checks live next to the code they
guard (e.g. `tests/test_aggregation.py` for the completed-vs-forming
candle boundary, `tests/test_event_leakage.py` for the economic-event
engine).

## Honesty guarantees baked into the design

- `DATA UNAVAILABLE` is reported — never fabricated — when the primary
  M5 feed can't be reached, parsed, or doesn't meet the minimum candle
  depth (`config.MIN_M5_CANDLES_REQUIRED`).
- `signal_score` is a transparent, weighted **quality** composite
  (`psygnal/signal/score.py`), not a win probability.
- `confidence` reflects how much the engine trusts its own read of the
  market (probability separation, model agreement, data completeness,
  regime clarity), not a guarantee of outcome.
- **V1/V2 note (superseded by V3):** earlier versions of this engine
  guaranteed no hard tradeability veto — a LONG/SHORT direction was always
  produced whenever there was sufficient primary data. **V3 deliberately
  reverses this** with the `tradeability` decision (`TRADEABLE` / `WAIT` /
  `NOT_TRADEABLE`) — see
  [Human-like reasoning (V3)](#human-like-reasoning-v3) for the full
  rationale. `direction` (LONG/SHORT) is still always computed and
  reported for full transparency; `tradeability` is a separate,
  orthogonal field layered on top, so nothing about the original
  direction/entry/stop/target machinery changed or lost information.
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

## Human-like reasoning (V3)

V1/V2 built a genuinely deep *technical* intelligence engine, but it was
still fundamentally a signal bot: compute everything, blend it into a
score, always announce LONG or SHORT. V3 adds a reasoning layer on top
that behaves more like an experienced discretionary trader — it forms a
hypothesis, actively looks for evidence against its own hypothesis, tracks
what would prove it wrong, and is willing to say "I'm not trading this"
when the evidence doesn't support acting. None of the V1/V2 technical
engine was removed or weakened; V3 is additive.

### Why `NO_TRADE` supersedes the original "always LONG/SHORT" rule

The original V1 constitution explicitly forbade a tradeability veto: the
engine had to always pick LONG or SHORT once there was enough primary
data, on the reasoning that withholding a call was itself a form of
dishonesty about what the technical evidence showed. That reasoning holds
for the *technical* read alone — but a real discretionary trader doesn't
act on every technical read. They stand aside around high-impact news,
when cross-market evidence contradicts the technical picture, when the
regime is genuinely chaotic, or when risk/reward doesn't justify the
trade. Refusing to ever say "not now" was itself a distortion once the
engine started reasoning about *more than* raw technicals.

V3 resolves this without deleting the original guarantee: `direction`
(LONG/SHORT) is **still always computed** from the same technical
machinery as before, with full entry/stop/target/score detail — nothing
about that pipeline changed. A new, independent field,
`tradeability` (`TRADEABLE` / `WAIT` / `NOT_TRADEABLE`,
`psygnal.models.Tradeability`), is layered on top by
`psygnal/intelligence/reasoning.py::assess_tradeability`, driven by
regime, event proximity, cross-market/contradiction evidence, and R:R.
Both fields are always present in every signal — a consumer who wants the
old always-a-direction behavior can ignore `tradeability` entirely; a
consumer who wants human-like discretion reads both.

### Economic-event intelligence (`psygnal/macro/`)

- `calendar.py` / `sources.py` — free public ForexFactory-mirror calendar
  adapter (unchanged from V2), now also carrying each event's `actual`
  value when the source reports one.
- `categories.py` — keyword-based `MacroCategory` classification
  (INFLATION/EMPLOYMENT/GROWTH/MONETARY_POLICY/...).
- `surprise.py` — `actual - forecast`, normalized by a documented,
  **explicitly heuristic** per-category scale. Returns `None` (never a
  fabricated `0.0`) whenever either input is missing.
- `transmission.py` — a `TransmissionHypothesis`: what a textbook would
  theoretically expect the USD to do given this category and surprise
  sign. Confidence is always `THEORETICAL`/`LOW`/`UNAVAILABLE` — **never**
  `HIGH` — because this is a labeled prior for comparison, not a claim
  about actual market behavior.
- `event_engine.py` — the proximity state machine
  (`PRE_EVENT`/`AT_EVENT`/`POST_EVENT`, T-60..T+60 buckets) and the
  leakage guard: `build_event_context()` only ever reacts to an event
  whose `time_utc <= now_utc` **and** which already has a reported
  `actual` value — proven by a dedicated leakage suite
  (`tests/test_event_leakage.py`), including a test where an `actual`
  value is maliciously pre-populated on a future event and confirmed to
  never leak into a context built before that event's release time.
- `providers.py` — provider-agnostic `Protocol` interfaces
  (`CalendarProvider`, `NewsProvider`, `MacroDataProvider`,
  `RatesProvider`) so a real macro-data or rates feed can be plugged in
  later **without changing the intelligence engine**. `MacroDataProvider`
  and `RatesProvider` currently ship only as `UnavailableMacroDataProvider`
  / `UnavailableRatesProvider` stubs — see
  [Data sources](#data-sources--what-is-real-vs-derived-vs-unavailable)
  below for exactly why, and what degrades as a result.

### News-theme interpretation (`psygnal/news/interpretation.py`)

Classifies the free GDELT news feed's themes into a deliberately
conservative directional prior (`interpret_news`) — every theme except
`geopolitics` carries a **zero** lean, because keyword tagging alone
cannot reliably determine valence (a headline mentioning "inflation" is
not itself bullish or bearish gold without knowing the number).
`reconcile_news_with_reaction()` compares that prior against the
*observed* market direction and reports `CONFIRMED` / `CONTRADICTED` /
`UNCLEAR` — this is exactly the "economic release was theoretically X but
the market reaction was Y, so don't blindly trade the theory" mechanism
the V3 brief asked for.

### Market-shock detection (`psygnal/intelligence/shock.py`)

`detect_shock()` requires **at least two independent triggers**
(range-percentile spike, relative-volume spike, N consecutive directional
candles, or ATR-normalized velocity) before flagging `is_shock=True` —
severity is `MODERATE` at 2 triggers, `SEVERE` at 3+. A single unusual
wick or one high-volume candle is never enough on its own; this was a
deliberate fix after an early draft flagged shocks on single-trigger
noise (see `tests/test_shock.py::test_single_trigger_alone_is_not_a_shock`).

### Cross-market confirmation / contradiction

`psygnal/intelligence/cross_market_confirmation.py::classify_cross_market_confirmation`
labels the *type* of move (`MACRO_DRIVEN` / `CROSS_ASSET_CONFIRMED` /
`TECHNICAL` / `LIQUIDITY_DRIVEN` / `NEWS_SHOCK` / `MIXED` / `UNCLEAR`) by
priority cascade, so a technical breakout during a quiet macro backdrop is
never mislabeled as macro-driven just because some indicator moved.

`psygnal/intelligence/contradiction.py::detect_contradictions` runs the
mission's explicit 8-point checklist against the standing hypothesis (USD
direction, yields — currently always `UNAVAILABLE`, see below — M5
structure, silver confirmation, momentum continuation/deceleration, volume
support, opposing liquidity reclaim, and macro-vs-observed-reaction
contradiction), each item weighted `HIGH`/`MEDIUM`/`LOW`
(`evidence_score()`) so a handful of low-weight nitpicks can never outvote
one or two high-weight contradictions — directly addressing the mission's
"no indicator soup" requirement. Every check is unit-tested in isolation
against controlled mocks in `tests/test_contradiction.py`.

### Hypothesis, evidence, and invalidation (`psygnal/intelligence/reasoning.py`)

`build_hypothesis()` assembles a `MarketHypothesis`: the standing
direction, `confirming_evidence`/`contradicting_evidence` (both weighted
`EvidenceItem` lists), `invalidation_conditions` (plain-language
statements of what would prove the hypothesis wrong — e.g. for a LONG,
"a confirmed close below the nearest support level would invalidate
this"; for a SHORT, the mirror image at resistance — verified directly by
`tests/test_reasoning.py::test_invalidation_condition_for_long_references_support_below_not_resistance_above`),
and the `tradeability` decision with its human-readable reasons.
`assess_tradeability()`'s cascade: insufficient data or `CHAOTIC` regime
→ `NOT_TRADEABLE`; a high-impact event releasing right now →
`NOT_TRADEABLE`; otherwise accumulate `WAIT` reasons (event proximity,
`TRANSITION` regime, `MIXED` cross-market signal, contradiction evidence
outweighing confirming evidence, poor R:R) → `WAIT` if any fired,
otherwise `TRADEABLE`.

### Regime taxonomy expansion (`psygnal/intelligence/regime.py`)

Three new regimes beyond V1/V2's set: `BREAKDOWN` (a bearish
break/continuation, previously indistinguishable from generic
`BREAKOUT`), `POST_NEWS` (price actively reacting to a just-released
high-impact event), and `TRANSITION` (a structural reclaim/failure with
weak directional conviction — neither trending nor cleanly ranging).
`classify_regime()` remains backward compatible: `shock_state` and
`event_context` are optional keyword arguments that, when omitted, leave
V1/V2 behavior completely unchanged.

### Daily market memory (`psygnal/memory/daily.py`)

A structured JSON journal per trading day —
`data/daily/YYYY-MM-DD.json`, named by the **display-timezone (IST)**
calendar date to match the terminal report's "what day is it" framing —
tracking each symbol's forecast history, regime transitions (logged only
on an actual change, not every run), recent liquidity sweeps, and session
levels. Safe against a missing file (starts fresh) and never raises on a
corrupt one (degrades to a fresh journal rather than crashing the
engine). `python -m psygnal` updates it automatically every run; render
today's accumulated context with `python -m psygnal --daily-brief`
(`psygnal/reporting/daily_brief.py`).

### Historical event-reaction memory (`psygnal/forecasting/event_reaction_memory.py`)

Given genuine local historical M5 OHLCV plus a local historical-events
file (neither ships with this repository), computes empirical
"under conditions similar to today, this category of event, surprising in
this direction, has historically moved price by X over 5/15/30/60
minutes" statistics — reporting `INSUFFICIENT_HISTORICAL_SAMPLE` rather
than a fabricated number whenever fewer than `MIN_SAMPLES_FOR_STATISTIC`
(20) historical instances exist for a given category/surprise-direction
pair.

### The narrative terminal report

`psygnal/reporting/terminal.py` was rewritten for V3 to read like a
market-intelligence brief rather than an indicator dump: MARKET STATE →
MACRO → CROSS-MARKET → STRUCTURE → LIQUIDITY → MARKET BEHAVIOR → CURRENT
HYPOTHESIS → EVIDENCE FOR → EVIDENCE AGAINST → INVALIDATION →
TRADEABILITY → 60-MINUTE FORECAST → WHY. All V1/V2 fields are still
present; they are now organized under the section a human trader would
actually look for them in.

## Data sources — what is real vs. derived vs. unavailable

| Source | What it feeds | Status | Notes |
|---|---|---|---|
| Primary M5 feed (Oracle) | Everything technical | **Live, VERIFIED** | See "live schema calibration" above. |
| ForexFactory-mirror calendar | `macro_state`, event proximity/surprise | **Live, free, keyless** | No official schema guarantee (see caveat above); degrades to `MACRO_STATUS=UNAVAILABLE` on any failure. |
| GDELT news | `news_state`, news-theme interpretation | **Live, free, keyless** | Theme classification is DERIVED (keyword-based), not a genuine NLP sentiment model; directional lean is deliberately near-zero for most themes (see above). |
| `MacroDataProvider` (CPI/NFP/GDP/etc. as first-class economic series) | Would feed richer surprise/transmission analysis | **UNAVAILABLE — no live implementation wired** | No free, keyless, reachable-from-this-build-sandbox official statistical-agency API (FRED/BLS/ECB/Fed) was found; the `Protocol` interface and stub (`UnavailableMacroDataProvider`) exist so a real one can be plugged in without touching the intelligence engine. Everything currently reported for `actual`/`forecast` values comes from the ForexFactory-mirror calendar above, not this provider. |
| `RatesProvider` (bond yields) | Contradiction-engine check #2 (yield reaction) | **UNAVAILABLE — no live implementation wired** | Same reasoning as above; `detect_contradictions()` always reports this check as an explicit, low-weight "could not be checked" item rather than silently skipping it or fabricating a yield reading. |
| Historical event-reaction memory | `event_reaction_memory.py` statistics | **User-supplied only** | This repository ships no historical events file or historical M5 archive; the module honestly reports `INSUFFICIENT_HISTORICAL_SAMPLE` until the user supplies real historical data. |

No API keys or environment variables are required to run
`python -m psygnal` — every wired-up source above is free and keyless,
and every not-yet-wired source degrades to an explicit `UNAVAILABLE`
status rather than requiring configuration that doesn't exist yet. If a
real `MacroDataProvider` or `RatesProvider` implementation is added later,
document any key/rate-limit requirements it introduces here.
