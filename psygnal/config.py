"""Central configuration for the Psygnal engine.

No secrets or credentials live here. Every value is either a public
endpoint, a well-known free-data URL, or a tunable analysis parameter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Live market data
# ---------------------------------------------------------------------------

# Primary live M5 endpoint (PSYGRID RealMarketAPI feed).
LIVE_M5_ENDPOINT = "http://140.245.226.102:8080/public/m5-live.json"

LIVE_FETCH_TIMEOUT_SECONDS = 10
LIVE_FETCH_RETRIES = 3
LIVE_FETCH_BACKOFF_SECONDS = 1.5

# Data is considered stale if the freshest candle is older than this many
# minutes relative to "now" (UTC). Weekends/market closures are handled
# separately by the session engine, not by this raw freshness check.
MAX_DATA_STALENESS_MINUTES = 20

# Minimum number of completed M5 candles required per symbol before the
# engine will attempt any analysis at all.
MIN_M5_CANDLES_REQUIRED = 120

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

DEFAULT_SYMBOL_UNIVERSE: tuple[str, ...] = (
    "XAUUSD",
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "GBPJPY",
    "AUDUSD",
    "USDCAD",
    "NZDUSD",
    "XAGUSD",
    "USOIL",
)

# Pairs used to build the internal (non-official) USD strength composite.
# Sign convention: +1 means "pair rising == USD strengthening",
# -1 means "pair rising == USD weakening" (USD is the base currency there).
USD_COMPOSITE_LEGS: dict[str, int] = {
    "EURUSD": -1,
    "GBPUSD": -1,
    "AUDUSD": -1,
    "NZDUSD": -1,
    "USDJPY": +1,
    "USDCAD": +1,
}

GOLD_SYMBOL = "XAUUSD"
SILVER_SYMBOL = "XAGUSD"
OIL_SYMBOL = "USOIL"
RISK_BAROMETER_SYMBOL = "GBPJPY"

# ---------------------------------------------------------------------------
# Timeframes
# ---------------------------------------------------------------------------

# Number of raw M5 candles that make up one higher-timeframe candle.
TIMEFRAME_M5_MULTIPLES: dict[str, int] = {
    "M5": 1,
    "M15": 3,
    "M30": 6,
    "H1": 12,
    "H4": 48,
}

FORECAST_HORIZON_M5_CANDLES = 12  # ~60 minutes

# ---------------------------------------------------------------------------
# Sequence windows (in M5 candles)
# ---------------------------------------------------------------------------

SEQUENCE_WINDOWS: tuple[int, ...] = (3, 6, 12, 24, 48, 96)

# ---------------------------------------------------------------------------
# Timezones
# ---------------------------------------------------------------------------

UTC = ZoneInfo("UTC")
DISPLAY_TIMEZONE = ZoneInfo("Asia/Kolkata")  # IST, no DST

# Session windows expressed in UTC hours (approximate, DST-adjusted via
# the exchange-city zoneinfo entries in intelligence/sessions.py rather
# than fixed UTC offsets).
SESSION_CITY_TIMEZONES: dict[str, str] = {
    "ASIA": "Asia/Tokyo",
    "LONDON": "Europe/London",
    "NEW_YORK": "America/New_York",
}

# Local session hours (24h, in the session's own civil time) used to derive
# UTC windows for "today" after DST conversion.
SESSION_LOCAL_HOURS: dict[str, tuple[int, int]] = {
    "ASIA": (9, 18),
    "LONDON": (8, 17),
    "NEW_YORK": (8, 17),
}

# ---------------------------------------------------------------------------
# Indicator parameters
# ---------------------------------------------------------------------------

EMA_PERIODS: tuple[int, ...] = (9, 20, 50, 200)
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ATR_PERIOD = 14
ADX_PERIOD = 14
BOLLINGER_PERIOD = 20
BOLLINGER_STD_MULTIPLIER = 2.0
STOCHASTIC_K_PERIOD = 14
STOCHASTIC_D_PERIOD = 3
ROC_PERIOD = 10
VOLUME_MA_PERIOD = 20
ATR_PERCENTILE_LOOKBACK = 200

# ---------------------------------------------------------------------------
# Structure / liquidity
# ---------------------------------------------------------------------------

SWING_LOOKBACK = 2  # bars each side for a fractal swing high/low
EQUAL_LEVEL_TOLERANCE_ATR_MULTIPLE = 0.15

# ---------------------------------------------------------------------------
# Macro / news (free, public, no API keys required)
# ---------------------------------------------------------------------------

# Free, widely used public mirror of the ForexFactory weekly calendar feed.
# No authentication required. Treated as a secondary/optional source: any
# failure degrades gracefully to MACRO_STATUS = UNAVAILABLE.
FOREX_FACTORY_WEEKLY_CALENDAR_URL = (
    "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
)

GDELT_DOC_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

MACRO_NEWS_FETCH_TIMEOUT_SECONDS = 8
MACRO_NEWS_FETCH_RETRIES = 2
MACRO_NEWS_CACHE_TTL_SECONDS = 900  # 15 minutes, do not hammer public feeds

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
HISTORICAL_DIR = Path(__file__).resolve().parent.parent / "data" / "historical"

# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------

MIN_HISTORICAL_SAMPLES_FOR_TRAINING = 500
PATTERN_MEMORY_MIN_SAMPLES = 200
PATTERN_MEMORY_K_NEIGHBORS = 25

MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"

# ---------------------------------------------------------------------------
# Daily market memory
# ---------------------------------------------------------------------------

DAILY_MEMORY_DIR = Path(__file__).resolve().parent.parent / "data" / "daily"


@dataclass(frozen=True)
class EngineConfig:
    """Bundle of runtime-overridable settings passed through the pipeline."""

    live_endpoint: str = LIVE_M5_ENDPOINT
    symbols: tuple[str, ...] = DEFAULT_SYMBOL_UNIVERSE
    fetch_timeout_seconds: int = LIVE_FETCH_TIMEOUT_SECONDS
    fetch_retries: int = LIVE_FETCH_RETRIES
    max_staleness_minutes: int = MAX_DATA_STALENESS_MINUTES
    min_candles_required: int = MIN_M5_CANDLES_REQUIRED
    historical_dir: Path = field(default_factory=lambda: HISTORICAL_DIR)
    model_dir: Path = field(default_factory=lambda: MODEL_DIR)
    enable_macro: bool = True
    enable_news: bool = True
    enable_models: bool = True
