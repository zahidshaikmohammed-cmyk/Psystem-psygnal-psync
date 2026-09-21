"""Layered news interpretation.

    NEWS TEXT
       -> EVENT CLASSIFICATION      (theme tagging, news/aggregator.py)
       -> MACRO INTERPRETATION      (this module: interpret_news)
       -> EXPECTED TRANSMISSION     (this module: a bare, low-confidence prior)
       -> OBSERVED CROSS-MARKET REACTION   (intelligence/cross_market_confirmation.py)
       -> CONFIRMED / CONTRADICTED / UNCLEAR   (this module: reconcile_news_with_reaction)

HONESTY NOTE: this is keyword/theme-based classification, not NLP that
understands financial causality. Most themes are deliberately given NO
reliable directional lean (0) because a bare theme tag like "usd" or
"inflation" doesn't tell you whether the news was dollar-bullish or
dollar-bearish — that requires reading comprehension this module does not
attempt. The one exception (geopolitics -> conventional safe-haven
demand) is still only a LOW-confidence prior, and every lean this module
produces is meant to be compared against the market's actual behavior
before being trusted at all, never acted on directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

# Expected XAUUSD directional lean implied by a theme being dominant in
# the current news window, as a bare theoretical prior — NOT a signal.
# +1 = conventionally bullish gold, -1 = conventionally bearish gold,
# 0 = no reliable lean derivable from the theme tag alone.
_THEME_GOLD_LEAN: dict[str, int] = {
    "usd": 0,
    "federal_reserve": 0,
    "inflation": 0,
    "interest_rates": 0,
    "treasury_yields": 0,
    "gold": 0,
    "central_banks": 0,
    "employment": 0,
    "geopolitics": +1,
    "oil": 0,
    "risk_sentiment": 0,
}


@dataclass
class NewsInterpretation:
    dominant_themes: list[str]
    expected_gold_lean: int  # -1 / 0 / +1 — a bare prior, never a signal
    confidence: str  # "LOW" | "UNAVAILABLE"
    rationale: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "dominant_themes": self.dominant_themes,
            "expected_gold_lean": self.expected_gold_lean,
            "confidence": self.confidence,
            "rationale": self.rationale,
        }


def interpret_news(news_state: dict[str, Any]) -> NewsInterpretation:
    if news_state.get("status") != "OK":
        return NewsInterpretation([], 0, "UNAVAILABLE", "News data unavailable; no interpretation formed.")

    dominant = news_state.get("dominant_themes", [])
    if not dominant:
        return NewsInterpretation([], 0, "UNAVAILABLE", "No dominant themes identified in the current news window.")

    leans = [_THEME_GOLD_LEAN.get(t, 0) for t in dominant]
    nonzero = [l for l in leans if l != 0]
    if not nonzero:
        return NewsInterpretation(
            dominant,
            0,
            "LOW",
            f"Dominant theme(s) ({', '.join(dominant)}) carry no reliable directional lean from keyword tagging alone.",
        )

    if all(l == nonzero[0] for l in nonzero):
        lean = nonzero[0]
        word = "bullish" if lean > 0 else "bearish"
        rationale = (
            f"Dominant theme(s) ({', '.join(dominant)}) conventionally lean {word} for gold "
            "(a keyword-based prior, not a rule)."
        )
        return NewsInterpretation(dominant, lean, "LOW", rationale)

    return NewsInterpretation(dominant, 0, "LOW", "Dominant themes disagree on directional lean; treated as unclear.")


def reconcile_news_with_reaction(interpretation: NewsInterpretation, observed_gold_direction: Optional[int]) -> str:
    """CONFIRMED / CONTRADICTED / UNCLEAR: compares the news-derived
    expectation against how gold is actually behaving right now. This is
    the mechanism behind statements like "the news was theoretically
    bearish for gold, but gold is actually rising — don't blindly short."""
    if interpretation.expected_gold_lean == 0 or not observed_gold_direction:
        return "UNCLEAR"
    if interpretation.expected_gold_lean == observed_gold_direction:
        return "CONFIRMED"
    return "CONTRADICTED"
