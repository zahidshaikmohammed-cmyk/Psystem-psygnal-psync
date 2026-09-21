"""Economic-event categorization.

Purely a text classifier over event titles reported by the calendar
source — it does not know anything about the release itself beyond its
name, and falls back to OTHER rather than guessing.
"""

from __future__ import annotations

from enum import Enum


class MacroCategory(str, Enum):
    INFLATION = "INFLATION"
    EMPLOYMENT = "EMPLOYMENT"
    GROWTH = "GROWTH"
    MONETARY_POLICY = "MONETARY_POLICY"
    CENTRAL_BANK_SPEECH = "CENTRAL_BANK_SPEECH"
    MANUFACTURING = "MANUFACTURING"
    SERVICES = "SERVICES"
    HOUSING = "HOUSING"
    TRADE = "TRADE"
    SENTIMENT = "SENTIMENT"
    OTHER = "OTHER"


# Order matters: more specific categories are checked before OTHER, and
# MONETARY_POLICY/CENTRAL_BANK_SPEECH are checked before the generic
# growth/employment buckets so e.g. "FOMC Press Conference" doesn't fall
# through to OTHER.
_CATEGORY_KEYWORDS: dict[MacroCategory, tuple[str, ...]] = {
    MacroCategory.INFLATION: ("cpi", "pce price", "core pce", "ppi", "producer price", "inflation"),
    MacroCategory.MONETARY_POLICY: (
        "fomc",
        "federal funds rate",
        "rate decision",
        "interest rate decision",
        "press conference",
        "dot plot",
        "sep",
        "fomc minutes",
        "ecb rate",
        "boe rate",
        "boj rate",
        "monetary policy statement",
        "rate statement",
    ),
    MacroCategory.CENTRAL_BANK_SPEECH: (
        "speaks",
        "speech",
        "testimony",
        "powell",
        "lagarde",
        "bailey",
        "ueda",
        "press conference q&a",
    ),
    MacroCategory.EMPLOYMENT: (
        "non-farm",
        "nonfarm",
        "nfp",
        "payrolls",
        "unemployment",
        "jobless claims",
        "jolts",
        "adp",
        "average hourly earnings",
        "continuing claims",
        "employment change",
    ),
    MacroCategory.MANUFACTURING: ("ism manufacturing", "manufacturing pmi", "manufacturing production", "durable goods"),
    MacroCategory.SERVICES: ("ism services", "services pmi", "non-manufacturing"),
    MacroCategory.HOUSING: ("housing starts", "building permits", "existing home sales", "new home sales", "pending home sales"),
    MacroCategory.TRADE: ("trade balance", "current account"),
    MacroCategory.SENTIMENT: ("consumer confidence", "consumer sentiment", "michigan"),
    MacroCategory.GROWTH: ("gdp", "industrial production", "retail sales"),
}


def classify_category(title: str) -> MacroCategory:
    lower = title.lower()
    for category, keywords in _CATEGORY_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return category
    return MacroCategory.OTHER
