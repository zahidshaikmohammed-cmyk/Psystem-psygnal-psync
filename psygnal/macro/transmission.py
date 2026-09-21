"""Theoretical macro-to-USD transmission priors.

These are conventional textbook relationships (e.g. "hotter-than-expected
inflation conventionally firms up hawkish Fed expectations and thus
USD") used ONLY as a comparison baseline against what the market
*actually* does — see `intelligence/cross_market_confirmation.py` and
`intelligence/contradiction.py`'s "market reacted opposite to the
textbook interpretation" check.

They are explicitly NEVER used to pick a trade direction directly. Doing
that is exactly the "CPI higher = gold short" mistake this project has
been explicitly told not to make — these relationships are conditional
on the prevailing regime and on what's already priced in, which this
module has no way of knowing from event metadata alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from psygnal.macro.categories import MacroCategory

# Sign of d(USD)/d(surprise) under the conventional, textbook transmission
# — a POSITIVE (stronger than forecast) surprise in this category
# conventionally pushes USD in this direction. 0 means "no reliable
# textbook sign" (e.g. a central-bank speech's direction depends entirely
# on content, not derivable from metadata).
_USD_TRANSMISSION_SIGN: dict[MacroCategory, int] = {
    MacroCategory.INFLATION: +1,
    MacroCategory.EMPLOYMENT: +1,
    MacroCategory.GROWTH: +1,
    MacroCategory.MANUFACTURING: +1,
    MacroCategory.SERVICES: +1,
    MacroCategory.MONETARY_POLICY: +1,
    MacroCategory.SENTIMENT: +1,
    MacroCategory.HOUSING: 0,
    MacroCategory.TRADE: 0,
    MacroCategory.CENTRAL_BANK_SPEECH: 0,
    MacroCategory.OTHER: 0,
}

# A normalized surprise inside +/- this band is treated as "in line" —
# too small relative to typical scale to form any directional prior.
_IN_LINE_BAND = 0.1


@dataclass
class TransmissionHypothesis:
    category: str
    surprise_sign: int  # +1 stronger than forecast, -1 weaker, 0 in-line/unknown
    expected_usd_direction: int  # +1 USD strength, -1 USD weakness, 0 no prior
    rationale: str
    confidence: str  # "THEORETICAL" | "LOW" | "UNAVAILABLE" — never "HIGH"; this is a prior, not a finding

    def as_dict(self) -> dict[str, object]:
        return {
            "category": self.category,
            "surprise_sign": self.surprise_sign,
            "expected_usd_direction": self.expected_usd_direction,
            "rationale": self.rationale,
            "confidence": self.confidence,
        }


def build_transmission_hypothesis(category: str, surprise_normalized: Optional[float]) -> TransmissionHypothesis:
    try:
        cat_enum = MacroCategory(category)
    except ValueError:
        cat_enum = MacroCategory.OTHER

    if surprise_normalized is None:
        return TransmissionHypothesis(
            category=cat_enum.value,
            surprise_sign=0,
            expected_usd_direction=0,
            rationale="No numeric surprise available; no directional prior formed.",
            confidence="UNAVAILABLE",
        )

    surprise_sign = 1 if surprise_normalized > _IN_LINE_BAND else (-1 if surprise_normalized < -_IN_LINE_BAND else 0)
    transmission_sign = _USD_TRANSMISSION_SIGN.get(cat_enum, 0)
    expected_usd_direction = surprise_sign * transmission_sign

    if transmission_sign == 0:
        rationale = f"{cat_enum.value} events have no reliable textbook USD transmission sign; treated as unclear."
        confidence = "UNAVAILABLE"
    elif surprise_sign == 0:
        rationale = f"{cat_enum.value} release was in line with forecast; no directional prior."
        confidence = "LOW"
    else:
        direction_word = "Stronger-than-expected" if surprise_sign > 0 else "Weaker-than-expected"
        usd_word = "USD strength" if expected_usd_direction > 0 else "USD weakness"
        rationale = (
            f"{direction_word} {cat_enum.value.lower()} conventionally implies {usd_word} "
            "(a theoretical prior, not a rule — compare against the actual observed reaction)."
        )
        confidence = "THEORETICAL"

    return TransmissionHypothesis(
        category=cat_enum.value,
        surprise_sign=surprise_sign,
        expected_usd_direction=expected_usd_direction,
        rationale=rationale,
        confidence=confidence,
    )
