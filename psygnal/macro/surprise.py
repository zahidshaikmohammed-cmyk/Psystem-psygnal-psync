"""Actual-vs-forecast surprise calculation.

HONESTY NOTE: genuine cross-indicator normalization requires each
indicator's own historical surprise distribution (the standard deviation
of actual-minus-forecast over many past releases), which this engine
does not have without real historical calendar data (see
`forecasting/event_reaction_memory.py` — it reports
INSUFFICIENT_HISTORICAL_SAMPLE until real history is supplied). Absent
that, `compute_normalized_surprise` divides by a small set of documented,
deliberately rough per-category "typical scale" priors, purely so a CPI
surprise and an NFP surprise land in roughly the same ballpark for
comparison. This is DERIVED, heuristic data — never a statistically
calibrated z-score, and callers must not present it as one.
"""

from __future__ import annotations

from typing import Optional

from psygnal.macro.categories import MacroCategory

# Rough "one typical surprise" scale per category, in the release's own
# native units (percentage points for most, thousands of jobs for
# EMPLOYMENT). These are textbook approximations, not fitted values.
_TYPICAL_SURPRISE_SCALE: dict[MacroCategory, float] = {
    MacroCategory.INFLATION: 0.15,
    MacroCategory.EMPLOYMENT: 60.0,
    MacroCategory.GROWTH: 0.4,
    MacroCategory.MANUFACTURING: 1.5,
    MacroCategory.SERVICES: 1.5,
    MacroCategory.HOUSING: 5.0,
    MacroCategory.TRADE: 3.0,
    MacroCategory.SENTIMENT: 3.0,
    MacroCategory.MONETARY_POLICY: 0.25,
}
_DEFAULT_SCALE = 1.0


def compute_surprise(actual: Optional[float], forecast: Optional[float]) -> Optional[float]:
    """Raw surprise in the release's own units. None if either input is
    missing — never defaulted to zero, which would fabricate an
    "in-line" reading that was never actually observed."""
    if actual is None or forecast is None:
        return None
    return actual - forecast


def compute_normalized_surprise(surprise: Optional[float], category: MacroCategory) -> Optional[float]:
    if surprise is None:
        return None
    scale = _TYPICAL_SURPRISE_SCALE.get(category, _DEFAULT_SCALE)
    if scale <= 0:
        return None
    return surprise / scale
