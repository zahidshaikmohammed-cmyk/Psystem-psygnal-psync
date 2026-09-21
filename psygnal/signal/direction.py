"""Direction selection.

The engine always chooses LONG or SHORT once there is sufficient primary
market data — there is no "insufficient edge, WAIT" veto. A weak edge is
reflected through low score/confidence elsewhere, never by withholding a
direction.
"""

from __future__ import annotations

from psygnal.models import Direction


def select_direction(probability_long: float, probability_short: float) -> str:
    if probability_long >= probability_short:
        return Direction.LONG.value
    return Direction.SHORT.value
