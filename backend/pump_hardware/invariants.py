from __future__ import annotations

import math


# CH1 is the continuous oil phase and CH2 is the dispersed water phase. This
# safety gap is deliberately not user-configurable.
STRICT_Q1_Q2_GAP_UL_MIN = 0.2


def effective_q1_q2_gap(configured_gap: float | None = None) -> float:
    if configured_gap is None:
        return STRICT_Q1_Q2_GAP_UL_MIN
    value = float(configured_gap)
    if not math.isfinite(value):
        return STRICT_Q1_Q2_GAP_UL_MIN
    return max(STRICT_Q1_Q2_GAP_UL_MIN, value)


def q1_is_strictly_above_q2(
    q1: float,
    q2: float,
    configured_gap: float | None = None,
) -> bool:
    q1_value = float(q1)
    q2_value = float(q2)
    minimum_q1 = q2_value + effective_q1_q2_gap(configured_gap)
    return bool(
        math.isfinite(q1_value)
        and math.isfinite(q2_value)
        and (
            q1_value >= minimum_q1
            or math.isclose(q1_value, minimum_q1, rel_tol=1e-12, abs_tol=1e-9)
        )
    )
