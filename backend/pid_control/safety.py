from __future__ import annotations

import math


def is_finite(value: float | int | None) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def rate_limit(value: float, previous: float, limit: float) -> float:
    limit = abs(float(limit))
    if limit <= 0:
        return float(value)
    delta = float(value) - float(previous)
    if delta > limit:
        return float(previous) + limit
    if delta < -limit:
        return float(previous) - limit
    return float(value)
