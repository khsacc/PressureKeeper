"""Monotonic time source, injectable so tests can drive a fake clock."""
from __future__ import annotations

import math
import time
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float:
        """Seconds on a monotonic clock. Never wall-clock time."""
        ...


class MonotonicClock:
    def now(self) -> float:
        return time.monotonic()


class FakeClock:
    """Manually-advanced clock for deterministic tests / simulation."""

    def __init__(self, start: float = 0.0) -> None:
        if not math.isfinite(start):
            raise ValueError("clock start must be finite")
        self._t = start

    def now(self) -> float:
        return self._t

    def advance(self, dt: float) -> float:
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("clock advance must be finite and non-negative")
        self._t += dt
        return self._t
