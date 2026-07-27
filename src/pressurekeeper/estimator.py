"""PressureEstimator: turns a noisy ruby-pressure stream into a filtered
value, a slope estimate, and a validity verdict.

Pipeline per `update()`:
  1. Reject samples that are not usable (`fit_success=False`, missing
     pressure, out-of-order timestamp, or below the configured fit-quality
     gate) — these count towards `consecutive_invalid` but never enter the
     filter buffers.
  2. Outlier suppression: push the raw value into a short ring buffer (5-9
     points) and take its median. A single spike does not move the median.
  3. Smoothing: average the median-filtered points that fall within the last
     `smoothing_window_s`.
  4. Slope: ordinary least-squares fit of (t, median-filtered pressure) over
     the last `slope_window_s`.
"""
from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass

from .config import EstimatorConfig
from .models import RubyPressureSample


@dataclass
class _Point:
    t: float
    p: float


class PressureEstimator:
    def __init__(self, config: EstimatorConfig) -> None:
        self._cfg = config
        self._raw_window: deque[_Point] = deque(maxlen=config.outlier_median_window)
        self._median_history: deque[_Point] = deque(maxlen=max(config.outlier_median_window * 20, 200))
        self._last_t: float | None = None
        self._last_valid_t: float | None = None
        self._consecutive_invalid: int = 0
        self._last_jump_flagged: bool = False
        self._last_sample: RubyPressureSample | None = None

    # ------------------------------------------------------------------ update

    def update(self, sample: RubyPressureSample) -> None:
        self._last_sample = sample
        self._last_jump_flagged = False

        if not self._passes_quality_gate(sample):
            self._consecutive_invalid += 1
            return

        assert sample.pressure_gpa is not None
        t, p = sample.t_mono, sample.pressure_gpa

        if self._last_t is not None and t < self._last_t:
            # Out-of-order arrival (e.g. a slow request completing after a
            # faster later one) — drop rather than corrupting slope math.
            self._consecutive_invalid += 1
            return
        self._last_t = t

        if self._median_history and abs(p - self._median_history[-1].p) > self._cfg.max_jump_gpa:
            self._last_jump_flagged = True

        self._raw_window.append(_Point(t, p))
        median_p = statistics.median(pt.p for pt in self._raw_window)
        self._median_history.append(_Point(t, median_p))
        self._prune_history(t)

        self._last_valid_t = t
        self._consecutive_invalid = 0

    def _passes_quality_gate(self, sample: RubyPressureSample) -> bool:
        if not sample.is_usable:
            return False
        # Fail-closed: once a threshold is configured, a *missing* value is
        # treated the same as a value that fails it, not as "no opinion".
        if self._cfg.min_r2 is not None and (sample.r2 is None or sample.r2 < self._cfg.min_r2):
            return False
        if self._cfg.min_intensity is not None and (sample.intensity is None or sample.intensity < self._cfg.min_intensity):
            return False
        return True

    def _prune_history(self, now: float) -> None:
        horizon = max(self._cfg.smoothing_window_s, self._cfg.slope_window_s)
        while self._median_history and now - self._median_history[0].t > horizon:
            self._median_history.popleft()

    # ---------------------------------------------------------------- outputs

    def filtered_pressure(self) -> float | None:
        pts = self._points_within(self._cfg.smoothing_window_s)
        if not pts:
            return None
        return sum(pt.p for pt in pts) / len(pts)

    def pressure_slope(self) -> float | None:
        pts = self._points_within(self._cfg.slope_window_s)
        if len(pts) < 2:
            return 0.0
        n = len(pts)
        t_mean = sum(pt.t for pt in pts) / n
        p_mean = sum(pt.p for pt in pts) / n
        num = sum((pt.t - t_mean) * (pt.p - p_mean) for pt in pts)
        den = sum((pt.t - t_mean) ** 2 for pt in pts)
        if den <= 0:
            return 0.0
        return num / den

    def measurement_std(self) -> float | None:
        pts = [pt.p for pt in self._raw_window]
        if len(pts) < 2:
            return None
        return statistics.pstdev(pts)

    def is_valid(self, now: float) -> bool:
        if self._last_valid_t is None:
            return False
        if len(self._median_history) < self._cfg.min_points_for_valid:
            return False
        if now - self._last_valid_t > self._cfg.max_sample_age_s:
            return False
        return True

    def last_valid_age_s(self, now: float) -> float | None:
        if self._last_valid_t is None:
            return None
        return now - self._last_valid_t

    @property
    def consecutive_invalid(self) -> int:
        return self._consecutive_invalid

    @property
    def last_jump_flagged(self) -> bool:
        return self._last_jump_flagged

    @property
    def last_sample(self) -> RubyPressureSample | None:
        return self._last_sample

    def predicted_pressure(self, horizon_s: float) -> float | None:
        filtered = self.filtered_pressure()
        slope = self.pressure_slope()
        if filtered is None or slope is None:
            return None
        return filtered + max(slope, 0.0) * horizon_s

    def _points_within(self, window_s: float) -> list[_Point]:
        if not self._median_history:
            return []
        newest = self._median_history[-1].t
        return [pt for pt in self._median_history if newest - pt.t <= window_s]
