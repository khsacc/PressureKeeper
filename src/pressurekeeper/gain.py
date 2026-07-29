"""Run-local estimates of dP_sample/dP_membrane.

Adaptive-local mode uses only nearby settled observations from this loading;
without one it explicitly reports ``probe`` so the controller sends a small
gas-side identification step. Legacy mode retains the pressure-binned,
configured-prior estimator for backwards compatibility. Interrupted
compression-rate responses remain a separate dynamic slew-limit signal and
are never mixed into settled/static step sizing.
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict

from .config import GainEstimationConfig
from .models import GainEstimate, GainRegion, StepRecord


def _percentile(values: list[float], pct: float) -> float:
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


class GainEstimator:
    def __init__(self, config: GainEstimationConfig) -> None:
        self._cfg = config
        self._bins: dict[int, list[float]] = defaultdict(list)
        # Ordered, run-local quasi-static observations. Unlike _bins these are
        # queried only near the current pressure, so a low-pressure gain is
        # never silently reused as the high-pressure estimate.
        self._local_observations: list[tuple[int, float, float]] = []
        # One mutable maximum per interrupted command. A PAUSE lasts several
        # ticks; replacing its peak prevents one physical response from being
        # counted as many independent samples.
        self._interrupted_rate_observations: dict[int, tuple[float, float]] = {}

    def update_config(self, config: GainEstimationConfig) -> None:
        """Hot-swap the config and discard all recorded gain observations --
        a changed `bin_width_gpa` would otherwise make existing bin indices
        meaningless. Only safe to call before any steps have been recorded
        (see gui/parameters_config_dialog.py: gated to before Start Control).
        """
        self.__init__(config)

    def _bin_index(self, sample_pressure_gpa: float) -> int:
        return int(sample_pressure_gpa // self._cfg.bin_width_gpa)

    def record_step(self, step: StepRecord) -> None:
        gain = step.observed_gain
        pivot = step.midpoint_sample_pressure_gpa
        if (
            gain is None
            or pivot is None
            or not math.isfinite(gain)
            or not math.isfinite(pivot)
            or gain <= 0
        ):
            return
        self._bins[self._bin_index(pivot)].append(gain)
        self._local_observations.append((step.step_id, pivot, gain))

    def record_interrupted_rate_observation(
        self,
        observation_id: int,
        sample_pressure_gpa: float,
        raw_rate_gain: float,
    ) -> None:
        """Record/update one clean interrupted response for rate limiting.

        This intentionally does not touch ``_bins``: an interrupted response
        has no trustworthy settled/static gain and must never change step
        sizing. It can only make the independent membrane slew cap stricter.
        """
        if self._cfg.interrupted_rate_learning_mode == "off":
            return
        if (
            not math.isfinite(sample_pressure_gpa)
            or not math.isfinite(raw_rate_gain)
            or raw_rate_gain <= 0
        ):
            return
        previous = self._interrupted_rate_observations.get(observation_id)
        if previous is None or raw_rate_gain > previous[1]:
            self._interrupted_rate_observations[observation_id] = (
                sample_pressure_gpa,
                raw_rate_gain,
            )

    def discard_interrupted_rate_observation(self, observation_id: int) -> None:
        """Remove an observation later contaminated by another safety cause."""
        self._interrupted_rate_observations.pop(observation_id, None)

    def _interrupted_rate_floor(self, sample_pressure_gpa: float) -> tuple[float, int]:
        if self._cfg.interrupted_rate_learning_mode == "off":
            return 0.0, 0
        if self._cfg.step_sizing_mode == "adaptive_local":
            relevant = [
                raw_gain
                for pressure, raw_gain in self._interrupted_rate_observations.values()
                if (
                    (
                        self._cfg.interrupted_rate_propagate_upward
                        and pressure <= sample_pressure_gpa
                    )
                    or abs(pressure - sample_pressure_gpa)
                    <= self._cfg.local_pressure_window_gpa
                )
            ]
        else:
            center = self._bin_index(sample_pressure_gpa)
            relevant = [
                raw_gain
                for pressure, raw_gain in self._interrupted_rate_observations.values()
                if (
                    self._bin_index(pressure) == center
                    or (
                        self._cfg.interrupted_rate_propagate_upward
                        and self._bin_index(pressure) <= center
                    )
                )
            ]
        if not relevant:
            return 0.0, 0
        return max(relevant) * self._cfg.interrupted_rate_safety_factor, len(relevant)

    def _with_rate_limit(
        self,
        estimate: GainEstimate,
        sample_pressure_gpa: float,
        configured_floor: float,
    ) -> GainEstimate:
        learned_floor, observation_count = self._interrupted_rate_floor(sample_pressure_gpa)
        settled_floor = estimate.safe_gain
        base_rate_limit = max(settled_floor, configured_floor)
        applied_rate_limit = base_rate_limit
        if settled_floor > configured_floor:
            rate_source: str = "settled"
        elif configured_floor > 0:
            rate_source = "configured"
        else:
            rate_source = "none"
        if (
            self._cfg.interrupted_rate_learning_mode == "enforce"
            and learned_floor > applied_rate_limit
        ):
            applied_rate_limit = learned_floor
            rate_source = "interrupted"
        return GainEstimate(
            safe_gain=estimate.safe_gain,
            estimated_gain=estimate.estimated_gain,
            gain_uncertainty=estimate.gain_uncertainty,
            source=estimate.source,
            n_samples=estimate.n_samples,
            rate_limit_gain=applied_rate_limit,
            rate_gain_source=rate_source,
            interrupted_rate_observation_count=observation_count,
            learned_rate_floor=learned_floor,
            local_gain_trend_per_gpa=estimate.local_gain_trend_per_gpa,
            local_observation_span_gpa=estimate.local_observation_span_gpa,
        )

    def _estimate_adaptive_local(
        self,
        sample_pressure_gpa: float,
        forward_sample_step_gpa: float,
    ) -> GainEstimate:
        candidates = [
            (step_id, pressure, gain)
            for step_id, pressure, gain in self._local_observations
            if abs(sample_pressure_gpa - pressure)
            <= self._cfg.local_pressure_window_gpa
        ]
        candidates = candidates[-self._cfg.local_max_observations :]
        if not candidates:
            estimate = GainEstimate(
                safe_gain=0.0,
                estimated_gain=0.0,
                gain_uncertainty=0.0,
                source="probe",
                n_samples=0,
                rate_limit_gain=0.0,
                rate_gain_source="none",
            )
            return self._with_rate_limit(estimate, sample_pressure_gpa, 0.0)

        gains = [gain for _, _, gain in candidates]
        median_gain = statistics.median(gains)
        upper = max(gains)
        spread = statistics.pstdev(gains) if len(gains) >= 2 else 0.0
        uncertainty = max(spread, upper - median_gain, 0.0)

        # A per-GPa curvature trend needs enough pressure spread across the
        # window to mean anything -- fit through *all* local candidates
        # (least squares, not just the two most recent) and only trust the
        # resulting slope once the window's observations span at least
        # local_trend_min_span_gpa. Below that span, ordinary observed_gain
        # noise (from a short settle window / measurement std comparable to
        # the step's own sample-pressure delta) divided by a tiny pressure
        # gap produces an arbitrarily large apparent slope with no relation
        # to the plant's real curvature -- see local_trend_min_span_gpa's
        # docstring in config.py for the real-hardware run that exposed this.
        ordered = sorted(candidates, key=lambda item: (item[1], item[0]))
        span = ordered[-1][1] - ordered[0][1]
        trend = 0.0
        if span >= self._cfg.local_trend_min_span_gpa:
            n = len(ordered)
            p_mean = sum(p for _, p, _ in ordered) / n
            g_mean = sum(g for _, _, g in ordered) / n
            covariance = sum((p - p_mean) * (g - g_mean) for _, p, g in ordered)
            variance = sum((p - p_mean) ** 2 for _, p, _ in ordered)
            if variance > 0:
                trend = max(0.0, covariance / variance)

        latest_gain = candidates[-1][2]
        local_upper = max(
            latest_gain,
            upper,
            median_gain
            + self._cfg.local_uncertainty_safety_factor * uncertainty,
        )
        safe_gain = (
            local_upper * self._cfg.local_gain_safety_factor
            + self._cfg.local_curvature_safety_factor
            * trend
            * max(forward_sample_step_gpa, 0.0)
        )
        estimate = GainEstimate(
            safe_gain=safe_gain,
            estimated_gain=latest_gain,
            gain_uncertainty=uncertainty,
            source="observed",
            n_samples=len(candidates),
            rate_limit_gain=safe_gain,
            rate_gain_source="settled",
            local_gain_trend_per_gpa=trend,
            local_observation_span_gpa=span,
        )
        return self._with_rate_limit(estimate, sample_pressure_gpa, 0.0)

    def estimate(
        self,
        sample_pressure_gpa: float,
        prior_region: GainRegion,
        *,
        forward_sample_step_gpa: float = 0.0,
    ) -> GainEstimate:
        if self._cfg.step_sizing_mode == "adaptive_local":
            return self._estimate_adaptive_local(
                sample_pressure_gpa,
                forward_sample_step_gpa,
            )

        # Floor for the dynamic gas-side rate cap (see GainRegion.rate_limit_gain's
        # docstring): independent of, and never lower than, whatever safe_gain
        # this call ends up returning below -- guards the case a configured
        # prior turns out to be optimistic relative to real hardware, which
        # the online estimate alone only corrects for once enough observations
        # accumulate near this band.
        rate_limit_floor = (
            prior_region.rate_limit_gain if prior_region.rate_limit_gain is not None else prior_region.safe_gain
        )

        center = self._bin_index(sample_pressure_gpa)
        gathered: list[float] = list(self._bins.get(center, []))
        for radius in range(1, self._cfg.neighbor_bins + 1):
            if len(gathered) >= self._cfg.min_samples_for_estimate:
                break
            gathered += self._bins.get(center - radius, []) + self._bins.get(center + radius, [])

        if len(gathered) < self._cfg.min_samples_for_estimate:
            estimate = GainEstimate(
                safe_gain=prior_region.safe_gain,
                estimated_gain=prior_region.safe_gain,
                gain_uncertainty=0.0,
                source="prior",
                n_samples=len(gathered),
                rate_limit_gain=max(prior_region.safe_gain, rate_limit_floor),
            )
            return self._with_rate_limit(estimate, sample_pressure_gpa, rate_limit_floor)

        median_gain = statistics.median(gathered)
        upper = _percentile(gathered, self._cfg.upper_percentile)
        spread = statistics.pstdev(gathered) if len(gathered) >= 2 else 0.0
        uncertainty = max(spread, upper - median_gain, 0.0)
        # Floored at the configured prior: neighbor_bins pulls in lower-pressure
        # bins (smaller true gain in a system where gain grows with pressure),
        # so a handful of observations can otherwise pull the online estimate
        # below the region's own conservative baseline — the opposite of
        # "biased conservatively high" this module promises, and unsafe
        # because a too-low gain makes the controller command an oversized
        # membrane step (membrane_step = requested_sample_step / safe_gain).
        safe_gain = max(median_gain + self._cfg.safety_factor * uncertainty, upper, prior_region.safe_gain)
        estimate = GainEstimate(
            safe_gain=safe_gain,
            estimated_gain=median_gain,
            gain_uncertainty=uncertainty,
            source="observed",
            n_samples=len(gathered),
            rate_limit_gain=max(safe_gain, rate_limit_floor),
        )
        return self._with_rate_limit(estimate, sample_pressure_gpa, rate_limit_floor)
