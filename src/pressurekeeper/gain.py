"""GainEstimator: online estimate of dP_sample/dP_membrane, binned by sample
pressure, biased conservatively high so step sizing never under-shoots the
true (larger, at high pressure) response.

Falls back to the configured prior (`GainRegion.safe_gain`) whenever a bin
does not yet have enough observations — this is what makes the very first
steps in a run, or steps into a never-before-visited pressure band, safe.
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
        # One mutable maximum per interrupted command. A PAUSE lasts several
        # ticks; replacing its peak prevents one physical response from being
        # counted as many independent samples.
        self._interrupted_rate_observations: dict[int, tuple[int, float]] = {}

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
        bin_index = self._bin_index(sample_pressure_gpa)
        previous = self._interrupted_rate_observations.get(observation_id)
        if previous is None or raw_rate_gain > previous[1]:
            self._interrupted_rate_observations[observation_id] = (bin_index, raw_rate_gain)

    def discard_interrupted_rate_observation(self, observation_id: int) -> None:
        """Remove an observation later contaminated by another safety cause."""
        self._interrupted_rate_observations.pop(observation_id, None)

    def _interrupted_rate_floor(self, sample_pressure_gpa: float) -> tuple[float, int]:
        if self._cfg.interrupted_rate_learning_mode == "off":
            return 0.0, 0
        center = self._bin_index(sample_pressure_gpa)
        relevant = [
            raw_gain
            for bin_index, raw_gain in self._interrupted_rate_observations.values()
            if (
                bin_index == center
                or (
                    self._cfg.interrupted_rate_propagate_upward
                    and bin_index <= center
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
        rate_source: str = "settled" if settled_floor > configured_floor else "configured"
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
        )

    def estimate(self, sample_pressure_gpa: float, prior_region: GainRegion) -> GainEstimate:
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
