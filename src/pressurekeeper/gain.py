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
            return GainEstimate(
                safe_gain=prior_region.safe_gain,
                estimated_gain=prior_region.safe_gain,
                gain_uncertainty=0.0,
                source="prior",
                n_samples=len(gathered),
                rate_limit_gain=max(prior_region.safe_gain, rate_limit_floor),
            )

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
        return GainEstimate(
            safe_gain=safe_gain,
            estimated_gain=median_gain,
            gain_uncertainty=uncertainty,
            source="observed",
            n_samples=len(gathered),
            rate_limit_gain=max(safe_gain, rate_limit_floor),
        )
