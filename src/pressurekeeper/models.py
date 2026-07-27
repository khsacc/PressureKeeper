"""Shared data models for PressureKeeper.

Units are fixed throughout the codebase to avoid silent conversion bugs:
  * sample (ruby) pressure  -> GPa
  * membrane gas pressure   -> MPa
  * time                    -> seconds, on `time.monotonic()` unless stated otherwise
  * gain (sample/membrane)  -> GPa / MPa
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class ControlState(str, Enum):
    APPROACH = "APPROACH"
    SETTLE = "SETTLE"
    HOLD = "HOLD"
    PAUSE = "PAUSE"
    ABORT = "ABORT"


@dataclass(frozen=True)
class RubyPressureSample:
    """One reading from the ruby fluorescence API.

    `t_mono`/`t_wall` are stamped by the client at receipt time, not parsed
    from the remote payload, so slope/staleness math is always comparable
    against our own monotonic clock regardless of the remote PC's clock skew.
    """

    t_mono: float
    t_wall: float
    pressure_gpa: float | None
    pressure_err_gpa: float | None
    fit_success: bool
    r2: float | None = None
    intensity: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_usable(self) -> bool:
        return (
            self.fit_success
            and self.pressure_gpa is not None
            and math.isfinite(self.pressure_gpa)
        )


@dataclass(frozen=True)
class MembraneStatus:
    t_mono: float
    connected: bool
    pressure_mpa: float | None = None
    target_pressure_mpa: float | None = None
    slew_rate_mpa_per_sec: float | None = None
    control_mode: bool | None = None
    source_pressure_positive_mpa: float | None = None
    effort_percent: float | None = None


@dataclass(frozen=True)
class GainRegion:
    """Gain-scheduling entry: physical behaviour of one sample-pressure band.

    `safe_gain` is the conservative prior (GPa of sample pressure per MPa of
    membrane pressure) used until enough online observations accumulate near
    this band. All numeric values are device-specific and must come from
    configuration, never be hardcoded.
    """

    sample_pressure_min_gpa: float
    sample_pressure_max_gpa: float
    safe_gain: float
    max_sample_step_gpa: float
    max_membrane_step: float
    minimum_settle_time_s: float
    settled_slope_threshold_gpa_s: float

    def __post_init__(self) -> None:
        values = (self.sample_pressure_min_gpa, self.sample_pressure_max_gpa, self.safe_gain,
                  self.max_sample_step_gpa, self.max_membrane_step, self.minimum_settle_time_s,
                  self.settled_slope_threshold_gpa_s)
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"GainRegion fields must all be finite (got {values})")
        if self.sample_pressure_min_gpa >= self.sample_pressure_max_gpa:
            raise ValueError(
                f"GainRegion sample_pressure_min_gpa ({self.sample_pressure_min_gpa}) must be < "
                f"sample_pressure_max_gpa ({self.sample_pressure_max_gpa})"
            )
        if self.safe_gain <= 0:
            # membrane_step = requested_sample_step / safe_gain in
            # controller._maybe_issue_step guards on safe_gain <= 0 by
            # silently returning every tick -- a non-positive prior here
            # freezes control in this band with no error ever logged.
            raise ValueError(f"GainRegion.safe_gain must be > 0 (got {self.safe_gain})")
        if self.max_sample_step_gpa <= 0:
            raise ValueError(f"GainRegion.max_sample_step_gpa must be > 0 (got {self.max_sample_step_gpa})")
        if self.max_membrane_step <= 0:
            raise ValueError(f"GainRegion.max_membrane_step must be > 0 (got {self.max_membrane_step})")
        if self.minimum_settle_time_s < 0:
            raise ValueError(f"GainRegion.minimum_settle_time_s must be >= 0 (got {self.minimum_settle_time_s})")
        if self.settled_slope_threshold_gpa_s <= 0:
            raise ValueError(
                f"GainRegion.settled_slope_threshold_gpa_s must be > 0 (got {self.settled_slope_threshold_gpa_s})"
            )

    def contains(self, sample_pressure_gpa: float) -> bool:
        return self.sample_pressure_min_gpa <= sample_pressure_gpa < self.sample_pressure_max_gpa


@dataclass
class StepRecord:
    """One membrane-pressure command, tracked as an independent response test
    from the moment it is issued until the next command is issued."""

    step_id: int
    t_command: float
    membrane_pressure_before: float
    membrane_pressure_after: float
    sample_pressure_before: float
    reason: str
    decision: dict[str, Any] = field(default_factory=dict)

    t_settled: float | None = None
    sample_pressure_after: float | None = None
    max_slope_gpa_s: float = 0.0
    measurement_std_gpa: float | None = None
    settled: bool = False
    # True if the command that opened this step raised a MembraneCommError
    # (response lost/timed out) -- the write may or may not have actually
    # applied on the device; see OneSidedPressureController._maybe_issue_step.
    ack_uncertain: bool = False
    # A recovery command is first staged while the PACE5000 is in Measure.
    # Physical response timing starts only after its readback is confirmed
    # and Control is enabled.
    t_drive_started: float | None = None

    @property
    def response_time_s(self) -> float | None:
        if self.t_settled is None:
            return None
        return self.t_settled - (self.t_drive_started or self.t_command)

    @property
    def observed_gain(self) -> float | None:
        if self.sample_pressure_after is None or not self.settled:
            return None
        d_membrane = self.membrane_pressure_after - self.membrane_pressure_before
        if d_membrane <= 0:
            return None
        return (self.sample_pressure_after - self.sample_pressure_before) / d_membrane

    @property
    def midpoint_sample_pressure_gpa(self) -> float | None:
        if self.sample_pressure_after is None:
            return None
        return 0.5 * (self.sample_pressure_before + self.sample_pressure_after)


@dataclass(frozen=True)
class GainEstimate:
    safe_gain: float
    estimated_gain: float
    gain_uncertainty: float
    source: Literal["prior", "observed"]
    n_samples: int


@dataclass(frozen=True)
class SafetyEvent:
    t_mono: float
    code: str
    severity: Literal["info", "warning", "pause", "abort"]
    message: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SafetyVerdict:
    level: Literal["ok", "pause", "abort"]
    events: tuple[SafetyEvent, ...] = ()

    @property
    def forbids_pressurization(self) -> bool:
        return self.level != "ok"


@dataclass(frozen=True)
class StateTransition:
    t_mono: float
    from_state: ControlState
    to_state: ControlState
    reason: str


@dataclass(frozen=True)
class ControllerSnapshot:
    """Everything the CLI / logger needs to display "what is happening now"."""

    t_mono: float
    state: ControlState
    user_target_gpa: float
    control_target_gpa: float
    raw_pressure_gpa: float | None
    filtered_pressure_gpa: float | None
    pressure_slope_gpa_s: float | None
    predicted_pressure_gpa: float | None
    measurement_std_gpa: float | None
    measurement_r2: float | None
    estimator_valid: bool
    membrane_setpoint_mpa: float | None
    membrane_actual_mpa: float | None
    safe_gain: float | None
    last_command_reason: str | None
    manual_pause: bool
    safety_level: Literal["ok", "pause", "abort"]
    safety_reasons: tuple[str, ...] = ()
    max_compression_rate_gpa_per_min: float | None = None
    membrane_rate_mpa_per_min: float | None = None
    source_pressure_positive_mpa: float | None = None
    logging_error: str | None = None
