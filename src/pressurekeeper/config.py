"""Configuration models and loader.

All device-specific numbers (gains, thresholds, limits, URLs) live in a
YAML or TOML file — nothing device-specific is hardcoded in the control
logic. See config/default.yaml for a documented example.
"""
from __future__ import annotations

import math
import os
import re
import tomllib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from .models import GainRegion


class RubyAcquisitionConfig(BaseModel):
    """Body fields sent to POST /acquire/pressure on the ruby API.

    Mirrors FluoraPressee's AcquirePressureRequest schema; kept here (rather
    than passed ad hoc) so an operator can retune fitting/exposure without
    touching code.
    """

    configuration_id: str | None = None
    axis_mode: Literal["calibrated", "pixel"] | None = None
    exposure_time_s: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    accumulations: int | None = Field(default=None, ge=1)
    dark_mode: Literal["none", "reuse_loaded", "provided"] = "reuse_loaded"

    @model_validator(mode="after")
    def _axis_mode_requires_explicit_configuration(self) -> "RubyAcquisitionConfig":
        # Mirrors FluoraPressee's own AcquireRequest validator
        # (lab_andor/FluoraPressee/src/api/schemas.py) -- it rejects
        # axis_mode set without an explicit configuration_id with HTTP 422.
        # Catching this at config-load time means a bad config fails fast
        # instead of 422ing every single /acquire/pressure call.
        if self.axis_mode is not None and self.configuration_id is None:
            raise ValueError("axis_mode can only be set together with an explicit configuration_id")
        return self

    fit_function: Literal["Pseudo Voigt", "Moffat", "Gauss", "Lorentz", "Diamond Raman Edge"] = "Pseudo Voigt"
    fit_peak_count: int = Field(default=1, ge=1, le=5)
    peak_sort_order: Literal["x_desc", "x_asc", "intensity_desc", "intensity_asc"] = "x_desc"
    baseline_model: Literal["constant", "linear", "quadratic", "auto_polynomial"] = "constant"
    fit_range: tuple[float, float] | None = None

    sensor: str = "ruby"
    pressure_scale: str = "ruby_shen_2020"
    zero_pressure_peak: float = Field(default=694.3, gt=0, allow_inf_nan=False)
    pressure_peak_index: int = Field(default=1, ge=1, le=5)

    @model_validator(mode="after")
    def _request_is_accepted_by_fluorapressee(self) -> "RubyAcquisitionConfig":
        if self.pressure_peak_index > self.fit_peak_count:
            raise ValueError("pressure_peak_index must be less than or equal to fit_peak_count")
        if self.fit_function == "Diamond Raman Edge" and self.fit_peak_count != 1:
            raise ValueError("Diamond Raman Edge fitting requires fit_peak_count=1")
        if self.dark_mode == "provided":
            # This client intentionally has no field for transmitting a dark
            # spectrum, and _build_body() sends data=None.  Accepting
            # dark_mode=provided here would therefore guarantee HTTP 422 on
            # every acquisition.
            raise ValueError("dark_mode='provided' is unsupported because no dark spectrum data is configured")
        if self.fit_range is not None:
            start, end = self.fit_range
            if not (math.isfinite(start) and math.isfinite(end) and start < end):
                raise ValueError("fit_range must contain two finite values with start < end")
        if not self.sensor.strip() or not self.pressure_scale.strip():
            raise ValueError("sensor and pressure_scale must be non-empty")
        return self


class RubyApiConfig(BaseModel):
    base_url: str = "http://127.0.0.1:8765"
    api_key: str
    timeout_s: float = Field(default=5.0, gt=0, allow_inf_nan=False)
    poll_interval_s: float = Field(
        default=0.25, gt=0, allow_inf_nan=False,
        description="Target cadence; do not exceed ~4 Hz (0.25 s).",
    )
    acquisition: RubyAcquisitionConfig = Field(default_factory=RubyAcquisitionConfig)

    @field_validator("poll_interval_s")
    @classmethod
    def _cap_poll_rate(cls, v: float) -> float:
        if v < 0.2:
            raise ValueError("poll_interval_s below 0.2 s exceeds the ~4 Hz measurement limit")
        return v


class Pace5000ApiConfig(BaseModel):
    base_url: str = "http://127.0.0.1:8765"
    api_key: str | None = None
    timeout_s: float = Field(default=5.0, gt=0, allow_inf_nan=False)
    status_poll_interval_s: float = Field(default=0.25, gt=0, allow_inf_nan=False)
    default_rate_mpa_per_min: float = Field(default=5.0, gt=0, allow_inf_nan=False)
    ensure_control_mode_enabled: bool = True
    # After we write set_control_mode(True), the PACE5000 control app's own
    # status endpoint can lag behind the write by more than one poll interval
    # before it reports control_mode=True (observed on real hardware: a
    # write can 200 OK before the device has actually latched into remote
    # control). Without this grace window, SafetySupervisor treats that
    # still-False readback as an external relinquish and PAUSEs, which
    # re-stops and re-arms every tick -- see membrane_control_mode_disabled.
    control_mode_resume_grace_s: float = Field(default=2.0, ge=0, allow_inf_nan=False)


class HysteresisConfig(BaseModel):
    reach_margin_gpa: float = Field(default=0.10, ge=0, allow_inf_nan=False)
    reapproach_margin_gpa: float = Field(default=0.15, ge=0, allow_inf_nan=False)
    overshoot_margin_gpa: float = Field(default=0.05, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _ordering(self) -> "HysteresisConfig":
        if not (self.reach_margin_gpa <= self.reapproach_margin_gpa):
            raise ValueError("reapproach_margin_gpa should be >= reach_margin_gpa (re-approach only once we've fallen back further than 'reached')")
        return self


class ApproachConfig(BaseModel):
    approach_margin_gpa: float = Field(default=0.05, ge=0, allow_inf_nan=False)
    approach_factor: float = Field(default=0.5, gt=0.0, le=1.0)
    prediction_horizon_s: float = Field(default=5.0, ge=0, allow_inf_nan=False)
    near_target_distance_gpa: float = Field(default=0.2, gt=0, allow_inf_nan=False)
    near_target_max_sample_step_gpa: float = Field(default=0.015, gt=0, allow_inf_nan=False)
    near_target_slope_threshold_scale: float = Field(default=0.5, gt=0.0, le=1.0)
    near_target_extra_settle_time_s: float = Field(default=5.0, ge=0, allow_inf_nan=False)
    min_membrane_step_mpa: float = Field(default=0.001, gt=0, allow_inf_nan=False)
    membrane_arrival_tolerance_mpa: float = Field(default=0.01, ge=0, allow_inf_nan=False)
    # Extra buffer added on top of a step's own physical ramp time (membrane_step
    # / gas-side slew rate) when computing that step's settle blackout -- see
    # OneSidedPressureController._update_pending_step. Keeps a slow
    # membrane_rate_mpa_per_min (e.g. one sized to respect
    # max_compression_rate_gpa_per_min at high gain) from ever letting settle
    # detection fire while the membrane is still mid-ramp toward this step's
    # own target, without requiring a fast fixed rate the way the old
    # rate-vs-minimum_settle_time_s floor did.
    ramp_time_margin_s: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    # Operator-facing ceiling on how fast sample pressure is allowed to rise,
    # independent of (and further restricting) gain_regions[].max_sample_step_gpa.
    # None = no additional cap (existing per-region step caps still apply).
    # Runtime-adjustable via OneSidedPressureController.set_max_compression_rate();
    # this is only the default a controller starts with.
    max_compression_rate_gpa_per_min: float | None = Field(default=0.5, allow_inf_nan=False)

    @field_validator("max_compression_rate_gpa_per_min")
    @classmethod
    def _positive_rate(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError(f"max_compression_rate_gpa_per_min must be > 0 when set (got {v})")
        return v


class GainEstimationConfig(BaseModel):
    # ``legacy_regions`` retains the original pressure-binned gain schedule.
    # ``adaptive_local`` identifies dP_sample/dP_membrane from the current
    # loading itself and uses direct gas-pressure probes until a local slope is
    # available.  The model default stays legacy for backwards-compatible
    # programmatic construction; config/default.yaml opts into adaptive_local.
    step_sizing_mode: Literal["legacy_regions", "adaptive_local"] = "legacy_regions"
    bin_width_gpa: float = Field(default=0.5, gt=0, allow_inf_nan=False)
    min_samples_for_estimate: int = Field(default=3, ge=1)
    safety_factor: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    upper_percentile: float = Field(default=90.0, gt=50.0, lt=100.0)
    neighbor_bins: int = Field(default=1, ge=0)
    # Adaptive-local static-gain estimation and probing.  These are
    # experiment-independent exploration bounds, not a calibrated DAC gain
    # curve. A no-response step is allowed to grow only after it has genuinely
    # settled and never by more than probe_growth_factor.
    local_pressure_window_gpa: float = Field(default=0.25, gt=0, allow_inf_nan=False)
    local_max_observations: int = Field(default=5, ge=1)
    local_gain_safety_factor: float = Field(default=1.25, ge=1.0, allow_inf_nan=False)
    local_uncertainty_safety_factor: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    local_curvature_safety_factor: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    # A real-hardware run (logs/run_20260729T164838_686358) showed the
    # curvature/trend term dominated by measurement noise between two
    # closely-spaced observations (e.g. 0.0068 GPa apart) rather than genuine
    # gain curvature, producing a "slope" over 9 GPa/MPa per GPa and pushing
    # safe_gain to ~4x the largest gain ever actually observed in that run.
    # Below this minimum pressure span across the local window's candidates,
    # the trend is treated as unknown (0.0) rather than trusted.
    local_trend_min_span_gpa: float = Field(default=0.05, ge=0, allow_inf_nan=False)
    response_detection_sigma: float = Field(default=3.0, gt=0, allow_inf_nan=False)
    response_detection_floor_gpa: float = Field(default=0.002, gt=0, allow_inf_nan=False)
    initial_probe_step_mpa: float = Field(default=0.02, gt=0, allow_inf_nan=False)
    probe_growth_factor: float = Field(default=1.6, gt=1.0, allow_inf_nan=False)
    max_probe_step_mpa: float = Field(default=0.20, gt=0, allow_inf_nan=False)
    # Global high-side physical envelope used only while the local static gain
    # is unknown. It ties every probe to the remaining sample-pressure budget:
    # probe <= requested_sample_step / this value.
    adaptive_probe_max_expected_gain: float = Field(
        default=5.0, gt=0, allow_inf_nan=False
    )
    # Extra observation time after the gas ramp has completed before a
    # statistically flat response may be classified as "not detected" and
    # permit the next probe to grow.
    adaptive_no_response_wait_s: float = Field(
        default=30.0, gt=0, allow_inf_nan=False
    )
    # SITE-SPECIFIC relaxation of the two probe controls above while the
    # membrane/gas pressure itself is still below a level where a sample-
    # pressure response is physically implausible (e.g. below the cell's
    # engagement threshold). Gated on actual membrane pressure, not sample
    # pressure or distance-to-target -- this is about the known-flat
    # low-gas-pressure regime specifically. None (default) disables this
    # entirely; probe sizing/wait then always use
    # adaptive_probe_max_expected_gain / adaptive_no_response_wait_s as
    # before, at every pressure.
    low_pressure_probe_membrane_ceiling_mpa: float | None = Field(
        default=None, gt=0, allow_inf_nan=False
    )
    low_pressure_probe_max_step_mpa: float = Field(default=0.05, gt=0, allow_inf_nan=False)
    low_pressure_probe_no_response_wait_s: float = Field(default=15.0, gt=0, allow_inf_nan=False)
    max_step_growth_factor: float = Field(default=1.5, ge=1.0, allow_inf_nan=False)
    rate_exceeded_step_backoff_factor: float = Field(default=0.5, gt=0, le=1.0)
    probe_rate_mpa_per_min: float = Field(default=0.5, gt=0, allow_inf_nan=False)
    adaptive_max_sample_step_gpa: float = Field(default=0.03, gt=0, allow_inf_nan=False)
    adaptive_max_membrane_step_mpa: float = Field(default=0.25, gt=0, allow_inf_nan=False)
    adaptive_minimum_settle_time_s: float = Field(default=12.0, ge=0, allow_inf_nan=False)
    adaptive_settled_slope_threshold_gpa_s: float = Field(
        default=0.002, gt=0, allow_inf_nan=False
    )
    # Interrupted steps are never mixed into the settled/static-gain data.
    # They can, however, provide a conservative lower bound for the separate
    # dynamic slew-rate limiter. ``observe`` computes/logs that bound without
    # changing commands; ``enforce`` applies it to subsequent commands.
    interrupted_rate_learning_mode: Literal["off", "observe", "enforce"] = "observe"
    interrupted_rate_safety_factor: float = Field(default=1.25, ge=1.0, allow_inf_nan=False)
    interrupted_rate_propagate_upward: bool = True


class EstimatorConfig(BaseModel):
    outlier_median_window: int = Field(default=7, ge=3)
    smoothing_window_s: float = Field(default=3.0, gt=0, allow_inf_nan=False)
    slope_window_s: float = Field(default=6.0, gt=0, allow_inf_nan=False)
    min_points_for_valid: int = Field(default=3, ge=2)
    max_sample_age_s: float = Field(default=3.0, gt=0, allow_inf_nan=False)
    max_jump_gpa: float = Field(default=0.3, gt=0, allow_inf_nan=False)
    min_r2: float | None = Field(default=None, le=1.0, allow_inf_nan=False)
    min_intensity: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class SafetyConfig(BaseModel):
    max_sample_pressure_gpa: float = Field(gt=0, allow_inf_nan=False)
    max_membrane_pressure_mpa: float = Field(default=6.0, gt=0, allow_inf_nan=False)
    max_membrane_step_mpa_hard: float = Field(default=0.5, gt=0, allow_inf_nan=False)
    max_cumulative_step_mpa: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    cumulative_window_s: float = Field(default=60.0, gt=0, allow_inf_nan=False)
    max_stale_sample_s: float = Field(default=5.0, gt=0, allow_inf_nan=False)
    max_consecutive_invalid: int = Field(default=5, ge=1)
    max_consecutive_comm_errors: int = Field(default=3, ge=1)
    sample_jump_hard_gpa: float = Field(default=0.5, gt=0, allow_inf_nan=False)
    max_consecutive_jump_flags: int = Field(default=3, ge=1)
    setpoint_mismatch_tol_mpa: float = Field(default=0.05, ge=0, allow_inf_nan=False)
    setpoint_mismatch_grace_s: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    max_setpoint_actual_gap_mpa: float = Field(default=0.5, gt=0, allow_inf_nan=False)
    minimum_source_pressure_headroom_mpa: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    ruby_error_pause_after_s: float = Field(default=3.0, ge=0, allow_inf_nan=False)
    membrane_error_pause_after_s: float = Field(default=3.0, ge=0, allow_inf_nan=False)


class LoggingConfig(BaseModel):
    directory: str = "./logs"
    run_name: str | None = None
    flush_every_n: int = Field(default=1, ge=1)


class ControlConfig(BaseModel):
    dry_run: bool = True
    loop_min_interval_s: float = Field(default=0.2, gt=0, allow_inf_nan=False)
    default_target_pressure_gpa: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class Configuration(BaseModel):
    ruby_api: RubyApiConfig
    pace5000_api: Pace5000ApiConfig = Field(default_factory=Pace5000ApiConfig)
    hysteresis: HysteresisConfig = Field(default_factory=HysteresisConfig)
    approach: ApproachConfig = Field(default_factory=ApproachConfig)
    gain_regions: list[GainRegion]
    gain_estimation: GainEstimationConfig = Field(default_factory=GainEstimationConfig)
    estimator: EstimatorConfig = Field(default_factory=EstimatorConfig)
    safety: SafetyConfig
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    control: ControlConfig = Field(default_factory=ControlConfig)

    @field_validator("gain_regions")
    @classmethod
    def _regions_nonempty_sorted(cls, v: list[GainRegion]) -> list[GainRegion]:
        if not v:
            raise ValueError("gain_regions must contain at least one region")
        return sorted(v, key=lambda r: r.sample_pressure_min_gpa)

    @model_validator(mode="after")
    def _region_steps_within_hard_cap(self) -> "Configuration":
        # A region allowing a bigger per-command step than the global hard cap
        # is not just "clamped" — SafetySupervisor.check_command() rejects the
        # whole command outright, and the controller silently retries the same
        # oversized step forever, freezing progress with no visible error.
        offending = [
            r for r in self.gain_regions
            if r.max_membrane_step > self.safety.max_membrane_step_mpa_hard
        ]
        if offending:
            bad = ", ".join(
                f"[{r.sample_pressure_min_gpa}-{r.sample_pressure_max_gpa}) GPa: "
                f"max_membrane_step={r.max_membrane_step} > hard cap"
                for r in offending
            )
            raise ValueError(
                f"safety.max_membrane_step_mpa_hard ({self.safety.max_membrane_step_mpa_hard}) "
                f"must be >= every gain_regions[].max_membrane_step, otherwise those steps are "
                f"unconditionally rejected and the controller can never progress: {bad}"
            )
        return self

    @model_validator(mode="after")
    def _adaptive_steps_within_hard_cap(self) -> "Configuration":
        gain = self.gain_estimation
        if gain.step_sizing_mode != "adaptive_local":
            return self
        if gain.initial_probe_step_mpa > gain.max_probe_step_mpa:
            raise ValueError(
                "gain_estimation.initial_probe_step_mpa must be <= max_probe_step_mpa"
            )
        if gain.max_probe_step_mpa > gain.adaptive_max_membrane_step_mpa:
            raise ValueError(
                "gain_estimation.max_probe_step_mpa must be <= "
                "adaptive_max_membrane_step_mpa"
            )
        if (
            gain.adaptive_max_membrane_step_mpa
            > self.safety.max_membrane_step_mpa_hard
        ):
            raise ValueError(
                "gain_estimation.adaptive_max_membrane_step_mpa must be <= "
                "safety.max_membrane_step_mpa_hard"
            )
        if (
            gain.low_pressure_probe_membrane_ceiling_mpa is not None
            and gain.low_pressure_probe_max_step_mpa > gain.adaptive_max_membrane_step_mpa
        ):
            raise ValueError(
                "gain_estimation.low_pressure_probe_max_step_mpa must be <= "
                "adaptive_max_membrane_step_mpa"
            )
        return self

    @model_validator(mode="after")
    def _active_setpoint_gap_accepts_legal_steps(self) -> "Configuration":
        if self.safety.max_setpoint_actual_gap_mpa < self.safety.max_membrane_step_mpa_hard:
            raise ValueError(
                "safety.max_setpoint_actual_gap_mpa must be >= "
                "safety.max_membrane_step_mpa_hard; otherwise a legal step can "
                "immediately PAUSE itself while the PACE5000 is ramping"
            )
        return self

    @model_validator(mode="after")
    def _default_target_within_absolute_limit(self) -> "Configuration":
        target = self.control.default_target_pressure_gpa
        if target is not None and target > self.safety.max_sample_pressure_gpa:
            raise ValueError(
                f"control.default_target_pressure_gpa ({target}) exceeds "
                f"safety.max_sample_pressure_gpa ({self.safety.max_sample_pressure_gpa})"
            )
        return self

    @model_validator(mode="after")
    def _regions_contiguous(self) -> "Configuration":
        # region_for() falls back to gain_regions[0] -- the region with the
        # loosest step cap -- for any pressure that doesn't land in a band.
        # A gap between bands would silently hand out that loosest cap deep
        # inside what looks like a covered range, so bands must tile the
        # axis with no gaps (and no overlaps, which sorting alone can't rule
        # out either).
        for prev, nxt in zip(self.gain_regions, self.gain_regions[1:]):
            if prev.sample_pressure_max_gpa != nxt.sample_pressure_min_gpa:
                raise ValueError(
                    f"gain_regions must be contiguous: region ending at "
                    f"{prev.sample_pressure_max_gpa} GPa is followed by one starting at "
                    f"{nxt.sample_pressure_min_gpa} GPa, leaving a gap/overlap where "
                    f"region_for() would fall back to gain_regions[0]"
                )
        return self

    @model_validator(mode="after")
    def _cumulative_cap_has_headroom(self) -> "Configuration":
        # If the sliding-window cumulative cap is no bigger than the single-
        # command hard cap, one max-size command exhausts the entire
        # cumulative_window_s budget outright, and every later command is
        # rejected with cumulative_step_cap_exceeded until that command ages
        # out of the window -- effectively "one command per window" instead
        # of the intended burst-limiting behaviour.
        if self.safety.max_cumulative_step_mpa <= self.safety.max_membrane_step_mpa_hard:
            raise ValueError(
                f"safety.max_cumulative_step_mpa ({self.safety.max_cumulative_step_mpa}) must be > "
                f"max_membrane_step_mpa_hard ({self.safety.max_membrane_step_mpa_hard}), otherwise a "
                f"single max-size command exhausts the whole cumulative_window_s budget and every "
                f"later command is rejected until it ages out"
            )
        return self

    def region_for(self, sample_pressure_gpa: float) -> GainRegion:
        for region in self.gain_regions:
            if region.contains(sample_pressure_gpa):
                return region
        # Above the last region's upper bound (e.g. pressure right at/above the
        # configured ceiling): fall back to the highest region rather than
        # raising, since SafetySupervisor's absolute cap is the real guard.
        if sample_pressure_gpa >= self.gain_regions[-1].sample_pressure_max_gpa:
            return self.gain_regions[-1]
        return self.gain_regions[0]


def _read_raw(path: Path) -> dict:
    suffix = path.suffix.lower()
    if suffix == ".toml":
        with path.open("rb") as f:
            return tomllib.load(f)
    if suffix in (".yaml", ".yml"):
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    raise ValueError(f"Unsupported config file extension: {suffix!r} (use .yaml/.yml/.toml)")


_ENV_VAR_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


def _expand_env_vars(value: object) -> object:
    """Recursively replace any string value that is *entirely* "${NAME}"
    with os.environ["NAME"], failing fast (rather than silently leaving the
    literal placeholder in place) if it's unset -- so secrets like
    ruby_api.api_key / pace5000_api.api_key never need to be committed to a
    config file (see config/default.yaml's placeholders and CLAUDE.md's
    "never commit the real key here"). Only a whole-string match expands
    (not substitution inside a larger string) to keep this unambiguous.
    """
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    if isinstance(value, str):
        m = _ENV_VAR_PATTERN.match(value)
        if m:
            name = m.group(1)
            if name not in os.environ:
                raise ValueError(
                    f"config value {value!r} references environment variable {name!r}, "
                    "which is not set"
                )
            return os.environ[name]
    return value


def load_config(path: str | Path) -> Configuration:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = _expand_env_vars(_read_raw(path))
    return Configuration.model_validate(raw)


_REDACTED_RUBY_API_KEY_PLACEHOLDER = "${PRESSUREKEEPER_RUBY_API_KEY}"
_REDACTED_PACE5000_API_KEY_PLACEHOLDER = "${PRESSUREKEEPER_PACE5000_API_KEY}"


def redact_api_keys(dumped: dict) -> dict:
    """Replace already-expanded API-key secrets in a `Configuration.model_dump()`
    dict with the same "${ENV_VAR}" placeholder convention config/default.yaml's
    own comments recommend (see `_expand_env_vars` above) -- for anywhere a
    dumped config is persisted outside the process's own memory (a GUI-saved
    YAML file, a run's manifest.json), where the literal secret
    `_expand_env_vars` resolved at load time must never land. The dict this
    produces still loads cleanly via `load_config()` as long as the named
    environment variable is set again; it is not meant to be loaded as-is.
    """
    dumped = dict(dumped)
    ruby = dict(dumped["ruby_api"])
    ruby["api_key"] = _REDACTED_RUBY_API_KEY_PLACEHOLDER
    dumped["ruby_api"] = ruby
    pace = dict(dumped["pace5000_api"])
    if pace.get("api_key"):
        pace["api_key"] = _REDACTED_PACE5000_API_KEY_PLACEHOLDER
        dumped["pace5000_api"] = pace
    return dumped
