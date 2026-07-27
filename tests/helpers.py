"""Shared test fixtures: a fast, deterministic Configuration and a helper to
wire the simulator through the exact same `build_app` path production uses.
"""
from __future__ import annotations

from pathlib import Path

from pressurekeeper.app import AppContext, build_app
from pressurekeeper.clock import FakeClock
from pressurekeeper.config import (
    ApproachConfig,
    Configuration,
    ControlConfig,
    EstimatorConfig,
    GainEstimationConfig,
    HysteresisConfig,
    LoggingConfig,
    Pace5000ApiConfig,
    RubyApiConfig,
    SafetyConfig,
)
from pressurekeeper.controller import OneSidedPressureController
from pressurekeeper.errors import MembraneCommError
from pressurekeeper.estimator import PressureEstimator
from pressurekeeper.gain import GainEstimator
from pressurekeeper.logging_sink import DataLogger
from pressurekeeper.models import GainRegion, MembraneStatus, RubyPressureSample
from pressurekeeper.safety import SafetySupervisor
from pressurekeeper.sim import DACPhysicsConfig


def make_config(tmp_path: Path, *, max_sample_pressure_gpa: float = 5.0, dry_run: bool = True) -> Configuration:
    # minimum_settle_time_s must clear the simulator's default dead_time_s
    # (1.5 s, see DACPhysicsConfig) with margin: if it didn't, the controller
    # could mistake "no response has arrived yet" (flat because the dead-time
    # delay hasn't elapsed) for "settled", and stack multiple oversized
    # commands before any of them has shown its real effect. Production
    # config (config/default.yaml) keeps this margin much larger (8-25 s).
    gain_regions = [
        GainRegion(0.0, 0.5, safe_gain=0.20, max_sample_step_gpa=0.10,
                   max_membrane_step=1.0, minimum_settle_time_s=3.0, settled_slope_threshold_gpa_s=0.01),
        GainRegion(0.5, 1.5, safe_gain=0.40, max_sample_step_gpa=0.06,
                   max_membrane_step=0.5, minimum_settle_time_s=4.0, settled_slope_threshold_gpa_s=0.008),
        GainRegion(1.5, 5.0, safe_gain=0.90, max_sample_step_gpa=0.03,
                   max_membrane_step=0.15, minimum_settle_time_s=5.0, settled_slope_threshold_gpa_s=0.006),
    ]
    return Configuration(
        ruby_api=RubyApiConfig(base_url="http://unused.invalid:8765", api_key="test-key", poll_interval_s=0.25),
        # default_rate_mpa_per_min is deliberately faster than the simulator's
        # own (independent) membrane_ramp_rate_mpa_per_min: it only needs to
        # satisfy Configuration's ramp-time-vs-minimum_settle_time_s check
        # below, not describe the simulated physics — tests that care about a
        # slow physical ramp override DACPhysicsConfig directly.
        pace5000_api=Pace5000ApiConfig(base_url="http://unused.invalid:8765", default_rate_mpa_per_min=24.0),
        hysteresis=HysteresisConfig(reach_margin_gpa=0.10, reapproach_margin_gpa=0.15, overshoot_margin_gpa=0.05),
        approach=ApproachConfig(
            approach_margin_gpa=0.05, approach_factor=0.5, prediction_horizon_s=2.0,
            near_target_distance_gpa=0.2, near_target_max_sample_step_gpa=0.015,
            near_target_slope_threshold_scale=0.5, near_target_extra_settle_time_s=1.0,
            min_membrane_step_mpa=0.0005,
        ),
        gain_regions=gain_regions,
        gain_estimation=GainEstimationConfig(bin_width_gpa=0.5, min_samples_for_estimate=2, safety_factor=1.0,
                                              upper_percentile=90.0, neighbor_bins=1),
        estimator=EstimatorConfig(outlier_median_window=5, smoothing_window_s=1.0, slope_window_s=2.0,
                                   min_points_for_valid=3, max_sample_age_s=1.5, max_jump_gpa=0.3,
                                   min_r2=None, min_intensity=None),
        safety=SafetyConfig(
            max_sample_pressure_gpa=max_sample_pressure_gpa, max_membrane_pressure_mpa=6.0,
            max_membrane_step_mpa_hard=1.0, max_cumulative_step_mpa=5.0, cumulative_window_s=30.0,
            max_stale_sample_s=1.5, max_consecutive_invalid=4, max_consecutive_comm_errors=3,
            sample_jump_hard_gpa=0.5, max_consecutive_jump_flags=3, setpoint_mismatch_tol_mpa=0.05,
            setpoint_mismatch_grace_s=0.0, max_setpoint_actual_gap_mpa=1.0,
            minimum_source_pressure_headroom_mpa=0.0,
            ruby_error_pause_after_s=1.0, membrane_error_pause_after_s=1.0,
        ),
        logging=LoggingConfig(directory=str(tmp_path / "logs"), run_name="test", flush_every_n=1000),
        control=ControlConfig(dry_run=dry_run, loop_min_interval_s=0.1, default_target_pressure_gpa=None),
    )


def build_sim_app(
    tmp_path: Path,
    *,
    dry_run: bool = False,
    max_sample_pressure_gpa: float = 5.0,
    physics: DACPhysicsConfig | None = None,
    seed: int = 0,
    start_t: float = 0.0,
) -> tuple[AppContext, FakeClock]:
    config = make_config(tmp_path, max_sample_pressure_gpa=max_sample_pressure_gpa, dry_run=dry_run)
    clock = FakeClock(start_t)
    physics = physics or DACPhysicsConfig(seed=seed, measurement_noise_std_gpa=0.002)
    ctx = build_app(config, use_simulator=True, dry_run=dry_run, clock=clock, sim_physics=physics)
    return ctx, clock


def run_ticks(ctx: AppContext, clock: FakeClock, n: int, dt: float):
    snap = None
    for _ in range(n):
        clock.advance(dt)
        snap = ctx.controller.step()
    return snap


def run_until(ctx: AppContext, clock: FakeClock, dt: float, max_ticks: int, predicate):
    """Advance until `predicate(snapshot)` is true or `max_ticks` is exhausted.

    Returns (last_snapshot, ticks_used, max_filtered_pressure_seen).
    """
    snap = None
    max_p = float("-inf")
    for i in range(max_ticks):
        clock.advance(dt)
        snap = ctx.controller.step()
        if snap.filtered_pressure_gpa is not None:
            max_p = max(max_p, snap.filtered_pressure_gpa)
        if predicate(snap):
            return snap, i + 1, max_p
    return snap, max_ticks, max_p


class ScriptedRubySource:
    """A ruby source whose readings are scripted directly by a test, rather
    than derived from simulated physics -- lets tests place `filtered` at an
    exact value without waiting out lag/settle dynamics."""

    def __init__(self, clock: FakeClock) -> None:
        self._clock = clock
        self._values: list[float] = [0.0]

    def push(self, value: float, n: int = 1) -> None:
        self._values.extend([value] * n)

    def read(self) -> RubyPressureSample:
        p = self._values.pop(0) if len(self._values) > 1 else self._values[0]
        now = self._clock.now()
        return RubyPressureSample(t_mono=now, t_wall=now, pressure_gpa=p, pressure_err_gpa=0.01,
                                   fit_success=True, r2=0.99)


class ScriptedMembrane:
    """A membrane with an instantaneous response only while in Control.

    While in Measure, set_pressure changes only the stored target. This is
    essential for verifying the safe staging order used on every re-arm.
    """

    def __init__(self, clock: FakeClock) -> None:
        self._clock = clock
        self.setpoint = 0.0
        self.actual = 0.0
        self.control_mode = True
        self.source_pressure = 10.0
        self.commands: list[tuple[float, float]] = []
        self.control_mode_commands: list[tuple[float, bool]] = []
        self.operations: list[tuple[str, float, float | bool]] = []
        self.fail_control_mode_writes = 0  # test hook: raise this many times before succeeding

    def read_status(self) -> MembraneStatus:
        now = self._clock.now()
        return MembraneStatus(t_mono=now, connected=True, pressure_mpa=self.actual,
                               target_pressure_mpa=self.setpoint, control_mode=self.control_mode,
                               source_pressure_positive_mpa=self.source_pressure)

    def set_pressure(self, pressure_mpa: float, rate_mpa_per_min: float) -> None:
        self.setpoint = pressure_mpa
        if self.control_mode:
            self.actual = pressure_mpa
        self.commands.append((self._clock.now(), pressure_mpa))
        self.operations.append(("set_pressure", self._clock.now(), pressure_mpa))

    def set_control_mode(self, enabled: bool) -> None:
        if self.fail_control_mode_writes > 0:
            self.fail_control_mode_writes -= 1
            raise MembraneCommError("scripted control_mode write failure")
        self.control_mode = enabled
        if enabled:
            self.actual = self.setpoint
        self.control_mode_commands.append((self._clock.now(), enabled))
        self.operations.append(("control_mode", self._clock.now(), enabled))


def build_scripted_controller(tmp_path: Path, dry_run: bool = False, target: float = 1.0):
    """A fully wired controller over ScriptedRubySource/ScriptedMembrane
    (not the physics simulator), for deterministic state-machine tests."""
    config = make_config(tmp_path, dry_run=dry_run)
    clock = FakeClock(0.0)
    ruby = ScriptedRubySource(clock)
    membrane = ScriptedMembrane(clock)
    estimator = PressureEstimator(config.estimator)
    gain_estimator = GainEstimator(config.gain_estimation)
    safety = SafetySupervisor(config.safety, start_t=0.0)
    logger = DataLogger(config.logging)
    controller = OneSidedPressureController(config, ruby, membrane, estimator, gain_estimator, safety,
                                             logger=logger, clock=clock)
    controller.set_target(target)
    return controller, ruby, membrane, clock, logger


def tick(controller: OneSidedPressureController, clock: FakeClock, n: int = 1, dt: float = 0.25):
    snap = None
    for _ in range(n):
        clock.advance(dt)
        snap = controller.step()
    return snap
