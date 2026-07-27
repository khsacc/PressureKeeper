"""Offline simulator: lets the full control stack be exercised without any
real hardware or network access.

`SimulatedDAC` is the physical model (shared, single source of truth for
simulated time); `SimulatedRubySource` and `SimulatedMembraneController` are
thin `RubyPressureSource` / `MembranePressureController` adapters around it,
so the controller cannot tell them apart from the real HTTP clients.

Physics, deliberately simple but qualitatively faithful:
  * nonlinear steady-state gain: target_sample(m) = base_gain*m*(1+coeff*m)
    -> local slope d(target)/dm grows with m, i.e. gain increases at high
    pressure, as membrane DACs do.
  * dead time: a commanded membrane pressure only starts influencing the
    sample after `dead_time_s`.
  * first-order lag + creep: response is a blend of a fast time constant
    (tau_s) and a much slower one (creep_tau_s), so a step shows a fast
    initial rise followed by continued slow creep.
  * irreversibility: a ratcheted high-water mark on the (dead-time-delayed)
    membrane pressure. Increases follow the normal curve; decreases only
    claw back a small `irreversibility` fraction of the drop, modelling a
    membrane that does not release its deformation.
  * measurement noise + occasional outliers on top of the true value.
"""
from __future__ import annotations

import math
import random
import time
from collections import deque
from dataclasses import dataclass

from ..errors import MembraneCommError, RubyCommError
from ..models import MembraneStatus, RubyPressureSample


@dataclass
class DACPhysicsConfig:
    base_gain_gpa_per_mpa: float = 0.30
    gain_pressure_coeff: float = 0.12
    tau_s: float = 6.0
    dead_time_s: float = 1.5
    creep_weight: float = 0.15
    creep_tau_s: float = 45.0
    irreversibility: float = 0.05
    membrane_ramp_rate_mpa_per_min: float = 5.0
    max_membrane_pressure_mpa: float = 6.0
    measurement_noise_std_gpa: float = 0.004
    outlier_probability: float = 0.0
    outlier_magnitude_gpa: float = 0.3
    physics_substep_s: float = 0.1
    seed: int | None = None

    def __post_init__(self) -> None:
        finite = {
            name: value
            for name, value in vars(self).items()
            if name != "seed"
        }
        if not all(math.isfinite(value) for value in finite.values()):
            raise ValueError(f"DACPhysicsConfig numeric fields must be finite (got {finite})")
        if self.base_gain_gpa_per_mpa <= 0:
            raise ValueError("base_gain_gpa_per_mpa must be > 0")
        if self.gain_pressure_coeff < 0:
            raise ValueError("gain_pressure_coeff must be >= 0")
        if self.tau_s <= 0 or self.creep_tau_s <= 0 or self.physics_substep_s <= 0:
            raise ValueError("tau_s, creep_tau_s, and physics_substep_s must be > 0")
        if self.dead_time_s < 0:
            raise ValueError("dead_time_s must be >= 0")
        if not 0.0 <= self.creep_weight <= 1.0:
            raise ValueError("creep_weight must be in [0, 1]")
        if not 0.0 <= self.irreversibility <= 1.0:
            raise ValueError("irreversibility must be in [0, 1]")
        if self.membrane_ramp_rate_mpa_per_min <= 0 or self.max_membrane_pressure_mpa <= 0:
            raise ValueError("membrane ramp rate and maximum pressure must be > 0")
        if self.measurement_noise_std_gpa < 0 or self.outlier_magnitude_gpa < 0:
            raise ValueError("measurement noise and outlier magnitude must be >= 0")
        if not 0.0 <= self.outlier_probability <= 1.0:
            raise ValueError("outlier_probability must be in [0, 1]")


class SimulatedDAC:
    """Owns simulated time; both adapters call `advance_to()` on the same
    instance, so whichever is called first in a tick integrates the physics
    and the second is a no-op (dt <= 0)."""

    def __init__(self, config: DACPhysicsConfig, start_t: float) -> None:
        self.cfg = config
        self._t = start_t
        self.membrane_setpoint_mpa = 0.0
        self.membrane_actual_mpa = 0.0
        # Mirrors the PACE5000's OUTP:STAT: while False, the device is in
        # measure-only mode and does not drive membrane_actual_mpa toward
        # membrane_setpoint_mpa at all, regardless of what the setpoint is --
        # this is what lets tests actually exercise "does STOP halt an
        # in-flight ramp" rather than trivially passing.
        self.driving = True
        self._delayed_membrane_mpa = 0.0
        self._high_water_mpa = 0.0
        self._fast = 0.0
        self._slow = 0.0
        self._dead_time_queue: deque[tuple[float, float]] = deque()

    def advance_to(self, now: float) -> None:
        remaining = now - self._t
        if remaining <= 0:
            return
        substep = self.cfg.physics_substep_s
        while remaining > 1e-9:
            dt = min(substep, remaining)
            self._substep(dt)
            remaining -= dt

    def _substep(self, dt: float) -> None:
        if self.driving:
            max_delta = self.cfg.membrane_ramp_rate_mpa_per_min / 60.0 * dt
            diff = self.membrane_setpoint_mpa - self.membrane_actual_mpa
            self.membrane_actual_mpa += max(-max_delta, min(max_delta, diff))

        arrival = self._t + dt + self.cfg.dead_time_s
        self._dead_time_queue.append((arrival, self.membrane_actual_mpa))
        while self._dead_time_queue and self._dead_time_queue[0][0] <= self._t + dt:
            self._delayed_membrane_mpa = self._dead_time_queue.popleft()[1]

        m = self._delayed_membrane_mpa
        if m >= self._high_water_mpa:
            self._high_water_mpa = m
            effective_input = m
        else:
            recovered = self.cfg.irreversibility * (self._high_water_mpa - m)
            effective_input = self._high_water_mpa - recovered

        target = self._target_sample(effective_input)
        self._fast += (target - self._fast) / self.cfg.tau_s * dt
        self._slow += (target - self._slow) / self.cfg.creep_tau_s * dt
        self._t += dt

    def _target_sample(self, membrane_mpa: float) -> float:
        return self.cfg.base_gain_gpa_per_mpa * membrane_mpa * (1.0 + self.cfg.gain_pressure_coeff * membrane_mpa)

    @property
    def sample_pressure_gpa(self) -> float:
        return (1.0 - self.cfg.creep_weight) * self._fast + self.cfg.creep_weight * self._slow


class _OutageMixin:
    _offline_until: float | None = None

    def simulate_outage(self, now: float, duration_s: float) -> None:
        self._offline_until = now + duration_s

    def _check_outage(self, now: float, exc_cls: type[Exception], message: str) -> None:
        if self._offline_until is None:
            return
        if now < self._offline_until:
            raise exc_cls(message)
        self._offline_until = None


class SimulatedMembraneController(_OutageMixin):
    def __init__(self, dac: SimulatedDAC, clock) -> None:
        self._dac = dac
        self._clock = clock
        self._setpoint_mpa = 0.0
        self._control_mode = True
        self.commands: list[tuple[float, float, float]] = []
        self.control_mode_commands: list[tuple[float, bool]] = []

    def read_status(self) -> MembraneStatus:
        now = self._clock.now()
        self._check_outage(now, MembraneCommError, "simulated PACE5000 API outage")
        self._dac.advance_to(now)
        return MembraneStatus(
            t_mono=now,
            connected=True,
            pressure_mpa=self._dac.membrane_actual_mpa,
            target_pressure_mpa=self._setpoint_mpa,
            slew_rate_mpa_per_sec=self._dac.cfg.membrane_ramp_rate_mpa_per_min / 60.0,
            control_mode=self._control_mode,
            source_pressure_positive_mpa=self._dac.cfg.max_membrane_pressure_mpa,
            effort_percent=None,
        )

    def set_pressure(self, pressure_mpa: float, rate_mpa_per_min: float) -> None:
        now = self._clock.now()
        self._check_outage(now, MembraneCommError, "simulated PACE5000 API outage")
        self._dac.advance_to(now)
        self._setpoint_mpa = min(pressure_mpa, self._dac.cfg.max_membrane_pressure_mpa)
        self._dac.membrane_setpoint_mpa = self._setpoint_mpa
        self.commands.append((now, pressure_mpa, rate_mpa_per_min))

    def set_control_mode(self, enabled: bool) -> None:
        now = self._clock.now()
        self._check_outage(now, MembraneCommError, "simulated PACE5000 API outage")
        self._dac.advance_to(now)
        self._control_mode = enabled
        self._dac.driving = enabled
        self.control_mode_commands.append((now, enabled))

    def close(self) -> None:
        pass


class SimulatedRubySource(_OutageMixin):
    def __init__(self, dac: SimulatedDAC, clock) -> None:
        self._dac = dac
        self._clock = clock
        self._rng = random.Random(dac.cfg.seed)
        self._pending_outlier_gpa: float | None = None
        self._forced_invalid_remaining = 0

    def inject_outlier(self, magnitude_gpa: float) -> None:
        self._pending_outlier_gpa = magnitude_gpa

    def force_invalid(self, n: int = 1) -> None:
        self._forced_invalid_remaining += n

    def read(self) -> RubyPressureSample:
        now = self._clock.now()
        self._check_outage(now, RubyCommError, "simulated ruby fluorescence API outage")
        self._dac.advance_to(now)

        if self._forced_invalid_remaining > 0:
            self._forced_invalid_remaining -= 1
            return RubyPressureSample(t_mono=now, t_wall=time.time(), pressure_gpa=None,
                                       pressure_err_gpa=None, fit_success=False, r2=None)

        true_p = self._dac.sample_pressure_gpa
        measured = true_p + self._rng.gauss(0.0, self._dac.cfg.measurement_noise_std_gpa)

        if self._pending_outlier_gpa is not None:
            measured += self._pending_outlier_gpa
            self._pending_outlier_gpa = None
        elif self._dac.cfg.outlier_probability > 0 and self._rng.random() < self._dac.cfg.outlier_probability:
            sign = self._rng.choice((-1.0, 1.0))
            measured += sign * self._dac.cfg.outlier_magnitude_gpa

        return RubyPressureSample(t_mono=now, t_wall=time.time(), pressure_gpa=measured,
                                   pressure_err_gpa=0.01, fit_success=True, r2=0.98)

    def close(self) -> None:
        pass
