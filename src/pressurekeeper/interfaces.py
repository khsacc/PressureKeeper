"""Abstract device I/O boundaries.

Real implementations live in `pressurekeeper.clients`; a fully self-contained
implementation lives in `pressurekeeper.sim`. Control/safety/estimation code
must only depend on these Protocols, never on `requests`, hardware SDKs, or
the simulator directly, so the two are interchangeable behind one interface.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import MembraneStatus, RubyPressureSample


@runtime_checkable
class RubyPressureSource(Protocol):
    def read(self) -> RubyPressureSample:
        """Trigger (or wait for) one acquisition and return its result.

        Must raise `pressurekeeper.errors.RubyCommError` on any transport or
        protocol failure rather than returning a sentinel value — callers
        rely on the exception to distinguish "no peak found" (a normal,
        successfully-parsed response with `fit_success=False`) from "could
        not talk to the spectrometer PC at all".
        """
        ...


@runtime_checkable
class MembranePressureController(Protocol):
    def read_status(self) -> MembraneStatus:
        """Raises `pressurekeeper.errors.MembraneCommError` on I/O failure."""
        ...

    def set_pressure(self, pressure_mpa: float, rate_mpa_per_min: float) -> None:
        """Command a new membrane setpoint and ramp rate.

        A recovery path may lower a stale device target while active control
        is disabled (Measure); it never requests an actual pressure decrease.
        Must raise `pressurekeeper.errors.MembraneCommError` on I/O failure
        or device-side rejection (e.g. target above source pressure). Must be
        a no-op write (but still succeed and be logged) when the controller
        was constructed in dry-run mode.
        """
        ...

    def set_control_mode(self, enabled: bool) -> None:
        """Enable (True) or disable (False) active pressure control.

        Disabling is this codebase's STOP mechanism: it halts any in-flight
        ramp regardless of setpoint math, by switching the device from
        actively driving a setpoint to measure-only. Must raise
        `pressurekeeper.errors.MembraneCommError` on I/O failure. Must be a
        no-op write (but still succeed and be logged) when the controller
        was constructed in dry-run mode.
        """
        ...
