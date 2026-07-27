"""ScheduleRunner: drives a controller through an ordered list of
set-pressure / wait steps.

Deliberately independent of Qt (and of any specific poll loop): a caller
already ticking a controller (the CLI's runner thread, a GUI worker thread,
or a test) calls `on_tick()` once per `ControllerSnapshot` it produces.
`ScheduleRunner` never calls `controller.step()` itself and owns no I/O or
timing thread of its own -- it only reads snapshots and calls
`set_target()`, exactly like an operator would.

Step semantics:
  * `SetPressureStep(target_gpa)` -- calls `set_target()` once, then waits
    until the controller reports `ControlState.HOLD` *for that target*
    before advancing. Reuses the controller's own hysteresis/settle logic;
    no separate "reached" tolerance is defined here.
  * `WaitStep(duration_s)` -- holds whatever target is already active and
    waits out a duration before advancing. Time only accumulates on ticks
    where the controller is not `PAUSE`d, so a safety pause (or a manual
    one) freezes the countdown rather than silently eating it.

A controller `ABORT` latches the schedule into `ABORTED` (mirrors the
controller's own sticky-abort philosophy: no automatic resumption). Safety
or manual `PAUSE` simply freezes progress in place and resumes on its own
once the controller resumes, since `on_tick` re-evaluates from scratch every
call.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol, Union, runtime_checkable

import yaml

from .models import ControllerSnapshot, ControlState


@dataclass(frozen=True)
class SetPressureStep:
    target_gpa: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.target_gpa) or self.target_gpa < 0:
            raise ValueError(
                f"SetPressureStep.target_gpa must be finite and >= 0 (got {self.target_gpa!r})"
            )


@dataclass(frozen=True)
class WaitStep:
    duration_s: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError(f"WaitStep.duration_s must be finite and > 0 (got {self.duration_s!r})")


ScheduleStep = Union[SetPressureStep, WaitStep]


class ScheduleRunStatus(str, Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    ABORTED = "ABORTED"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


@runtime_checkable
class TargetSettable(Protocol):
    def set_target(self, target_gpa: float) -> None:
        """Raises ValueError if target_gpa is rejected (e.g. above the
        configured absolute safety limit)."""
        ...


@dataclass(frozen=True)
class ScheduleSnapshot:
    status: ScheduleRunStatus
    step_index: int | None          # index into the original step list, or None if idle/finished
    step: ScheduleStep | None
    active_elapsed_s: float = 0.0   # WaitStep only: unpaused seconds elapsed on the current step
    reason: str | None = None       # why ABORTED/STOPPED/ERROR


class ScheduleRunner:
    def __init__(self, steps: list[ScheduleStep]) -> None:
        self._steps: list[ScheduleStep] = list(steps)
        self._status = ScheduleRunStatus.IDLE
        self._index = 0
        self._active_elapsed_s = 0.0
        self._target_issued_for_current_step = False
        self._last_tick_t: float | None = None
        self._reason: str | None = None

    @property
    def status(self) -> ScheduleRunStatus:
        return self._status

    @property
    def steps(self) -> list[ScheduleStep]:
        return list(self._steps)

    def start(self) -> None:
        if not self._steps:
            raise ValueError("cannot start an empty schedule")
        self._status = ScheduleRunStatus.RUNNING
        self._index = 0
        self._active_elapsed_s = 0.0
        self._target_issued_for_current_step = False
        self._last_tick_t = None
        self._reason = None

    def stop(self, reason: str = "operator stopped schedule") -> None:
        if self._status == ScheduleRunStatus.RUNNING:
            self._status = ScheduleRunStatus.STOPPED
            self._reason = reason

    def on_tick(self, snap: ControllerSnapshot, target: TargetSettable) -> ScheduleSnapshot:
        if self._status != ScheduleRunStatus.RUNNING:
            return self._snapshot()

        now = snap.t_mono
        dt = 0.0 if self._last_tick_t is None else max(0.0, now - self._last_tick_t)
        self._last_tick_t = now

        if snap.state == ControlState.ABORT:
            self._status = ScheduleRunStatus.ABORTED
            self._reason = "controller entered ABORT"
            return self._snapshot()

        if snap.state == ControlState.PAUSE:
            # Frozen: no target issuance, no wait-timer progress, no
            # reached-target check while paused (safety or manual).
            return self._snapshot()

        step = self._steps[self._index]
        if isinstance(step, SetPressureStep):
            if not self._target_issued_for_current_step:
                if step.target_gpa < snap.user_target_gpa - 1e-9:
                    # A schedule has no operator present to decide what should
                    # happen to the old, higher device setpoint.  Treating the
                    # controller's safety HOLD after a reduction as "target
                    # reached" would silently advance the schedule without
                    # ever reaching that lower pressure.
                    self._status = ScheduleRunStatus.ERROR
                    self._reason = (
                        f"step {self._index}: target {step.target_gpa:.6g} GPa is below "
                        f"the active one-sided target {snap.user_target_gpa:.6g} GPa"
                    )
                    return self._snapshot()
                try:
                    target.set_target(step.target_gpa)
                except ValueError as e:
                    self._status = ScheduleRunStatus.ERROR
                    self._reason = f"step {self._index}: {e}"
                    return self._snapshot()
                self._target_issued_for_current_step = True
                # Don't check "reached" against this same snapshot: it was
                # computed against the *previous* target, before set_target()
                # was just called above.
                return self._snapshot()
            reached = snap.state == ControlState.HOLD and abs(snap.user_target_gpa - step.target_gpa) < 1e-9
            if reached:
                self._advance()
        else:  # WaitStep
            self._active_elapsed_s += dt
            if self._active_elapsed_s >= step.duration_s:
                self._advance()

        return self._snapshot()

    def _advance(self) -> None:
        self._index += 1
        self._active_elapsed_s = 0.0
        self._target_issued_for_current_step = False
        if self._index >= len(self._steps):
            self._status = ScheduleRunStatus.COMPLETED

    def _snapshot(self) -> ScheduleSnapshot:
        step = self._steps[self._index] if 0 <= self._index < len(self._steps) else None
        return ScheduleSnapshot(
            status=self._status,
            step_index=self._index if step is not None else None,
            step=step,
            active_elapsed_s=self._active_elapsed_s,
            reason=self._reason,
        )


def load_schedule(path: str | Path) -> list[ScheduleStep]:
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    except yaml.YAMLError as e:
        raise ValueError(f"invalid schedule YAML: {e}") from e
    if not isinstance(raw, list):
        raise ValueError("schedule YAML must contain a top-level list")
    steps: list[ScheduleStep] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"schedule entry {i} must be a mapping/object")
        kind = entry.get("type")
        if kind == "set_pressure":
            if "target_gpa" not in entry:
                raise ValueError(f"schedule entry {i}: set_pressure requires target_gpa")
            try:
                target_gpa = float(entry["target_gpa"])
            except (TypeError, ValueError) as e:
                raise ValueError(f"schedule entry {i}: target_gpa must be numeric") from e
            steps.append(SetPressureStep(target_gpa=target_gpa))
        elif kind == "wait":
            if "duration_s" not in entry:
                raise ValueError(f"schedule entry {i}: wait requires duration_s")
            try:
                duration_s = float(entry["duration_s"])
            except (TypeError, ValueError) as e:
                raise ValueError(f"schedule entry {i}: duration_s must be numeric") from e
            steps.append(WaitStep(duration_s=duration_s))
        else:
            raise ValueError(f"schedule entry {i}: unknown step type {kind!r} (expected 'set_pressure' or 'wait')")
    return steps


def save_schedule(path: str | Path, steps: list[ScheduleStep]) -> None:
    raw: list[dict] = []
    for step in steps:
        if isinstance(step, SetPressureStep):
            raw.append({"type": "set_pressure", "target_gpa": step.target_gpa})
        else:
            raw.append({"type": "wait", "duration_s": step.duration_s})
    Path(path).write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
