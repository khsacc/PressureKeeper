"""Deterministic tests for ScheduleRunner, in the same scripted-input style as
test_controller_unit.py: a real OneSidedPressureController drives readings
declared by the test rather than simulator physics, so exact timing/HOLD
transitions are fully controlled.
"""
from __future__ import annotations

import math

import pytest

from pressurekeeper.scheduler import (
    ScheduleRunner,
    ScheduleRunStatus,
    SetPressureStep,
    WaitStep,
    load_schedule,
    save_schedule,
)

from .helpers import build_scripted_controller


def run_schedule_ticks(controller, runner, clock, n, dt=0.25):
    sched = None
    for _ in range(n):
        clock.advance(dt)
        snap = controller.step()
        sched = runner.on_tick(snap, controller)
    return sched


def test_schedule_full_cycle_set_pressure_then_wait(tmp_path):
    controller, ruby, membrane, clock, logger = build_scripted_controller(tmp_path, target=0.0)
    controller.set_max_compression_rate(None)
    runner = ScheduleRunner([SetPressureStep(0.4), WaitStep(2.0)])
    runner.start()

    clock.advance(0.25)
    snap = controller.step()
    sched = runner.on_tick(snap, controller)
    assert sched.status == ScheduleRunStatus.RUNNING
    assert isinstance(sched.step, SetPressureStep)
    assert controller.user_target_gpa == 0.4, "runner must call set_target() on the first tick of a SetPressureStep"

    # Within reach margin of 0.4 (hysteresis.reach_margin_gpa=0.10 -> target-0.10=0.30).
    ruby.push(0.36, n=50)
    sched = None
    for _ in range(20):
        clock.advance(0.25)
        snap = controller.step()
        sched = runner.on_tick(snap, controller)
        if sched.step_index == 1:
            break
    assert isinstance(sched.step, WaitStep), f"expected to have advanced to WaitStep, got {sched.step!r}"
    assert sched.active_elapsed_s == 0.0

    # 7 more ticks (1.75s) must stay below the 2.0s wait duration.
    sched = run_schedule_ticks(controller, runner, clock, n=7)
    assert sched.status == ScheduleRunStatus.RUNNING
    assert sched.active_elapsed_s < 2.0

    # The 8th tick crosses 2.0s -> schedule completes (no more steps).
    sched = run_schedule_ticks(controller, runner, clock, n=1)
    logger.close()
    assert sched.status == ScheduleRunStatus.COMPLETED


def test_schedule_pause_freezes_wait_step_timer(tmp_path):
    controller, ruby, membrane, clock, logger = build_scripted_controller(tmp_path, target=0.0)
    runner = ScheduleRunner([WaitStep(2.0)])
    runner.start()

    # The first on_tick after start() only seeds the internal time baseline
    # (no prior tick to diff against), so n=4 ticks yields 3 dt intervals.
    sched = run_schedule_ticks(controller, runner, clock, n=4)  # 0.75s active
    assert sched.active_elapsed_s == 0.75

    controller.pause("test pause")
    sched = run_schedule_ticks(controller, runner, clock, n=8)  # would be +2.0s if not frozen
    assert sched.status == ScheduleRunStatus.RUNNING
    assert sched.active_elapsed_s == 0.75, "wait timer must not advance while the controller is PAUSEd"

    controller.resume()
    sched = run_schedule_ticks(controller, runner, clock, n=3)  # +0.75s
    logger.close()
    assert sched.active_elapsed_s == 1.5


def test_schedule_aborts_when_controller_aborts(tmp_path):
    controller, ruby, membrane, clock, logger = build_scripted_controller(tmp_path, target=0.0)
    runner = ScheduleRunner([WaitStep(10.0)])
    runner.start()

    sched = run_schedule_ticks(controller, runner, clock, n=1)
    assert sched.status == ScheduleRunStatus.RUNNING

    controller.abort("test abort")
    sched = run_schedule_ticks(controller, runner, clock, n=1)
    logger.close()
    assert sched.status == ScheduleRunStatus.ABORTED
    assert sched.reason is not None


def test_schedule_stop_is_explicit_and_does_not_fight_manual_target(tmp_path):
    controller, ruby, membrane, clock, logger = build_scripted_controller(tmp_path, target=0.0)
    runner = ScheduleRunner([SetPressureStep(0.4), WaitStep(5.0)])
    runner.start()

    run_schedule_ticks(controller, runner, clock, n=1)
    assert controller.user_target_gpa == 0.4

    runner.stop("operator stopped")
    assert runner.status == ScheduleRunStatus.STOPPED

    controller.set_target(0.1)  # Tab1 is unlocked once the schedule is stopped
    sched = run_schedule_ticks(controller, runner, clock, n=1)
    logger.close()
    assert sched.status == ScheduleRunStatus.STOPPED
    assert controller.user_target_gpa == 0.1, "a stopped schedule must not overwrite a manual target change"


def test_schedule_marks_error_on_target_rejected_by_controller(tmp_path):
    controller, ruby, membrane, clock, logger = build_scripted_controller(tmp_path, target=0.0)
    runner = ScheduleRunner([SetPressureStep(999.0)])  # above make_config()'s default 5.0 GPa limit
    runner.start()

    sched = run_schedule_ticks(controller, runner, clock, n=1)
    logger.close()
    assert sched.status == ScheduleRunStatus.ERROR
    assert sched.reason is not None and "999" in sched.reason


def test_schedule_rejects_decreasing_target_instead_of_falsely_completing(tmp_path):
    controller, ruby, membrane, clock, logger = build_scripted_controller(tmp_path, target=1.0)
    runner = ScheduleRunner([SetPressureStep(0.5)])
    runner.start()

    sched = run_schedule_ticks(controller, runner, clock, n=1)
    logger.close()
    assert sched.status == ScheduleRunStatus.ERROR
    assert sched.reason is not None and "below" in sched.reason
    assert controller.user_target_gpa == 1.0


def test_empty_schedule_cannot_start():
    runner = ScheduleRunner([])
    try:
        runner.start()
        assert False, "expected ValueError starting an empty schedule"
    except ValueError:
        pass


def test_step_validation_rejects_bad_values():
    try:
        SetPressureStep(-1.0)
        assert False, "expected ValueError for a negative target"
    except ValueError:
        pass
    try:
        WaitStep(0.0)
        assert False, "expected ValueError for a non-positive duration"
    except ValueError:
        pass


def test_schedule_yaml_roundtrip(tmp_path):
    steps = [SetPressureStep(0.4), WaitStep(600.0), SetPressureStep(1.0), WaitStep(1800.0)]
    path = tmp_path / "schedule.yaml"
    save_schedule(path, steps)
    assert load_schedule(path) == steps


def test_load_schedule_rejects_unknown_step_type(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- type: nonsense\n  value: 1\n", encoding="utf-8")
    try:
        load_schedule(path)
        assert False, "expected ValueError for an unknown step type"
    except ValueError:
        pass


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_schedule_steps_reject_non_finite_values(bad):
    with pytest.raises(ValueError):
        SetPressureStep(bad)
    with pytest.raises(ValueError):
        WaitStep(bad)


@pytest.mark.parametrize(
    "text",
    [
        "type: wait\nduration_s: 1\n",
        "- not-a-mapping\n",
        "- type: wait\n",
        "- type: set_pressure\n  target_gpa: nope\n",
        "- type: [\n",
    ],
)
def test_load_schedule_reports_malformed_yaml_as_value_error(tmp_path, text):
    path = tmp_path / "malformed.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_schedule(path)
