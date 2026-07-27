"""Deterministic, scripted-input tests of the OneSidedPressureController state
machine — no simulator physics, so exact pressure trajectories are fully
controlled by the test. Full closed-loop behaviour against realistic DAC
physics is covered separately in test_scenarios.py.
"""
from __future__ import annotations

from pressurekeeper.errors import MembraneCommError
from pressurekeeper.models import ControlState

from .helpers import build_scripted_controller as build
from .helpers import build_sim_app, make_config, run_until, tick


def test_membrane_setpoint_never_decreases(tmp_path):
    # Uses the real closed-loop simulator (not the scripted double above):
    # discrete scripted jumps create artificial slope spikes at each level
    # change that don't reflect a real continuous physical response.
    ctx, clock = build_sim_app(tmp_path, dry_run=False, seed=42)
    ctx.controller.set_target(1.0)
    run_until(ctx, clock, dt=0.25, max_ticks=20_000, predicate=lambda s: s.state == ControlState.HOLD)
    ctx.close()

    assert len(ctx.membrane.commands) >= 2, "expected multiple membrane steps over this run"
    setpoints = [p for _, p, _ in ctx.membrane.commands]
    assert all(a <= b + 1e-9 for a, b in zip(setpoints, setpoints[1:])), \
        f"membrane setpoint must never decrease in normal control: {setpoints}"


def test_settle_blackout_enforced_between_commands(tmp_path):
    controller, ruby, membrane, clock, logger = build(tmp_path, target=2.0)
    for level in [0.1, 0.2, 0.3, 0.4]:
        ruby.push(level, n=32)
        tick(controller, clock, n=32)
    logger.close()

    times = [t for t, _ in membrane.commands]
    assert len(times) >= 2
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert all(g >= 3.0 - 1e-6 for g in gaps), f"a new command was issued before region 0's minimum_settle_time_s (3.0s): {gaps}"


def test_dry_run_never_calls_set_pressure_and_stays_approach(tmp_path):
    controller, ruby, membrane, clock, logger = build(tmp_path, dry_run=True, target=1.0)
    ruby.push(0.3, n=20)  # far from target - reach margin, so a step attempt is guaranteed
    snap = tick(controller, clock, n=20)
    logger.close()

    assert membrane.commands == [], "dry_run must never call MembranePressureController.set_pressure"
    assert snap.state == ControlState.APPROACH
    assert snap.last_command_reason is not None and "[dry-run]" in snap.last_command_reason


def test_dry_run_can_rehearse_while_real_output_is_measure_only(tmp_path):
    controller, ruby, membrane, clock, logger = build(tmp_path, dry_run=True, target=1.0)
    membrane.control_mode = False
    ruby.push(0.1, n=20)
    snap = tick(controller, clock, n=20)
    logger.close()

    assert snap.state == ControlState.APPROACH
    assert snap.safety_level == "ok"
    assert membrane.commands == []
    assert membrane.control_mode_commands == []
    assert snap.last_command_reason is not None and "[dry-run]" in snap.last_command_reason


def test_startup_rebases_stale_setpoint_in_measure_before_control(tmp_path):
    controller, ruby, membrane, clock, logger = build(tmp_path, target=1.0)
    controller.set_max_compression_rate(None)
    membrane.actual = 0.2
    membrane.setpoint = 5.0
    membrane.operations.clear()

    tick(controller, clock, n=8)
    logger.close()

    kinds = [op[0] for op in membrane.operations]
    stop_i = next(i for i, op in enumerate(membrane.operations)
                  if op[0] == "control_mode" and op[2] is False)
    stage_i = kinds.index("set_pressure")
    arm_i = next(i for i, op in enumerate(membrane.operations)
                 if op[0] == "control_mode" and op[2] is True)
    staged = float(membrane.operations[stage_i][2])
    assert stop_i < stage_i < arm_i
    assert 0.2 < staged <= 0.2 + controller.config.safety.max_membrane_step_mpa_hard
    assert staged < 5.0, "the stale target must be lowered while Measure before Control"


def test_reset_stays_measure_then_recalculates_before_control(tmp_path):
    controller, ruby, membrane, clock, logger = build(tmp_path, target=1.0)
    tick(controller, clock, n=8)
    controller.abort("test reset sequence")
    assert membrane.control_mode is False

    # Simulate a stale target remaining in the PACE5000 after the interrupted
    # ramp. Reset must not briefly re-arm it.
    membrane.setpoint = 4.0
    actual_at_reset = membrane.actual
    controller.reset()
    membrane.operations.clear()
    assert membrane.control_mode is False

    tick(controller, clock, n=5)
    logger.close()

    stage_i = next(i for i, op in enumerate(membrane.operations) if op[0] == "set_pressure")
    arm_i = next(i for i, op in enumerate(membrane.operations)
                 if op[0] == "control_mode" and op[2] is True)
    staged = float(membrane.operations[stage_i][2])
    assert stage_i < arm_i
    assert staged < 4.0
    assert staged >= actual_at_reset


def test_manual_control_mode_waits_until_safe_setpoint_readback(tmp_path):
    controller, ruby, membrane, clock, logger = build(tmp_path, target=1.0)
    controller.config.pace5000_api.ensure_control_mode_enabled = False
    membrane.setpoint = 4.0
    membrane.actual = 0.2
    tick(controller, clock, n=8)

    assert membrane.commands
    assert membrane.control_mode is False
    assert not any(enabled for _, enabled in membrane.control_mode_commands)
    staged = membrane.setpoint
    assert staged < 4.0

    # The operator enables Control only after seeing the safely staged target.
    membrane.set_control_mode(True)
    tick(controller, clock, n=1)
    logger.close()
    assert membrane.control_mode is True
    assert controller.safety.membrane_stop_intended is False


def test_insufficient_supply_blocks_setpoint_and_pauses(tmp_path):
    controller, ruby, membrane, clock, logger = build(tmp_path, target=1.0)
    membrane.source_pressure = 0.05
    snap = tick(controller, clock, n=8)

    assert snap.state == ControlState.PAUSE
    assert membrane.commands == []
    assert not any(op[0] == "control_mode" and op[2] is True for op in membrane.operations)

    membrane.source_pressure = 1.0
    controller.resume()
    snap = tick(controller, clock, n=6)
    logger.close()
    assert membrane.commands
    assert membrane.control_mode is True
    assert membrane.source_pressure > membrane.setpoint


def test_pressure_write_waits_for_fresh_same_tick_supply_status(tmp_path):
    controller, ruby, membrane, clock, logger = build(tmp_path, target=1.0)
    controller.config.pace5000_api.status_poll_interval_s = 1.0

    # First status at 0.25 s establishes startup state/Measure. Ruby feedback
    # becomes valid before the next status is due, but cached supply pressure
    # must not authorize a write.
    tick(controller, clock, n=4, dt=0.25)
    assert membrane.commands == []

    tick(controller, clock, n=1, dt=0.25)  # fresh status at 1.25 s
    logger.close()
    assert membrane.commands, "a fresh same-tick supply status should now authorize staging"


def test_logging_initialization_failure_is_visible_without_pause(tmp_path, monkeypatch):
    from pressurekeeper import app as app_module
    from pressurekeeper.clock import FakeClock

    def fail_logger(_config):
        raise OSError("simulated unwritable log directory")

    monkeypatch.setattr(app_module, "DataLogger", fail_logger)
    clock = FakeClock(0.0)
    ctx = app_module.build_app(
        make_config(tmp_path, dry_run=False),
        use_simulator=True,
        dry_run=False,
        clock=clock,
    )
    clock.advance(0.25)
    snap = ctx.controller.step()
    ctx.close()

    assert ctx.logger is None
    assert snap.logging_error is not None
    assert "simulated unwritable" in snap.logging_error
    assert snap.safety_level != "pause"


def test_hold_reached_and_no_overshoot_beyond_margin(tmp_path):
    controller, ruby, membrane, clock, logger = build(tmp_path, target=1.0)
    ruby.push(0.96, n=20)  # already within reach margin (target - 0.10 = 0.90)
    snap = tick(controller, clock, n=20)
    logger.close()

    assert snap.state == ControlState.HOLD
    assert snap.filtered_pressure_gpa <= 1.0 + 0.05 + 1e-6


def test_small_dip_during_hold_does_not_reapproach(tmp_path):
    controller, ruby, membrane, clock, logger = build(tmp_path, target=1.0)
    ruby.push(0.96, n=16)
    tick(controller, clock, n=16)
    assert controller.state == ControlState.HOLD

    # Dip to 0.90: below the reach margin (0.90) but still above the wider
    # re-approach margin (target - 0.15 = 0.85) -> must stay HOLD.
    ruby.push(0.90, n=16)
    snap = tick(controller, clock, n=16)
    logger.close()
    assert snap.state == ControlState.HOLD
    assert membrane.commands == [], "a small dip within the hysteresis band must not trigger a new step"


def test_large_dip_during_hold_triggers_reapproach(tmp_path):
    controller, ruby, membrane, clock, logger = build(tmp_path, target=1.0)
    ruby.push(0.96, n=16)
    tick(controller, clock, n=16)
    assert controller.state == ControlState.HOLD

    # Dip to 0.80: below target - reapproach_margin (0.85) -> must resume APPROACH.
    ruby.push(0.80, n=16)
    snap = tick(controller, clock, n=16)
    logger.close()
    assert snap.state in (ControlState.APPROACH, ControlState.SETTLE)


def test_manual_pause_blocks_new_commands_and_resume_continues(tmp_path):
    controller, ruby, membrane, clock, logger = build(tmp_path, target=1.0)
    controller.pause("test pause")
    ruby.push(0.5, n=20)
    snap = tick(controller, clock, n=20)
    assert snap.state == ControlState.PAUSE
    assert membrane.commands == []

    controller.resume()
    snap = tick(controller, clock, n=20)
    logger.close()
    assert snap.state != ControlState.PAUSE
    assert len(membrane.commands) >= 1


def test_set_target_above_absolute_limit_rejected(tmp_path):
    controller, ruby, membrane, clock, logger = build(tmp_path, target=1.0)
    logger.close()
    try:
        controller.set_target(999.0)
        assert False, "expected ValueError for a target above the configured absolute safety limit"
    except ValueError:
        pass


def test_set_target_rejects_non_finite_and_negative_values(tmp_path):
    controller, ruby, membrane, clock, logger = build(tmp_path, target=1.0)
    logger.close()
    for bad in (float("nan"), float("inf"), float("-inf"), -0.1):
        try:
            controller.set_target(bad)
            assert False, f"expected ValueError for target={bad!r}"
        except ValueError:
            pass


def test_raising_target_from_hold_rearms_and_continues(tmp_path):
    controller, ruby, membrane, clock, logger = build(tmp_path, target=0.5)
    ruby.push(0.46, n=20)
    snap = tick(controller, clock, n=20)
    assert snap.state == ControlState.HOLD
    assert membrane.control_mode is False

    controller.set_target(1.0)
    ruby.push(0.46, n=20)
    snap = tick(controller, clock, n=20)
    logger.close()
    assert snap.state in (ControlState.APPROACH, ControlState.SETTLE)
    assert membrane.control_mode is True
    assert membrane.commands, "the higher target must not leave the output permanently stopped"


def test_max_compression_rate_caps_step_size(tmp_path):
    # Region 0 (0.0-0.5 GPa) allows up to 0.10 GPa/step with
    # minimum_settle_time_s=3.0s -> an unrestricted step here could reach an
    # average rate of up to 0.10/3.0*60 = 2.0 GPa/min. Capping at 0.5 GPa/min
    # must shrink the first commanded step below what an uncapped run would
    # issue.
    controller, ruby, membrane, clock, logger = build(tmp_path, target=0.45)
    controller.set_max_compression_rate(0.5)
    ruby.push(0.0, n=20)
    snap = tick(controller, clock, n=20)
    logger.close()

    assert len(membrane.commands) >= 1, "expected at least one membrane command"
    first_setpoint = membrane.commands[0][1]
    safe_gain = 0.20  # region 0's configured prior in make_config()
    implied_sample_step = first_setpoint * safe_gain
    max_allowed_step = (0.5 / 60.0) * 3.0  # rate_gpa_per_min/60 * minimum_settle_time_s
    assert implied_sample_step <= max_allowed_step + 1e-9, (
        f"first step implies a {implied_sample_step:.4f} GPa sample step, "
        f"exceeding the {max_allowed_step:.4f} GPa cap from a 0.5 GPa/min compression-rate limit"
    )
    assert snap.max_compression_rate_gpa_per_min == 0.5


def test_max_compression_rate_none_leaves_behavior_unchanged(tmp_path):
    controller, ruby, membrane, clock, logger = build(tmp_path, target=0.45)
    assert controller.max_compression_rate_gpa_per_min == 0.5
    controller.set_max_compression_rate(None)
    ruby.push(0.0, n=20)
    snap = tick(controller, clock, n=20)
    logger.close()
    assert snap.max_compression_rate_gpa_per_min is None


def test_set_max_compression_rate_rejects_non_positive(tmp_path):
    controller, ruby, membrane, clock, logger = build(tmp_path, target=1.0)
    logger.close()
    for bad in (0.0, -1.0):
        try:
            controller.set_max_compression_rate(bad)
            assert False, f"expected ValueError for max_compression_rate_gpa_per_min={bad}"
        except ValueError:
            pass


def test_manual_pause_stops_membrane_control_mode_immediately(tmp_path):
    # pause()'s emergency-stop attempt runs synchronously, independent of
    # step() -- control_mode should already be off before any tick.
    controller, ruby, membrane, clock, logger = build(tmp_path, target=1.0)
    # n=20 (not a handful of ticks): a scripted step-function jump from the
    # initial 0.0 to 0.5 produces a huge transient slope until enough ticks
    # have passed for the filter/prediction to settle -- too few ticks and
    # that artificial slope spike alone can (correctly) trigger HOLD, which
    # isn't what this test is about.
    ruby.push(0.5, n=20)
    tick(controller, clock, n=20)
    assert membrane.control_mode is True
    controller.pause("test pause")
    assert membrane.control_mode is False, "pause's direct emergency stop must not wait for a tick"
    snap = tick(controller, clock, n=1)
    logger.close()
    assert snap.state == ControlState.PAUSE


def test_hold_stops_membrane_control_mode(tmp_path):
    controller, ruby, membrane, clock, logger = build(tmp_path, target=1.0)
    ruby.push(0.96, n=20)  # already within reach margin (target - 0.10 = 0.90)
    snap = tick(controller, clock, n=20)
    logger.close()
    assert snap.state == ControlState.HOLD
    assert membrane.control_mode is False, "HOLD must stop the membrane, not just withhold new commands"


def test_resume_from_pause_re_arms_control_mode_before_next_step(tmp_path):
    controller, ruby, membrane, clock, logger = build(tmp_path, target=1.0)
    controller.pause("test pause")
    assert membrane.control_mode is False
    ruby.push(0.1, n=5)
    tick(controller, clock, n=5)
    assert membrane.control_mode is False, "must stay stopped while manually paused"

    controller.resume()
    snap = tick(controller, clock, n=10)
    logger.close()
    assert membrane.control_mode is True, "must re-arm control mode before issuing a new setpoint"
    assert len(membrane.commands) >= 1


def test_abort_stop_retries_after_a_failed_first_attempt(tmp_path):
    # Regression: the old one-shot abort freeze latched "aborted" even if
    # the freeze write itself failed, and never tried again.
    controller, ruby, membrane, clock, logger = build(tmp_path, target=1.0)
    ruby.push(0.5, n=20)  # let the artificial step-jump transient settle first
    tick(controller, clock, n=20)

    membrane.fail_control_mode_writes = 2
    controller.abort("test abort")
    assert membrane.control_mode is True, "first (direct) attempt was made to fail"

    snap = tick(controller, clock, n=1)
    assert snap.state == ControlState.ABORT
    assert membrane.control_mode is True, "second attempt (step()'s retry) was also made to fail"

    snap = tick(controller, clock, n=1)
    logger.close()
    assert snap.state == ControlState.ABORT
    assert membrane.control_mode is False, "third attempt must succeed -- the retry must not give up"


def test_logger_failure_does_not_prevent_hold_stop(tmp_path):
    controller, ruby, membrane, clock, logger = build(tmp_path, target=1.0)

    def raise_on_log_event(*args, **kwargs):
        raise RuntimeError("simulated disk-full logging failure")
    logger.log_event = raise_on_log_event

    ruby.push(0.96, n=20)  # within reach margin -> HOLD
    snap = tick(controller, clock, n=20)
    logger.close()
    assert snap.state == ControlState.HOLD
    assert membrane.control_mode is False, "a logging failure must never prevent the STOP action"
    assert snap.logging_error is not None
    assert "simulated disk-full" in snap.logging_error
    assert snap.safety_level != "pause", "logging failure itself must not cause PAUSE"


def test_ambiguous_write_outcome_is_not_stacked_on(tmp_path):
    controller, ruby, membrane, clock, logger = build(tmp_path, target=2.0)
    # This test scripts an instantaneous sample-pressure jump; disable the
    # independent dP/dt interlock so it isolates ambiguous write handling.
    controller.set_max_compression_rate(None)
    ruby.push(0.1, n=5)
    tick(controller, clock, n=5)

    original_set_pressure = membrane.set_pressure

    def failing_set_pressure(pressure_mpa, rate_mpa_per_min):
        original_set_pressure(pressure_mpa, rate_mpa_per_min)  # actually applies...
        raise MembraneCommError("response lost")  # ...but the response is lost

    membrane.set_pressure = failing_set_pressure
    snap = tick(controller, clock, n=1)
    assert snap.state == ControlState.SETTLE
    commands_after_ambiguous = len(membrane.commands)
    assert commands_after_ambiguous >= 1

    membrane.set_pressure = original_set_pressure
    snap = tick(controller, clock, n=3)
    logger.close()
    assert len(membrane.commands) == commands_after_ambiguous, (
        "must not stack a new command on top of an ambiguous (comm-error) write instead of "
        "waiting for its settle/mismatch reconciliation"
    )


def test_step_clamped_at_region_boundary(tmp_path):
    # Region 0 spans 0.0-0.5 GPa (safe_gain=0.20); filtered sits only 0.05
    # GPa from that boundary, well under region 0's own max_sample_step_gpa
    # (0.10) -- without clamping to the boundary, the controller would size
    # a step assuming region 0's gain applies all the way across a step that
    # actually crosses into region 1's higher gain.
    controller, ruby, membrane, clock, logger = build(tmp_path, target=2.0)
    ruby.push(0.45, n=20)
    tick(controller, clock, n=20)
    logger.close()
    assert len(membrane.commands) >= 1
    first_setpoint = membrane.commands[0][1]
    safe_gain = 0.20  # region 0's configured prior in make_config()
    implied_sample_step = first_setpoint * safe_gain  # current_setpoint starts at 0.0
    assert implied_sample_step <= 0.05 + 1e-9, (
        f"step must not cross the 0.5 GPa region boundary using region 0's gain: "
        f"implied sample-pressure step {implied_sample_step:.4f} GPa"
    )
