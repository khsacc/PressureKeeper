"""Deterministic, scripted-input tests of the OneSidedPressureController state
machine — no simulator physics, so exact pressure trajectories are fully
controlled by the test. Full closed-loop behaviour against realistic DAC
physics is covered separately in test_scenarios.py.
"""
from __future__ import annotations

from dataclasses import replace as dataclass_replace

from pressurekeeper.errors import MembraneCommError
from pressurekeeper.models import ControlState

from .helpers import build_scripted_controller as build
from .helpers import build_sim_app, make_config, run_until, tick


def test_membrane_setpoint_never_retreats_more_than_one_step(tmp_path):
    # Uses the real closed-loop simulator (not the scripted double above):
    # discrete scripted jumps create artificial slope spikes at each level
    # change that don't reflect a real continuous physical response.
    #
    # A compression-rate-only PAUSE may retreat the setpoint down to the
    # fresh actual membrane pressure (never below it) rather than continuing
    # an in-flight step that is producing too fast a sample-pressure rise --
    # see OneSidedPressureController._try_hold_at_current_pressure. Since
    # that PAUSE typically fires mid-ramp, the retreat can land below the
    # just-written target. This is intentional (never an unbounded/actual
    # depressurization -- the design's premise against reversing course is
    # about the predictor's model, not a hard safety wall): bound it to at
    # most one step's worth of give-back, not a strict non-decrease.
    config = make_config(tmp_path, dry_run=False)
    ctx, clock = build_sim_app(tmp_path, dry_run=False, seed=42)
    ctx.controller.set_target(1.0)
    run_until(ctx, clock, dt=0.25, max_ticks=20_000, predicate=lambda s: s.state == ControlState.HOLD)
    ctx.close()

    assert len(ctx.membrane.commands) >= 2, "expected multiple membrane steps over this run"
    setpoints = [p for _, p, _ in ctx.membrane.commands]
    max_step = config.safety.max_membrane_step_mpa_hard
    assert all(a - max_step <= b + 1e-9 for a, b in zip(setpoints, setpoints[1:])), \
        f"membrane setpoint retreated by more than one step's worth: {setpoints}"


def test_settle_blackout_enforced_between_commands(tmp_path):
    controller, ruby, membrane, clock, logger = build(tmp_path, target=2.0)
    initial_actual = membrane.actual
    for level in [0.1, 0.2, 0.3, 0.4]:
        ruby.push(level, n=32)
        tick(controller, clock, n=32)
    logger.close()

    # A compression-rate-only PAUSE may insert a zero-net-effect hold write
    # (see OneSidedPressureController._try_hold_at_current_pressure) that
    # repeats the current actual pressure rather than sizing a new approach
    # step. It carries no unresolved physical motion to wait out, so it does
    # not need to respect the settle blackout the way a real step does --
    # only gaps between commands that actually advance beyond the highest
    # setpoint reached so far matter (a hold write, including a possible
    # first one before any real step, never exceeds that high-water mark).
    running_max = initial_actual
    real_steps = []
    for t, p in membrane.commands:
        if p > running_max + 1e-9:
            real_steps.append((t, p))
            running_max = p
    times = [t for t, _ in real_steps]
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

    def fail_logger(_config, **_kwargs):
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


def test_lowering_safety_ceiling_stops_approach_toward_a_stale_higher_target(tmp_path):
    """Regression: apply_config_update() (GUI "Configure Parameters" -> Save &
    Apply) used to only warn when the operator's existing target exceeded a
    newly-lowered safety.max_sample_pressure_gpa -- user_target_gpa itself was
    left untouched and nothing re-checked it before sizing the next step, so
    approach continued toward the stale target until the hard
    sample_pressure_over_limit abort in safety.py eventually tripped.
    """
    # 0.45 GPa sits within the new 0.5 GPa ceiling's reach margin (target -
    # reach_margin_gpa = 0.40) but nowhere near the stale 1.0 GPa target's
    # (0.90), and stays safely under both the hard 0.5 GPa abort ceiling and
    # the hard-jump/compression-rate thresholds a single scripted jump from
    # 0.0 could otherwise trip.
    controller, ruby, membrane, clock, logger = build(tmp_path, target=1.0)
    ruby.push(0.45, n=5)
    tick(controller, clock, n=5)
    commands_before = len(membrane.commands)

    lowered = make_config(tmp_path, max_sample_pressure_gpa=0.5, dry_run=False)
    controller.apply_config_update(lowered)
    assert controller.user_target_gpa == 1.0, "apply_config_update must not silently rewrite the operator's target"

    ruby.push(0.45, n=30)
    snap = tick(controller, clock, n=30)
    logger.close()

    assert snap.state == ControlState.HOLD, (
        "0.45 GPa is already within the new 0.5 GPa ceiling's reach margin -- approach must stop, not keep "
        "driving toward the stale 1.0 GPa target"
    )
    assert len(membrane.commands) == commands_before, (
        "no further membrane step should be issued once the effective (ceiling-clamped) target is reached"
    )


def test_raising_safety_ceiling_back_up_lets_approach_resume(tmp_path):
    # The ceiling clamp (_effective_target_gpa) never latches like a target
    # reduction does: raising safety.max_sample_pressure_gpa back up must let
    # approach continue toward the still-unchanged operator target without
    # requiring the operator to re-enter it.
    controller, ruby, membrane, clock, logger = build(tmp_path, target=1.0)
    ruby.push(0.45, n=5)
    tick(controller, clock, n=5)

    controller.apply_config_update(make_config(tmp_path, max_sample_pressure_gpa=0.5, dry_run=False))
    ruby.push(0.45, n=30)
    snap = tick(controller, clock, n=30)
    assert snap.state == ControlState.HOLD

    controller.apply_config_update(make_config(tmp_path, max_sample_pressure_gpa=5.0, dry_run=False))
    ruby.push(0.45, n=30)
    snap = tick(controller, clock, n=30)
    logger.close()
    assert snap.state != ControlState.HOLD, "raising the ceiling back up must let approach resume automatically"


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


def test_set_membrane_rate_mpa_per_min_updates_and_is_used_on_next_command(tmp_path):
    controller, ruby, membrane, clock, logger = build(tmp_path, target=1.0)
    assert controller.membrane_rate_mpa_per_min == 24.0  # make_config()'s default_rate_mpa_per_min
    # This test is about the raw operator-set rate being honoured, not its
    # interaction with max_compression_rate_gpa_per_min (see
    # test_max_compression_rate_caps_the_effective_membrane_rate for that).
    controller.set_max_compression_rate(None)

    controller.set_membrane_rate_mpa_per_min(30.0)
    assert controller.membrane_rate_mpa_per_min == 30.0

    seen_rates: list[float] = []
    original_set_pressure = membrane.set_pressure

    def spying_set_pressure(pressure_mpa, rate_mpa_per_min):
        seen_rates.append(rate_mpa_per_min)
        original_set_pressure(pressure_mpa, rate_mpa_per_min)

    membrane.set_pressure = spying_set_pressure
    ruby.push(0.0, n=5)
    snap = tick(controller, clock, n=5)
    logger.close()

    assert seen_rates, "expected at least one set_pressure command"
    assert all(r == 30.0 for r in seen_rates)
    assert snap.membrane_rate_mpa_per_min == 30.0


def test_max_compression_rate_caps_the_effective_membrane_rate(tmp_path):
    # Regression for the real-hardware compression-rate overshoot: sizing the
    # requested *sample* step for the cap (test_max_compression_rate_caps_step_size)
    # only bounds the average rate over minimum_settle_time_s -- the real,
    # front-loaded response can still spike well above it. Capping the actual
    # gas-side slew rate sent with each command protects the instantaneous
    # rate too. See OneSidedPressureController._maybe_issue_step.
    controller, ruby, membrane, clock, logger = build(tmp_path, target=1.0)
    controller.set_membrane_rate_mpa_per_min(1000.0)  # operator configured a fast rate
    controller.set_max_compression_rate(0.5)  # GPa/min

    seen_rates: list[float] = []
    original_set_pressure = membrane.set_pressure

    def spying_set_pressure(pressure_mpa, rate_mpa_per_min):
        seen_rates.append(rate_mpa_per_min)
        original_set_pressure(pressure_mpa, rate_mpa_per_min)

    membrane.set_pressure = spying_set_pressure
    ruby.push(0.0, n=5)
    tick(controller, clock, n=5)
    logger.close()

    assert seen_rates, "expected at least one set_pressure command"
    region0_safe_gain = controller.config.gain_regions[0].safe_gain
    expected_cap = 0.5 / region0_safe_gain
    assert all(r <= expected_cap + 1e-9 for r in seen_rates), \
        f"effective rate must respect max_compression_rate_gpa_per_min / safe_gain: {seen_rates} vs cap {expected_cap}"
    assert all(r < 1000.0 for r in seen_rates), \
        "the operator-configured rate alone must not be sent uncapped when a compression cap is active"


def test_max_compression_rate_uses_rate_limit_gain_not_safe_gain(tmp_path):
    # Regression: the dynamic gas-side rate cap used to divide by safe_gain,
    # so a region's safe_gain prior turning out to be optimistic relative to
    # real hardware silently made the rate cap optimistic too (see
    # config/default.yaml's gain_regions note). A region with an explicit
    # rate_limit_gain above its safe_gain must size the cap from the former.
    controller, ruby, membrane, clock, logger = build(tmp_path, target=1.0)
    controller.set_membrane_rate_mpa_per_min(1000.0)
    controller.set_max_compression_rate(0.5)  # GPa/min

    region0 = controller.config.gain_regions[0]
    assert region0.rate_limit_gain is None, "test assumes helpers.make_config() leaves this unset"
    inflated_rate_limit_gain = region0.safe_gain * 2.0
    widened_config = controller.config.model_copy(update={
        "gain_regions": [
            dataclass_replace(region0, rate_limit_gain=inflated_rate_limit_gain),
            *controller.config.gain_regions[1:],
        ],
    })
    controller.apply_config_update(widened_config)
    controller.set_membrane_rate_mpa_per_min(1000.0)
    controller.set_max_compression_rate(0.5)

    seen_rates: list[float] = []
    original_set_pressure = membrane.set_pressure

    def spying_set_pressure(pressure_mpa, rate_mpa_per_min):
        seen_rates.append(rate_mpa_per_min)
        original_set_pressure(pressure_mpa, rate_mpa_per_min)

    membrane.set_pressure = spying_set_pressure
    ruby.push(0.0, n=5)
    tick(controller, clock, n=5)
    logger.close()

    assert seen_rates, "expected at least one set_pressure command"
    expected_cap = 0.5 / inflated_rate_limit_gain
    stale_cap_using_safe_gain = 0.5 / region0.safe_gain
    assert all(r <= expected_cap + 1e-9 for r in seen_rates), \
        f"effective rate must respect max_compression_rate_gpa_per_min / rate_limit_gain: {seen_rates} vs cap {expected_cap}"
    assert all(r < stale_cap_using_safe_gain for r in seen_rates), \
        "rate_limit_gain must actually tighten the cap below what plain safe_gain would have allowed"


def test_set_membrane_rate_mpa_per_min_rejects_non_positive(tmp_path):
    controller, ruby, membrane, clock, logger = build(tmp_path, target=1.0)
    logger.close()
    for bad in (0.0, -1.0):
        try:
            controller.set_membrane_rate_mpa_per_min(bad)
            assert False, f"expected ValueError for membrane_rate_mpa_per_min={bad}"
        except ValueError:
            pass


def test_set_membrane_rate_mpa_per_min_accepts_a_rate_slower_than_the_old_settle_floor(tmp_path):
    # make_config()'s region 0 has max_membrane_step=1.0, minimum_settle_time_s=3.0
    # -- a rate slower than 1.0/3.0*60 = 20.0 MPa/min used to be rejected
    # outright (Configuration._region_ramp_time_within_settle). That floor
    # made it impossible to slow the gas-side slew enough to respect
    # max_compression_rate_gpa_per_min at high gain. It's gone now: see
    # test_slow_membrane_rate_extends_settle_blackout_to_cover_its_own_ramp_time
    # for how the settle blackout protects the same invariant per-step instead.
    controller, ruby, membrane, clock, logger = build(tmp_path, target=1.0)
    logger.close()
    controller.set_membrane_rate_mpa_per_min(1.0)  # far below the old ~20.0 MPa/min floor
    assert controller.membrane_rate_mpa_per_min == 1.0


def test_slow_membrane_rate_extends_settle_blackout_to_cover_its_own_ramp_time(tmp_path):
    # Regression for the compression-rate overshoot seen against real
    # hardware: a membrane_rate_mpa_per_min slow enough to respect
    # max_compression_rate_gpa_per_min at high gain must not let settle
    # detection (and a second, stacked command) fire while the membrane is
    # still physically mid-ramp toward this step's own target, even once
    # region.minimum_settle_time_s alone has elapsed -- see
    # OneSidedPressureController._update_pending_step.
    controller, ruby, membrane, clock, logger = build(tmp_path, target=1.0)
    controller.set_max_compression_rate(None)
    controller.set_membrane_rate_mpa_per_min(1.0)  # deliberately far slower than region 0's old ~20.0 MPa/min floor
    ruby.push(0.0, n=300)  # hold flat so the slope-threshold side of settle detection is trivially satisfied

    dt = 0.25
    tick(controller, clock, n=5)  # a few ticks for the guarded Measure -> stage -> Control startup sequence
    assert len(membrane.commands) == 1, "expected exactly one startup step issued by now"
    step_mpa = membrane.commands[0][1]
    region0 = controller.config.gain_regions[0]
    ramp_time_s = step_mpa / (1.0 / 60.0)
    assert ramp_time_s > region0.minimum_settle_time_s, \
        "test setup: pick a rate slow enough that ramp time dominates the old fixed floor"

    # Advance to just past the old fixed minimum_settle_time_s alone -- under
    # the old behaviour this would already be enough to mark the step settled.
    tick(controller, clock, n=int(region0.minimum_settle_time_s / dt) + 4)
    assert len(membrane.commands) == 1, \
        "a second command must not be issued while the membrane is still mid-ramp, " \
        "even after the old fixed minimum_settle_time_s alone has elapsed"

    # Advance well past this step's own full ramp time + margin.
    total_needed_s = ramp_time_s + controller.config.approach.ramp_time_margin_s
    tick(controller, clock, n=int(total_needed_s / dt) + 8)
    logger.close()
    assert len(membrane.commands) >= 2, \
        "the step must eventually settle once its own ramp time has actually elapsed"


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


def test_resume_grace_absorbs_a_short_control_mode_readback_lag(tmp_path):
    # Regression for the STOP/re-arm flapping seen against real hardware: a
    # set_control_mode(True) write can 200 OK before the PACE5000 control
    # app's own status endpoint reports control_mode=True. Within
    # pace5000_api.control_mode_resume_grace_s (default 2.0s) that lag must
    # not be mistaken for an external relinquish and PAUSE.
    controller, ruby, membrane, clock, logger = build(tmp_path, target=1.0)
    assert controller.config.pace5000_api.control_mode_resume_grace_s > 0
    # A discrete scripted jump in the ruby reading creates an artificial slope
    # spike unrelated to what this test checks (see
    # test_membrane_setpoint_never_retreats_more_than_one_step); disable the
    # unrelated compression-rate cap so it can't also PAUSE here.
    controller.set_max_compression_rate(None)
    controller.pause("test pause")
    ruby.push(0.1, n=5)
    tick(controller, clock, n=5)

    # dt=0.25s * delay=3 reads = 0.75s of lag, comfortably inside the 2.0s grace.
    # Set the delay before resume(): the staged-setpoint confirm sequence
    # takes a few ticks before the actual set_control_mode(True) write, and
    # the delay only starts counting from that write.
    membrane.control_mode_readback_delay_reads = 3
    controller.resume()
    for _ in range(20):
        snap = tick(controller, clock, n=1, dt=0.25)
        assert snap.state != ControlState.PAUSE, \
            f"resume-grace lag must not PAUSE (state={snap.state}, reasons={snap.safety_reasons})"
        if membrane.control_mode is True:
            break
    logger.close()
    assert membrane.control_mode is True, "control mode must eventually be confirmed once the lag resolves"


def test_resume_grace_still_pauses_on_a_genuine_control_mode_relinquish(tmp_path):
    # The grace window must not mask a real external relinquish (operator
    # takes the front panel to local control) -- only a short, expected lag.
    controller, ruby, membrane, clock, logger = build(tmp_path, target=1.0)
    grace_s = controller.config.pace5000_api.control_mode_resume_grace_s
    controller.set_max_compression_rate(None)
    controller.pause("test pause")
    ruby.push(0.1, n=5)
    tick(controller, clock, n=5)

    dt = 0.25
    # Delay long enough (in read_status() calls) to outlast the grace window,
    # with generous headroom for the staged-setpoint confirm sequence that
    # precedes the actual set_control_mode(True) write.
    membrane.control_mode_readback_delay_reads = int(grace_s / dt) + 20
    controller.resume()
    saw_pause = False
    for _ in range(int(grace_s / dt) + 30):
        snap = tick(controller, clock, n=1, dt=dt)
        if snap.state == ControlState.PAUSE and snap.safety_reasons and "membrane_control_mode_disabled" in snap.safety_reasons:
            saw_pause = True
            break
    logger.close()
    assert saw_pause, "a control_mode readback that never recovers must still PAUSE once the grace window expires"


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
