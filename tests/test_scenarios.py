"""Closed-loop scenario tests against the full simulator (nonlinear gain,
lag, dead time, creep, noise, outliers, hysteresis/irreversibility) — the
required-by-spec scenario list.
"""
from __future__ import annotations

from dataclasses import replace as dataclass_replace

from pressurekeeper.models import ControlState
from pressurekeeper.sim import DACPhysicsConfig

from .helpers import build_sim_app, run_until

DT = 0.25


def test_low_pressure_target_reached(tmp_path):
    ctx, clock = build_sim_app(tmp_path, dry_run=False, seed=1)
    ctx.controller.set_target(0.8)

    snap, ticks, max_p = run_until(
        ctx, clock, DT, max_ticks=20_000,
        predicate=lambda s: s.state == ControlState.HOLD,
    )
    ctx.close()

    assert snap.state == ControlState.HOLD, f"did not reach HOLD within {ticks} ticks"
    assert max_p <= 0.8 + 0.05 + 1e-3, f"overshot target+margin: max observed {max_p}"
    # HOLD triggers off the *predicted* pressure (filtered + slope*horizon),
    # which is allowed to run ahead of the filtered value near the transition.
    assert snap.predicted_pressure_gpa >= 0.8 - 0.10 - 1e-3


def test_high_pressure_region_gain_surge(tmp_path):
    physics = DACPhysicsConfig(seed=2, base_gain_gpa_per_mpa=0.30, gain_pressure_coeff=0.35,
                                measurement_noise_std_gpa=0.002)
    ctx, clock = build_sim_app(tmp_path, dry_run=False, max_sample_pressure_gpa=5.0, physics=physics, seed=2)
    ctx.controller.set_target(3.0)  # crosses several gain_regions bands

    snap, ticks, max_p = run_until(
        ctx, clock, DT, max_ticks=60_000,
        predicate=lambda s: s.state == ControlState.HOLD,
    )
    ctx.close()

    assert snap.state == ControlState.HOLD, f"did not reach HOLD within {ticks} ticks (max_p={max_p})"
    assert max_p <= 3.0 + 0.05 + 1e-3, f"gain surge caused an overshoot beyond target+margin: {max_p}"
    assert max_p <= 5.0  # never anywhere near the absolute hard limit


def test_outliers_in_ruby_measurement_do_not_derail_control(tmp_path):
    ctx, clock = build_sim_app(tmp_path, dry_run=False, seed=3)
    ctx.controller.set_target(0.7)

    def predicate(snap):
        # occasionally inject a bounded, single-shot outlier (below the hard-jump
        # threshold and spaced out, so it never trips the safety supervisor)
        if int(clock.now() * 4) % 97 == 0:
            ctx.ruby.inject_outlier(0.15)
        return snap.state == ControlState.HOLD

    snap, ticks, max_p = run_until(ctx, clock, DT, max_ticks=20_000, predicate=predicate)
    ctx.close()

    assert snap.state == ControlState.HOLD, f"outliers prevented reaching HOLD within {ticks} ticks"
    assert snap.safety_level == "ok"
    assert max_p <= 0.7 + 0.05 + 0.02, f"an outlier leaked through the median filter and caused overshoot: {max_p}"


def test_ruby_api_outage_pauses_and_auto_resumes(tmp_path):
    ctx, clock = build_sim_app(tmp_path, dry_run=False, seed=4)
    ctx.controller.set_target(0.8)

    # Let it start approaching normally first.
    run_until(ctx, clock, DT, max_ticks=200, predicate=lambda s: False)
    commands_before_outage = len(ctx.membrane.commands)

    outage_s = 6.0  # well beyond max_stale_sample_s
    ctx.ruby.simulate_outage(clock.now(), outage_s)

    paused_seen = False
    n_during_outage = int(outage_s / DT) + 4
    for _ in range(n_during_outage):
        clock.advance(DT)
        snap = ctx.controller.step()
        if snap.state == ControlState.PAUSE:
            paused_seen = True
    assert paused_seen, "controller must PAUSE while the ruby API is unreachable/stale"
    assert len(ctx.membrane.commands) == commands_before_outage, "no new membrane command may be issued during an outage"

    # Give it time to recover and resume automatically (no manual resume call:
    # this was a safety-triggered pause, not an operator pause).
    snap, ticks, _ = run_until(
        ctx, clock, DT, max_ticks=2_000,
        predicate=lambda s: s.state != ControlState.PAUSE,
    )
    ctx.close()
    assert snap.state != ControlState.PAUSE, "controller should auto-resume once ruby data is fresh again"
    assert not ctx.controller.safety.is_manually_paused


def test_pace5000_slow_response_does_not_trigger_premature_commands(tmp_path):
    # Fast membrane ramp (arrives at setpoint almost immediately) isolates
    # the effect under test: a slow *sample* response (large tau_s) after
    # arrival, with a bigger-than-usual gain so the initial slope is well
    # above the settle threshold and takes a while to decay below it.
    physics = DACPhysicsConfig(seed=5, membrane_ramp_rate_mpa_per_min=60.0, dead_time_s=1.0,
                                tau_s=15.0, base_gain_gpa_per_mpa=0.6)
    ctx, clock = build_sim_app(tmp_path, dry_run=False, physics=physics, seed=5)
    ctx.controller.set_target(0.6)
    ctx.controller.set_max_compression_rate(None)
    # The simulator now honours the rate sent with each PACE5000 command;
    # retain this scenario's intended fast-membrane/slow-sample separation.
    ctx.controller.set_membrane_rate_mpa_per_min(60.0)

    run_until(ctx, clock, DT, max_ticks=40_000, predicate=lambda s: len(ctx.membrane.commands) >= 2)
    ctx.close()

    assert len(ctx.membrane.commands) >= 2
    t0 = ctx.membrane.commands[0][0]
    t1 = ctx.membrane.commands[1][0]
    # region 0's minimum_settle_time_s (blackout) is 3.0 s in the test config;
    # the real (slow) response must push the actual wait well past that,
    # proving the controller waited for genuine settling, not just the
    # blackout timer.
    assert (t1 - t0) > 8.0, f"second command issued too soon given the simulated slow response: {t1 - t0:.2f}s"


def test_creep_near_target_does_not_cause_overshoot(tmp_path):
    physics = DACPhysicsConfig(seed=6, creep_weight=0.35, creep_tau_s=25.0, tau_s=4.0)
    ctx, clock = build_sim_app(tmp_path, dry_run=False, physics=physics, seed=6)
    ctx.controller.set_target(0.6)

    snap, ticks, max_p = run_until(ctx, clock, DT, max_ticks=20_000, predicate=lambda s: s.state == ControlState.HOLD)
    assert snap.state == ControlState.HOLD

    # Now let creep continue running for a long time after HOLD is reached.
    for _ in range(4000):
        clock.advance(DT)
        snap = ctx.controller.step()
        if snap.filtered_pressure_gpa is not None:
            max_p = max(max_p, snap.filtered_pressure_gpa)
    ctx.close()

    assert max_p <= 0.6 + 0.05 + 1e-3, f"post-HOLD creep overshot target+margin: {max_p}"


def test_actual_response_larger_than_estimated_gain_adapts_and_stays_bounded(tmp_path):
    # Config prior (safe_gain) is deliberately far below the simulator's true
    # gain, so every step under-predicts the real response. Target is set far
    # enough away that convergence still needs several steps (region step
    # caps limit how much *predicted* sample pressure any one command chases,
    # regardless of the gain surprise), giving the online estimator data to
    # adapt from within this run.
    physics = DACPhysicsConfig(seed=7, base_gain_gpa_per_mpa=0.9, gain_pressure_coeff=0.05,
                                measurement_noise_std_gpa=0.002)
    ctx, clock = build_sim_app(tmp_path, dry_run=False, physics=physics, seed=7)
    ctx.controller.set_target(1.8)

    snap, ticks, max_p = run_until(ctx, clock, DT, max_ticks=40_000, predicate=lambda s: s.state == ControlState.HOLD)
    ctx.close()

    assert snap.state == ControlState.HOLD, "controller must still converge (not oscillate/crash) despite the surprise"
    assert len(ctx.membrane.commands) >= 2, "target was chosen so more than one step is needed"
    # A persistent gain surprise this large (config prior 0.20 vs actual ~0.9
    # GPa/MPa) is exactly what the near-target step caps + hysteresis exist to
    # bound — it is not expected to be perfectly zero-overshoot, but it must
    # not blow far past the target.
    assert max_p <= 1.8 + 0.30, f"unbounded overshoot from underestimated gain: {max_p}"
    # And the online estimator must have picked up the true (higher) gain for
    # future steps, rather than staying stuck on the too-low prior.
    region = ctx.controller.config.region_for(snap.filtered_pressure_gpa)
    estimate = ctx.controller.gain_estimator.estimate(snap.filtered_pressure_gpa, region)
    assert estimate.source == "observed"
    assert estimate.estimated_gain > region.safe_gain


def test_adaptive_local_tracks_one_loading_without_region_calibration(tmp_path):
    physics = DACPhysicsConfig(
        seed=17,
        base_gain_gpa_per_mpa=0.25,
        gain_pressure_coeff=0.40,
        measurement_noise_std_gpa=0.001,
    )
    ctx, clock = build_sim_app(
        tmp_path,
        dry_run=False,
        max_sample_pressure_gpa=3.0,
        physics=physics,
        seed=17,
    )
    # Deliberately nonsensical legacy priors: adaptive-local must neither use
    # them as a floor nor require repeated observations inside fixed bins.
    adaptive = ctx.controller.config.model_copy(update={
        "gain_regions": [
            dataclass_replace(region, safe_gain=50.0, rate_limit_gain=None)
            for region in ctx.controller.config.gain_regions
        ],
        "gain_estimation": ctx.controller.config.gain_estimation.model_copy(update={
            "step_sizing_mode": "adaptive_local",
            "initial_probe_step_mpa": 0.03,
            "probe_growth_factor": 1.6,
            "max_probe_step_mpa": 0.12,
            "adaptive_probe_max_expected_gain": 2.0,
            "adaptive_no_response_wait_s": 3.0,
            "adaptive_max_membrane_step_mpa": 0.15,
            "adaptive_max_sample_step_gpa": 0.03,
            "probe_rate_mpa_per_min": 2.0,
            "adaptive_minimum_settle_time_s": 3.0,
            "adaptive_settled_slope_threshold_gpa_s": 0.008,
            "response_detection_floor_gpa": 0.002,
            "local_gain_safety_factor": 1.35,
            "local_pressure_window_gpa": 0.30,
            "interrupted_rate_learning_mode": "enforce",
        }),
    })
    ctx.controller.apply_config_update(adaptive)
    ctx.controller.set_target(1.2)

    snap, ticks, max_p = run_until(
        ctx,
        clock,
        DT,
        max_ticks=60_000,
        predicate=lambda s: s.state == ControlState.HOLD,
    )
    ctx.close()

    assert snap.state == ControlState.HOLD, f"adaptive run did not converge in {ticks} ticks"
    assert max_p <= 1.2 + 0.05 + 0.01
    region = ctx.controller.config.region_for(snap.filtered_pressure_gpa)
    estimate = ctx.controller.gain_estimator.estimate(
        snap.filtered_pressure_gpa,
        region,
    )
    assert estimate.source == "observed"
    assert estimate.safe_gain < 50.0
    assert any(command[2] <= 2.0 for command in ctx.membrane.commands)
    assert (
        ctx.dac.cfg.membrane_ramp_rate_mpa_per_min
        == ctx.membrane.commands[-1][2]
    ), "simulator physics must apply the slew rate sent with the latest command"


def test_approaching_hard_sample_pressure_limit_is_respected(tmp_path):
    ctx, clock = build_sim_app(tmp_path, dry_run=False, max_sample_pressure_gpa=1.0, seed=8)
    ctx.controller.set_target(0.9)  # comfortably below the 1.0 GPa hard limit

    snap, ticks, max_p = run_until(ctx, clock, DT, max_ticks=30_000, predicate=lambda s: s.state == ControlState.HOLD)
    ctx.close()

    assert snap.state == ControlState.HOLD
    assert max_p <= 1.0, f"exceeded the absolute safety limit: {max_p}"
    assert snap.safety_level == "ok"


def test_raw_jump_past_hard_limit_triggers_abort(tmp_path):
    ctx, clock = build_sim_app(tmp_path, dry_run=False, max_sample_pressure_gpa=1.0, seed=9)
    ctx.controller.set_target(0.9)

    run_until(ctx, clock, DT, max_ticks=200, predicate=lambda s: False)
    ctx.ruby.inject_outlier(5.0)  # forces a raw reading far past the 1.0 GPa hard limit

    clock.advance(DT)
    snap = ctx.controller.step()
    assert snap.state == ControlState.ABORT

    commands_at_abort = len(ctx.membrane.commands)
    for _ in range(200):
        clock.advance(DT)
        snap = ctx.controller.step()
    ctx.close()
    assert snap.state == ControlState.ABORT, "ABORT must be sticky"
    assert len(ctx.membrane.commands) == commands_at_abort, "no further commands may be issued once aborted"


def test_small_noise_fluctuation_during_hold_issues_no_new_commands(tmp_path):
    ctx, clock = build_sim_app(tmp_path, dry_run=False, seed=10)
    ctx.controller.set_target(0.6)

    snap, ticks, _ = run_until(ctx, clock, DT, max_ticks=20_000, predicate=lambda s: s.state == ControlState.HOLD)
    assert snap.state == ControlState.HOLD
    commands_at_hold = len(ctx.membrane.commands)

    for _ in range(400):  # 100 s of ordinary measurement noise while holding
        clock.advance(DT)
        snap = ctx.controller.step()
    ctx.close()

    assert snap.state == ControlState.HOLD
    assert len(ctx.membrane.commands) == commands_at_hold, "ordinary measurement noise while HOLD must not trigger a new step"


def test_manual_stop_mid_control(tmp_path):
    ctx, clock = build_sim_app(tmp_path, dry_run=False, seed=11)
    ctx.controller.set_target(0.8)

    run_until(ctx, clock, DT, max_ticks=200, predicate=lambda s: False)
    ctx.controller.pause("operator manual stop")

    commands_before = len(ctx.membrane.commands)
    for _ in range(400):
        clock.advance(DT)
        snap = ctx.controller.step()
    assert snap.state == ControlState.PAUSE
    assert len(ctx.membrane.commands) == commands_before, "no commands may be issued while manually paused"

    ctx.controller.resume()
    snap, ticks, _ = run_until(
        ctx, clock, DT, max_ticks=20_000,
        predicate=lambda s: len(ctx.membrane.commands) > commands_before,
    )
    ctx.close()
    assert len(ctx.membrane.commands) > commands_before, "control must resume issuing steps after an operator resume"


def test_abort_mid_ramp_halts_in_flight_membrane_ramp(tmp_path):
    # Regression test: a command already sent to the PACE5000 keeps ramping
    # toward its setpoint under the device's own rate/target, independent of
    # this controller. Blocking *new* steps (what PAUSE/ABORT did before this
    # fix) does not stop that ramp -- ABORT must additionally switch the
    # PACE5000 out of control mode (STOP), which the simulator models as
    # actually halting membrane_actual_mpa's advance toward its setpoint.
    ctx, clock = build_sim_app(tmp_path, dry_run=False, seed=12)
    ctx.controller.set_target(0.8)

    snap, ticks, _ = run_until(
        ctx, clock, DT, max_ticks=20_000,
        predicate=lambda s: s.state == ControlState.SETTLE and len(ctx.membrane.commands) >= 1,
    )
    assert snap.state == ControlState.SETTLE, "expected a step in flight before aborting"
    clock.advance(DT)
    snap = ctx.controller.step()
    assert snap.state == ControlState.SETTLE, "step must still be in flight one tick later"
    commanded_setpoint = snap.membrane_setpoint_mpa
    actual_before_abort = snap.membrane_actual_mpa
    assert actual_before_abort < commanded_setpoint - 1e-3, "must abort while still mid-ramp, not after it settled"

    ctx.controller.abort("test abort mid-ramp")
    assert ctx.membrane.control_mode_commands and ctx.membrane.control_mode_commands[-1][1] is False, \
        "abort's direct emergency stop must disable control mode immediately, before any further tick"

    actual_at_abort_command = ctx.membrane.read_status().pressure_mpa
    for _ in range(int(120 / DT)):  # long enough to have fully ramped if not actually stopped
        clock.advance(DT)
        snap = ctx.controller.step()
    ctx.close()

    assert snap.state == ControlState.ABORT, "ABORT must remain sticky"
    assert snap.membrane_actual_mpa <= actual_at_abort_command + 0.02, (
        f"membrane kept ramping toward the pre-abort setpoint after ABORT: "
        f"{actual_at_abort_command:.4f} -> {snap.membrane_actual_mpa:.4f} MPa, "
        f"pre-abort commanded setpoint was {commanded_setpoint:.4f} MPa"
    )
    assert snap.membrane_actual_mpa < commanded_setpoint - 1e-3, \
        "sanity check: without the fix the membrane would have fully reached the pre-abort setpoint by now"


def test_hold_halts_in_flight_membrane_creep_not_just_new_commands(tmp_path):
    # The existing HOLD/pause tests only ever checked "no new commands are
    # issued" -- they never confirmed a step already in flight when HOLD is
    # reached actually stops climbing. Use a slow membrane ramp so there is
    # still real distance left to climb at the moment HOLD triggers.
    physics = DACPhysicsConfig(seed=7, membrane_ramp_rate_mpa_per_min=0.5, dead_time_s=1.0)
    ctx, clock = build_sim_app(tmp_path, dry_run=False, physics=physics, seed=7)
    ctx.controller.set_target(0.5)

    snap, ticks, _ = run_until(
        ctx, clock, DT, max_ticks=40_000,
        predicate=lambda s: s.state == ControlState.HOLD,
    )
    assert snap.state == ControlState.HOLD
    actual_at_hold = snap.membrane_actual_mpa
    setpoint_at_hold = snap.membrane_setpoint_mpa
    assert actual_at_hold < setpoint_at_hold - 1e-3, \
        "must reach HOLD while still mid-ramp, not after the membrane already arrived"

    for _ in range(int(300 / DT)):  # long enough for a still-ramping membrane to fully arrive
        clock.advance(DT)
        snap = ctx.controller.step()
    ctx.close()

    assert snap.state == ControlState.HOLD
    assert snap.membrane_actual_mpa <= actual_at_hold + 0.02, (
        f"membrane kept climbing after HOLD instead of being stopped: "
        f"{actual_at_hold:.4f} -> {snap.membrane_actual_mpa:.4f} MPa (setpoint was {setpoint_at_hold:.4f} MPa)"
    )


def test_manual_pause_mid_ramp_halts_actual_pressure_not_just_new_commands(tmp_path):
    # Companion to test_manual_stop_mid_control in this file, which only ever
    # checked that no *new* commands were issued during a manual pause --
    # never that a step already in flight actually stopped climbing.
    physics = DACPhysicsConfig(seed=14, membrane_ramp_rate_mpa_per_min=2.0, dead_time_s=1.0)
    ctx, clock = build_sim_app(tmp_path, dry_run=False, physics=physics, seed=14)
    ctx.controller.set_target(0.8)

    snap, ticks, _ = run_until(
        ctx, clock, DT, max_ticks=40_000,
        predicate=lambda s: s.state == ControlState.SETTLE and len(ctx.membrane.commands) >= 1,
    )
    assert snap.state == ControlState.SETTLE, "expected a step in flight before pausing"
    # membrane_status in this snapshot is one tick stale relative to the
    # command just issued this same tick (read_status() runs before
    # _maybe_issue_step within step()) -- advance once more so it reflects
    # the setpoint actually sent.
    clock.advance(DT)
    snap = ctx.controller.step()
    actual_before_pause = snap.membrane_actual_mpa
    setpoint_before_pause = snap.membrane_setpoint_mpa
    assert actual_before_pause < setpoint_before_pause - 1e-3, "must pause while still mid-ramp"

    ctx.controller.pause("test manual pause mid-ramp")
    for _ in range(int(120 / DT)):
        clock.advance(DT)
        snap = ctx.controller.step()
    ctx.close()

    assert snap.state == ControlState.PAUSE
    assert snap.membrane_actual_mpa <= actual_before_pause + 0.02, (
        f"membrane kept climbing after a manual pause instead of being stopped: "
        f"{actual_before_pause:.4f} -> {snap.membrane_actual_mpa:.4f} MPa"
    )


def test_target_lowered_mid_ramp_halts_the_ramp(tmp_path):
    # Lowering the target while a step is in flight must re-trigger HOLD (the
    # filtered-pressure-above-target+margin check runs every tick regardless
    # of prior state) and stop the membrane, not let the in-flight ramp
    # finish climbing toward the old, higher target.
    physics = DACPhysicsConfig(seed=15, membrane_ramp_rate_mpa_per_min=2.0, dead_time_s=1.0)
    ctx, clock = build_sim_app(tmp_path, dry_run=False, physics=physics, seed=15)
    ctx.controller.set_target(1.0)

    snap, ticks, _ = run_until(
        ctx, clock, DT, max_ticks=40_000,
        predicate=lambda s: s.state == ControlState.SETTLE and len(ctx.membrane.commands) >= 1,
    )
    assert snap.state == ControlState.SETTLE, "expected a step in flight before lowering the target"
    clock.advance(DT)  # membrane_status here is one tick stale; see the pause test above
    snap = ctx.controller.step()
    actual_before_lower = snap.membrane_actual_mpa

    ctx.controller.set_target(0.2)  # already exceeded -> must HOLD and stop, not finish the old ramp
    for _ in range(int(120 / DT)):
        clock.advance(DT)
        snap = ctx.controller.step()
    ctx.close()

    assert snap.state == ControlState.HOLD
    assert snap.membrane_actual_mpa <= actual_before_lower + 0.02, (
        f"membrane kept climbing toward the old, higher target after it was lowered: "
        f"{actual_before_lower:.4f} -> {snap.membrane_actual_mpa:.4f} MPa"
    )
