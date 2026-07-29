import dataclasses

from pressurekeeper.config import EstimatorConfig, SafetyConfig
from pressurekeeper.estimator import PressureEstimator
from pressurekeeper.models import MembraneStatus, RubyPressureSample
from pressurekeeper.safety import SafetySupervisor


def _estimator_cfg(**overrides) -> EstimatorConfig:
    base = dict(outlier_median_window=5, smoothing_window_s=2.0, slope_window_s=4.0,
                min_points_for_valid=3, max_sample_age_s=1.0, max_jump_gpa=0.3,
                min_r2=None, min_intensity=None)
    base.update(overrides)
    return EstimatorConfig(**base)


def _safety_cfg(**overrides) -> SafetyConfig:
    base = dict(
        max_sample_pressure_gpa=3.0, max_membrane_pressure_mpa=6.0, max_membrane_step_mpa_hard=0.5,
        max_cumulative_step_mpa=1.0, cumulative_window_s=10.0, max_stale_sample_s=1.0,
        max_consecutive_invalid=3, max_consecutive_comm_errors=3, sample_jump_hard_gpa=0.3,
        max_consecutive_jump_flags=2, setpoint_mismatch_tol_mpa=0.05, setpoint_mismatch_grace_s=1.0,
        ruby_error_pause_after_s=0.5, membrane_error_pause_after_s=0.5,
    )
    base.update(overrides)
    return SafetyConfig(**base)


def sample(t, p, fit_success=True):
    return RubyPressureSample(t_mono=t, t_wall=t, pressure_gpa=p, pressure_err_gpa=0.01, fit_success=fit_success)


def fresh_estimator(cfg, n=5, dt=0.25, p=1.0, start=0.0):
    est = PressureEstimator(cfg)
    for i in range(n):
        est.update(sample(start + i * dt, p))
    return est, start + (n - 1) * dt


def test_ok_when_everything_nominal():
    est_cfg = _estimator_cfg()
    safety = SafetySupervisor(_safety_cfg(), start_t=0.0)
    est, now = fresh_estimator(est_cfg)
    safety.on_ruby_sample(1.0, 1.0, now)
    safety.on_membrane_status(now)
    verdict = safety.evaluate(
        est, MembraneStatus(t_mono=now, connected=True, pressure_mpa=1.0,
                            target_pressure_mpa=1.0, control_mode=True,
                            source_pressure_positive_mpa=6.0), now
    )
    assert verdict.level == "ok"


def test_absolute_sample_pressure_over_limit_aborts_and_is_sticky():
    est_cfg = _estimator_cfg()
    safety = SafetySupervisor(_safety_cfg(max_sample_pressure_gpa=2.0), start_t=0.0)
    est, now = fresh_estimator(est_cfg, p=2.5)  # already over the limit
    verdict = safety.evaluate(est, None, now)
    assert verdict.level == "abort"
    # sticky: even if we now feed it something that looks fine, it stays aborted
    est2, now2 = fresh_estimator(est_cfg, p=0.1, start=now + 1)
    verdict2 = safety.evaluate(est2, None, now2)
    assert verdict2.level == "abort"
    safety.force_reset()
    verdict3 = safety.evaluate(est2, None, now2)
    assert verdict3.level == "ok"


def test_membrane_pressure_over_limit_aborts():
    safety = SafetySupervisor(_safety_cfg(max_membrane_pressure_mpa=6.0), start_t=0.0)
    est_cfg = _estimator_cfg()
    est, now = fresh_estimator(est_cfg)
    status = MembraneStatus(t_mono=now, connected=True, pressure_mpa=6.5, target_pressure_mpa=6.5)
    verdict = safety.evaluate(est, status, now)
    assert verdict.level == "abort"


def test_membrane_disconnected_triggers_pause():
    safety = SafetySupervisor(_safety_cfg(), start_t=0.0)
    est_cfg = _estimator_cfg()
    est, now = fresh_estimator(est_cfg)
    safety.on_membrane_status(now)  # app answered HTTP 200, but hardware isn't connected
    status = MembraneStatus(t_mono=now, connected=False)
    verdict = safety.evaluate(est, status, now)
    assert verdict.level == "pause"
    assert any(e.code == "membrane_disconnected" for e in verdict.events)


def test_membrane_control_mode_disabled_triggers_pause():
    safety = SafetySupervisor(_safety_cfg(), start_t=0.0)
    est_cfg = _estimator_cfg()
    est, now = fresh_estimator(est_cfg)
    safety.on_membrane_status(now)
    status = MembraneStatus(t_mono=now, connected=True, pressure_mpa=1.0, target_pressure_mpa=1.0, control_mode=False)
    verdict = safety.evaluate(est, status, now)
    assert verdict.level == "pause"
    assert any(e.code == "membrane_control_mode_disabled" for e in verdict.events)


def test_stale_measurement_triggers_pause():
    safety = SafetySupervisor(_safety_cfg(max_stale_sample_s=1.0), start_t=0.0)
    est_cfg = _estimator_cfg(max_sample_age_s=100.0)  # keep estimator itself "valid", only safety checks staleness independently...
    est, now = fresh_estimator(est_cfg)
    later = now + 5.0  # far beyond max_stale_sample_s
    verdict = safety.evaluate(est, None, later)
    codes = [e.code for e in verdict.events]
    assert verdict.level == "pause"
    assert "stale_measurement" in codes


def test_consecutive_invalid_measurements_triggers_pause():
    est_cfg = _estimator_cfg()
    est = PressureEstimator(est_cfg)
    safety = SafetySupervisor(_safety_cfg(max_consecutive_invalid=3), start_t=0.0)
    t = 0.0
    for _ in range(3):
        est.update(sample(t, None, fit_success=False))
        t += 0.25
    verdict = safety.evaluate(est, None, t)
    assert verdict.level == "pause"
    assert any(e.code == "consecutive_invalid_measurements" for e in verdict.events)


def test_hard_sample_jump_triggers_pause():
    est_cfg = _estimator_cfg()
    safety = SafetySupervisor(_safety_cfg(sample_jump_hard_gpa=0.2), start_t=0.0)
    est, now = fresh_estimator(est_cfg, p=1.0)
    events = safety.on_ruby_sample(1.5, prior_filtered_gpa=1.0, now=now)  # jump of 0.5 > 0.2
    assert any(e.code == "hard_sample_jump" for e in events)
    verdict = safety.evaluate(est, None, now, extra_events=events)
    assert verdict.level == "pause"


def test_ruby_comm_errors_pause_after_threshold():
    safety = SafetySupervisor(_safety_cfg(max_consecutive_comm_errors=3), start_t=0.0)
    est_cfg = _estimator_cfg()
    est, now = fresh_estimator(est_cfg)
    for _ in range(3):
        safety.on_ruby_error(now)
    verdict = safety.evaluate(est, None, now)
    assert verdict.level == "pause"
    assert any(e.code == "ruby_api_unreachable" for e in verdict.events)


def test_membrane_comm_errors_pause_after_threshold():
    safety = SafetySupervisor(_safety_cfg(max_consecutive_comm_errors=2), start_t=0.0)
    est_cfg = _estimator_cfg()
    est, now = fresh_estimator(est_cfg)
    for _ in range(2):
        safety.on_membrane_error(now)
    verdict = safety.evaluate(est, None, now)
    assert verdict.level == "pause"
    assert any(e.code == "pace5000_unreachable" for e in verdict.events)


def test_setpoint_mismatch_after_grace_period_pauses():
    safety = SafetySupervisor(_safety_cfg(setpoint_mismatch_tol_mpa=0.05, setpoint_mismatch_grace_s=1.0), start_t=0.0)
    est_cfg = _estimator_cfg()
    est, now = fresh_estimator(est_cfg)
    safety.on_command_issued(membrane_step_mpa=0.5, new_setpoint_mpa=1.0, now=now)
    mismatched_status = MembraneStatus(t_mono=now, connected=True, pressure_mpa=0.4,
                                       target_pressure_mpa=0.4, control_mode=True,
                                       source_pressure_positive_mpa=6.0)
    verdict_immediate = safety.evaluate(est, mismatched_status, now + 0.1)  # still in grace period
    assert verdict_immediate.level == "ok"
    verdict_late = safety.evaluate(est, mismatched_status, now + 2.0)  # past grace period
    assert verdict_late.level == "pause"
    assert any(e.code == "setpoint_mismatch" for e in verdict_late.events)


def test_setpoint_mismatch_with_zero_grace_pauses_on_first_status():
    safety = SafetySupervisor(
        _safety_cfg(setpoint_mismatch_tol_mpa=0.05, setpoint_mismatch_grace_s=0.0),
        start_t=0.0,
    )
    est, now = fresh_estimator(_estimator_cfg())
    safety.on_command_issued(0.2, 1.0, now)
    status = MembraneStatus(
        t_mono=now + 0.25,
        connected=True,
        pressure_mpa=0.8,
        target_pressure_mpa=0.8,
        control_mode=True,
        source_pressure_positive_mpa=6.0,
    )
    verdict = safety.evaluate(est, status, now + 0.25)
    assert verdict.level == "pause"
    assert any(e.code == "setpoint_mismatch" for e in verdict.events)


def test_large_active_setpoint_actual_gap_pauses_immediately():
    safety = SafetySupervisor(
        _safety_cfg(max_setpoint_actual_gap_mpa=0.5),
        start_t=0.0,
    )
    est, now = fresh_estimator(_estimator_cfg())
    status = MembraneStatus(
        t_mono=now,
        connected=True,
        pressure_mpa=0.2,
        target_pressure_mpa=1.0,
        control_mode=True,
        source_pressure_positive_mpa=6.0,
    )
    verdict = safety.evaluate(est, status, now)
    assert verdict.level == "pause"
    assert any(e.code == "setpoint_actual_gap_too_large" for e in verdict.events)


def test_source_pressure_must_strictly_exceed_active_setpoint():
    safety = SafetySupervisor(_safety_cfg(), start_t=0.0)
    est, now = fresh_estimator(_estimator_cfg())
    status = MembraneStatus(
        t_mono=now,
        connected=True,
        pressure_mpa=1.0,
        target_pressure_mpa=1.0,
        control_mode=True,
        source_pressure_positive_mpa=1.0,
    )
    verdict = safety.evaluate(est, status, now)
    assert verdict.level == "pause"
    assert any(e.code == "source_pressure_insufficient" for e in verdict.events)


def test_manual_pause_and_resume():
    safety = SafetySupervisor(_safety_cfg(), start_t=0.0)
    est_cfg = _estimator_cfg()
    est, now = fresh_estimator(est_cfg)
    safety.request_manual_pause("operator break")
    verdict = safety.evaluate(est, None, now)
    assert verdict.level == "pause"
    assert safety.is_manually_paused
    safety.clear_manual_pause()
    verdict2 = safety.evaluate(est, None, now)
    assert verdict2.level == "ok"


def test_manual_abort_is_sticky_until_force_reset():
    safety = SafetySupervisor(_safety_cfg(), start_t=0.0)
    est_cfg = _estimator_cfg()
    est, now = fresh_estimator(est_cfg)
    safety.request_manual_abort("emergency stop")
    assert safety.evaluate(est, None, now).level == "abort"
    safety.force_reset()
    assert safety.evaluate(est, None, now).level == "ok"


def test_check_command_hard_per_command_cap():
    safety = SafetySupervisor(_safety_cfg(max_membrane_step_mpa_hard=0.3), start_t=0.0)
    allowed, reason = safety.check_command(membrane_step_mpa=0.4, new_setpoint_mpa=1.0, now=0.0)
    assert not allowed and reason == "step_exceeds_hard_per_command_cap"


def test_check_command_negative_step_rejected():
    safety = SafetySupervisor(_safety_cfg(), start_t=0.0)
    allowed, reason = safety.check_command(membrane_step_mpa=-0.1, new_setpoint_mpa=1.0, now=0.0)
    assert not allowed and reason == "membrane_step_negative"


def test_check_command_cumulative_cap():
    safety = SafetySupervisor(_safety_cfg(max_cumulative_step_mpa=0.5, cumulative_window_s=10.0,
                                           max_membrane_step_mpa_hard=1.0), start_t=0.0)
    allowed1, _ = safety.check_command(
        membrane_step_mpa=0.3, new_setpoint_mpa=0.3, now=0.0,
        source_pressure_mpa=6.0,
    )
    assert allowed1
    safety.on_command_issued(0.3, 0.3, now=0.0)
    allowed2, reason2 = safety.check_command(
        membrane_step_mpa=0.3, new_setpoint_mpa=0.6, now=0.5,
        source_pressure_mpa=6.0,
    )
    assert not allowed2 and reason2 == "cumulative_step_cap_exceeded"
    # outside the window, the old step ages out and the budget is free again
    allowed3, _ = safety.check_command(
        membrane_step_mpa=0.3, new_setpoint_mpa=0.6, now=11.0,
        source_pressure_mpa=6.0,
    )
    assert allowed3


def test_check_command_ceiling():
    safety = SafetySupervisor(_safety_cfg(max_membrane_pressure_mpa=2.0, max_membrane_step_mpa_hard=5.0), start_t=0.0)
    allowed, reason = safety.check_command(membrane_step_mpa=0.5, new_setpoint_mpa=2.5, now=0.0)
    assert not allowed and reason == "setpoint_would_exceed_membrane_ceiling"


def test_check_command_blocks_setpoint_at_or_above_source_pressure():
    safety = SafetySupervisor(_safety_cfg(), start_t=0.0)
    for setpoint in (1.0, 1.1):
        allowed, reason = safety.check_command(
            membrane_step_mpa=0.1,
            new_setpoint_mpa=setpoint,
            now=0.0,
            source_pressure_mpa=1.0,
        )
        assert not allowed
        assert reason == "setpoint_not_below_source_pressure"


def test_check_command_rejects_non_finite_values():
    safety = SafetySupervisor(_safety_cfg(), start_t=0.0)
    for bad_step, bad_setpoint in [(float("nan"), 1.0), (0.1, float("nan")),
                                    (float("inf"), 1.0), (0.1, float("inf"))]:
        allowed, reason = safety.check_command(membrane_step_mpa=bad_step, new_setpoint_mpa=bad_setpoint, now=0.0)
        assert not allowed and reason == "non_finite_command_value"


def test_non_finite_sample_pressure_aborts():
    safety = SafetySupervisor(_safety_cfg(), start_t=0.0)
    est_cfg = _estimator_cfg()
    est, now = fresh_estimator(est_cfg, p=1.0)
    # Bypass the estimator's own filtering to exercise evaluate()'s
    # defense-in-depth check directly against a corrupted last_sample.
    est._last_sample = dataclasses.replace(est._last_sample, pressure_gpa=float("nan"))
    verdict = safety.evaluate(est, None, now)
    assert verdict.level == "abort"
    assert any(e.code == "non_finite_sample_pressure" for e in verdict.events)


def test_non_finite_membrane_pressure_aborts():
    safety = SafetySupervisor(_safety_cfg(), start_t=0.0)
    est_cfg = _estimator_cfg()
    est, now = fresh_estimator(est_cfg)
    status = MembraneStatus(t_mono=now, connected=True, pressure_mpa=float("nan"), target_pressure_mpa=1.0)
    verdict = safety.evaluate(est, status, now)
    assert verdict.level == "abort"
    assert any(e.code == "non_finite_membrane_pressure" for e in verdict.events)


def test_membrane_status_incomplete_while_connected_pauses():
    safety = SafetySupervisor(_safety_cfg(), start_t=0.0)
    est_cfg = _estimator_cfg()
    est, now = fresh_estimator(est_cfg)
    status = MembraneStatus(t_mono=now, connected=True, pressure_mpa=None, target_pressure_mpa=None)
    verdict = safety.evaluate(est, status, now)
    assert verdict.level == "pause"
    assert any(e.code == "membrane_status_incomplete" for e in verdict.events)


def test_dangerous_target_over_limit_aborts_even_below_actual_limit():
    # Actual pressure is comfortably under the limit, but the reported
    # *target* is already past it -- must not wait for the ramp to actually
    # cross the limit before reacting.
    safety = SafetySupervisor(_safety_cfg(max_membrane_pressure_mpa=6.0), start_t=0.0)
    est_cfg = _estimator_cfg()
    est, now = fresh_estimator(est_cfg)
    status = MembraneStatus(t_mono=now, connected=True, pressure_mpa=1.0, target_pressure_mpa=7.0)
    verdict = safety.evaluate(est, status, now)
    assert verdict.level == "abort"
    assert any(e.code == "membrane_target_over_limit" for e in verdict.events)


def test_dangerous_target_over_limit_aborts_even_when_actual_is_missing():
    safety = SafetySupervisor(_safety_cfg(max_membrane_pressure_mpa=6.0), start_t=0.0)
    est, now = fresh_estimator(_estimator_cfg())
    status = MembraneStatus(
        t_mono=now,
        connected=True,
        pressure_mpa=None,
        target_pressure_mpa=7.0,
        control_mode=None,
    )
    verdict = safety.evaluate(est, status, now)
    assert verdict.level == "abort"
    assert any(e.code == "membrane_target_over_limit" for e in verdict.events)


def test_command_gate_rechecks_concurrent_operator_stop_flags():
    safety = SafetySupervisor(_safety_cfg(), start_t=0.0)
    safety.request_manual_pause()
    allowed, reason = safety.check_command(0.1, 1.0, now=0.0)
    assert not allowed and reason == "manual_pause_requested"

    safety.clear_manual_pause()
    safety.request_manual_abort()
    allowed, reason = safety.check_command(0.1, 1.0, now=0.0)
    assert not allowed and reason == "abort_requested"


def test_ruby_error_streak_pauses_after_configured_grace_even_with_continuous_errors():
    # Regression: on_ruby_error used to overwrite _last_ruby_error_t on every
    # call, so a continuous stream of errors (one per tick) never let
    # `now - last_error_t` grow past the grace window. Track the streak's
    # first error time instead.
    safety = SafetySupervisor(_safety_cfg(ruby_error_pause_after_s=1.0, max_consecutive_comm_errors=1000), start_t=0.0)
    est_cfg = _estimator_cfg()
    est, now = fresh_estimator(est_cfg)
    t = now
    for _ in range(20):
        t += 0.1
        safety.on_ruby_error(t)
    verdict = safety.evaluate(est, None, t)
    assert verdict.level == "pause"
    assert any(e.code == "ruby_api_unreachable" for e in verdict.events)


def test_membrane_error_pause_after_s_is_wired_in():
    safety = SafetySupervisor(_safety_cfg(membrane_error_pause_after_s=1.0, max_consecutive_comm_errors=1000), start_t=0.0)
    est_cfg = _estimator_cfg()
    est, now = fresh_estimator(est_cfg)
    t = now
    for _ in range(20):
        t += 0.1
        safety.on_membrane_error(t)
    verdict = safety.evaluate(est, None, t)
    assert verdict.level == "pause"
    assert any(e.code == "pace5000_unreachable" for e in verdict.events)


def test_compression_rate_cap_pauses_on_observed_rate_not_just_step_size():
    safety = SafetySupervisor(_safety_cfg(), start_t=0.0)
    est_cfg = _estimator_cfg()
    est = PressureEstimator(est_cfg)
    # Rising fast enough to exceed a 0.5 GPa/min cap regardless of how any
    # step was sized -- this check watches reality, not the request.
    for i in range(20):
        est.update(sample(i * 0.25, 1.0 + 0.05 * i))
    now = 19 * 0.25
    verdict = safety.evaluate(est, None, now, max_compression_rate_gpa_per_min=0.5)
    assert verdict.level == "pause"
    assert any(e.code == "compression_rate_exceeded" for e in verdict.events)


def test_compression_rate_cap_none_never_pauses_for_it():
    safety = SafetySupervisor(_safety_cfg(), start_t=0.0)
    est_cfg = _estimator_cfg()
    est = PressureEstimator(est_cfg)
    for i in range(20):
        est.update(sample(i * 0.25, 1.0 + 0.05 * i))
    now = 19 * 0.25
    verdict = safety.evaluate(est, None, now, max_compression_rate_gpa_per_min=None)
    assert not any(e.code == "compression_rate_exceeded" for e in verdict.events)


def test_membrane_control_mode_disabled_suppressed_when_stop_intended():
    safety = SafetySupervisor(_safety_cfg(), start_t=0.0)
    est_cfg = _estimator_cfg()
    est, now = fresh_estimator(est_cfg)
    safety.on_membrane_status(now)
    safety.set_membrane_stop_intended(True)
    status = MembraneStatus(t_mono=now, connected=True, pressure_mpa=1.0, target_pressure_mpa=1.0, control_mode=False)
    verdict = safety.evaluate(est, status, now)
    assert not any(e.code == "membrane_control_mode_disabled" for e in verdict.events), \
        "control mode off must not be flagged as unsafe when we intentionally disabled it ourselves"


def test_membrane_control_mode_disabled_still_flagged_when_not_intended():
    safety = SafetySupervisor(_safety_cfg(), start_t=0.0)
    est_cfg = _estimator_cfg()
    est, now = fresh_estimator(est_cfg)
    safety.on_membrane_status(now)
    assert not safety.membrane_stop_intended
    status = MembraneStatus(t_mono=now, connected=True, pressure_mpa=1.0, target_pressure_mpa=1.0, control_mode=False)
    verdict = safety.evaluate(est, status, now)
    assert any(e.code == "membrane_control_mode_disabled" for e in verdict.events)


def test_membrane_control_mode_disabled_suppressed_within_resume_grace():
    # Real PACE5000 status reads can lag a set_control_mode(True) write by
    # more than one poll interval -- a still-False readback in the first
    # tick or two after a resume request must not be mistaken for a genuine
    # external relinquish (see OneSidedPressureController._reconcile_membrane_drive).
    safety = SafetySupervisor(_safety_cfg(), start_t=0.0)
    est_cfg = _estimator_cfg()
    est, now = fresh_estimator(est_cfg)
    safety.on_membrane_status(now)
    status = MembraneStatus(t_mono=now, connected=True, pressure_mpa=1.0, target_pressure_mpa=1.0, control_mode=False)
    verdict = safety.evaluate(est, status, now, control_mode_resume_grace_until=now + 2.0)
    assert not any(e.code == "membrane_control_mode_disabled" for e in verdict.events), \
        "a still-False readback within the resume grace window must not PAUSE"


def test_membrane_control_mode_disabled_flagged_after_resume_grace_expires():
    safety = SafetySupervisor(_safety_cfg(), start_t=0.0)
    est_cfg = _estimator_cfg()
    est, now = fresh_estimator(est_cfg)
    safety.on_membrane_status(now)
    status = MembraneStatus(t_mono=now, connected=True, pressure_mpa=1.0, target_pressure_mpa=1.0, control_mode=False)
    verdict = safety.evaluate(est, status, now, control_mode_resume_grace_until=now - 0.001)
    assert any(e.code == "membrane_control_mode_disabled" for e in verdict.events), \
        "a still-False readback after the resume grace window expires must still PAUSE " \
        "(a genuine external relinquish must still be caught)"
