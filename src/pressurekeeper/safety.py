"""SafetySupervisor: the single place allowed to veto pressurization.

Everything else (controller, gain estimator) assumes it is safe to act
unless told otherwise. This module owns no device I/O; it only consumes
events pushed to it by the control loop and answers two questions each
tick: "what is the current safety level" (ok / pause / abort) and, before
any write, "is this specific command allowed".

ABORT is sticky: once raised by an automatic trigger, `evaluate()` keeps
returning "abort" until `force_reset()` is called explicitly by an operator
(never by the control loop itself). PAUSE from an automatic trigger clears
itself once the triggering condition clears; PAUSE requested manually via
`request_manual_pause()` only clears via `clear_manual_pause()`.

The manual pause/abort/reset flags (and the membrane-stop-intent flag) are
touched from two threads: the control loop's polling thread (inside
`evaluate()`) and an operator-command thread (`request_manual_abort()` etc,
called directly from `OneSidedPressureController.abort()`/`pause()` without
waiting on the controller's own lock, so an emergency abort is never blocked
behind a stuck ruby/PACE5000 read). `_flag_lock` protects exactly these
fields; it is only ever held for a few attribute assignments, never across
any I/O, so it cannot itself become a source of the latency this design is
meant to avoid.
"""
from __future__ import annotations

import math
import threading
from collections import deque

from .config import SafetyConfig
from .estimator import PressureEstimator
from .models import MembraneStatus, SafetyEvent, SafetyVerdict


class SafetySupervisor:
    def __init__(self, config: SafetyConfig, start_t: float) -> None:
        self._cfg = config
        self._start_t = start_t

        self._flag_lock = threading.Lock()
        self._manual_pause = False
        self._manual_pause_reason: str | None = None
        self._aborted = False
        self._abort_reason: str | None = None
        self._membrane_stop_intended = False

        self._last_ruby_ok_t: float | None = None
        self._consecutive_ruby_errors = 0
        self._first_ruby_error_in_streak_t: float | None = None
        self._last_ruby_error_t: float | None = None

        self._last_membrane_ok_t: float | None = None
        self._consecutive_membrane_errors = 0
        self._first_membrane_error_in_streak_t: float | None = None
        self._last_membrane_error_t: float | None = None

        self._consecutive_jump_flags = 0

        self._last_commanded_setpoint_mpa: float | None = None
        self._last_command_t: float | None = None

        self._recent_steps: deque[tuple[float, float]] = deque()

    def update_config(self, config: SafetyConfig, start_t: float) -> None:
        """Hot-swap the config, resetting all tracked state (manual
        pause/abort flags, comm-error streaks, cumulative-step window, etc.)
        as if freshly constructed. Only safe to call before the control loop
        has started ticking -- see gui/parameters_config_dialog.py, gated to
        before Start Control -- since this discards the abort/pause latch
        along with everything else.
        """
        self.__init__(config, start_t)

    # ------------------------------------------------------------- event feed

    def on_ruby_sample(self, sample_gpa: float | None, prior_filtered_gpa: float | None, now: float) -> list[SafetyEvent]:
        self._last_ruby_ok_t = now
        self._consecutive_ruby_errors = 0
        self._first_ruby_error_in_streak_t = None
        events: list[SafetyEvent] = []
        if sample_gpa is not None and prior_filtered_gpa is not None and math.isfinite(sample_gpa):
            if abs(sample_gpa - prior_filtered_gpa) > self._cfg.sample_jump_hard_gpa:
                events.append(SafetyEvent(now, "hard_sample_jump", "pause",
                                           f"raw reading {sample_gpa:.4f} GPa jumped "
                                           f"{sample_gpa - prior_filtered_gpa:+.4f} GPa from filtered value"))
        return events

    def on_ruby_error(self, now: float) -> None:
        self._consecutive_ruby_errors += 1
        if self._first_ruby_error_in_streak_t is None:
            self._first_ruby_error_in_streak_t = now
        self._last_ruby_error_t = now

    def on_membrane_status(self, now: float) -> None:
        self._last_membrane_ok_t = now
        self._consecutive_membrane_errors = 0
        self._first_membrane_error_in_streak_t = None

    def on_membrane_error(self, now: float) -> None:
        self._consecutive_membrane_errors += 1
        if self._first_membrane_error_in_streak_t is None:
            self._first_membrane_error_in_streak_t = now
        self._last_membrane_error_t = now

    def on_command_issued(self, membrane_step_mpa: float, new_setpoint_mpa: float, now: float) -> None:
        self._recent_steps.append((now, membrane_step_mpa))
        self._last_commanded_setpoint_mpa = new_setpoint_mpa
        self._last_command_t = now

    def request_manual_pause(self, reason: str = "operator requested pause") -> None:
        with self._flag_lock:
            self._manual_pause = True
            self._manual_pause_reason = reason

    def clear_manual_pause(self) -> None:
        with self._flag_lock:
            self._manual_pause = False
            self._manual_pause_reason = None

    def request_manual_abort(self, reason: str = "operator requested abort") -> None:
        with self._flag_lock:
            self._aborted = True
            self._abort_reason = reason

    def force_reset(self) -> None:
        """Operator-only escape hatch. Never call this from automatic logic."""
        with self._flag_lock:
            self._aborted = False
            self._abort_reason = None
            self._manual_pause = False
            self._manual_pause_reason = None
        self._consecutive_ruby_errors = 0
        self._first_ruby_error_in_streak_t = None
        self._last_ruby_error_t = None
        self._consecutive_membrane_errors = 0
        self._first_membrane_error_in_streak_t = None
        self._last_membrane_error_t = None
        self._consecutive_jump_flags = 0
        # A reset starts a new target reconciliation while remaining in
        # Measure. Do not let the pre-abort target expectation prevent that
        # safe rebase; cumulative physical motion remains charged below.
        self._last_commanded_setpoint_mpa = None
        self._last_command_t = None

    def set_membrane_stop_intended(self, intended: bool) -> None:
        """Tell `evaluate()` whether a control-mode-off report from the
        PACE5000 is expected (our own STOP mechanism) or not (e.g. an
        operator took the panel to local control, still unsafe)."""
        with self._flag_lock:
            self._membrane_stop_intended = intended

    @property
    def membrane_stop_intended(self) -> bool:
        with self._flag_lock:
            return self._membrane_stop_intended

    @property
    def is_manually_paused(self) -> bool:
        with self._flag_lock:
            return self._manual_pause

    @property
    def last_command_t(self) -> float | None:
        """Timestamp of the most recent command accepted via on_command_issued
        (regular step or staged rearm), or None if none has been issued yet.
        Only ever written from the polling thread, so unlike the manual
        pause/abort flags this needs no lock."""
        return self._last_command_t

    def comm_errors_recent(self, now: float, within_s: float) -> bool:
        """Whether a ruby or PACE5000 read has failed within the last
        `within_s` seconds, even if the streak itself has since ended.

        A single failed read (or a streak that just recovered) doesn't by
        itself reach the consecutive-error or elapsed-time thresholds that
        raise their own PAUSE event, but a pressure-slope reading computed
        from a window that straddles a recent gap/discontinuity in the data
        is not yet a trustworthy basis for a *pure* compression-rate
        condition (see OneSidedPressureController._try_hold_at_current_pressure).
        Callers should pass their slope estimator's own smoothing window as
        `within_s`, since that's the span the slope reading actually draws on."""
        for last_error_t in (self._last_ruby_error_t, self._last_membrane_error_t):
            if last_error_t is not None and now - last_error_t < within_s:
                return True
        return False

    # ------------------------------------------------------------ pre-command

    def check_command(
        self,
        membrane_step_mpa: float,
        new_setpoint_mpa: float,
        now: float,
        *,
        source_pressure_mpa: float | None = None,
        allow_while_stopped: bool = False,
    ) -> tuple[bool, str | None]:
        # Operator flags are deliberately writable without the controller's
        # main lock so pause/abort is not delayed by sensor I/O.  Re-check them
        # at the last command gate as well as at the start of the tick; an
        # emergency request may have arrived after evaluate().
        with self._flag_lock:
            if self._aborted:
                return False, "abort_requested"
            if self._manual_pause:
                return False, "manual_pause_requested"
            if self._membrane_stop_intended and not allow_while_stopped:
                return False, "membrane_stop_intended"
        if not (math.isfinite(membrane_step_mpa) and math.isfinite(new_setpoint_mpa)):
            return False, "non_finite_command_value"
        if membrane_step_mpa < 0:
            return False, "membrane_step_negative"
        if membrane_step_mpa > self._cfg.max_membrane_step_mpa_hard:
            return False, "step_exceeds_hard_per_command_cap"
        if new_setpoint_mpa > self._cfg.max_membrane_pressure_mpa + 1e-9:
            return False, "setpoint_would_exceed_membrane_ceiling"
        if source_pressure_mpa is None or not math.isfinite(source_pressure_mpa):
            return False, "source_pressure_unavailable"
        required_source = new_setpoint_mpa + self._cfg.minimum_source_pressure_headroom_mpa
        if source_pressure_mpa <= required_source:
            return False, "setpoint_not_below_source_pressure"

        cutoff = now - self._cfg.cumulative_window_s
        while self._recent_steps and self._recent_steps[0][0] < cutoff:
            self._recent_steps.popleft()
        cumulative = sum(s for _, s in self._recent_steps) + membrane_step_mpa
        if cumulative > self._cfg.max_cumulative_step_mpa:
            return False, "cumulative_step_cap_exceeded"

        return True, None

    # ------------------------------------------------------------------ tick

    def evaluate(
        self,
        estimator: PressureEstimator,
        membrane_status: MembraneStatus | None,
        now: float,
        extra_events: list[SafetyEvent] | None = None,
        max_compression_rate_gpa_per_min: float | None = None,
        control_mode_resume_grace_until: float | None = None,
    ) -> SafetyVerdict:
        events: list[SafetyEvent] = list(extra_events or [])

        with self._flag_lock:
            aborted, abort_reason = self._aborted, self._abort_reason
            manual_pause, manual_pause_reason = self._manual_pause, self._manual_pause_reason
            stop_intended = self._membrane_stop_intended

        if aborted:
            events.append(SafetyEvent(now, "sticky_abort", "abort", abort_reason or "aborted"))
            return SafetyVerdict("abort", tuple(events))

        if manual_pause:
            events.append(SafetyEvent(now, "manual_pause", "pause", manual_pause_reason or "manual pause"))

        def _abort_here(code: str, reason: str) -> SafetyVerdict:
            with self._flag_lock:
                self._aborted = True
                self._abort_reason = reason
            events.append(SafetyEvent(now, code, "abort", reason))
            return SafetyVerdict("abort", tuple(events))

        last_sample = estimator.last_sample
        if last_sample is not None and last_sample.pressure_gpa is not None:
            p = last_sample.pressure_gpa
            if not math.isfinite(p):
                return _abort_here("non_finite_sample_pressure",
                                    f"ruby sample pressure is non-finite ({p!r})")
            if p > self._cfg.max_sample_pressure_gpa:
                return _abort_here(
                    "sample_pressure_over_limit",
                    f"sample pressure {p:.3f} GPa exceeds absolute limit {self._cfg.max_sample_pressure_gpa:.3f} GPa",
                )

        if membrane_status is not None and membrane_status.connected:
            pressure = membrane_status.pressure_mpa
            target = membrane_status.target_pressure_mpa
            source = membrane_status.source_pressure_positive_mpa

            # Missing pressure_mpa/control_mode are independent conditions
            # from the pressure/target-limit checks below -- a status
            # missing control_mode (but with a perfectly good, over-limit
            # pressure reading) must still abort on that reading, not have
            # the abort masked by only reporting "status incomplete".
            if pressure is None:
                events.append(SafetyEvent(now, "membrane_status_incomplete", "pause",
                                           "PACE5000 reports connected but pressure_mpa is missing"))
            if membrane_status.control_mode is None:
                events.append(SafetyEvent(now, "membrane_status_incomplete", "pause",
                                           "PACE5000 reports connected but control_mode is missing"))
            if source is None:
                events.append(SafetyEvent(now, "source_pressure_unavailable", "pause",
                                           "PACE5000 status does not include the positive supply pressure"))

            if (
                (pressure is not None and not math.isfinite(pressure))
                or (target is not None and not math.isfinite(target))
                or (source is not None and not math.isfinite(source))
            ):
                return _abort_here("non_finite_membrane_pressure",
                                    "membrane status has a non-finite pressure value "
                                    f"(pressure={pressure!r}, target={target!r}, source={source!r})")
            if pressure is not None:
                if pressure > self._cfg.max_membrane_pressure_mpa + 1e-6:
                    return _abort_here(
                        "membrane_pressure_over_limit",
                        f"membrane pressure {pressure:.3f} MPa exceeds absolute limit "
                        f"{self._cfg.max_membrane_pressure_mpa:.3f} MPa",
                    )
            if target is not None and target > self._cfg.max_membrane_pressure_mpa + 1e-6:
                # Catches a hazardous setpoint left over from before this
                # controller started polling (or set by something else
                # entirely) even while actual pressure is missing -- don't
                # wait for the ramp to cross the limit before reacting.
                return _abort_here(
                    "membrane_target_over_limit",
                    f"PACE5000 target {target:.3f} MPa exceeds absolute limit "
                    f"{self._cfg.max_membrane_pressure_mpa:.3f} MPa",
                )
            if (
                membrane_status.control_mode is True
                and pressure is not None
                and target is not None
            ):
                gap = abs(target - pressure)
                if gap > self._cfg.max_setpoint_actual_gap_mpa:
                    events.append(SafetyEvent(
                        now,
                        "setpoint_actual_gap_too_large",
                        "pause",
                        f"PACE5000 target/actual gap {gap:.4f} MPa exceeds the immediate-PAUSE "
                        f"limit {self._cfg.max_setpoint_actual_gap_mpa:.4f} MPa",
                    ))
            if (
                membrane_status.control_mode is True
                and target is not None
                and source is not None
                and source <= target + self._cfg.minimum_source_pressure_headroom_mpa
            ):
                events.append(SafetyEvent(
                    now,
                    "source_pressure_insufficient",
                    "pause",
                    f"positive supply pressure {source:.4f} MPa must exceed active setpoint "
                    f"{target:.4f} MPa by more than "
                    f"{self._cfg.minimum_source_pressure_headroom_mpa:.4f} MPa",
                ))

        if membrane_status is not None and not membrane_status.connected:
            # The control app answered (HTTP 200) but reports it has no live
            # connection to the PACE5000 itself — a distinct failure mode from
            # a comm error and just as unsafe to act on, since every reading
            # on the status is None and _current_membrane_setpoint() would
            # otherwise let the controller sit silently forever.
            events.append(SafetyEvent(now, "membrane_disconnected", "pause",
                                       "PACE5000 control app reports the membrane controller is not connected"))

        within_resume_grace = (
            control_mode_resume_grace_until is not None and now < control_mode_resume_grace_until
        )
        if (
            membrane_status is not None
            and membrane_status.control_mode is False
            and not stop_intended
            and not within_resume_grace
        ):
            # If remote control is relinquished (e.g. an operator takes the
            # panel to local control) any set_pressure() call we
            # issue is silently ignored by the device, so this must pause
            # just like a lost connection. Suppressed while `stop_intended`
            # is set: that means *we* disabled control mode on purpose (see
            # OneSidedPressureController._stop_membrane), not an external
            # relinquish. Also suppressed for a short grace window right
            # after we asked to re-enable Control (control_mode_resume_grace_until,
            # from OneSidedPressureController._reconcile_membrane_drive): on
            # real hardware, the PACE5000 control app's write can 200 OK
            # before its own status endpoint reports control_mode=True, so a
            # still-False readback in the first tick or two after a resume
            # request is expected lag, not a real external relinquish -- see
            # membrane_control_mode_disabled in the README/CLAUDE.md history.
            events.append(SafetyEvent(now, "membrane_control_mode_disabled", "pause",
                                       "PACE5000 is not in remote control mode; commands would be ignored"))

        had_first_sample = estimator.last_valid_age_s(now) is not None
        if not had_first_sample:
            if now - self._start_t > self._cfg.max_stale_sample_s:
                events.append(SafetyEvent(now, "no_valid_sample_yet", "pause", "no valid ruby reading since startup"))
        else:
            age = estimator.last_valid_age_s(now)
            if age is not None and age > self._cfg.max_stale_sample_s:
                events.append(SafetyEvent(now, "stale_measurement", "pause",
                                           f"last valid ruby sample is {age:.1f} s old"))

        if estimator.consecutive_invalid >= self._cfg.max_consecutive_invalid:
            events.append(SafetyEvent(now, "consecutive_invalid_measurements", "pause",
                                       f"{estimator.consecutive_invalid} consecutive invalid ruby measurements"))

        if estimator.last_jump_flagged:
            self._consecutive_jump_flags += 1
        else:
            self._consecutive_jump_flags = 0
        if self._consecutive_jump_flags >= self._cfg.max_consecutive_jump_flags:
            events.append(SafetyEvent(now, "abnormal_sample_jump_streak", "pause",
                                       f"{self._consecutive_jump_flags} consecutive flagged jumps in ruby reading"))

        if self._consecutive_ruby_errors >= self._cfg.max_consecutive_comm_errors:
            events.append(SafetyEvent(now, "ruby_api_unreachable", "pause",
                                       f"{self._consecutive_ruby_errors} consecutive ruby API errors"))
        elif self._first_ruby_error_in_streak_t is not None and now - self._first_ruby_error_in_streak_t >= self._cfg.ruby_error_pause_after_s:
            events.append(SafetyEvent(now, "ruby_api_unreachable", "pause", "ruby API not recovering"))

        if self._consecutive_membrane_errors >= self._cfg.max_consecutive_comm_errors:
            events.append(SafetyEvent(now, "pace5000_unreachable", "pause",
                                       f"{self._consecutive_membrane_errors} consecutive PACE5000 API errors"))
        elif self._first_membrane_error_in_streak_t is not None and now - self._first_membrane_error_in_streak_t >= self._cfg.membrane_error_pause_after_s:
            events.append(SafetyEvent(now, "pace5000_unreachable", "pause", "PACE5000 API not recovering"))

        if (
            self._last_commanded_setpoint_mpa is not None
            and self._last_command_t is not None
            and membrane_status is not None
            and membrane_status.target_pressure_mpa is not None
            and now - self._last_command_t >= self._cfg.setpoint_mismatch_grace_s
        ):
            mismatch = abs(membrane_status.target_pressure_mpa - self._last_commanded_setpoint_mpa)
            if mismatch > self._cfg.setpoint_mismatch_tol_mpa:
                events.append(SafetyEvent(now, "setpoint_mismatch", "pause",
                                           f"PACE5000 target {membrane_status.target_pressure_mpa:.4f} MPa != "
                                           f"commanded {self._last_commanded_setpoint_mpa:.4f} MPa"))

        if max_compression_rate_gpa_per_min is not None:
            slope = estimator.pressure_slope()
            if slope is not None and math.isfinite(slope):
                cap_gpa_per_s = max_compression_rate_gpa_per_min / 60.0
                if slope > cap_gpa_per_s:
                    events.append(SafetyEvent(now, "compression_rate_exceeded", "pause",
                                               f"observed sample-pressure rate {slope * 60.0:.4f} GPa/min exceeds "
                                               f"configured cap {max_compression_rate_gpa_per_min:.4f} GPa/min"))

        level = "pause" if any(e.severity in ("pause", "abort") for e in events) else "ok"
        return SafetyVerdict(level, tuple(events))
