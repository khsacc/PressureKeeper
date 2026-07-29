"""OneSidedPressureController: predictive, one-directional, stepwise state
machine that drives actual membrane pressure toward a user target without
requesting a physical decrease and without overshooting.

This is deliberately not a PID: every step size is derived from a
safety-biased sensitivity estimate and a hysteresis band around the target,
and no new step is ever issued until the previous one has demonstrably
settled. See README.md for the full state diagram and rationale.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import replace

from .clock import Clock, MonotonicClock
from .config import Configuration
from .errors import MembraneCommError, RubyCommError
from .gain import GainEstimator
from .estimator import PressureEstimator
from .interfaces import MembranePressureController, RubyPressureSource
from .logging_sink import DataLogger
from .models import (
    ControllerSnapshot,
    ControlState,
    MembraneStatus,
    SafetyEvent,
    SafetyVerdict,
    StateTransition,
    StepRecord,
)
from .safety import SafetySupervisor


class OneSidedPressureController:
    def __init__(
        self,
        config: Configuration,
        ruby: RubyPressureSource,
        membrane: MembranePressureController,
        estimator: PressureEstimator,
        gain_estimator: GainEstimator,
        safety: SafetySupervisor,
        logger: DataLogger | None = None,
        clock: Clock | None = None,
        initial_logging_error: str | None = None,
    ) -> None:
        self._cfg = config
        self._ruby = ruby
        self._membrane = membrane
        self._estimator = estimator
        self._gain_estimator = gain_estimator
        self._safety = safety
        self._logger = logger
        self._clock = clock or MonotonicClock()

        # step() runs on a background polling thread while operator commands
        # (set_target/pause/resume/abort/reset) arrive from the CLI's main
        # thread; both mutate controller state, so they serialize on this
        # lock. Blocking device I/O (ruby.read/membrane.read_status, and the
        # STOP/step writes issued from within a tick) deliberately happens
        # without holding this lock across the call, and abort()/pause()
        # issue their own emergency-stop attempt entirely independent of it
        # (see _emergency_stop_attempt) -- an operator command must never be
        # stuck waiting on a hung ruby/PACE5000 read.
        self._lock = threading.RLock()

        self.state = ControlState.APPROACH
        self.user_target_gpa = config.control.default_target_pressure_gpa or 0.0
        self._max_compression_rate_gpa_per_min = config.approach.max_compression_rate_gpa_per_min
        self._membrane_rate_mpa_per_min = config.pace5000_api.default_rate_mpa_per_min

        self._membrane_status: MembraneStatus | None = None
        self._membrane_status_fresh_this_tick = False
        self._last_membrane_poll_t: float | None = None
        self._pending_step: StepRecord | None = None
        self._staged_rearm_step: StepRecord | None = None
        self._next_step_id = 1
        self._settled_since: float | None = None
        self._prior_state_before_pause: ControlState | None = None
        # Compression-rate-only PAUSE recovery (see _try_hold_at_current_pressure):
        # once the fresh-actual-pressure hold target below has been written and
        # confirmed, _rate_pause_holding latches so Control stays armed against
        # it (countering leaks) instead of re-stopping every tick while the
        # observed slope remains above the cap. _rate_pause_eligible tracks
        # whether *every* tick since this PAUSE episode began has been caused
        # solely by compression_rate_exceeded -- a transient sensor artefact
        # (e.g. a hard_sample_jump) whose slope-window echo outlives the jump
        # itself must never retroactively qualify.
        self._rate_pause_eligible = False
        self._rate_pause_holding = False
        self._rate_pause_hold_setpoint: float | None = None
        self._rate_pause_hold_t_command: float | None = None
        # Set once per resume episode, the first tick _reconcile_membrane_drive
        # starts attempting to re-arm Control (stop_intended False but
        # control_mode not yet confirmed True) -- never renewed on later ticks
        # of the same episode, so a control_mode that never actually comes
        # back (a genuine external relinquish) still PAUSEs once this expires.
        # See safety.py's membrane_control_mode_disabled check.
        self._control_mode_resume_grace_until: float | None = None
        self._last_command_reason: str | None = None
        self._last_command_decision: dict | None = None
        self._logging_error = initial_logging_error
        # Lowering a one-sided target cannot be followed by an automatic
        # "correction" in the other direction.  It means stop where we are and
        # remain held until the operator restores at least the pre-reduction
        # target; a smaller new target could still be overshot by the stopped
        # but not cancelled device setpoint.
        self._target_reduction_hold = False
        self._target_reduction_resume_floor_gpa: float | None = None
        # dry_run never actually writes, so _membrane_status.control_mode
        # never changes -- track our own believed intent just to avoid
        # re-logging the same "would stop/resume" event every single tick.
        self._dry_run_driving_believed = True
        # Startup is a recovery boundary too.  Never trust a target left in
        # the PACE5000: first confirm Measure, stage a current-pressure-based
        # safe target, confirm its readback, and only then enter Control.
        self._safety.set_membrane_stop_intended(True)

    # ------------------------------------------------------------ read-only

    @property
    def config(self) -> Configuration:
        return self._cfg

    @property
    def safety(self) -> SafetySupervisor:
        return self._safety

    @property
    def gain_estimator(self) -> GainEstimator:
        return self._gain_estimator

    @property
    def estimator(self) -> PressureEstimator:
        return self._estimator

    @property
    def logging_error(self) -> str | None:
        return self._logging_error

    # --------------------------------------------------------------- control

    def apply_config_update(self, config: Configuration) -> None:
        """Hot-swap the entire Configuration this controller, its estimator,
        gain_estimator, and safety supervisor use -- without reconstructing
        the ruby/membrane clients, the logger, or restarting the process.

        Only meant to be called from the GUI's "Configure Parameters" dialog
        (see gui/parameters_config_dialog.py), and only before Start Control:
        `estimator`/`gain_estimator`/`safety` are reset to their fresh-
        construction state (see their own `update_config()`), which would
        corrupt an in-progress run's buffered history, gain observations, and
        pause/abort latch. Does not touch `self.user_target_gpa` -- an
        operator-entered target (e.g. via the Single Target tab) must not be
        silently overwritten by config.control.default_target_pressure_gpa.
        A lowered `config.safety.max_sample_pressure_gpa` still takes effect
        immediately even though `user_target_gpa` itself is left alone: every
        approach/hold/step-sizing decision reads the target through
        `_effective_target_gpa()`, which clamps to the current ceiling (see
        its docstring). Callers may still want to warn the operator that
        `user_target_gpa` exceeds the new ceiling so they know why approach
        stopped short of the value they originally entered.
        """
        with self._lock:
            self._cfg = config
            self._estimator.update_config(config.estimator)
            self._gain_estimator.update_config(config.gain_estimation)
            self._safety.update_config(config.safety, self._clock.now())
            # SafetySupervisor.update_config() resets membrane_stop_intended
            # to its own construction default (False) -- restore the
            # startup invariant that nothing re-arms Control until a fresh
            # safe setpoint has been staged and confirmed.
            self._safety.set_membrane_stop_intended(True)
        # Reapply the operator-adjustable rate caps against the new config
        # (new gain_regions in particular) using the existing setters, so
        # membrane_rate_mpa_per_min's settle-time-vs-ramp-time invariant is
        # re-validated against the new gain_regions rather than silently
        # left checked against the old ones.
        self.set_max_compression_rate(config.approach.max_compression_rate_gpa_per_min)
        self.set_membrane_rate_mpa_per_min(config.pace5000_api.default_rate_mpa_per_min)

    def _effective_target_gpa(self) -> float:
        """`user_target_gpa` clamped to the current safety ceiling.

        `set_target()` rejects a target above `safety.max_sample_pressure_gpa`
        at entry, but the ceiling itself can be lowered afterwards via
        `apply_config_update()` while an operator target set under the old,
        higher ceiling is still in effect. Every approach/hold/step-sizing
        decision must route through this (rather than reading
        `user_target_gpa` directly) so a lowered ceiling caps forward motion
        on the very next tick -- not only once the hard
        `sample_pressure_over_limit` abort in safety.py trips. Unlike a
        target *reduction* (see `set_target`'s `_target_reduction_hold`),
        this never latches: raising the ceiling back up immediately raises
        the effective target again, since nothing here changes when the
        ceiling changes, only what it's compared against.
        """
        return min(self.user_target_gpa, self._cfg.safety.max_sample_pressure_gpa)

    def set_target(self, target_gpa: float) -> None:
        if not math.isfinite(target_gpa) or target_gpa < 0:
            raise ValueError(f"target must be a finite, non-negative pressure (got {target_gpa!r})")
        if target_gpa > self._cfg.safety.max_sample_pressure_gpa:
            raise ValueError(
                f"target {target_gpa:.3f} GPa exceeds the configured absolute "
                f"sample-pressure limit {self._cfg.safety.max_sample_pressure_gpa:.3f} GPa"
            )
        stop_now = False
        with self._lock:
            old_target = self.user_target_gpa
            self.user_target_gpa = target_gpa
            if target_gpa < old_target and (
                self._pending_step is not None
                or self._staged_rearm_step is not None
                or self._target_reduction_hold
            ):
                self._target_reduction_hold = True
                floor = self._target_reduction_resume_floor_gpa
                self._target_reduction_resume_floor_gpa = old_target if floor is None else max(floor, old_target)
                self._safety.set_membrane_stop_intended(True)
                self._abandon_motion(self._clock.now(), "target reduction interrupted the active/staged step")
                self._rate_pause_eligible = False
                self._rate_pause_holding = False
                self._rate_pause_hold_setpoint = None
                if self.state not in (ControlState.PAUSE, ControlState.ABORT):
                    self._set_state(ControlState.HOLD, self._clock.now(), "target was lowered; holding one-sided actuator")
                stop_now = True
            elif (
                target_gpa > old_target
                and self._target_reduction_resume_floor_gpa is not None
                and target_gpa >= self._target_reduction_resume_floor_gpa
            ):
                self._target_reduction_hold = False
                self._target_reduction_resume_floor_gpa = None
            # Deliberately leave HOLD latched here.  The next tick, using a
            # fresh ruby reading, decides whether the new target is far enough
            # above the sample to cross the re-approach hysteresis and re-arm
            # the membrane.  Switching to APPROACH immediately used to strand
            # `_membrane_stop_intended=True` forever (no code path remained to
            # clear it), and clearing the stop here would re-arm an old
            # in-flight setpoint before fresh feedback had been evaluated.
        if stop_now:
            # Like pause()/abort(), a lower target must cancel forward motion
            # immediately rather than waiting behind a slow sensor read.
            self._emergency_stop_attempt()

    @property
    def max_compression_rate_gpa_per_min(self) -> float | None:
        return self._max_compression_rate_gpa_per_min

    def set_max_compression_rate(self, gpa_per_min: float | None) -> None:
        """Operator-adjustable ceiling on sample-pressure rise rate.

        None clears the cap (only the per-region step caps in config still
        apply). Further restricts, never loosens, gain_regions[].max_sample_step_gpa.
        """
        if gpa_per_min is not None and (not math.isfinite(gpa_per_min) or gpa_per_min <= 0):
            raise ValueError(
                f"max_compression_rate_gpa_per_min must be finite and > 0 (got {gpa_per_min!r})"
            )
        with self._lock:
            self._max_compression_rate_gpa_per_min = gpa_per_min

    @property
    def membrane_rate_mpa_per_min(self) -> float:
        return self._membrane_rate_mpa_per_min

    def set_membrane_rate_mpa_per_min(self, mpa_per_min: float) -> None:
        """Operator-adjustable slew rate sent to the PACE5000 with every
        set_pressure() command (i.e. how fast the membrane/gas pressure
        setpoint itself is allowed to ramp). Independent of
        max_compression_rate_gpa_per_min, which caps the resulting sample
        pressure rise rather than the gas-side ramp -- satisfying the latter
        at high gain can require a rate much slower than gain_regions was
        sized around. A slower rate no longer needs rejecting here: each
        step's settle blackout now extends to cover that step's own actual
        ramp time at whatever rate it was commanded with (see
        _update_pending_step), so settle detection can no longer fire while
        the membrane is still mid-ramp regardless of how slow this is.
        """
        if not math.isfinite(mpa_per_min) or mpa_per_min <= 0:
            raise ValueError(
                f"membrane_rate_mpa_per_min must be finite and > 0 (got {mpa_per_min!r})"
            )
        with self._lock:
            self._membrane_rate_mpa_per_min = mpa_per_min

    def pause(self, reason: str = "operator requested pause") -> None:
        # Deliberately not gated by self._lock: SafetySupervisor's own flag
        # lock makes this safe to call while the polling thread is mid-tick,
        # and _emergency_stop_attempt() issues its own independent HTTP call
        # so a stuck ruby/PACE5000 read never delays this.
        self._safety.request_manual_pause(reason)
        self._safety.set_membrane_stop_intended(True)
        self._emergency_stop_attempt()
        with self._lock:
            self._abandon_motion(self._clock.now(), reason)
            self._rate_pause_holding = False
            self._rate_pause_hold_setpoint = None

    def resume(self) -> None:
        self._safety.clear_manual_pause()

    def abort(self, reason: str = "operator requested abort") -> None:
        self._safety.request_manual_abort(reason)
        self._safety.set_membrane_stop_intended(True)
        self._emergency_stop_attempt()
        with self._lock:
            self._abandon_motion(self._clock.now(), reason)
            self._rate_pause_holding = False
            self._rate_pause_hold_setpoint = None

    def reset(self) -> None:
        """Operator-only recovery from ABORT. Never called automatically.

        Reset deliberately remains in Measure.  A later valid feedback tick
        recalculates and stages a safe target before Control can be enabled.
        """
        with self._lock:
            self._safety.force_reset()
            self._safety.set_membrane_stop_intended(True)
            self.state = ControlState.APPROACH
            self._abandon_motion(self._clock.now(), "operator reset")
            self._rate_pause_holding = False
            self._rate_pause_hold_setpoint = None
            self._prior_state_before_pause = None
        # Usually ABORT has already done this, but Reset is exposed to an
        # operator and must be safe even if invoked outside ABORT.
        self._emergency_stop_attempt()

    def stop_and_confirm(self, timeout_s: float = 5.0, poll_interval_s: float = 0.25) -> bool:
        """Blocking shutdown sequence: disable PACE5000 control mode and
        wait for read-back confirmation before the caller (CLI/GUI) exits.

        Must be called before closing the HTTP session/logger on quit,
        Ctrl-C, or window-close -- otherwise the PACE5000 keeps ramping
        toward its last setpoint after this process is gone. Uses real
        wall-clock timing (not self._clock) since this waits on real
        hardware regardless of what clock the control loop itself is using
        (e.g. a FakeClock in tests/sim).

        Returns True once confirmed stopped, False if it could not confirm
        within timeout_s (the caller should still exit, but should tell the
        operator plainly that it could not confirm).
        """
        self.abort("process shutting down")
        if self._cfg.control.dry_run:
            return True
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                status = self._membrane.read_status()
            except MembraneCommError:
                status = None
            if status is not None and status.control_mode is False:
                return True
            if time.monotonic() >= deadline:
                return False
            self._emergency_stop_attempt()
            time.sleep(poll_interval_s)

    def _emergency_stop_attempt(self) -> None:
        """Best-effort immediate PACE5000 stop, independent of self._lock and
        of whatever the polling thread's current step() is doing. This is
        what actually bounds emergency-stop latency; step()'s own
        _reconcile_membrane_drive (run every tick under self._lock) retries
        this regardless of whether this direct attempt succeeds, so a
        failure here is never the only chance.
        """
        if self._cfg.control.dry_run:
            return
        try:
            self._membrane.set_control_mode(False)
        except MembraneCommError:
            pass

    # ------------------------------------------------------------------ tick

    def step(self, now: float | None = None) -> ControllerSnapshot:
        t_start = self._clock.now() if now is None else now

        due_for_status_poll = (
            self._last_membrane_poll_t is None
            or (t_start - self._last_membrane_poll_t) >= self._cfg.pace5000_api.status_poll_interval_s
        )

        # I/O outside the lock: only this polling thread ever calls step(),
        # so these calls don't need to be serialized against each other --
        # they only must not make an operator command on another thread
        # (abort/pause/set_target/resume) wait out a slow or timed-out read
        # to acquire self._lock.
        ruby_sample = None
        ruby_error: RubyCommError | None = None
        try:
            ruby_sample = self._ruby.read()
        except RubyCommError as e:
            ruby_error = e

        status: MembraneStatus | None = None
        status_error: MembraneCommError | None = None
        if due_for_status_poll:
            try:
                status = self._membrane.read_status()
            except MembraneCommError as e:
                status_error = e

        # Re-read the clock after I/O completes: a ruby/PACE5000 sample is
        # timestamped by its client at receipt, which can be well after
        # whatever `now` was taken before a slow (or timed-out) call --
        # using that earlier value for staleness/settle-blackout math can
        # otherwise produce negative ages and mask real staleness.
        now = self._clock.now()

        with self._lock:
            self._membrane_status_fresh_this_tick = False
            prior_filtered = self._estimator.filtered_pressure()
            extra_events: list[SafetyEvent] = []

            if ruby_error is not None:
                self._safety.on_ruby_error(now)
                self._log_event(SafetyEvent(now, "ruby_read_error", "warning", str(ruby_error)))
            else:
                assert ruby_sample is not None
                self._estimator.update(ruby_sample)
                extra_events += self._safety.on_ruby_sample(ruby_sample.pressure_gpa, prior_filtered, now)

            if due_for_status_poll:
                self._last_membrane_poll_t = now
                if status_error is not None:
                    self._safety.on_membrane_error(now)
                    self._log_event(SafetyEvent(now, "membrane_read_error", "warning", str(status_error)))
                    status = self._membrane_status
                else:
                    self._safety.on_membrane_status(now)
                    self._membrane_status_fresh_this_tick = True
            else:
                status = self._membrane_status

            if (
                self._cfg.control.dry_run
                and status is not None
                and status.connected
            ):
                # Rehearsal mode must never toggle the real PACE5000 output.
                # Evaluate the state machine against our simulated drive
                # intent instead -- re-applied every tick, not only ticks
                # with a fresh status poll: _dry_run_driving_believed can
                # flip on any tick (_reconcile_membrane_drive runs every
                # tick) while status polls are comparatively infrequent.
                # Baking the override in only at poll time left a stale
                # real control_mode=False cached in self._membrane_status
                # between polls, spuriously tripping
                # membrane_control_mode_disabled and abandoning a
                # just-staged step on every rearm cycle.
                status = replace(status, control_mode=self._dry_run_driving_believed)

            self._membrane_status = status

            verdict = self._safety.evaluate(
                self._estimator, status, now, extra_events,
                max_compression_rate_gpa_per_min=self._max_compression_rate_gpa_per_min,
                control_mode_resume_grace_until=self._control_mode_resume_grace_until,
            )

            # Safety action before logging: a logging failure (disk full,
            # sink exception) must never be able to suppress a freeze/stop.
            self._apply_verdict(verdict, now)

            for ev in verdict.events:
                self._log_event(ev)

            snap = self._snapshot(now, verdict)
            if self._logger is not None:
                self._safe_call(lambda: self._logger.log_tick(snap))
                if self._logging_error != snap.logging_error:
                    snap = replace(snap, logging_error=self._logging_error)
            return snap

    # ------------------------------------------------------------- internals

    def _apply_verdict(self, verdict: SafetyVerdict, now: float) -> None:
        if verdict.level == "abort":
            self._safety.set_membrane_stop_intended(True)
            self._abandon_motion(now, "safety abort interrupted the active/staged step")
            self._rate_pause_holding = False
            self._rate_pause_hold_setpoint = None
            self._set_state(ControlState.ABORT, now, "safety abort")
        elif verdict.level == "pause":
            entering_pause = self.state != ControlState.PAUSE
            if entering_pause:
                self._prior_state_before_pause = self.state
                self._rate_pause_eligible = True
            # Every PAUSE is a Measure boundary: an old device setpoint is
            # never re-armed. The one exception is a PAUSE episode caused,
            # every tick since it began, solely by compression_rate_exceeded:
            # once _try_hold_at_current_pressure has written and confirmed a
            # *freshly computed* current-pressure hold target, Control stays
            # armed against it so the PACE5000 itself counteracts small leaks
            # for the rest of the episode, instead of repeating a
            # stop/measure/rebase cycle every tick while the observed slope
            # stays above the cap. Requiring purity since entry (not just this
            # tick) matters because a hard_sample_jump's effect on the slope
            # estimate can outlive the jump event itself: the first tick or
            # two of such an episode carry both events, but by the time the
            # jump flag itself stops firing, treating the lingering
            # compression_rate_exceeded alone as "rate-only" would hold-rebase
            # against a cause that was never purely a rate issue. A ruby/
            # PACE5000 comm error within the slope window disqualifies it the
            # same way, even after the streak itself has ended and even
            # before it was ever old/long enough to raise its own PAUSE
            # event: the slope estimate spans slope_window_s, so a gap inside
            # that window (e.g. right after a ruby outage clears) still
            # taints the reading it's computed from.
            rate_only_this_tick = (
                all(e.code == "compression_rate_exceeded" for e in verdict.events)
                and not self._safety.comm_errors_recent(now, self._cfg.estimator.slope_window_s)
            )
            self._rate_pause_eligible = self._rate_pause_eligible and rate_only_this_tick
            if self._rate_pause_eligible and self._rate_pause_holding:
                pass
            else:
                if not self._rate_pause_eligible:
                    self._rate_pause_holding = False
                    self._rate_pause_hold_setpoint = None
                self._safety.set_membrane_stop_intended(True)
                self._abandon_motion(now, "pause interrupted the active/staged step")
                if self._rate_pause_eligible:
                    self._try_hold_at_current_pressure(now)
            reason = "; ".join(e.message for e in verdict.events) or "paused"
            self._set_state(ControlState.PAUSE, now, reason)
        else:
            # evaluate() always returns "pause" while manually paused, so
            # reaching this branch already implies the manual pause has
            # been cleared.
            self._rate_pause_holding = False
            self._rate_pause_hold_setpoint = None
            if self.state == ControlState.PAUSE:
                resumed = self._prior_state_before_pause or ControlState.APPROACH
                self._prior_state_before_pause = None
                if resumed == ControlState.SETTLE:
                    resumed = ControlState.APPROACH
                self._set_state(resumed, now, "safety condition cleared; resuming")
            self._advance(now)

        # Runs every tick, regardless of which branch above fired: reconciles
        # the PACE5000's actual control-mode state with whatever we currently
        # intend, retrying a failed stop/resume write indefinitely rather
        # than giving up after one attempt (see _stop_membrane).
        self._reconcile_membrane_drive(now)

    def _reconcile_membrane_drive(self, now: float) -> None:
        if self._safety.membrane_stop_intended:
            self._control_mode_resume_grace_until = None
            if self._membrane_status is None or self._membrane_status.control_mode is not False:
                self._stop_membrane(now)
        else:
            control_mode_confirmed_true = (
                self._membrane_status is not None and self._membrane_status.control_mode is True
            )
            if control_mode_confirmed_true:
                self._control_mode_resume_grace_until = None
            elif self._control_mode_resume_grace_until is None:
                # First tick of this resume episode that still needs a
                # write/wait -- start the grace window now, and never renew
                # it on later ticks of the same episode (see __init__).
                self._control_mode_resume_grace_until = (
                    now + self._cfg.pace5000_api.control_mode_resume_grace_s
                )
            if not control_mode_confirmed_true:
                self._resume_membrane_drive(now)

    def _stop_membrane(self, now: float) -> None:
        """Disable PACE5000 control mode (switch it from actively driving a
        setpoint to measure-only), halting any in-flight ramp regardless of
        setpoint math or measurement staleness. Safe to call every tick:
        no-ops once read-back confirms control mode is off, and otherwise
        retries the write -- this is what makes an abort-freeze retry after
        a failed first attempt instead of latching "aborted but still
        ramping" forever.
        """
        if self._cfg.control.dry_run:
            if self._dry_run_driving_believed:
                self._dry_run_driving_believed = False
                self._log_event(SafetyEvent(now, "dry_run_command_suppressed", "info",
                                             "would disable PACE5000 control mode (STOP); write suppressed by dry_run"))
            return
        try:
            self._membrane.set_control_mode(False)
        except MembraneCommError as e:
            self._log_event(SafetyEvent(now, "membrane_stop_failed", "warning",
                                         f"failed to disable PACE5000 control mode; will retry: {e}"))
            return
        self._log_event(SafetyEvent(now, "membrane_stopped", "info",
                                     "disabled PACE5000 control mode to halt any in-flight ramp"))

    def _resume_membrane_drive(self, now: float) -> bool:
        """Re-arm PACE5000 control mode after the safe Measure-mode target
        has already been written and confirmed. Returns True once the write
        succeeds; mode readback comes from a later status poll."""
        if not self._cfg.pace5000_api.ensure_control_mode_enabled:
            return False
        if self._cfg.control.dry_run:
            if not self._dry_run_driving_believed:
                self._dry_run_driving_believed = True
                self._log_event(SafetyEvent(now, "dry_run_command_suppressed", "info",
                                             "would re-enable PACE5000 control mode; write suppressed by dry_run"))
            return True
        try:
            self._membrane.set_control_mode(True)
        except MembraneCommError as e:
            self._log_event(SafetyEvent(now, "membrane_resume_failed", "warning",
                                         f"failed to re-enable PACE5000 control mode; will retry: {e}"))
            return False
        self._log_event(SafetyEvent(now, "membrane_control_resumed", "info",
                                     "re-enabled PACE5000 control mode"))
        return True

    def _try_hold_at_current_pressure(self, now: float) -> None:
        """Recovery specific to a compression-rate-only PAUSE (see
        _apply_verdict): once Measure is confirmed, stage the *current,
        fresh* actual membrane pressure -- no forward step -- as the new
        setpoint and re-enable Control, so the PACE5000's own regulation
        counteracts small leaks for the rest of the episode instead of
        sitting inertly in Measure until the observed slope drops back under
        the configured cap. Never re-arms the stale pre-pause setpoint: the
        target used here is always the actual pressure read this tick.
        """
        if self._rate_pause_hold_setpoint is not None:
            self._confirm_rate_pause_hold(now)
            return
        if self._membrane_status is None or self._membrane_status.control_mode is not False:
            return  # awaiting Measure confirmation; retried every tick
        if not self._membrane_status_fresh_this_tick or self._membrane_status.pressure_mpa is None:
            return
        current_actual = self._membrane_status.pressure_mpa
        filtered = self._estimator.filtered_pressure()
        slope = self._estimator.pressure_slope()
        if filtered is None or slope is None:
            return
        sizing_pressure = self._conservative_sample_pressure(filtered)
        # If _advance() would put/keep this at HOLD once the pause clears
        # anyway (at/above the overshoot margin, or already predicted to
        # reach the target), its own hysteresis already protects against
        # small dips -- mirrors the two HOLD checks at the top of _advance().
        # Skipping the rebase here matters because ordinary measurement
        # noise can transiently trip compression_rate_exceeded right at
        # HOLD; without this, every such blip would write a redundant hold
        # command that plain Measure-and-clear would never have issued.
        # Uses the same ceiling-clamped target _advance() does (see
        # _effective_target_gpa) so a lowered safety ceiling isn't re-armed
        # against here either.
        target = self._effective_target_gpa()
        hyst = self._cfg.hysteresis
        if sizing_pressure > target + hyst.overshoot_margin_gpa:
            return
        predicted = max(
            filtered + max(slope, 0.0) * self._cfg.approach.prediction_horizon_s,
            sizing_pressure,
        )
        if predicted >= target - hyst.reach_margin_gpa:
            return
        # Respect the same settle blackout as any other command: issuing this
        # write before the previous command's minimum_settle_time_s has
        # elapsed would repeat the exact hazard the blackout exists to
        # prevent (see the module docstring's tuning note) -- stacking a new
        # command before the last one's real response is even observable.
        last_command_t = self._safety.last_command_t
        if last_command_t is not None:
            region = self._cfg.region_for(sizing_pressure)
            if now - last_command_t < region.minimum_settle_time_s:
                return
        allowed, reason = self._safety.check_command(
            0.0, current_actual, now,
            source_pressure_mpa=self._membrane_status.source_pressure_positive_mpa,
            allow_while_stopped=True,
        )
        if not allowed:
            self._log_event(SafetyEvent(now, "command_blocked", "warning", reason or "blocked"))
            return
        if self._cfg.control.dry_run:
            self._log_event(SafetyEvent(now, "dry_run_command_suppressed", "info",
                                         f"would hold membrane setpoint at {current_actual:.4f} MPa "
                                         "to counteract leaks during a compression-rate pause; "
                                         "write suppressed by dry_run"))
            self._rate_pause_holding = True
            self._safety.set_membrane_stop_intended(False)
            return
        try:
            self._membrane.set_pressure(current_actual, self._membrane_rate_mpa_per_min)
        except MembraneCommError as e:
            self._log_event(SafetyEvent(now, "membrane_write_ambiguous", "warning",
                                         f"hold-at-current-pressure write may or may not have applied: {e}"))
        self._safety.on_command_issued(0.0, current_actual, now)
        self._rate_pause_hold_setpoint = current_actual
        self._rate_pause_hold_t_command = now
        self._log_event(SafetyEvent(now, "rate_pause_hold_staged", "info",
                                     f"staged hold setpoint {current_actual:.4f} MPa in Measure "
                                     "to counteract leaks during compression-rate pause"))

    def _confirm_rate_pause_hold(self, now: float) -> None:
        """Confirm the hold-at-current-pressure write before re-enabling
        Control -- mirrors _confirm_staged_rearm's fresh-readback gate."""
        status = self._membrane_status
        setpoint = self._rate_pause_hold_setpoint
        assert setpoint is not None
        if (
            not self._membrane_status_fresh_this_tick
            or status is None
            or status.t_mono <= self._rate_pause_hold_t_command
            or status.target_pressure_mpa is None
            or status.pressure_mpa is None
        ):
            return

        tolerance = self._cfg.safety.setpoint_mismatch_tol_mpa
        if abs(status.target_pressure_mpa - setpoint) > tolerance:
            # SafetySupervisor emits its own PAUSE event for the same
            # mismatch. Remain in Measure and do not overwrite evidence with
            # another write.
            return

        source = status.source_pressure_positive_mpa
        required_source = setpoint + self._cfg.safety.minimum_source_pressure_headroom_mpa
        if source is None or not math.isfinite(source) or source <= required_source:
            return

        if (
            not self._cfg.pace5000_api.ensure_control_mode_enabled
            and status.control_mode is not True
        ):
            return  # wait for an operator to arm Control manually

        self._rate_pause_holding = True
        self._safety.set_membrane_stop_intended(False)
        self._log_event(SafetyEvent(now, "rate_pause_hold_confirmed", "info",
                                     f"confirmed hold setpoint {setpoint:.4f} MPa; re-enabling Control "
                                     "while the compression-rate pause continues"))

    def _advance(self, now: float) -> None:
        if not self._estimator.is_valid(now):
            return
        filtered = self._estimator.filtered_pressure()
        slope = self._estimator.pressure_slope()
        if filtered is None or slope is None:
            return
        sizing_pressure = self._conservative_sample_pressure(filtered)
        predicted = max(
            filtered + max(slope, 0.0) * self._cfg.approach.prediction_horizon_s,
            sizing_pressure,
        )

        if self._pending_step is not None and not self._safety.membrane_stop_intended:
            self._update_pending_step(filtered, slope, now)

        target = self._effective_target_gpa()
        hyst = self._cfg.hysteresis
        if self._target_reduction_hold:
            self._safety.set_membrane_stop_intended(True)
            self._abandon_motion(now, "target reduction entered HOLD")
            self._set_state(ControlState.HOLD, now, "target was lowered; one-sided control remains stopped")
            return
        if sizing_pressure > target + hyst.overshoot_margin_gpa:
            self._safety.set_membrane_stop_intended(True)
            self._abandon_motion(now, "sample pressure exceeded target")
            self._set_state(ControlState.HOLD, now, "sample pressure above target + overshoot margin")
            return

        if predicted >= target - hyst.reach_margin_gpa:
            self._safety.set_membrane_stop_intended(True)
            self._abandon_motion(now, "target reach prediction entered HOLD")
            self._set_state(ControlState.HOLD, now, "predicted pressure within reach margin of target")
            return

        if self.state == ControlState.HOLD:
            if filtered < target - hyst.reapproach_margin_gpa:
                self._set_state(ControlState.APPROACH, now, "pressure fell below re-approach margin")
            else:
                return

        if self._safety.membrane_stop_intended:
            if self._staged_rearm_step is not None:
                manually_armed_after_staging = (
                    not self._cfg.pace5000_api.ensure_control_mode_enabled
                    and self._membrane_status is not None
                    and self._membrane_status.control_mode is True
                )
                if (
                    self._membrane_status is not None
                    and (
                        self._membrane_status.control_mode is False
                        or manually_armed_after_staging
                    )
                ):
                    self._confirm_staged_rearm(now)
                return
            if self._membrane_status is None or self._membrane_status.control_mode is not False:
                # First establish Measure. _reconcile_membrane_drive retries
                # the STOP write after this state-machine pass.
                return
            self._maybe_issue_step(
                filtered, sizing_pressure, slope, predicted, target, now,
                stage_for_rearm=True,
            )
            return

        if self._pending_step is not None:
            self._set_state(ControlState.SETTLE, now, "awaiting response to previous membrane step")
            return

        if self._membrane_status is None or self._membrane_status.control_mode is not True:
            # Not yet confirmed armed (control mode re-enabled and read back
            # as such) -- _reconcile_membrane_drive is retrying the arm write
            # every tick; wait for it rather than racing a set_pressure call
            # against a possibly-still-disabled output.
            return

        self._maybe_issue_step(filtered, sizing_pressure, slope, predicted, target, now)

    def _update_pending_step(self, filtered: float, slope: float, now: float) -> None:
        step = self._pending_step
        assert step is not None
        step.max_slope_gpa_s = max(step.max_slope_gpa_s, slope)

        region = self._cfg.region_for(filtered)
        threshold = region.settled_slope_threshold_gpa_s
        min_settle = region.minimum_settle_time_s
        near_target = (self.user_target_gpa - filtered) < self._cfg.approach.near_target_distance_gpa
        if near_target:
            threshold *= self._cfg.approach.near_target_slope_threshold_scale
            min_settle += self._cfg.approach.near_target_extra_settle_time_s

        if abs(slope) < threshold:
            if self._settled_since is None:
                self._settled_since = now
        else:
            self._settled_since = None

        settle_confirmed = self._settled_since is not None and (now - self._settled_since) >= min_settle
        response_start_t = step.t_drive_started or step.t_command

        # A step commanded at a slower membrane_rate_mpa_per_min than
        # region.minimum_settle_time_s was written for (e.g. a slew
        # deliberately slowed to respect max_compression_rate_gpa_per_min --
        # see set_membrane_rate_mpa_per_min) can still be physically mid-ramp
        # once minimum_settle_time_s alone has elapsed. Extend the blackout
        # for *this step* to also cover its own actual ramp time, using the
        # rate that was in effect when it was commanded (step.decision),
        # not whatever the operator may have changed it to since.
        commanded_rate = step.decision.get("membrane_rate_mpa_per_min") or self._membrane_rate_mpa_per_min
        membrane_step_mpa = step.membrane_pressure_after - step.membrane_pressure_before
        ramp_time_s = 0.0
        if commanded_rate and commanded_rate > 0 and membrane_step_mpa > 0:
            ramp_time_s = membrane_step_mpa / (commanded_rate / 60.0) + self._cfg.approach.ramp_time_margin_s
        blackout_elapsed = (now - response_start_t) >= max(min_settle, ramp_time_s)

        # A flat slope only means "settled" if the membrane has actually
        # finished ramping to the commanded setpoint — otherwise it can just
        # mean the response hasn't arrived yet (dead time, or the ramp itself
        # still in progress). Without this, the next step's baseline
        # (_current_membrane_setpoint) stacks on top of a setpoint the
        # membrane hasn't physically reached, compounding into overshoot.
        membrane_arrived = (
            self._membrane_status is not None
            and self._membrane_status.pressure_mpa is not None
            and self._membrane_status.pressure_mpa
            >= step.membrane_pressure_after - self._cfg.approach.membrane_arrival_tolerance_mpa
        )

        if settle_confirmed and blackout_elapsed and membrane_arrived:
            step.settled = True
            step.t_settled = now
            step.sample_pressure_after = filtered
            step.measurement_std_gpa = self._estimator.measurement_std()
            self._gain_estimator.record_step(step)
            if self._logger is not None:
                self._safe_call(lambda: self._logger.log_step_record(step))
            self._pending_step = None
            self._settled_since = None

    def _maybe_issue_step(
        self,
        filtered: float,
        sizing_pressure: float,
        slope: float,
        predicted: float,
        target_gpa: float,
        now: float,
        *,
        stage_for_rearm: bool = False,
    ) -> None:
        # A pressure write must use supply pressure and actual/setpoint values
        # read in this same control tick, never a cached status from an older
        # poll interval.
        if not self._membrane_status_fresh_this_tick:
            return

        # A median/mean filter necessarily lags a genuine upward change.  For
        # one-sided pressure control an upward raw reading can safely make a
        # step smaller (or suppress it), but must never be ignored in favour of
        # a lower filtered value: doing so can select a lower-gain region and
        # size a command that crosses into the accelerating part of the plant.
        region = self._cfg.region_for(sizing_pressure)
        gain_est = self._gain_estimator.estimate(sizing_pressure, region)

        approach = self._cfg.approach
        control_target = target_gpa - approach.approach_margin_gpa
        predicted_error = control_target - predicted
        if predicted_error <= 0:
            return

        max_sample_step = region.max_sample_step_gpa
        if (target_gpa - predicted) < approach.near_target_distance_gpa:
            max_sample_step = min(max_sample_step, approach.near_target_max_sample_step_gpa)
        # Never let a step's implied sample-pressure rise cross into a
        # higher-gain region than the one `gain_est` was just computed for --
        # otherwise part of the step would be sized using a lower gain than
        # actually applies once the sample pressure crosses the boundary.
        max_sample_step = min(max_sample_step, max(0.0, region.sample_pressure_max_gpa - sizing_pressure))
        if self._max_compression_rate_gpa_per_min is not None:
            # A step is expected to take roughly region.minimum_settle_time_s
            # to show its full response (see the settle-detection blackout in
            # _update_pending_step), so bounding requested_sample_step by
            # rate * that duration bounds this step's average sample-pressure
            # rate to the operator-configured ceiling.
            rate_cap_step = (self._max_compression_rate_gpa_per_min / 60.0) * region.minimum_settle_time_s
            max_sample_step = min(max_sample_step, rate_cap_step)

        requested_sample_step = max(0.0, min(approach.approach_factor * predicted_error, max_sample_step))
        if requested_sample_step <= 0.0:
            return

        safe_gain = gain_est.safe_gain
        if safe_gain <= 0:
            return

        membrane_step = max(0.0, min(requested_sample_step / safe_gain, region.max_membrane_step))

        # Tighten (never loosen) the gas-side slew for *this* command so the
        # resulting sample-pressure rate can't exceed max_compression_rate_
        # gpa_per_min even at this region's (possibly much higher, near the
        # top of the plant's nonlinear gain) gain -- sizing the step itself
        # for the compression cap (max_sample_step above) only bounds the
        # *average* rate over minimum_settle_time_s, not the real,
        # front-loaded response actually observed against hardware. The
        # extended settle blackout above already tolerates a slower rate.
        # Uses gain_est.rate_limit_gain, not safe_gain: see
        # GainRegion.rate_limit_gain's docstring -- a step-sizing prior that
        # turns out to be optimistic relative to real hardware must not also
        # under-restrict the one independent, physically-grounded rate cap.
        effective_rate_mpa_per_min = self._membrane_rate_mpa_per_min
        if self._max_compression_rate_gpa_per_min is not None:
            rate_cap_mpa_per_min = self._max_compression_rate_gpa_per_min / gain_est.rate_limit_gain
            effective_rate_mpa_per_min = min(effective_rate_mpa_per_min, rate_cap_mpa_per_min)

        current_setpoint = (
            self._membrane_status.pressure_mpa
            if stage_for_rearm and self._membrane_status is not None
            else self._current_membrane_setpoint()
        )
        if current_setpoint is None:
            return  # no known baseline yet — wait for a status read

        new_setpoint = min(current_setpoint + membrane_step, self._cfg.safety.max_membrane_pressure_mpa)
        membrane_step = new_setpoint - current_setpoint
        if membrane_step < approach.min_membrane_step_mpa:
            return

        source_pressure = (
            self._membrane_status.source_pressure_positive_mpa
            if self._membrane_status is not None
            else None
        )
        allowed, reason = self._safety.check_command(
            membrane_step,
            new_setpoint,
            now,
            source_pressure_mpa=source_pressure,
            allow_while_stopped=stage_for_rearm,
        )
        if not allowed:
            self._log_event(SafetyEvent(now, "command_blocked", "warning", reason or "blocked"))
            if reason in {"source_pressure_unavailable", "setpoint_not_below_source_pressure"}:
                pause_reason = (
                    "cannot safely change membrane pressure: positive supply pressure "
                    "is unavailable or does not exceed the proposed setpoint"
                )
                self._safety.request_manual_pause(pause_reason)
                self._safety.set_membrane_stop_intended(True)
                self._set_state(ControlState.PAUSE, now, pause_reason)
            return

        decision = {
            "filtered_pressure_gpa": filtered,
            "sizing_pressure_gpa": sizing_pressure,
            "pressure_slope_gpa_s": slope,
            "predicted_pressure_gpa": predicted,
            "control_target_gpa": control_target,
            "predicted_error_gpa": predicted_error,
            "gain_source": gain_est.source,
            "estimated_gain": gain_est.estimated_gain,
            "gain_uncertainty": gain_est.gain_uncertainty,
            "safe_gain": safe_gain,
            "rate_limit_gain": gain_est.rate_limit_gain,
            "requested_sample_step_gpa": requested_sample_step,
            "membrane_step_mpa": membrane_step,
            "membrane_rate_mpa_per_min": effective_rate_mpa_per_min,
            "region_min_gpa": region.sample_pressure_min_gpa,
            "region_max_gpa": region.sample_pressure_max_gpa,
            "source_pressure_positive_mpa": source_pressure,
            "staged_in_measure": stage_for_rearm,
        }
        reason_str = (
            "safe setpoint staged in Measure before Control re-arm"
            if stage_for_rearm
            else "one-sided approach step toward user target"
        )

        if self._cfg.control.dry_run:
            # Rehearsal mode: never write, never open a pending step (nothing
            # physically moves, so there is nothing to settle and recording
            # a "step" here would poison the online gain estimator with a
            # bogus near-zero observed_gain). Log what WOULD have been sent
            # and re-evaluate fresh next tick.
            self._log_event(SafetyEvent(now, "dry_run_command_suppressed", "info",
                                         f"would set membrane setpoint to {new_setpoint:.4f} MPa "
                                         f"(+{membrane_step:.4f}); write suppressed by dry_run"))
            self._last_command_reason = f"[dry-run] {reason_str}"
            self._last_command_decision = decision
            if stage_for_rearm:
                self._safety.set_membrane_stop_intended(False)
            return

        step = StepRecord(
            step_id=self._next_step_id,
            t_command=now,
            membrane_pressure_before=current_setpoint,
            membrane_pressure_after=new_setpoint,
            sample_pressure_before=filtered,
            reason=reason_str,
            decision=decision,
        )

        try:
            self._membrane.set_pressure(new_setpoint, effective_rate_mpa_per_min)
        except MembraneCommError as e:
            # The write may have applied on the device even though we never
            # got the HTTP response back. Commit the same bookkeeping a
            # success would get -- treating this as "not issued" instead
            # would let the cumulative-step budget and settle-wait be
            # bypassed by stacking another command on top of an unknown real
            # state. The setpoint_mismatch safety check is the authority
            # that reconciles this once the next status read-back arrives.
            self._log_event(SafetyEvent(now, "membrane_write_ambiguous", "warning",
                                         f"set_pressure may or may not have applied before this error: {e}"))
            step.ack_uncertain = True
            self._commit_step(step, membrane_step, now, staged_for_rearm=stage_for_rearm)
            return

        # read_status() runs before command issuance in this tick, so its
        # target field is now stale even though the actual-pressure field is
        # still the best reading we have.  Preserve the actual value but expose
        # the successfully acknowledged target immediately in snapshots and
        # subsequent baseline calculations.
        if self._membrane_status is not None and not stage_for_rearm:
            self._membrane_status = replace(
                self._membrane_status,
                target_pressure_mpa=new_setpoint,
            )
        self._commit_step(step, membrane_step, now, staged_for_rearm=stage_for_rearm)

    def _commit_step(
        self,
        step: StepRecord,
        membrane_step_mpa: float,
        now: float,
        *,
        staged_for_rearm: bool = False,
    ) -> None:
        self._next_step_id += 1
        self._safety.on_command_issued(membrane_step_mpa, step.membrane_pressure_after, now)
        if staged_for_rearm:
            self._staged_rearm_step = step
        else:
            self._pending_step = step
        self._settled_since = None
        self._last_command_reason = f"[ambiguous ack] {step.reason}" if step.ack_uncertain else step.reason
        self._last_command_decision = step.decision
        if self._logger is not None:
            self._safe_call(lambda: self._logger.log_command(step))
        if staged_for_rearm:
            note = (
                "staged safe setpoint in Measure (ambiguous ack); awaiting fresh readback"
                if step.ack_uncertain
                else "staged safe setpoint in Measure; awaiting fresh readback"
            )
            self._set_state(ControlState.APPROACH, now, note)
        else:
            note = ("issued membrane step (ambiguous ack); awaiting settle" if step.ack_uncertain
                    else "issued membrane step; awaiting settle")
            self._set_state(ControlState.SETTLE, now, note)

    def _confirm_staged_rearm(self, now: float) -> None:
        """Confirm the Measure-mode target write before allowing Control.

        The cached status from the command tick is not evidence: a response
        timeout can mean that the write did not apply.  Only a later status
        sample with a matching target can open this gate.
        """
        step = self._staged_rearm_step
        status = self._membrane_status
        assert step is not None
        if (
            not self._membrane_status_fresh_this_tick
            or status is None
            or status.t_mono <= step.t_command
            or status.target_pressure_mpa is None
            or status.pressure_mpa is None
        ):
            return

        tolerance = self._cfg.safety.setpoint_mismatch_tol_mpa
        if abs(status.target_pressure_mpa - step.membrane_pressure_after) > tolerance:
            # SafetySupervisor emits the immediate PAUSE event for the same
            # mismatch.  Remain in Measure and do not overwrite evidence with
            # another write.
            return

        source = status.source_pressure_positive_mpa
        reported_target = status.target_pressure_mpa
        required_source = (
            reported_target
            + self._cfg.safety.minimum_source_pressure_headroom_mpa
        )
        if source is None or not math.isfinite(source) or source <= required_source:
            return

        # Measure normally freezes actual pressure, but use the fresh actual
        # readback as the authority. If it moved, never arm a target that
        # would lower it or that is now a larger jump than the step we sized.
        actual_gap = reported_target - status.pressure_mpa
        staged_gap = float(step.decision.get("membrane_step_mpa", 0.0))
        arrival_tol = self._cfg.approach.membrane_arrival_tolerance_mpa
        if actual_gap < -arrival_tol or actual_gap > staged_gap + arrival_tol:
            self._log_event(SafetyEvent(
                now,
                "staged_setpoint_recalculation_required",
                "warning",
                "membrane actual pressure changed while Measure; discarded the staged "
                "re-arm command and will recalculate from fresh actual pressure",
            ))
            self._staged_rearm_step = None
            return

        # With automatic arming disabled, retain the safely staged target and
        # accept an operator-entered Control state only after it is observed.
        if (
            not self._cfg.pace5000_api.ensure_control_mode_enabled
            and status.control_mode is not True
        ):
            step.decision["rearm_readback_confirmed"] = True
            return

        step.t_drive_started = now
        self._staged_rearm_step = None
        self._pending_step = step
        self._settled_since = None
        self._safety.set_membrane_stop_intended(False)
        self._set_state(ControlState.SETTLE, now, "safe staged setpoint confirmed; enabling Control")

    def _abandon_motion(self, now: float, reason: str) -> None:
        """Forget response tracking whenever Measure interrupts a ramp.

        The physical command budget remains charged in SafetySupervisor; only
        the now-invalid settle/gain observation is discarded.
        """
        if self._pending_step is None and self._staged_rearm_step is None:
            self._settled_since = None
            return
        abandoned_ids = [
            step.step_id
            for step in (self._pending_step, self._staged_rearm_step)
            if step is not None
        ]
        self._pending_step = None
        self._staged_rearm_step = None
        self._settled_since = None
        self._log_event(SafetyEvent(
            now,
            "step_tracking_abandoned",
            "info",
            f"discarded interrupted step tracking {abandoned_ids}: {reason}",
        ))

    def _conservative_sample_pressure(self, filtered: float) -> float:
        """Upper pressure bound used for one-sided command decisions.

        The latest raw value is receipt-time current while the robust filter
        intentionally lags.  Taking the higher of the two means an upward
        reading can only reduce pressurization; a downward outlier can never
        justify a larger command.
        """
        last = self._estimator.last_sample
        if (
            last is not None
            and last.pressure_gpa is not None
            and isinstance(last.pressure_gpa, (int, float))
            and math.isfinite(last.pressure_gpa)
        ):
            return max(filtered, float(last.pressure_gpa))
        return filtered

    def _current_membrane_setpoint(self) -> float | None:
        if self._membrane_status is None:
            return None
        if self._membrane_status.target_pressure_mpa is not None:
            return self._membrane_status.target_pressure_mpa
        return self._membrane_status.pressure_mpa

    def _set_state(self, new_state: ControlState, now: float, reason: str) -> None:
        if new_state != self.state:
            if self._logger is not None:
                transition = StateTransition(now, self.state, new_state, reason)
                self._safe_call(lambda: self._logger.log_transition(transition))
            self.state = new_state

    def _log_event(self, event: SafetyEvent) -> None:
        if self._logger is not None:
            self._safe_call(lambda: self._logger.log_event(event))

    def _safe_call(self, fn) -> None:
        # A logging-sink failure (disk full, I/O error) must never propagate
        # into the control loop -- it must never be able to prevent (or
        # crash out of) a safety action.
        try:
            fn()
        except Exception as e:
            # Logging is deliberately not a pressure-control interlock, but
            # silently swallowing audit loss is unacceptable. Keep a sticky
            # operator-visible error in every subsequent snapshot.
            self._logging_error = f"{type(e).__name__}: {e}"

    def _snapshot(self, now: float, verdict: SafetyVerdict) -> ControllerSnapshot:
        filtered = self._estimator.filtered_pressure()
        sizing_pressure = self._conservative_sample_pressure(filtered) if filtered is not None else None
        region = self._cfg.region_for(sizing_pressure) if sizing_pressure is not None else self._cfg.gain_regions[0]
        gain_est = self._gain_estimator.estimate(sizing_pressure, region) if sizing_pressure is not None else None
        predicted = self._estimator.predicted_pressure(self._cfg.approach.prediction_horizon_s)
        if predicted is not None and sizing_pressure is not None:
            predicted = max(predicted, sizing_pressure)
        last_sample = self._estimator.last_sample
        return ControllerSnapshot(
            t_mono=now,
            state=self.state,
            user_target_gpa=self.user_target_gpa,
            control_target_gpa=self._effective_target_gpa() - self._cfg.approach.approach_margin_gpa,
            raw_pressure_gpa=last_sample.pressure_gpa if last_sample else None,
            filtered_pressure_gpa=filtered,
            pressure_slope_gpa_s=self._estimator.pressure_slope(),
            predicted_pressure_gpa=predicted,
            measurement_std_gpa=self._estimator.measurement_std(),
            measurement_r2=last_sample.r2 if last_sample else None,
            estimator_valid=self._estimator.is_valid(now),
            membrane_setpoint_mpa=self._membrane_status.target_pressure_mpa if self._membrane_status else None,
            membrane_actual_mpa=self._membrane_status.pressure_mpa if self._membrane_status else None,
            safe_gain=gain_est.safe_gain if gain_est else None,
            last_command_reason=self._last_command_reason,
            manual_pause=self._safety.is_manually_paused,
            safety_level=verdict.level,
            safety_reasons=tuple(e.code for e in verdict.events),
            max_compression_rate_gpa_per_min=self._max_compression_rate_gpa_per_min,
            membrane_rate_mpa_per_min=self._membrane_rate_mpa_per_min,
            source_pressure_positive_mpa=(
                self._membrane_status.source_pressure_positive_mpa
                if self._membrane_status
                else None
            ),
            logging_error=self._logging_error,
        )
