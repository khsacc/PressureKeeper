"""DataLogger: time-series CSV logging.

Five files per run, all timestamp-keyed so they can be joined in analysis:
  * ticks.csv     — one row per control-loop iteration (§ログ requirements)
  * commands.csv  — one row per PACE5000 write, with the reason and every
                     control-decision value that produced it
  * steps.csv     — one row per settled step (before/after, response time,
                     observed gain) — the online sensitivity data set
  * interrupted_steps.csv — start/final records for steps interrupted before
                     settling, including rate-learning eligibility
  * events.csv    — safety events and state transitions

Plus manifest.json, a small run-level record (git SHA, mode, whether Control
was ever started, and how the run ended) -- unlike the five CSVs above, this
is written even when a run never produces a single tick (e.g. the operator
opened the GUI and closed it without pressing Start, or the very first
step() call raised before anything else could log), so a completely empty
CSV set is no longer a total dead end when reviewing a run after the fact.

CSV was chosen over SQLite to keep the dependency footprint at zero; any
other sink (e.g. SQLite) can be swapped in by implementing the same method
surface, since callers only depend on this class's public API.

close() also makes a best-effort attempt to write summary_plots.png (see
report_plots.py) from the just-closed ticks.csv -- unlike the files above,
this is a convenience artifact, not part of the audit trail, so a failure
there (e.g. matplotlib not installed) is swallowed and never affects
end_reason or any other manifest field.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TextIO

from .config import LoggingConfig
from .models import (
    ControllerSnapshot,
    InterruptedStepObservation,
    SafetyEvent,
    StateTransition,
    StepRecord,
)


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _echo_to_terminal(tag: str, code: str, message: str) -> None:
    # CSV rows are the audit trail, but they're silent until someone opens the
    # file after the fact -- an operator watching the terminal next to the GUI
    # (which only ever surfaces the bare event *code*, see main_window.py)
    # needs the actual message live to tell a real device fault apart from,
    # e.g., a client-side timeout that fired while the remote acquisition
    # actually succeeded.
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {tag:<5s} {code}: {message}", file=sys.stderr, flush=True)


_TICK_FIELDS = [
    "t_mono", "t_wall", "state",
    "raw_pressure_gpa", "filtered_pressure_gpa", "pressure_slope_gpa_s",
    "predicted_pressure_gpa", "user_target_gpa", "control_target_gpa",
    "membrane_setpoint_mpa", "membrane_actual_mpa",
    "safe_gain", "measurement_std_gpa", "measurement_r2", "estimator_valid",
    "manual_pause", "safety_level", "safety_reasons", "last_command_reason",
    "max_compression_rate_gpa_per_min", "membrane_rate_mpa_per_min", "source_pressure_positive_mpa",
    "logging_error",
]

_COMMAND_FIELDS = [
    "t_mono", "t_wall", "step_id", "reason",
    "membrane_pressure_before", "membrane_pressure_after", "membrane_step_mpa", "membrane_rate_mpa_per_min",
    "sample_pressure_before", "filtered_pressure_gpa", "sizing_pressure_gpa", "pressure_slope_gpa_s",
    "predicted_pressure_gpa", "control_target_gpa", "predicted_error_gpa",
    "gain_source", "estimated_gain", "gain_uncertainty", "safe_gain", "rate_limit_gain",
    "local_gain_trend_per_gpa", "step_sizing_mode", "adaptive_probe",
    "adaptive_probe_max_expected_gain", "probe_target_cap_mpa",
    "rate_gain_source", "interrupted_rate_observation_count", "learned_rate_floor",
    "requested_sample_step_gpa", "region_min_gpa", "region_max_gpa",
    "source_pressure_positive_mpa", "staged_in_measure",
]

_STEP_FIELDS = [
    "t_mono", "t_wall", "step_id", "t_command", "t_drive_started",
    "membrane_pressure_before", "membrane_pressure_after",
    "membrane_actual_before", "membrane_actual_after",
    "sample_pressure_before", "sample_pressure_after",
    "response_time_s", "max_slope_gpa_s", "measurement_std_gpa",
    "response_detection_threshold_gpa", "response_detected", "observed_gain",
]

_INTERRUPTED_STEP_FIELDS = [
    "t_mono", "t_wall", "phase", "step_id", "cause_code",
    "eligible_for_rate_learning", "exclusion_reason",
    "t_command", "t_drive_started", "t_interrupted", "t_observation_end",
    "observation_end_reason", "sizing_pressure_gpa",
    "sample_pressure_before", "sample_pressure_at_interrupt",
    "max_sample_pressure_gpa", "max_slope_gpa_s",
    "commanded_membrane_rate_mpa_per_min", "raw_rate_gain",
    "applied_safety_factor", "learned_rate_floor",
    "membrane_pressure_before", "membrane_pressure_after",
    "membrane_actual_at_interrupt", "max_membrane_actual_mpa",
    "ack_uncertain",
]

_EVENT_FIELDS = ["t_mono", "t_wall", "kind", "code", "severity", "message", "from_state", "to_state"]


class DataLogger:
    def __init__(self, config: LoggingConfig, *, mode: str | None = None) -> None:
        run_name = config.run_name or datetime.now().strftime("run_%Y%m%dT%H%M%S_%f")
        self.directory = Path(config.directory) / run_name
        # Logs are part of the safety/audit trail.  Reusing a run name used to
        # truncate all four CSV files silently; fail before opening anything
        # instead of destroying an earlier run.
        self.directory.mkdir(parents=True, exist_ok=False)

        self._manifest: dict[str, object] = {
            "run_name": run_name,
            "pid": os.getpid(),
            "git_sha": _git_sha(),
            "mode": mode,
            "log_schema_version": 3,
            "started_at_wall": time.time(),
            "started_control": False,
            "ended_at_wall": None,
            "end_reason": None,
            "crash_traceback": None,
        }
        self._write_manifest()

        opened: list[TextIO] = []
        try:
            self._tick_f = self._open("ticks.csv", _TICK_FIELDS)
            opened.append(self._tick_f)
            self._command_f = self._open("commands.csv", _COMMAND_FIELDS)
            opened.append(self._command_f)
            self._step_f = self._open("steps.csv", _STEP_FIELDS)
            opened.append(self._step_f)
            self._interrupted_step_f = self._open(
                "interrupted_steps.csv", _INTERRUPTED_STEP_FIELDS
            )
            opened.append(self._interrupted_step_f)
            self._event_f = self._open("events.csv", _EVENT_FIELDS)
            opened.append(self._event_f)

            self._tick_w = csv.DictWriter(self._tick_f, fieldnames=_TICK_FIELDS)
            self._command_w = csv.DictWriter(self._command_f, fieldnames=_COMMAND_FIELDS)
            self._step_w = csv.DictWriter(self._step_f, fieldnames=_STEP_FIELDS)
            self._interrupted_step_w = csv.DictWriter(
                self._interrupted_step_f,
                fieldnames=_INTERRUPTED_STEP_FIELDS,
            )
            self._event_w = csv.DictWriter(self._event_f, fieldnames=_EVENT_FIELDS)
            for writer in (
                self._tick_w,
                self._command_w,
                self._step_w,
                self._interrupted_step_w,
                self._event_w,
            ):
                writer.writeheader()
        except Exception:
            for file in opened:
                file.close()
            raise

        self._flush_every = max(1, config.flush_every_n)
        self._n_since_flush = 0

    def _open(self, name: str, _fields: list[str]) -> TextIO:
        return (self.directory / name).open("w", newline="", encoding="utf-8")

    def _write_manifest(self) -> None:
        # Write-then-rename so a reader never sees a half-written file, and a
        # crash mid-write can't corrupt the previous, still-valid manifest.
        tmp = self.directory / "manifest.json.tmp"
        tmp.write_text(json.dumps(self._manifest, indent=2), encoding="utf-8")
        tmp.replace(self.directory / "manifest.json")

    def mark_control_started(self) -> None:
        """Call once Control is actually started (CLI: as soon as the polling
        thread starts; GUI: on the operator's "Start Control" click) -- this
        is what lets a later read of the manifest tell "opened but never
        started" apart from every other outcome."""
        self._manifest["started_control"] = True
        self._write_manifest()

    def mark_end(self, reason: str, *, crash_traceback: str | None = None) -> None:
        """Record why the run ended. Safe to call more than once (e.g. a
        crash handler calling this before the normal close() path also
        runs); the first call wins so a crash reason is never silently
        overwritten by close()'s own default."""
        if self._manifest["end_reason"] is not None:
            return
        self._manifest["end_reason"] = reason
        self._manifest["ended_at_wall"] = time.time()
        if crash_traceback is not None:
            self._manifest["crash_traceback"] = crash_traceback
        self._write_manifest()

    def log_tick(self, snap: ControllerSnapshot) -> None:
        self._tick_w.writerow({
            "t_mono": snap.t_mono, "t_wall": time.time(), "state": snap.state.value,
            "raw_pressure_gpa": snap.raw_pressure_gpa,
            "filtered_pressure_gpa": snap.filtered_pressure_gpa,
            "pressure_slope_gpa_s": snap.pressure_slope_gpa_s,
            "predicted_pressure_gpa": snap.predicted_pressure_gpa,
            "user_target_gpa": snap.user_target_gpa,
            "control_target_gpa": snap.control_target_gpa,
            "membrane_setpoint_mpa": snap.membrane_setpoint_mpa,
            "membrane_actual_mpa": snap.membrane_actual_mpa,
            "safe_gain": snap.safe_gain,
            "measurement_std_gpa": snap.measurement_std_gpa,
            "measurement_r2": snap.measurement_r2,
            "estimator_valid": snap.estimator_valid,
            "manual_pause": snap.manual_pause,
            "safety_level": snap.safety_level,
            "safety_reasons": ";".join(snap.safety_reasons),
            "last_command_reason": snap.last_command_reason,
            "max_compression_rate_gpa_per_min": snap.max_compression_rate_gpa_per_min,
            "membrane_rate_mpa_per_min": snap.membrane_rate_mpa_per_min,
            "source_pressure_positive_mpa": snap.source_pressure_positive_mpa,
            "logging_error": snap.logging_error,
        })
        self._maybe_flush()

    def log_command(self, step: StepRecord) -> None:
        d = step.decision
        self._command_w.writerow({
            "t_mono": step.t_command, "t_wall": time.time(), "step_id": step.step_id,
            "reason": step.reason,
            "membrane_pressure_before": step.membrane_pressure_before,
            "membrane_pressure_after": step.membrane_pressure_after,
            "membrane_step_mpa": d.get("membrane_step_mpa"),
            "membrane_rate_mpa_per_min": d.get("membrane_rate_mpa_per_min"),
            "sample_pressure_before": step.sample_pressure_before,
            "filtered_pressure_gpa": d.get("filtered_pressure_gpa"),
            "sizing_pressure_gpa": d.get("sizing_pressure_gpa"),
            "pressure_slope_gpa_s": d.get("pressure_slope_gpa_s"),
            "predicted_pressure_gpa": d.get("predicted_pressure_gpa"),
            "control_target_gpa": d.get("control_target_gpa"),
            "predicted_error_gpa": d.get("predicted_error_gpa"),
            "gain_source": d.get("gain_source"),
            "estimated_gain": d.get("estimated_gain"),
            "gain_uncertainty": d.get("gain_uncertainty"),
            "safe_gain": d.get("safe_gain"),
            "rate_limit_gain": d.get("rate_limit_gain"),
            "local_gain_trend_per_gpa": d.get("local_gain_trend_per_gpa"),
            "step_sizing_mode": d.get("step_sizing_mode"),
            "adaptive_probe": d.get("adaptive_probe"),
            "adaptive_probe_max_expected_gain": d.get(
                "adaptive_probe_max_expected_gain"
            ),
            "probe_target_cap_mpa": d.get("probe_target_cap_mpa"),
            "rate_gain_source": d.get("rate_gain_source"),
            "interrupted_rate_observation_count": d.get(
                "interrupted_rate_observation_count"
            ),
            "learned_rate_floor": d.get("learned_rate_floor"),
            "requested_sample_step_gpa": d.get("requested_sample_step_gpa"),
            "region_min_gpa": d.get("region_min_gpa"),
            "region_max_gpa": d.get("region_max_gpa"),
            "source_pressure_positive_mpa": d.get("source_pressure_positive_mpa"),
            "staged_in_measure": d.get("staged_in_measure"),
        })
        self._maybe_flush()

    def log_step_record(self, step: StepRecord) -> None:
        self._step_w.writerow({
            "t_mono": step.t_settled, "t_wall": time.time(), "step_id": step.step_id,
            "t_command": step.t_command,
            "t_drive_started": step.t_drive_started,
            "membrane_pressure_before": step.membrane_pressure_before,
            "membrane_pressure_after": step.membrane_pressure_after,
            "membrane_actual_before": step.membrane_actual_before,
            "membrane_actual_after": step.membrane_actual_after,
            "sample_pressure_before": step.sample_pressure_before,
            "sample_pressure_after": step.sample_pressure_after,
            "response_time_s": step.response_time_s,
            "max_slope_gpa_s": step.max_slope_gpa_s,
            "measurement_std_gpa": step.measurement_std_gpa,
            "response_detection_threshold_gpa": (
                step.response_detection_threshold_gpa
            ),
            "response_detected": step.response_detected,
            "observed_gain": step.observed_gain,
        })
        self._maybe_flush()

    def log_interrupted_step(
        self,
        observation: InterruptedStepObservation,
        *,
        phase: str,
        safety_factor: float,
    ) -> None:
        raw_rate_gain = observation.raw_rate_gain
        learned_rate_floor = (
            raw_rate_gain * safety_factor
            if raw_rate_gain is not None
            else None
        )
        self._interrupted_step_w.writerow({
            "t_mono": (
                observation.t_interrupted
                if phase == "started"
                else observation.t_observation_end
            ),
            "t_wall": time.time(),
            "phase": phase,
            "step_id": observation.step_id,
            "cause_code": observation.cause_code,
            "eligible_for_rate_learning": observation.eligible_for_rate_learning,
            "exclusion_reason": observation.exclusion_reason,
            "t_command": observation.t_command,
            "t_drive_started": observation.t_drive_started,
            "t_interrupted": observation.t_interrupted,
            "t_observation_end": observation.t_observation_end,
            "observation_end_reason": observation.observation_end_reason,
            "sizing_pressure_gpa": observation.sizing_pressure_gpa,
            "sample_pressure_before": observation.sample_pressure_before,
            "sample_pressure_at_interrupt": observation.sample_pressure_at_interrupt,
            "max_sample_pressure_gpa": observation.max_sample_pressure_gpa,
            "max_slope_gpa_s": observation.max_positive_slope_gpa_s,
            "commanded_membrane_rate_mpa_per_min": (
                observation.commanded_membrane_rate_mpa_per_min
            ),
            "raw_rate_gain": raw_rate_gain,
            "applied_safety_factor": safety_factor,
            "learned_rate_floor": learned_rate_floor,
            "membrane_pressure_before": observation.membrane_pressure_before,
            "membrane_pressure_after": observation.membrane_pressure_after,
            "membrane_actual_at_interrupt": observation.membrane_actual_at_interrupt,
            "max_membrane_actual_mpa": observation.max_membrane_actual_mpa,
            "ack_uncertain": observation.ack_uncertain,
        })
        self._maybe_flush()

    def log_event(self, event: SafetyEvent) -> None:
        self._event_w.writerow({
            "t_mono": event.t_mono, "t_wall": time.time(), "kind": "safety_event",
            "code": event.code, "severity": event.severity, "message": event.message,
            "from_state": "", "to_state": "",
        })
        self._maybe_flush()
        if event.severity != "info":
            _echo_to_terminal(event.severity.upper(), event.code, event.message)

    def log_transition(self, transition: StateTransition) -> None:
        self._event_w.writerow({
            "t_mono": transition.t_mono, "t_wall": time.time(), "kind": "state_transition",
            "code": "", "severity": "info", "message": transition.reason,
            "from_state": transition.from_state.value, "to_state": transition.to_state.value,
        })
        self._maybe_flush()
        _echo_to_terminal("STATE", f"{transition.from_state.value}->{transition.to_state.value}", transition.reason)

    def _maybe_flush(self) -> None:
        self._n_since_flush += 1
        if self._n_since_flush >= self._flush_every:
            self._n_since_flush = 0
            for f in (
                self._tick_f,
                self._command_f,
                self._step_f,
                self._interrupted_step_f,
                self._event_f,
            ):
                f.flush()

    def close(self) -> None:
        # A caller that never had a specific reason to report (the ordinary
        # process-exit path) still leaves a meaningful record: distinguishing
        # a run that reached Control from one that was opened and closed
        # without ever starting is exactly what turns an all-empty CSV set
        # from a dead end into "this one was never started."
        if self._manifest["end_reason"] is None:
            self.mark_end("completed" if self._manifest["started_control"] else "not_started")
        for f in (
            self._tick_f,
            self._command_f,
            self._step_f,
            self._interrupted_step_f,
            self._event_f,
        ):
            f.flush()
            f.close()
        self._write_summary_plot()

    def _write_summary_plot(self) -> None:
        # Best-effort, and deliberately after the CSVs/manifest above are
        # already flushed and closed: the audit trail this class exists for
        # (see module docstring) must never depend on this succeeding.
        # matplotlib is an optional dependency (pyproject.toml's `plotting`
        # extra) not required to run this class at all, hence the lazy
        # import here rather than at module load.
        try:
            from .report_plots import write_summary_plots
            write_summary_plots(self.directory)
        except Exception as e:
            _echo_to_terminal("WARN", "plot_export_failed", f"{type(e).__name__}: {e}")

    def __enter__(self) -> "DataLogger":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
