"""DataLogger: time-series CSV logging.

Four files per run, all timestamp-keyed so they can be joined in analysis:
  * ticks.csv     — one row per control-loop iteration (§ログ requirements)
  * commands.csv  — one row per PACE5000 write, with the reason and every
                     control-decision value that produced it
  * steps.csv     — one row per settled step (before/after, response time,
                     observed gain) — the online sensitivity data set
  * events.csv    — safety events and state transitions

CSV was chosen over SQLite to keep the dependency footprint at zero; any
other sink (e.g. SQLite) can be swapped in by implementing the same method
surface, since callers only depend on this class's public API.
"""
from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path
from typing import TextIO

from .config import LoggingConfig
from .models import ControllerSnapshot, SafetyEvent, StateTransition, StepRecord

_TICK_FIELDS = [
    "t_mono", "t_wall", "state",
    "raw_pressure_gpa", "filtered_pressure_gpa", "pressure_slope_gpa_s",
    "predicted_pressure_gpa", "user_target_gpa", "control_target_gpa",
    "membrane_setpoint_mpa", "membrane_actual_mpa",
    "safe_gain", "measurement_std_gpa", "measurement_r2", "estimator_valid",
    "manual_pause", "safety_level", "safety_reasons", "last_command_reason",
    "max_compression_rate_gpa_per_min", "source_pressure_positive_mpa",
    "logging_error",
]

_COMMAND_FIELDS = [
    "t_mono", "t_wall", "step_id", "reason",
    "membrane_pressure_before", "membrane_pressure_after", "membrane_step_mpa",
    "sample_pressure_before", "filtered_pressure_gpa", "sizing_pressure_gpa", "pressure_slope_gpa_s",
    "predicted_pressure_gpa", "control_target_gpa", "predicted_error_gpa",
    "gain_source", "estimated_gain", "gain_uncertainty", "safe_gain",
    "requested_sample_step_gpa", "region_min_gpa", "region_max_gpa",
    "source_pressure_positive_mpa", "staged_in_measure",
]

_STEP_FIELDS = [
    "t_mono", "t_wall", "step_id", "t_command", "t_drive_started",
    "membrane_pressure_before", "membrane_pressure_after",
    "sample_pressure_before", "sample_pressure_after",
    "response_time_s", "max_slope_gpa_s", "measurement_std_gpa", "observed_gain",
]

_EVENT_FIELDS = ["t_mono", "t_wall", "kind", "code", "severity", "message", "from_state", "to_state"]


class DataLogger:
    def __init__(self, config: LoggingConfig) -> None:
        run_name = config.run_name or datetime.now().strftime("run_%Y%m%dT%H%M%S_%f")
        self.directory = Path(config.directory) / run_name
        # Logs are part of the safety/audit trail.  Reusing a run name used to
        # truncate all four CSV files silently; fail before opening anything
        # instead of destroying an earlier run.
        self.directory.mkdir(parents=True, exist_ok=False)

        opened: list[TextIO] = []
        try:
            self._tick_f = self._open("ticks.csv", _TICK_FIELDS)
            opened.append(self._tick_f)
            self._command_f = self._open("commands.csv", _COMMAND_FIELDS)
            opened.append(self._command_f)
            self._step_f = self._open("steps.csv", _STEP_FIELDS)
            opened.append(self._step_f)
            self._event_f = self._open("events.csv", _EVENT_FIELDS)
            opened.append(self._event_f)

            self._tick_w = csv.DictWriter(self._tick_f, fieldnames=_TICK_FIELDS)
            self._command_w = csv.DictWriter(self._command_f, fieldnames=_COMMAND_FIELDS)
            self._step_w = csv.DictWriter(self._step_f, fieldnames=_STEP_FIELDS)
            self._event_w = csv.DictWriter(self._event_f, fieldnames=_EVENT_FIELDS)
            for writer in (self._tick_w, self._command_w, self._step_w, self._event_w):
                writer.writeheader()
        except Exception:
            for file in opened:
                file.close()
            raise

        self._flush_every = max(1, config.flush_every_n)
        self._n_since_flush = 0

    def _open(self, name: str, _fields: list[str]) -> TextIO:
        return (self.directory / name).open("w", newline="", encoding="utf-8")

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
            "sample_pressure_before": step.sample_pressure_before,
            "sample_pressure_after": step.sample_pressure_after,
            "response_time_s": step.response_time_s,
            "max_slope_gpa_s": step.max_slope_gpa_s,
            "measurement_std_gpa": step.measurement_std_gpa,
            "observed_gain": step.observed_gain,
        })
        self._maybe_flush()

    def log_event(self, event: SafetyEvent) -> None:
        self._event_w.writerow({
            "t_mono": event.t_mono, "t_wall": time.time(), "kind": "safety_event",
            "code": event.code, "severity": event.severity, "message": event.message,
            "from_state": "", "to_state": "",
        })
        self._maybe_flush()

    def log_transition(self, transition: StateTransition) -> None:
        self._event_w.writerow({
            "t_mono": transition.t_mono, "t_wall": time.time(), "kind": "state_transition",
            "code": "", "severity": "info", "message": transition.reason,
            "from_state": transition.from_state.value, "to_state": transition.to_state.value,
        })
        self._maybe_flush()

    def _maybe_flush(self) -> None:
        self._n_since_flush += 1
        if self._n_since_flush >= self._flush_every:
            self._n_since_flush = 0
            for f in (self._tick_f, self._command_f, self._step_f, self._event_f):
                f.flush()

    def close(self) -> None:
        for f in (self._tick_f, self._command_f, self._step_f, self._event_f):
            f.flush()
            f.close()

    def __enter__(self) -> "DataLogger":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
