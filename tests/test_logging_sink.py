"""manifest.json is the run-level record that survives even when a run
never produces a single ticks.csv row (opened but never started, or crashed
before the first tick) -- see DataLogger's module docstring.
"""
from __future__ import annotations

import csv
import json

from pressurekeeper.config import LoggingConfig
from pressurekeeper.logging_sink import DataLogger
from pressurekeeper.models import InterruptedStepObservation


def _manifest(directory) -> dict:
    return json.loads((directory / "manifest.json").read_text(encoding="utf-8"))


def test_manifest_written_on_construction_before_any_tick(tmp_path):
    logger = DataLogger(LoggingConfig(directory=str(tmp_path), run_name="run1"), mode="dry-run")
    m = _manifest(logger.directory)
    assert m["mode"] == "dry-run"
    assert m["started_control"] is False
    assert m["end_reason"] is None
    assert m["crash_traceback"] is None
    assert m["run_name"] == "run1"
    assert isinstance(m["pid"], int)
    logger.close()


def test_close_without_starting_control_records_not_started(tmp_path):
    logger = DataLogger(LoggingConfig(directory=str(tmp_path), run_name="run2"))
    logger.close()
    m = _manifest(logger.directory)
    assert m["end_reason"] == "not_started", \
        "a run that never called mark_control_started() must not be recorded as completed"
    assert m["ended_at_wall"] is not None


def test_close_after_starting_control_records_completed(tmp_path):
    logger = DataLogger(LoggingConfig(directory=str(tmp_path), run_name="run3"))
    logger.mark_control_started()
    m = _manifest(logger.directory)
    assert m["started_control"] is True
    logger.close()
    m = _manifest(logger.directory)
    assert m["end_reason"] == "completed"


def test_mark_end_crashed_persists_traceback_and_is_not_overwritten_by_close(tmp_path):
    logger = DataLogger(LoggingConfig(directory=str(tmp_path), run_name="run4"))
    logger.mark_control_started()
    logger.mark_end("crashed", crash_traceback="Traceback (most recent call last):\n  ...\nRuntimeError: boom")
    logger.close()  # must not overwrite the crash reason with "completed"
    m = _manifest(logger.directory)
    assert m["end_reason"] == "crashed"
    assert "RuntimeError: boom" in m["crash_traceback"]


def test_git_sha_recorded_when_running_inside_this_repo(tmp_path):
    logger = DataLogger(LoggingConfig(directory=str(tmp_path), run_name="run5"))
    m = _manifest(logger.directory)
    assert m["git_sha"] is None or (isinstance(m["git_sha"], str) and len(m["git_sha"]) == 40)
    logger.close()


def test_interrupted_step_start_and_final_records_are_additive_to_existing_csvs(tmp_path):
    logger = DataLogger(LoggingConfig(directory=str(tmp_path), run_name="run6"))
    observation = InterruptedStepObservation(
        step_id=6,
        cause_code="compression_rate_exceeded",
        eligible_for_rate_learning=True,
        exclusion_reason=None,
        t_command=10.0,
        t_drive_started=11.0,
        t_interrupted=15.0,
        sizing_pressure_gpa=0.3,
        sample_pressure_before=0.2,
        sample_pressure_at_interrupt=0.35,
        max_sample_pressure_gpa=0.35,
        max_positive_slope_gpa_s=0.02,
        commanded_membrane_rate_mpa_per_min=3.0,
        membrane_pressure_before=2.0,
        membrane_pressure_after=2.6,
        membrane_actual_at_interrupt=2.4,
        max_membrane_actual_mpa=2.4,
        ack_uncertain=False,
    )
    logger.log_interrupted_step(observation, phase="started", safety_factor=1.25)
    observation.t_observation_end = 20.0
    observation.observation_end_reason = "safety_condition_cleared"
    observation.max_sample_pressure_gpa = 0.42
    logger.log_interrupted_step(observation, phase="final", safety_factor=1.25)
    logger.close()

    with (logger.directory / "interrupted_steps.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        rows = list(csv.DictReader(f))
    assert [row["phase"] for row in rows] == ["started", "final"]
    assert rows[-1]["cause_code"] == "compression_rate_exceeded"
    assert abs(float(rows[-1]["raw_rate_gain"]) - 0.4) < 1e-9
    assert abs(float(rows[-1]["learned_rate_floor"]) - 0.5) < 1e-9
    assert (logger.directory / "steps.csv").exists(), \
        "the settled-step audit file must retain its old role and path"
