"""manifest.json is the run-level record that survives even when a run
never produces a single ticks.csv row (opened but never started, or crashed
before the first tick) -- see DataLogger's module docstring.
"""
from __future__ import annotations

import json

from pressurekeeper.config import LoggingConfig
from pressurekeeper.logging_sink import DataLogger


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
