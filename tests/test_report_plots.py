"""report_plots.write_summary_plots reads ticks.csv back from disk (see its
module docstring), so these tests go through DataLogger.log_tick() rather
than hand-writing CSV rows, and separately confirm DataLogger.close() treats
a plot-export failure as best-effort -- never allowed to affect end_reason
or raise out of close().
"""
from __future__ import annotations

import json

from pressurekeeper.config import LoggingConfig
from pressurekeeper.logging_sink import DataLogger
from pressurekeeper.models import ControlState, ControllerSnapshot
from pressurekeeper.report_plots import write_summary_plots


def _snapshot(t_mono: float, **overrides) -> ControllerSnapshot:
    fields = dict(
        t_mono=t_mono,
        state=ControlState.APPROACH,
        user_target_gpa=5.0,
        control_target_gpa=4.9,
        raw_pressure_gpa=1.0,
        filtered_pressure_gpa=1.0,
        pressure_slope_gpa_s=0.0,
        predicted_pressure_gpa=1.0,
        measurement_std_gpa=0.01,
        measurement_r2=0.99,
        estimator_valid=True,
        membrane_setpoint_mpa=0.6,
        membrane_actual_mpa=0.5,
        safe_gain=0.08,
        last_command_reason=None,
        manual_pause=False,
        safety_level="ok",
    )
    fields.update(overrides)
    return ControllerSnapshot(**fields)


def test_no_ticks_returns_none_and_writes_nothing(tmp_path):
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()
    assert write_summary_plots(run_dir) is None
    assert not (run_dir / "summary_plots.png").exists()


def test_missing_ticks_csv_returns_none(tmp_path):
    assert write_summary_plots(tmp_path / "no_such_dir") is None


def test_writes_png_from_logged_ticks(tmp_path):
    logger = DataLogger(LoggingConfig(directory=str(tmp_path), run_name="run1"))
    logger.log_tick(_snapshot(0.0, membrane_setpoint_mpa=0.0, membrane_actual_mpa=0.0))
    logger.log_tick(_snapshot(1.0, membrane_setpoint_mpa=0.6, membrane_actual_mpa=0.3))
    logger.log_tick(_snapshot(2.0, membrane_setpoint_mpa=0.6, membrane_actual_mpa=0.6))
    logger.close()

    out_path = logger.directory / "summary_plots.png"
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_close_survives_plot_export_failure(tmp_path, monkeypatch):
    def _boom(_run_dir):
        raise RuntimeError("simulated plotting failure")

    monkeypatch.setattr(
        "pressurekeeper.report_plots.write_summary_plots", _boom,
    )
    logger = DataLogger(LoggingConfig(directory=str(tmp_path), run_name="run2"))
    logger.mark_control_started()
    logger.log_tick(_snapshot(0.0))
    logger.close()  # must not raise

    manifest = json.loads((logger.directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["end_reason"] == "completed"
    assert not (logger.directory / "summary_plots.png").exists()
