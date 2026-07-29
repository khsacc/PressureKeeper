"""Configure Parameters dialog: lets an operator edit every non-connection
value in config/default.yaml (hysteresis, approach, gain_regions, gain
estimation, estimator, safety, logging, control, and the ruby/PACE5000 API
sections' non-connection fields) from the GUI instead of hand-editing YAML.

Deliberately excludes ruby_api/pace5000_api's base_url/api_key -- those stay
in ApiConfigDialog ("Configure API") to avoid two widgets fighting over the
same value. Like ApiConfigDialog, MainWindow only offers this dialog before
"Start Control" has been pressed: "Save & Apply" both writes the edited
values to a new local YAML file (never config/default.yaml -- see
main_window.py's save guard) and hot-swaps them into the running
estimator/gain_estimator/safety/controller (see
OneSidedPressureController.apply_config_update()), which resets buffered
history/observations/latched flags -- safe only because nothing has ticked
yet.

This module only collects and validates form input (`collect_overlay()`
returns a dict overlay, raising ValueError on unparseable text); all the
policy around confirming risky transitions, choosing a save path, and
actually applying the result to the session lives in main_window.py.
"""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..config import Configuration


# ---------------------------------------------------------------- widget helpers

def _line(value: object) -> QLineEdit:
    w = QLineEdit()
    if value is not None:
        w.setText(str(value))
    return w


def _req_float(w: QLineEdit, label: str) -> float:
    text = w.text().strip()
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"{label}: {text!r} is not a valid number") from None


def _req_int(w: QLineEdit, label: str) -> int:
    text = w.text().strip()
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"{label}: {text!r} is not a valid integer") from None


def _opt_float(w: QLineEdit, label: str) -> float | None:
    return None if not w.text().strip() else _req_float(w, label)


def _opt_int(w: QLineEdit, label: str) -> int | None:
    return None if not w.text().strip() else _req_int(w, label)


def _opt_str(w: QLineEdit) -> str | None:
    text = w.text().strip()
    return text or None


def _combo(options: list[str], current: str) -> QComboBox:
    w = QComboBox()
    w.addItems(options)
    if current in options:
        w.setCurrentIndex(options.index(current))
    return w


def _scrollable(inner: QWidget) -> QScrollArea:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(inner)
    return area


# ---------------------------------------------------------------------- tabs

class _RubyApiTab(QWidget):
    def __init__(self, cfg) -> None:  # cfg: RubyApiConfig
        super().__init__()
        acq = cfg.acquisition
        form = QFormLayout(self)

        self.timeout_s = _line(cfg.timeout_s)
        form.addRow("timeout_s:", self.timeout_s)
        self.poll_interval_s = _line(cfg.poll_interval_s)
        # RubyPressureClient never reads this value itself -- the control
        # loop's actual tick cadence is ControllerWorker's own
        # poll_interval_s, computed once at launch from this and
        # control.loop_min_interval_s and never re-read afterward. Unlike
        # pace5000_api.status_poll_interval_s (which controller.py re-reads
        # from the live config every tick), editing this has no effect on
        # the current session.
        form.addRow("poll_interval_s (>= 0.2, next launch only):", self.poll_interval_s)

        form.addRow(QLabel("--- acquisition ---"))
        self.configuration_id = _line(acq.configuration_id)
        form.addRow("configuration_id (blank = null):", self.configuration_id)
        self.axis_mode = _combo(["(null)", "calibrated", "pixel"], acq.axis_mode or "(null)")
        form.addRow("axis_mode (only with configuration_id set):", self.axis_mode)
        self.exposure_time_s = _line(acq.exposure_time_s)
        form.addRow("exposure_time_s (blank = null):", self.exposure_time_s)
        self.accumulations = _line(acq.accumulations)
        form.addRow("accumulations (blank = null):", self.accumulations)
        self.dark_mode = _combo(["none", "reuse_loaded", "provided"], acq.dark_mode)
        form.addRow("dark_mode:", self.dark_mode)
        self.fit_function = _combo(
            ["Pseudo Voigt", "Moffat", "Gauss", "Lorentz", "Diamond Raman Edge"], acq.fit_function,
        )
        form.addRow("fit_function:", self.fit_function)
        self.fit_peak_count = _line(acq.fit_peak_count)
        form.addRow("fit_peak_count (1-5):", self.fit_peak_count)
        self.peak_sort_order = _combo(
            ["x_desc", "x_asc", "intensity_desc", "intensity_asc"], acq.peak_sort_order,
        )
        form.addRow("peak_sort_order:", self.peak_sort_order)
        self.baseline_model = _combo(
            ["constant", "linear", "quadratic", "auto_polynomial"], acq.baseline_model,
        )
        form.addRow("baseline_model:", self.baseline_model)
        self.fit_range_start = _line(acq.fit_range[0] if acq.fit_range else None)
        self.fit_range_end = _line(acq.fit_range[1] if acq.fit_range else None)
        fit_range_row = QHBoxLayout()
        fit_range_row.addWidget(QLabel("start:"))
        fit_range_row.addWidget(self.fit_range_start)
        fit_range_row.addWidget(QLabel("end:"))
        fit_range_row.addWidget(self.fit_range_end)
        form.addRow("fit_range (both blank = null):", fit_range_row)
        self.sensor = _line(acq.sensor)
        form.addRow("sensor:", self.sensor)
        self.pressure_scale = _line(acq.pressure_scale)
        form.addRow("pressure_scale:", self.pressure_scale)
        self.zero_pressure_peak = _line(acq.zero_pressure_peak)
        form.addRow("zero_pressure_peak:", self.zero_pressure_peak)
        self.pressure_peak_index = _line(acq.pressure_peak_index)
        form.addRow("pressure_peak_index (1-5):", self.pressure_peak_index)

    def values(self) -> dict:
        axis_mode_text = self.axis_mode.currentText()
        fit_range = None
        start_text, end_text = self.fit_range_start.text().strip(), self.fit_range_end.text().strip()
        if start_text or end_text:
            fit_range = [
                _req_float(self.fit_range_start, "fit_range.start"),
                _req_float(self.fit_range_end, "fit_range.end"),
            ]
        return {
            "timeout_s": _req_float(self.timeout_s, "ruby_api.timeout_s"),
            "poll_interval_s": _req_float(self.poll_interval_s, "ruby_api.poll_interval_s"),
            "acquisition": {
                "configuration_id": _opt_str(self.configuration_id),
                "axis_mode": None if axis_mode_text == "(null)" else axis_mode_text,
                "exposure_time_s": _opt_float(self.exposure_time_s, "acquisition.exposure_time_s"),
                "accumulations": _opt_int(self.accumulations, "acquisition.accumulations"),
                "dark_mode": self.dark_mode.currentText(),
                "fit_function": self.fit_function.currentText(),
                "fit_peak_count": _req_int(self.fit_peak_count, "acquisition.fit_peak_count"),
                "peak_sort_order": self.peak_sort_order.currentText(),
                "baseline_model": self.baseline_model.currentText(),
                "fit_range": fit_range,
                "sensor": self.sensor.text().strip(),
                "pressure_scale": self.pressure_scale.text().strip(),
                "zero_pressure_peak": _req_float(self.zero_pressure_peak, "acquisition.zero_pressure_peak"),
                "pressure_peak_index": _req_int(self.pressure_peak_index, "acquisition.pressure_peak_index"),
            },
        }


class _Pace5000ApiTab(QWidget):
    def __init__(self, cfg) -> None:  # cfg: Pace5000ApiConfig
        super().__init__()
        form = QFormLayout(self)
        self.timeout_s = _line(cfg.timeout_s)
        form.addRow("timeout_s:", self.timeout_s)
        self.status_poll_interval_s = _line(cfg.status_poll_interval_s)
        form.addRow("status_poll_interval_s:", self.status_poll_interval_s)
        self.default_rate_mpa_per_min = _line(cfg.default_rate_mpa_per_min)
        form.addRow("default_rate_mpa_per_min:", self.default_rate_mpa_per_min)
        self.ensure_control_mode_enabled = QCheckBox()
        self.ensure_control_mode_enabled.setChecked(cfg.ensure_control_mode_enabled)
        form.addRow("ensure_control_mode_enabled:", self.ensure_control_mode_enabled)

    def values(self) -> dict:
        return {
            "timeout_s": _req_float(self.timeout_s, "pace5000_api.timeout_s"),
            "status_poll_interval_s": _req_float(self.status_poll_interval_s, "pace5000_api.status_poll_interval_s"),
            "default_rate_mpa_per_min": _req_float(self.default_rate_mpa_per_min, "pace5000_api.default_rate_mpa_per_min"),
            "ensure_control_mode_enabled": self.ensure_control_mode_enabled.isChecked(),
        }


class _HysteresisApproachTab(QWidget):
    def __init__(self, hysteresis, approach) -> None:
        super().__init__()
        form = QFormLayout(self)

        form.addRow(QLabel("--- hysteresis ---"))
        self.reach_margin_gpa = _line(hysteresis.reach_margin_gpa)
        form.addRow("reach_margin_gpa:", self.reach_margin_gpa)
        self.reapproach_margin_gpa = _line(hysteresis.reapproach_margin_gpa)
        form.addRow("reapproach_margin_gpa:", self.reapproach_margin_gpa)
        self.overshoot_margin_gpa = _line(hysteresis.overshoot_margin_gpa)
        form.addRow("overshoot_margin_gpa:", self.overshoot_margin_gpa)

        form.addRow(QLabel("--- approach ---"))
        self.approach_margin_gpa = _line(approach.approach_margin_gpa)
        form.addRow("approach_margin_gpa:", self.approach_margin_gpa)
        self.approach_factor = _line(approach.approach_factor)
        form.addRow("approach_factor (0-1):", self.approach_factor)
        self.prediction_horizon_s = _line(approach.prediction_horizon_s)
        form.addRow("prediction_horizon_s:", self.prediction_horizon_s)
        self.near_target_distance_gpa = _line(approach.near_target_distance_gpa)
        form.addRow("near_target_distance_gpa:", self.near_target_distance_gpa)
        self.near_target_max_sample_step_gpa = _line(approach.near_target_max_sample_step_gpa)
        form.addRow("near_target_max_sample_step_gpa:", self.near_target_max_sample_step_gpa)
        self.near_target_slope_threshold_scale = _line(approach.near_target_slope_threshold_scale)
        form.addRow("near_target_slope_threshold_scale (0-1):", self.near_target_slope_threshold_scale)
        self.near_target_extra_settle_time_s = _line(approach.near_target_extra_settle_time_s)
        form.addRow("near_target_extra_settle_time_s:", self.near_target_extra_settle_time_s)
        self.min_membrane_step_mpa = _line(approach.min_membrane_step_mpa)
        form.addRow("min_membrane_step_mpa:", self.min_membrane_step_mpa)
        self.membrane_arrival_tolerance_mpa = _line(approach.membrane_arrival_tolerance_mpa)
        form.addRow("membrane_arrival_tolerance_mpa:", self.membrane_arrival_tolerance_mpa)
        self.max_compression_rate_gpa_per_min = _line(approach.max_compression_rate_gpa_per_min)
        form.addRow("max_compression_rate_gpa_per_min (blank = no cap):", self.max_compression_rate_gpa_per_min)

    def values(self) -> dict:
        return {
            "hysteresis": {
                "reach_margin_gpa": _req_float(self.reach_margin_gpa, "hysteresis.reach_margin_gpa"),
                "reapproach_margin_gpa": _req_float(self.reapproach_margin_gpa, "hysteresis.reapproach_margin_gpa"),
                "overshoot_margin_gpa": _req_float(self.overshoot_margin_gpa, "hysteresis.overshoot_margin_gpa"),
            },
            "approach": {
                "approach_margin_gpa": _req_float(self.approach_margin_gpa, "approach.approach_margin_gpa"),
                "approach_factor": _req_float(self.approach_factor, "approach.approach_factor"),
                "prediction_horizon_s": _req_float(self.prediction_horizon_s, "approach.prediction_horizon_s"),
                "near_target_distance_gpa": _req_float(self.near_target_distance_gpa, "approach.near_target_distance_gpa"),
                "near_target_max_sample_step_gpa": _req_float(
                    self.near_target_max_sample_step_gpa, "approach.near_target_max_sample_step_gpa"),
                "near_target_slope_threshold_scale": _req_float(
                    self.near_target_slope_threshold_scale, "approach.near_target_slope_threshold_scale"),
                "near_target_extra_settle_time_s": _req_float(
                    self.near_target_extra_settle_time_s, "approach.near_target_extra_settle_time_s"),
                "min_membrane_step_mpa": _req_float(self.min_membrane_step_mpa, "approach.min_membrane_step_mpa"),
                "membrane_arrival_tolerance_mpa": _req_float(
                    self.membrane_arrival_tolerance_mpa, "approach.membrane_arrival_tolerance_mpa"),
                "max_compression_rate_gpa_per_min": _opt_float(
                    self.max_compression_rate_gpa_per_min, "approach.max_compression_rate_gpa_per_min"),
            },
        }


_GAIN_REGION_COLUMNS = [
    ("sample_pressure_min_gpa", "min (GPa)"),
    ("sample_pressure_max_gpa", "max (GPa)"),
    ("safe_gain", "safe_gain"),
    ("rate_limit_gain", "rate_limit_gain (blank = safe_gain)"),
    ("max_sample_step_gpa", "max_sample_step (GPa)"),
    ("max_membrane_step", "max_membrane_step (MPa)"),
    ("minimum_settle_time_s", "minimum_settle_time_s"),
    ("settled_slope_threshold_gpa_s", "settled_slope_threshold (GPa/s)"),
]
# Only this column may be left blank (falls back to safe_gain -- see
# GainRegion.rate_limit_gain's docstring); every other column is required.
_GAIN_REGION_OPTIONAL_FIELDS = {"rate_limit_gain"}


class _GainRegionsTab(QWidget):
    def __init__(self, gain_regions: list) -> None:  # list[GainRegion]
        super().__init__()
        layout = QVBoxLayout(self)

        self.table = QTableWidget(len(gain_regions), len(_GAIN_REGION_COLUMNS))
        self.table.setHorizontalHeaderLabels([label for _, label in _GAIN_REGION_COLUMNS])
        for row, region in enumerate(gain_regions):
            for col, (field, _) in enumerate(_GAIN_REGION_COLUMNS):
                value = getattr(region, field)
                self.table.setItem(row, col, QTableWidgetItem("" if value is None else str(value)))
        layout.addWidget(self.table)

        buttons_row = QHBoxLayout()
        add_btn = QPushButton("+ Add region")
        add_btn.clicked.connect(self._add_row)
        remove_btn = QPushButton("- Remove selected")
        remove_btn.clicked.connect(self._remove_selected_rows)
        buttons_row.addWidget(add_btn)
        buttons_row.addWidget(remove_btn)
        buttons_row.addStretch(1)
        layout.addLayout(buttons_row)

    def _add_row(self) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        for col, (field, _) in enumerate(_GAIN_REGION_COLUMNS):
            # "0.0" would fail validation for rate_limit_gain (must be >=
            # safe_gain, which itself must be > 0) -- blank (falls back to
            # safe_gain) is the correct default, not an arbitrary number.
            default_text = "" if field in _GAIN_REGION_OPTIONAL_FIELDS else "0.0"
            self.table.setItem(row, col, QTableWidgetItem(default_text))

    def _remove_selected_rows(self) -> None:
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.table.removeRow(row)

    def values(self) -> list[dict]:
        regions = []
        for row in range(self.table.rowCount()):
            region: dict = {}
            for col, (field, label) in enumerate(_GAIN_REGION_COLUMNS):
                item = self.table.item(row, col)
                text = item.text().strip() if item is not None else ""
                if not text and field in _GAIN_REGION_OPTIONAL_FIELDS:
                    region[field] = None
                    continue
                try:
                    region[field] = float(text)
                except ValueError:
                    raise ValueError(f"gain_regions row {row + 1}, {label}: {text!r} is not a valid number") from None
            regions.append(region)
        if not regions:
            raise ValueError("gain_regions must contain at least one region")
        return regions


class _GainEstimationEstimatorTab(QWidget):
    def __init__(self, gain_estimation, estimator) -> None:
        super().__init__()
        form = QFormLayout(self)

        form.addRow(QLabel("--- gain_estimation ---"))
        self.bin_width_gpa = _line(gain_estimation.bin_width_gpa)
        form.addRow("bin_width_gpa:", self.bin_width_gpa)
        self.min_samples_for_estimate = _line(gain_estimation.min_samples_for_estimate)
        form.addRow("min_samples_for_estimate:", self.min_samples_for_estimate)
        self.safety_factor = _line(gain_estimation.safety_factor)
        form.addRow("safety_factor:", self.safety_factor)
        self.upper_percentile = _line(gain_estimation.upper_percentile)
        form.addRow("upper_percentile (50-100):", self.upper_percentile)
        self.neighbor_bins = _line(gain_estimation.neighbor_bins)
        form.addRow("neighbor_bins:", self.neighbor_bins)

        form.addRow(QLabel("--- estimator ---"))
        self.outlier_median_window = _line(estimator.outlier_median_window)
        form.addRow("outlier_median_window (>= 3):", self.outlier_median_window)
        self.smoothing_window_s = _line(estimator.smoothing_window_s)
        form.addRow("smoothing_window_s:", self.smoothing_window_s)
        self.slope_window_s = _line(estimator.slope_window_s)
        form.addRow("slope_window_s:", self.slope_window_s)
        self.min_points_for_valid = _line(estimator.min_points_for_valid)
        form.addRow("min_points_for_valid (>= 2):", self.min_points_for_valid)
        self.max_sample_age_s = _line(estimator.max_sample_age_s)
        form.addRow("max_sample_age_s:", self.max_sample_age_s)
        self.max_jump_gpa = _line(estimator.max_jump_gpa)
        form.addRow("max_jump_gpa:", self.max_jump_gpa)
        self.min_r2 = _line(estimator.min_r2)
        form.addRow("min_r2 (blank = disabled):", self.min_r2)
        self.min_intensity = _line(estimator.min_intensity)
        form.addRow("min_intensity (blank = disabled):", self.min_intensity)

    def values(self) -> dict:
        return {
            "gain_estimation": {
                "bin_width_gpa": _req_float(self.bin_width_gpa, "gain_estimation.bin_width_gpa"),
                "min_samples_for_estimate": _req_int(self.min_samples_for_estimate, "gain_estimation.min_samples_for_estimate"),
                "safety_factor": _req_float(self.safety_factor, "gain_estimation.safety_factor"),
                "upper_percentile": _req_float(self.upper_percentile, "gain_estimation.upper_percentile"),
                "neighbor_bins": _req_int(self.neighbor_bins, "gain_estimation.neighbor_bins"),
            },
            "estimator": {
                "outlier_median_window": _req_int(self.outlier_median_window, "estimator.outlier_median_window"),
                "smoothing_window_s": _req_float(self.smoothing_window_s, "estimator.smoothing_window_s"),
                "slope_window_s": _req_float(self.slope_window_s, "estimator.slope_window_s"),
                "min_points_for_valid": _req_int(self.min_points_for_valid, "estimator.min_points_for_valid"),
                "max_sample_age_s": _req_float(self.max_sample_age_s, "estimator.max_sample_age_s"),
                "max_jump_gpa": _req_float(self.max_jump_gpa, "estimator.max_jump_gpa"),
                "min_r2": _opt_float(self.min_r2, "estimator.min_r2"),
                "min_intensity": _opt_float(self.min_intensity, "estimator.min_intensity"),
            },
        }


class _SafetyTab(QWidget):
    def __init__(self, cfg) -> None:  # cfg: SafetyConfig
        super().__init__()
        form = QFormLayout(self)
        self.max_sample_pressure_gpa = _line(cfg.max_sample_pressure_gpa)
        form.addRow("max_sample_pressure_gpa:", self.max_sample_pressure_gpa)
        self.max_membrane_pressure_mpa = _line(cfg.max_membrane_pressure_mpa)
        form.addRow("max_membrane_pressure_mpa:", self.max_membrane_pressure_mpa)
        self.max_membrane_step_mpa_hard = _line(cfg.max_membrane_step_mpa_hard)
        form.addRow("max_membrane_step_mpa_hard:", self.max_membrane_step_mpa_hard)
        self.max_cumulative_step_mpa = _line(cfg.max_cumulative_step_mpa)
        form.addRow("max_cumulative_step_mpa:", self.max_cumulative_step_mpa)
        self.cumulative_window_s = _line(cfg.cumulative_window_s)
        form.addRow("cumulative_window_s:", self.cumulative_window_s)
        self.max_stale_sample_s = _line(cfg.max_stale_sample_s)
        form.addRow("max_stale_sample_s:", self.max_stale_sample_s)
        self.max_consecutive_invalid = _line(cfg.max_consecutive_invalid)
        form.addRow("max_consecutive_invalid:", self.max_consecutive_invalid)
        self.max_consecutive_comm_errors = _line(cfg.max_consecutive_comm_errors)
        form.addRow("max_consecutive_comm_errors:", self.max_consecutive_comm_errors)
        self.sample_jump_hard_gpa = _line(cfg.sample_jump_hard_gpa)
        form.addRow("sample_jump_hard_gpa:", self.sample_jump_hard_gpa)
        self.max_consecutive_jump_flags = _line(cfg.max_consecutive_jump_flags)
        form.addRow("max_consecutive_jump_flags:", self.max_consecutive_jump_flags)
        self.setpoint_mismatch_tol_mpa = _line(cfg.setpoint_mismatch_tol_mpa)
        form.addRow("setpoint_mismatch_tol_mpa:", self.setpoint_mismatch_tol_mpa)
        self.setpoint_mismatch_grace_s = _line(cfg.setpoint_mismatch_grace_s)
        form.addRow("setpoint_mismatch_grace_s:", self.setpoint_mismatch_grace_s)
        self.max_setpoint_actual_gap_mpa = _line(cfg.max_setpoint_actual_gap_mpa)
        form.addRow("max_setpoint_actual_gap_mpa:", self.max_setpoint_actual_gap_mpa)
        self.minimum_source_pressure_headroom_mpa = _line(cfg.minimum_source_pressure_headroom_mpa)
        form.addRow("minimum_source_pressure_headroom_mpa:", self.minimum_source_pressure_headroom_mpa)
        self.ruby_error_pause_after_s = _line(cfg.ruby_error_pause_after_s)
        form.addRow("ruby_error_pause_after_s:", self.ruby_error_pause_after_s)
        self.membrane_error_pause_after_s = _line(cfg.membrane_error_pause_after_s)
        form.addRow("membrane_error_pause_after_s:", self.membrane_error_pause_after_s)

    def values(self) -> dict:
        return {
            "max_sample_pressure_gpa": _req_float(self.max_sample_pressure_gpa, "safety.max_sample_pressure_gpa"),
            "max_membrane_pressure_mpa": _req_float(self.max_membrane_pressure_mpa, "safety.max_membrane_pressure_mpa"),
            "max_membrane_step_mpa_hard": _req_float(self.max_membrane_step_mpa_hard, "safety.max_membrane_step_mpa_hard"),
            "max_cumulative_step_mpa": _req_float(self.max_cumulative_step_mpa, "safety.max_cumulative_step_mpa"),
            "cumulative_window_s": _req_float(self.cumulative_window_s, "safety.cumulative_window_s"),
            "max_stale_sample_s": _req_float(self.max_stale_sample_s, "safety.max_stale_sample_s"),
            "max_consecutive_invalid": _req_int(self.max_consecutive_invalid, "safety.max_consecutive_invalid"),
            "max_consecutive_comm_errors": _req_int(self.max_consecutive_comm_errors, "safety.max_consecutive_comm_errors"),
            "sample_jump_hard_gpa": _req_float(self.sample_jump_hard_gpa, "safety.sample_jump_hard_gpa"),
            "max_consecutive_jump_flags": _req_int(self.max_consecutive_jump_flags, "safety.max_consecutive_jump_flags"),
            "setpoint_mismatch_tol_mpa": _req_float(self.setpoint_mismatch_tol_mpa, "safety.setpoint_mismatch_tol_mpa"),
            "setpoint_mismatch_grace_s": _req_float(self.setpoint_mismatch_grace_s, "safety.setpoint_mismatch_grace_s"),
            "max_setpoint_actual_gap_mpa": _req_float(self.max_setpoint_actual_gap_mpa, "safety.max_setpoint_actual_gap_mpa"),
            "minimum_source_pressure_headroom_mpa": _req_float(
                self.minimum_source_pressure_headroom_mpa, "safety.minimum_source_pressure_headroom_mpa"),
            "ruby_error_pause_after_s": _req_float(self.ruby_error_pause_after_s, "safety.ruby_error_pause_after_s"),
            "membrane_error_pause_after_s": _req_float(
                self.membrane_error_pause_after_s, "safety.membrane_error_pause_after_s"),
        }


class _LoggingControlTab(QWidget):
    """logging.* and control.* other than dry_run.

    dry_run is deliberately absent from this form (and from `values()`'s
    overlay): it is shown read-only for reference only. Two independent
    dry_run gates exist (this controller's own config.control.dry_run, and
    Pace5000Client's own `self.dry_run`, set once at construction from the
    CLI's --dry-run/--live and never resynchronized afterward -- see
    Pace5000Client.update_config()'s docstring); hot-swapping only the
    controller's copy via apply_config_update() would let an operator
    confirm "enable real writes" while the client's own independent gate
    kept silently suppressing every write. Making dry_run startup-only here
    removes the mutation path entirely rather than trying to keep two
    separately-owned flags in sync. Change it via --dry-run/--live at
    launch instead.

    Every other field here takes effect only on the *next* launch, not this
    session: the DataLogger and ControllerWorker this session is using were
    already constructed (in build_app()/gui/app.py) from the config in
    effect at startup, and apply_config_update() does not reconstruct
    either -- see main_window.py's _on_save_apply_parameters.
    """

    def __init__(self, logging_cfg, control) -> None:
        super().__init__()
        form = QFormLayout(self)

        form.addRow(QLabel("--- logging (next launch only) ---"))
        self.directory = _line(logging_cfg.directory)
        form.addRow("directory (next launch only):", self.directory)
        self.run_name = _line(logging_cfg.run_name)
        form.addRow("run_name (next launch only, blank = auto timestamp):", self.run_name)
        self.flush_every_n = _line(logging_cfg.flush_every_n)
        form.addRow("flush_every_n (next launch only):", self.flush_every_n)

        form.addRow(QLabel("--- control ---"))
        dry_run_display = QCheckBox()
        dry_run_display.setChecked(control.dry_run)
        dry_run_display.setEnabled(False)
        form.addRow("dry_run (read-only here -- set via --dry-run/--live at launch):", dry_run_display)
        self.loop_min_interval_s = _line(control.loop_min_interval_s)
        form.addRow("loop_min_interval_s (next launch only):", self.loop_min_interval_s)
        self.default_target_pressure_gpa = _line(control.default_target_pressure_gpa)
        form.addRow("default_target_pressure_gpa (next launch only, blank = null):", self.default_target_pressure_gpa)

    def values(self) -> dict:
        return {
            "logging": {
                "directory": self.directory.text().strip(),
                "run_name": _opt_str(self.run_name),
                "flush_every_n": _req_int(self.flush_every_n, "logging.flush_every_n"),
            },
            "control": {
                "loop_min_interval_s": _req_float(self.loop_min_interval_s, "control.loop_min_interval_s"),
                "default_target_pressure_gpa": _opt_float(
                    self.default_target_pressure_gpa, "control.default_target_pressure_gpa"),
            },
        }


# -------------------------------------------------------------------- dialog

class ParametersConfigDialog(QDialog):
    """Pure form: collects edits and validates parseability of raw text, but
    does not itself validate cross-field invariants (contiguous gain_regions,
    hard-cap ordering, etc.), save to disk, or apply anything -- see
    main_window.py's `_on_save_apply_parameters` for that.
    """

    def __init__(self, config: Configuration, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Configure Parameters")
        self.resize(720, 640)

        self._ruby_tab = _RubyApiTab(config.ruby_api)
        self._pace_tab = _Pace5000ApiTab(config.pace5000_api)
        self._hysteresis_approach_tab = _HysteresisApproachTab(config.hysteresis, config.approach)
        self._gain_regions_tab = _GainRegionsTab(config.gain_regions)
        self._gain_estimator_tab = _GainEstimationEstimatorTab(config.gain_estimation, config.estimator)
        self._safety_tab = _SafetyTab(config.safety)
        self._logging_control_tab = _LoggingControlTab(config.logging, config.control)

        tabs = QTabWidget()
        tabs.addTab(_scrollable(self._ruby_tab), "Ruby API")
        tabs.addTab(_scrollable(self._pace_tab), "PACE5000 API")
        tabs.addTab(_scrollable(self._hysteresis_approach_tab), "Hysteresis / Approach")
        tabs.addTab(self._gain_regions_tab, "Gain Regions")
        tabs.addTab(_scrollable(self._gain_estimator_tab), "Gain Estimation / Estimator")
        tabs.addTab(_scrollable(self._safety_tab), "Safety")
        tabs.addTab(_scrollable(self._logging_control_tab), "Logging / Control")

        note = QLabel(
            "base_url / api_key are not shown here -- use \"Configure API\" for those. "
            "Values are re-validated exactly as at config load; an invalid combination "
            "(e.g. non-contiguous gain_regions) is rejected with an explanation."
        )
        note.setWordWrap(True)

        self.save_apply_btn = QPushButton("Save && Apply parameters")
        self.save_apply_btn.setStyleSheet("font-weight:bold;")
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.Close).clicked.connect(self.reject)

        button_row = QHBoxLayout()
        button_row.addWidget(self.save_apply_btn)
        button_row.addWidget(buttons)

        layout = QVBoxLayout(self)
        layout.addWidget(note)
        layout.addWidget(tabs, 1)
        layout.addLayout(button_row)

    def collect_overlay(self) -> dict:
        """Nested dict overlay covering every field this dialog exposes.

        Raises ValueError (with a field-identifying message) if any text
        field can't be parsed to the type it needs to be. Does not check
        cross-field/model-level invariants -- the caller re-validates the
        merged result via `Configuration.model_validate()`.
        """
        overlay: dict = {}
        overlay["ruby_api"] = self._ruby_tab.values()
        overlay["pace5000_api"] = self._pace_tab.values()
        overlay.update(self._hysteresis_approach_tab.values())
        overlay["gain_regions"] = self._gain_regions_tab.values()
        overlay.update(self._gain_estimator_tab.values())
        overlay["safety"] = self._safety_tab.values()
        overlay.update(self._logging_control_tab.values())
        return overlay
