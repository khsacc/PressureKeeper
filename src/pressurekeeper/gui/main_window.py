"""Top-level window: the common Live Plot + safety controls area (always
visible, unaffected by which tab is active) plus the Tab1/Tab2 QTabWidget.

Wires everything to a single ControllerWorker: every snapshot it emits feeds
the live plot, the status line, and (if running) the schedule tab's runner,
all on the GUI thread (Qt queues cross-thread signal delivery automatically).

The worker is built but deliberately not started in __init__: control.step()
first runs only once the operator clicks "Start Control", so nothing reads
sensors or writes to the membrane merely because the window opened.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..app import AppContext
from ..clients import Pace5000Client, RubyPressureClient
from ..config import Configuration, redact_api_keys
from ..instance_lock import InstanceAlreadyRunning, SingleInstanceLock, lock_path_for
from ..models import ControllerSnapshot
from .api_config_dialog import ApiConfigDialog
from .live_plot import LivePlotWidget
from .membrane_rate_panel import MembraneRatePanel
from .parameters_config_dialog import ParametersConfigDialog
from .tab_schedule import ScheduleTab
from .tab_single_target import SingleTargetTab
from .worker import ControllerWorker


def _deep_update(base: dict, overlay: dict) -> dict:
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _format_validation_error(e: ValidationError) -> str:
    lines = []
    for err in e.errors():
        loc = ".".join(str(p) for p in err["loc"])
        lines.append(f"{loc}: {err['msg']}")
    return "\n".join(lines)


def _fmt(x: float | None, spec: str = "{:.4f}") -> str:
    return spec.format(x) if x is not None else "n/a"


def _format_status(snap: ControllerSnapshot) -> str:
    reasons = ", ".join(snap.safety_reasons)
    rate = _fmt(snap.max_compression_rate_gpa_per_min, "{:.4f}") + " GPa/min" if snap.max_compression_rate_gpa_per_min else "no limit"
    return (
        f"[{snap.state.value}]  target={_fmt(snap.user_target_gpa, '{:.3f}')} GPa  "
        f"filtered={_fmt(snap.filtered_pressure_gpa)} GPa  slope={_fmt(snap.pressure_slope_gpa_s, '{:.5f}')} GPa/s  "
        f"membrane={_fmt(snap.membrane_setpoint_mpa, '{:.4f}')}/{_fmt(snap.membrane_actual_mpa, '{:.4f}')} MPa (set/act)  "
        f"supply={_fmt(snap.source_pressure_positive_mpa, '{:.4f}')} MPa  "
        f"max rate={rate}  gas rate={_fmt(snap.membrane_rate_mpa_per_min, '{:.3f}')} MPa/min  safety={snap.safety_level}"
        f"{f' [{reasons}]' if reasons else ''}"
        f"{'  MANUAL-PAUSE' if snap.manual_pause else ''}"
        f"{f'  LOGGING ERROR: {snap.logging_error}' if snap.logging_error else ''}"
    )


class MainWindow(QMainWindow):
    def __init__(self, ctx: AppContext, config: Configuration, config_path: Path, poll_interval_s: float, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("PressureKeeper")
        self._ctx = ctx
        self._controller = ctx.controller
        # The Configuration currently in effect for this session -- starts as
        # whatever was loaded from --config, and is replaced wholesale by
        # _on_save_apply_parameters() every time "Save & Apply" succeeds, so
        # reopening Configure Parameters always prefills from the latest
        # applied state rather than the original file.
        self._current_config = config
        self._config_path = config_path
        self._last_params_save_path: Path | None = None

        # --sim mode: ctx.ruby/ctx.membrane are simulator objects with no
        # host/port/key to configure. Whether Configure API can ever be
        # enabled at all (independent of the run/stop toggling below, in
        # _on_start_clicked/_on_stop_clicked).
        self._configure_api_available = isinstance(ctx.ruby, RubyPressureClient) and isinstance(ctx.membrane, Pace5000Client)
        self.configure_api_action = self.menuBar().addAction("Configure API")
        self.configure_api_action.triggered.connect(self._on_configure_api)
        if not self._configure_api_available:
            self.configure_api_action.setEnabled(False)
            self.configure_api_action.setToolTip("Not available in simulator mode")

        self.configure_parameters_action = self.menuBar().addAction("Configure Parameters")
        self.configure_parameters_action.triggered.connect(self._on_configure_parameters)

        self.live_plot = LivePlotWidget()
        self.tab1 = SingleTargetTab(self._controller)
        self.tab2 = ScheduleTab(self._controller)
        self.tab2.running_changed.connect(self._on_schedule_running_changed)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.tab1, "Single Target")
        self.tabs.addTab(self.tab2, "Schedule")
        # The schedule tab has wide control rows, whose size hint would
        # otherwise override the column stretch factors and widen this pane.
        tabs_policy = self.tabs.sizePolicy()
        tabs_policy.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
        self.tabs.setSizePolicy(tabs_policy)

        # A small section below the tabs, independent of which tab is active,
        # for the gas-side slew rate -- unlike tab1/tab2 it is never locked
        # while a schedule runs (see _on_schedule_running_changed).
        self.membrane_rate_panel = MembraneRatePanel(self._controller)
        rate_panel_policy = self.membrane_rate_panel.sizePolicy()
        rate_panel_policy.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
        self.membrane_rate_panel.setSizePolicy(rate_panel_policy)

        self.start_btn = QPushButton("Start Control")
        self.start_btn.setStyleSheet("background-color:#22c55e; color:white; font-weight:bold;")
        self.start_btn.clicked.connect(self._on_start_clicked)

        self.stop_btn = QPushButton("Stop Control")
        self.stop_btn.setStyleSheet("background-color:#ef4444; color:white; font-weight:bold;")
        self.stop_btn.clicked.connect(self._on_stop_clicked)
        self.stop_btn.setEnabled(False)

        # Set once worker.start() has actually been called, so a later Start
        # Control click (after Stop Control) knows to just clear the sticky
        # abort latch (see _on_start_clicked) instead of starting the
        # ControllerWorker QThread a second time.
        self._control_started = False

        initial_status = 'not started — press "Start Control" to begin the control loop'
        if self._controller.logging_error:
            initial_status += f" — LOGGING ERROR: {self._controller.logging_error}"
        self.status_label = QLabel(initial_status)
        self.status_label.setWordWrap(True)

        controls_row = QHBoxLayout()
        for w in (self.start_btn, self.stop_btn):
            controls_row.addWidget(w)
        controls_row.addWidget(self.status_label, 1)

        right_column = QVBoxLayout()
        right_column.addWidget(self.tabs)
        right_column.addWidget(self.membrane_rate_panel)

        columns = QHBoxLayout()
        columns.addWidget(self.live_plot, 7)
        columns.addLayout(right_column, 3)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addLayout(controls_row)
        layout.addLayout(columns, 1)
        self.setCentralWidget(central)
        self.resize(1800, 800)

        self.worker = ControllerWorker(self._controller, poll_interval_s, logger=self._ctx.logger)
        self.worker.snapshot_ready.connect(self._on_snapshot)
        self.worker.crashed.connect(self._on_worker_crashed)
        # Not started here — see _on_start_clicked.

    def _on_start_clicked(self) -> None:
        if self._control_started:
            # Restarting after Stop Control: the ControllerWorker QThread
            # never stopped (it keeps polling/logging through a stopped
            # controller so status/plot stay live), only the sticky abort
            # latch set by _on_stop_clicked needs clearing.
            self._controller.reset()
        else:
            self._control_started = True
            if self._ctx.logger is not None:
                self._ctx.logger.mark_control_started()
            self.worker.start()
        self.start_btn.setEnabled(False)
        self.start_btn.setText("Running")
        self.stop_btn.setEnabled(True)
        # Repointing a client at a different device mid-control would do so
        # without the safety supervisor ever having characterized it.
        self.configure_api_action.setEnabled(False)
        # Same reasoning: apply_config_update() resets buffered
        # history/observations/latched flags, which would corrupt an
        # in-progress run.
        self.configure_parameters_action.setEnabled(False)
        self.status_label.setText("control loop starting...")

    def _on_stop_clicked(self) -> None:
        # Immediately halts pressurization (sticky abort latch -- see
        # controller.abort()/safety.py's module docstring). The worker
        # thread is deliberately left running: it keeps polling/logging so
        # the plot and status line stay live while stopped, and Start
        # Control simply clears the latch again (see _on_start_clicked)
        # rather than restarting the thread.
        self._controller.abort("operator requested stop")
        self.start_btn.setEnabled(True)
        self.start_btn.setText("Start Control")
        self.stop_btn.setEnabled(False)
        # Nothing is actively controlling the membrane while stopped, so
        # it's safe to repoint clients or hot-swap config again.
        if self._configure_api_available:
            self.configure_api_action.setEnabled(True)
        self.configure_parameters_action.setEnabled(True)

    def _on_configure_api(self) -> None:
        dialog = ApiConfigDialog(self._ctx.ruby, self._ctx.membrane, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._ctx.ruby.update_connection(base_url=dialog.ruby_url, api_key=dialog.ruby_api_key)
        self._apply_pace_endpoint_change(dialog.pace_url, dialog.pace_api_key)

    def _apply_pace_endpoint_change(self, base_url: str, api_key: str | None) -> None:
        """Repoint `self._ctx.membrane` at `base_url`, re-acquiring the
        single-instance lock (see instance_lock.py) against the new endpoint
        first and releasing the old one only after that succeeds.

        Without this, changing the PACE5000 endpoint left the lock keyed to
        the *old* endpoint forever: no lock was ever held against the new
        one (a second process could start against it undetected), and the
        stale lock on the old endpoint was never released either.
        """
        old_lock = self._ctx.instance_lock
        new_lock_path = lock_path_for(base_url)
        if old_lock is None or new_lock_path == old_lock.path:
            self._ctx.membrane.update_connection(base_url=base_url, api_key=api_key)
            return

        new_lock = SingleInstanceLock(new_lock_path)
        try:
            new_lock.acquire()
        except InstanceAlreadyRunning as e:
            QMessageBox.critical(
                self, "Configure API",
                f"Could not switch to the new PACE5000 endpoint: {e}\n\nKeeping the previous connection.",
            )
            return
        self._ctx.membrane.update_connection(base_url=base_url, api_key=api_key)
        old_lock.release()
        self._ctx.instance_lock = new_lock

    def _on_configure_parameters(self) -> None:
        dialog = ParametersConfigDialog(self._current_config, parent=self)
        dialog.save_apply_btn.clicked.connect(lambda: self._on_save_apply_parameters(dialog))
        dialog.exec()

    def _on_save_apply_parameters(self, dialog: ParametersConfigDialog) -> None:
        try:
            overlay = dialog.collect_overlay()
        except ValueError as e:
            QMessageBox.warning(self, "Configure Parameters", str(e))
            return

        merged = _deep_update(self._current_config.model_dump(), overlay)
        try:
            new_config = Configuration.model_validate(merged)
        except ValidationError as e:
            QMessageBox.critical(
                self, "Configure Parameters",
                "These values are inconsistent (same checks as loading a config file "
                f"at startup) and were not saved or applied:\n\n{_format_validation_error(e)}",
            )
            return

        # dry_run itself is not editable from this dialog (see
        # _LoggingControlTab's docstring) -- new_config.control.dry_run is
        # therefore always identical to self._current_config.control.dry_run,
        # so there is nothing to confirm here.
        is_real_devices = self._configure_api_available

        chosen_path = self._prompt_params_save_path()
        if chosen_path is None:
            return  # operator cancelled -- save & apply is all-or-nothing

        try:
            with chosen_path.open("w", encoding="utf-8") as f:
                # Never persist an already-expanded API key to disk:
                # ParametersConfigDialog deliberately excludes ruby_api/
                # pace5000_api's base_url/api_key (see its docstring), but
                # model_dump() still carries whatever secret config.py's
                # _expand_env_vars() resolved at load time. chosen_path is
                # not guaranteed gitignored (see .gitignore's config/*.yaml
                # exception for default.yaml only), so writing that literal
                # value here would risk committing a real credential.
                yaml.safe_dump(redact_api_keys(new_config.model_dump()), f, sort_keys=False)
        except OSError as e:
            QMessageBox.critical(self, "Configure Parameters", f"Could not write {chosen_path}: {e}")
            return
        self._last_params_save_path = chosen_path

        self._controller.apply_config_update(new_config)
        if is_real_devices:
            self._ctx.ruby.update_config(new_config.ruby_api)
            self._ctx.membrane.update_config(new_config.pace5000_api)
        self._current_config = new_config

        message = (
            f"Saved to {chosen_path} (API keys redacted to environment-variable placeholders "
            "in the saved file; the real key(s) remain in effect for this session) and applied "
            "to this session."
        )
        if self._controller.user_target_gpa > new_config.safety.max_sample_pressure_gpa:
            message += (
                f"\n\nCurrent target {self._controller.user_target_gpa:.3f} GPa now exceeds the new "
                f"safety ceiling {new_config.safety.max_sample_pressure_gpa:.3f} GPa -- approach is "
                "automatically capped at the new ceiling; set a new target explicitly if you want to "
                "record what you actually intend to reach."
            )
        QMessageBox.information(self, "Configure Parameters", message)

    def _prompt_params_save_path(self) -> Path | None:
        """Repeats the save-file picker until the operator either chooses a
        path other than the one loaded at startup (config/default.yaml is
        git-tracked and shared across deployments -- it must not be silently
        overwritten from the GUI) or cancels."""
        default_path = self._last_params_save_path or (self._config_path.parent / "local.yaml")
        while True:
            chosen, _ = QFileDialog.getSaveFileName(
                self, "Save parameters as", str(default_path), "YAML files (*.yaml *.yml)",
            )
            if not chosen:
                return None
            chosen_path = Path(chosen)
            if chosen_path.resolve() == self._config_path.resolve():
                QMessageBox.warning(
                    self, "Configure Parameters",
                    f"{self._config_path} is the config this session was launched with and must not "
                    "be overwritten from the GUI -- choose a different filename.",
                )
                default_path = self._config_path.parent / "local.yaml"
                continue
            return chosen_path

    def _on_schedule_running_changed(self, running: bool) -> None:
        self.tabs.setTabEnabled(0, not running)
        self.tab1.setEnabled(not running)
        if running:
            self.tabs.setCurrentWidget(self.tab2)

    def _on_snapshot(self, snap: ControllerSnapshot) -> None:
        self.live_plot.add_snapshot(snap)
        self.tab2.on_tick(snap, self._controller)
        self.status_label.setText(_format_status(snap))

    def _on_worker_crashed(self, message: str) -> None:
        QMessageBox.critical(
            self,
            "Controller crashed",
            "The control loop crashed and has been stopped.\n\n"
            f"{message}\n\n"
            "The PACE5000 has been switched out of control mode (STOP) to halt any in-flight ramp. "
            "Restart the application to recover.",
        )

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self.worker.request_stop()
        # Stop the PACE5000 before closing anything: a window close must not
        # leave the membrane ramping toward its last setpoint unattended.
        if not self._controller.stop_and_confirm():
            QMessageBox.warning(
                self, "Could not confirm stop",
                "Could not confirm the PACE5000 stopped before closing -- check it manually.",
            )
        # A tick can still be unwinding bounded HTTP timeouts.  Do not close
        # its requests sessions or CSV handles underneath it; that races the
        # worker and can corrupt the final log or crash Qt with a live QThread.
        self.worker.wait()
        self._ctx.close()
        super().closeEvent(event)
