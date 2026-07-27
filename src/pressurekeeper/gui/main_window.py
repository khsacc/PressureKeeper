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

from PyQt6.QtWidgets import (
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
from ..models import ControllerSnapshot
from .api_config_dialog import ApiConfigDialog
from .live_plot import LivePlotWidget
from .membrane_rate_panel import MembraneRatePanel
from .tab_schedule import ScheduleTab
from .tab_single_target import SingleTargetTab
from .worker import ControllerWorker


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
    def __init__(self, ctx: AppContext, poll_interval_s: float, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("PressureKeeper")
        self._ctx = ctx
        self._controller = ctx.controller

        self.configure_api_action = self.menuBar().addAction("Configure API")
        self.configure_api_action.triggered.connect(self._on_configure_api)
        if not (isinstance(ctx.ruby, RubyPressureClient) and isinstance(ctx.membrane, Pace5000Client)):
            # --sim mode: ctx.ruby/ctx.membrane are simulator objects with no
            # host/port/key to configure.
            self.configure_api_action.setEnabled(False)
            self.configure_api_action.setToolTip("Not available in simulator mode")

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

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(lambda: self._controller.pause("operator GUI pause"))
        self.resume_btn = QPushButton("Resume")
        self.resume_btn.clicked.connect(self._controller.resume)
        self.abort_btn = QPushButton("ABORT")
        self.abort_btn.setStyleSheet("background-color:#ef4444; color:white; font-weight:bold;")
        self.abort_btn.clicked.connect(lambda: self._controller.abort("operator GUI abort"))
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self._controller.reset)
        for w in (self.pause_btn, self.resume_btn, self.abort_btn, self.reset_btn):
            w.setEnabled(False)

        initial_status = 'not started — press "Start Control" to begin the control loop'
        if self._controller.logging_error:
            initial_status += f" — LOGGING ERROR: {self._controller.logging_error}"
        self.status_label = QLabel(initial_status)
        self.status_label.setWordWrap(True)

        controls_row = QHBoxLayout()
        for w in (self.start_btn, self.pause_btn, self.resume_btn, self.abort_btn, self.reset_btn):
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

        self.worker = ControllerWorker(self._controller, poll_interval_s)
        self.worker.snapshot_ready.connect(self._on_snapshot)
        self.worker.crashed.connect(self._on_worker_crashed)
        # Not started here — see _on_start_clicked.

    def _on_start_clicked(self) -> None:
        self.start_btn.setEnabled(False)
        self.start_btn.setText("Running")
        for w in (self.pause_btn, self.resume_btn, self.abort_btn, self.reset_btn):
            w.setEnabled(True)
        # Repointing a client at a different device mid-control would do so
        # without the safety supervisor ever having characterized it.
        self.configure_api_action.setEnabled(False)
        self.status_label.setText("control loop starting...")
        self.worker.start()

    def _on_configure_api(self) -> None:
        dialog = ApiConfigDialog(self._ctx.ruby, self._ctx.membrane, parent=self)
        dialog.exec()

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
