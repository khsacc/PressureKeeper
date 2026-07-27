"""Tab 2: build an ordered set-pressure/wait schedule and run it.

This widget owns the step list and the `ScheduleRunner` that executes it; it
never touches the controller's target directly except through the runner
(`ScheduleRunner.on_tick`), so there is exactly one code path that can call
`set_target()` while a schedule is running.
"""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..controller import OneSidedPressureController
from ..models import ControllerSnapshot
from ..scheduler import (
    ScheduleRunner,
    ScheduleRunStatus,
    SetPressureStep,
    WaitStep,
    load_schedule,
    save_schedule,
)

_ACTIVE_ROW_COLOR = QColor("#1d4ed8")
_HEADERS = ["#", "Type", "Target [GPa]", "Duration [min]"]


class ScheduleTab(QWidget):
    # Emitted whenever the schedule starts/stops running, so MainWindow can
    # lock/unlock Tab 1 and the tab bar.
    running_changed = pyqtSignal(bool)

    def __init__(self, controller: OneSidedPressureController, parent=None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._steps: list[SetPressureStep | WaitStep] = []
        self._runner: ScheduleRunner | None = None
        self._max_gpa = controller.config.safety.max_sample_pressure_gpa

        self.table = QTableWidget(0, len(_HEADERS))
        self.table.setHorizontalHeaderLabels(_HEADERS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

        self.target_spin = QDoubleSpinBox()
        self.target_spin.setDecimals(3)
        self.target_spin.setRange(0.0, self._max_gpa)
        self.target_spin.setSingleStep(0.01)
        self.target_spin.setSuffix(" GPa")
        add_target_btn = QPushButton("Add Set-Pressure Step")
        add_target_btn.clicked.connect(self._add_set_pressure_step)

        self.wait_spin = QDoubleSpinBox()
        self.wait_spin.setDecimals(2)
        self.wait_spin.setRange(0.01, 10_000.0)
        self.wait_spin.setSingleStep(1.0)
        self.wait_spin.setSuffix(" min")
        self.wait_spin.setValue(10.0)
        add_wait_btn = QPushButton("Add Wait Step")
        add_wait_btn.clicked.connect(self._add_wait_step)

        add_row = QHBoxLayout()
        add_row.addWidget(self.target_spin)
        add_row.addWidget(add_target_btn)
        add_row.addSpacing(20)
        add_row.addWidget(self.wait_spin)
        add_row.addWidget(add_wait_btn)
        add_row.addStretch(1)

        self.remove_btn = QPushButton("Remove Selected")
        self.remove_btn.clicked.connect(self._remove_selected)
        self.up_btn = QPushButton("Move Up")
        self.up_btn.clicked.connect(lambda: self._move(-1))
        self.down_btn = QPushButton("Move Down")
        self.down_btn.clicked.connect(lambda: self._move(1))
        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self._clear_all)
        self.load_btn = QPushButton("Load...")
        self.load_btn.clicked.connect(self._load)
        self.save_btn = QPushButton("Save...")
        self.save_btn.clicked.connect(self._save)

        edit_row = QHBoxLayout()
        for w in (self.remove_btn, self.up_btn, self.down_btn, self.clear_btn, self.load_btn, self.save_btn):
            edit_row.addWidget(w)
        edit_row.addStretch(1)
        self._edit_widgets = [
            self.remove_btn, self.up_btn, self.down_btn, self.clear_btn, self.load_btn,
            self.target_spin, add_target_btn, self.wait_spin, add_wait_btn,
        ]

        self.start_btn = QPushButton("Start Schedule")
        self.start_btn.clicked.connect(self._start)
        self.stop_btn = QPushButton("Stop Schedule")
        self.stop_btn.clicked.connect(self._stop)
        self.stop_btn.setEnabled(False)
        self.status_label = QLabel("idle")
        self.status_label.setWordWrap(True)

        run_row = QHBoxLayout()
        run_row.addWidget(self.start_btn)
        run_row.addWidget(self.stop_btn)
        run_row.addStretch(1)

        outer = QVBoxLayout(self)
        outer.addWidget(self.table)
        outer.addLayout(add_row)
        outer.addLayout(edit_row)
        outer.addLayout(run_row)
        outer.addWidget(self.status_label)

    # --------------------------------------------------------------- editing

    def _add_set_pressure_step(self) -> None:
        self._steps.append(SetPressureStep(target_gpa=self.target_spin.value()))
        self._refresh_table()

    def _add_wait_step(self) -> None:
        self._steps.append(WaitStep(duration_s=self.wait_spin.value() * 60.0))
        self._refresh_table()

    def _selected_row(self) -> int | None:
        rows = self.table.selectionModel().selectedRows()
        return rows[0].row() if rows else None

    def _remove_selected(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        del self._steps[row]
        self._refresh_table()

    def _move(self, delta: int) -> None:
        row = self._selected_row()
        if row is None:
            return
        new_row = row + delta
        if not (0 <= new_row < len(self._steps)):
            return
        self._steps[row], self._steps[new_row] = self._steps[new_row], self._steps[row]
        self._refresh_table()
        self.table.selectRow(new_row)

    def _clear_all(self) -> None:
        self._steps = []
        self._refresh_table()

    def _load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load schedule", "", "YAML (*.yaml *.yml)")
        if not path:
            return
        try:
            self._steps = load_schedule(path)
        except (OSError, ValueError) as e:
            QMessageBox.warning(self, "Load failed", str(e))
            return
        self._refresh_table()

    def _save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save schedule", "", "YAML (*.yaml *.yml)")
        if not path:
            return
        try:
            save_schedule(path, self._steps)
        except OSError as e:
            QMessageBox.warning(self, "Save failed", str(e))

    def _refresh_table(self) -> None:
        self.table.setRowCount(len(self._steps))
        for i, step in enumerate(self._steps):
            self.table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            if isinstance(step, SetPressureStep):
                self.table.setItem(i, 1, QTableWidgetItem("Set Pressure"))
                self.table.setItem(i, 2, QTableWidgetItem(f"{step.target_gpa:.3f}"))
                self.table.setItem(i, 3, QTableWidgetItem(""))
            else:
                self.table.setItem(i, 1, QTableWidgetItem("Wait"))
                self.table.setItem(i, 2, QTableWidgetItem(""))
                self.table.setItem(i, 3, QTableWidgetItem(f"{step.duration_s / 60.0:.2f}"))

    # ------------------------------------------------------------------ run

    def _set_editing_enabled(self, enabled: bool) -> None:
        for w in self._edit_widgets:
            w.setEnabled(enabled)

    def _start(self) -> None:
        if not self._steps:
            QMessageBox.information(self, "Empty schedule", "Add at least one step before starting.")
            return
        self._runner = ScheduleRunner(list(self._steps))
        self._runner.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._set_editing_enabled(False)
        self.status_label.setText("running")
        self.running_changed.emit(True)

    def _stop(self) -> None:
        if self._runner is not None:
            self._runner.stop("operator stopped schedule")
        self._on_run_finished(ScheduleRunStatus.STOPPED, "operator stopped schedule")

    def _on_run_finished(self, status: ScheduleRunStatus, reason: str | None = None) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._set_editing_enabled(True)
        self._clear_row_highlight()
        text = status.value.lower()
        if reason:
            text += f" ({reason})"
        self.status_label.setText(text)
        self.running_changed.emit(False)

    def _clear_row_highlight(self) -> None:
        for r in range(self.table.rowCount()):
            for c in range(self.table.columnCount()):
                item = self.table.item(r, c)
                if item is not None:
                    item.setBackground(QColor(0, 0, 0, 0))

    # ---------------------------------------------------------------- ticks

    def on_tick(self, snap: ControllerSnapshot, controller: OneSidedPressureController) -> None:
        if self._runner is None or self._runner.status != ScheduleRunStatus.RUNNING:
            return
        sched = self._runner.on_tick(snap, controller)

        self._clear_row_highlight()
        if sched.step_index is not None and sched.step_index < self.table.rowCount():
            for c in range(self.table.columnCount()):
                item = self.table.item(sched.step_index, c)
                if item is not None:
                    item.setBackground(_ACTIVE_ROW_COLOR)

        if isinstance(sched.step, WaitStep):
            remaining = max(0.0, sched.step.duration_s - sched.active_elapsed_s)
            self.status_label.setText(f"step {sched.step_index + 1}/{len(self._steps)}: waiting, {remaining:.0f}s left")
        elif isinstance(sched.step, SetPressureStep):
            self.status_label.setText(
                f"step {sched.step_index + 1}/{len(self._steps)}: approaching {sched.step.target_gpa:.3f} GPa"
            )

        if sched.status != ScheduleRunStatus.RUNNING:
            self._on_run_finished(sched.status, sched.reason)
