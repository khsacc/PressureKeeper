"""Tab 1: a single sample-pressure target plus an operator-adjustable cap on
how fast the sample pressure is allowed to rise.

Locked (via `setEnabled(False)`) by MainWindow whenever Tab 2's schedule is
RUNNING; a manual target change during a schedule must go through explicitly
stopping the schedule first, so the two can never fight over set_target().
"""
from __future__ import annotations

from PyQt6.QtWidgets import QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ..controller import OneSidedPressureController


class SingleTargetTab(QWidget):
    def __init__(self, controller: OneSidedPressureController, parent=None) -> None:
        super().__init__(parent)
        self._controller = controller
        max_gpa = controller.config.safety.max_sample_pressure_gpa

        self.target_spin = QDoubleSpinBox()
        self.target_spin.setDecimals(3)
        self.target_spin.setRange(0.0, max_gpa)
        self.target_spin.setSingleStep(0.01)
        self.target_spin.setSuffix(" GPa")
        self.target_spin.setValue(controller.user_target_gpa)

        self.rate_spin = QDoubleSpinBox()
        self.rate_spin.setDecimals(4)
        self.rate_spin.setRange(0.0, 50.0)
        self.rate_spin.setSingleStep(0.001)
        self.rate_spin.setSuffix(" GPa/min")
        self.rate_spin.setSpecialValueText("no limit")
        current_rate = controller.max_compression_rate_gpa_per_min
        self.rate_spin.setValue(current_rate if current_rate is not None else 0.0)

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self._apply)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)

        form = QFormLayout()
        form.addRow("Target sample pressure:", self.target_spin)
        form.addRow("Max compression rate:", self.rate_spin)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.apply_btn)
        btn_row.addStretch(1)

        outer = QVBoxLayout(self)
        outer.addLayout(form)
        outer.addLayout(btn_row)
        outer.addWidget(self.status_label)
        outer.addStretch(1)

    def _apply(self) -> None:
        target = self.target_spin.value()
        rate = self.rate_spin.value()
        rate_or_none = None if rate <= 0.0 else rate
        try:
            self._controller.set_target(target)
            self._controller.set_max_compression_rate(rate_or_none)
        except ValueError as e:
            self.status_label.setText(f"rejected: {e}")
            return
        rate_text = "no limit" if rate_or_none is None else f"{rate_or_none:.4f} GPa/min"
        self.status_label.setText(f"applied: target={target:.3f} GPa, max compression rate={rate_text}")
