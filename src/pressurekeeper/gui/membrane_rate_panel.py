"""Small section, independent of the Single Target / Schedule tabs above it,
for setting the slew rate PressureKeeper commands the PACE5000 to ramp the
membrane (gas) pressure setpoint at with every write.

Distinct from Tab 1's "max compression rate": that caps the resulting
*sample* pressure rise rate (GPa/min, enforced by the controller/safety
layer). This panel instead sets the gas-side ramp rate (MPa/min) sent
verbatim to the PACE5000 alongside each new setpoint. Never locked by
MainWindow while a schedule runs -- both tabs' set_pressure() calls use
whatever rate is current here, so it's safe to retune at any time.
"""
from __future__ import annotations

from PyQt6.QtWidgets import QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ..controller import OneSidedPressureController


class MembraneRatePanel(QWidget):
    def __init__(self, controller: OneSidedPressureController, parent=None) -> None:
        super().__init__(parent)
        self._controller = controller

        self.rate_spin = QDoubleSpinBox()
        self.rate_spin.setDecimals(3)
        self.rate_spin.setRange(0.001, 100.0)
        self.rate_spin.setSingleStep(0.1)
        self.rate_spin.setSuffix(" MPa/min")
        self.rate_spin.setValue(controller.membrane_rate_mpa_per_min)

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self._apply)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)

        row = QHBoxLayout()
        row.addWidget(QLabel("Gas pressure slew rate:"))
        row.addWidget(self.rate_spin)
        row.addWidget(self.apply_btn)
        row.addStretch(1)

        outer = QVBoxLayout(self)
        outer.addLayout(row)
        outer.addWidget(self.status_label)

    def _apply(self) -> None:
        rate = self.rate_spin.value()
        try:
            self._controller.set_membrane_rate_mpa_per_min(rate)
        except ValueError as e:
            self.status_label.setText(f"rejected: {e}")
            return
        self.status_label.setText(f"applied: gas pressure slew rate={rate:.3f} MPa/min")
