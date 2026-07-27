"""Common Live Plot: sample (ruby) pressure and membrane (gas) pressure vs
time, plus sample-vs-membrane pressure, shared by both tabs. The two
vs-time plots are X-linked pyqtgraph plots side by side (different units, so
a shared Y-axis would distort one series) inside a rolling time window --
plotting a full multi-hour run at ~4 Hz point-for-point gets slow, and the
full-resolution data is already on disk in ticks.csv for later analysis. The
third plot re-uses that same rolling window, just axed by membrane pressure
instead of time, to show the sample/membrane response curve.
"""
from __future__ import annotations

from collections import deque

import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from ..models import ControllerSnapshot

_STATE_COLORS = {
    "APPROACH": "#3b82f6",
    "SETTLE": "#f59e0b",
    "HOLD": "#22c55e",
    "PAUSE": "#eab308",
    "ABORT": "#ef4444",
}


class LivePlotWidget(QWidget):
    def __init__(self, history_window_s: float = 1800.0, parent=None) -> None:
        super().__init__(parent)
        self._window_s = history_window_s
        self._t: deque[float] = deque()
        self._sample: deque[float] = deque()
        self._target: deque[float] = deque()
        self._membrane_set: deque[float] = deque()
        self._membrane_act: deque[float] = deque()
        self._t0: float | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        glw = pg.GraphicsLayoutWidget()
        layout.addWidget(glw)

        self.sample_plot = glw.addPlot(row=0, col=0, title="Sample pressure (ruby)")
        self.sample_plot.setLabel("left", "Sample pressure", "GPa")
        self.sample_plot.setLabel("bottom", "elapsed time", "s")
        self.sample_plot.getAxis("left").enableAutoSIPrefix(False)
        self.sample_plot.showGrid(x=True, y=True, alpha=0.2)
        self.sample_plot.addLegend(offset=(10, 10))
        self.sample_curve = self.sample_plot.plot(pen=pg.mkPen("#3b82f6", width=2), name="filtered")
        self.target_curve = self.sample_plot.plot(
            pen=pg.mkPen("#94a3b8", width=1, style=Qt.PenStyle.DashLine), name="target"
        )

        self.membrane_plot = glw.addPlot(row=0, col=1, title="Membrane (gas) pressure")
        self.membrane_plot.setLabel("left", "Gas pressure", "MPa")
        self.membrane_plot.setLabel("bottom", "elapsed time", "s")
        self.membrane_plot.getAxis("left").enableAutoSIPrefix(False)
        self.membrane_plot.showGrid(x=True, y=True, alpha=0.2)
        self.membrane_plot.addLegend(offset=(10, 10))
        self.membrane_plot.setXLink(self.sample_plot)
        self.membrane_set_curve = self.membrane_plot.plot(pen=pg.mkPen("#94a3b8", width=1), name="setpoint")
        self.membrane_act_curve = self.membrane_plot.plot(pen=pg.mkPen("#a855f7", width=2), name="actual")

        self.corr_plot = glw.addPlot(row=0, col=2, title="Sample vs Membrane pressure")
        self.corr_plot.setLabel("left", "Sample pressure", "GPa")
        self.corr_plot.setLabel("bottom", "Membrane pressure", "MPa")
        self.corr_plot.getAxis("left").enableAutoSIPrefix(False)
        self.corr_plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.corr_plot.showGrid(x=True, y=True, alpha=0.2)
        self.corr_trace = self.corr_plot.plot(pen=pg.mkPen("#3b82f6", width=1))
        self.corr_current = pg.ScatterPlotItem(size=10, brush=pg.mkBrush("#ef4444"), pen=None)
        self.corr_plot.addItem(self.corr_current)

    def add_snapshot(self, snap: ControllerSnapshot) -> None:
        if self._t0 is None:
            self._t0 = snap.t_mono
        t = snap.t_mono - self._t0

        self._t.append(t)
        self._sample.append(_or_nan(snap.filtered_pressure_gpa))
        self._target.append(snap.user_target_gpa)
        self._membrane_set.append(_or_nan(snap.membrane_setpoint_mpa))
        self._membrane_act.append(_or_nan(snap.membrane_actual_mpa))

        cutoff = t - self._window_s
        while self._t and self._t[0] < cutoff:
            self._t.popleft()
            self._sample.popleft()
            self._target.popleft()
            self._membrane_set.popleft()
            self._membrane_act.popleft()

        xs = list(self._t)
        sample_ys = list(self._sample)
        membrane_act_xs = list(self._membrane_act)
        self.sample_curve.setData(xs, sample_ys)
        self.target_curve.setData(xs, list(self._target))
        self.membrane_set_curve.setData(xs, list(self._membrane_set))
        self.membrane_act_curve.setData(xs, membrane_act_xs)

        self.corr_trace.setData(membrane_act_xs, sample_ys)
        last_x, last_y = membrane_act_xs[-1], sample_ys[-1]
        if last_x == last_x and last_y == last_y:  # not NaN
            self.corr_current.setData([last_x], [last_y])
        else:
            self.corr_current.setData([], [])

        color = _STATE_COLORS.get(snap.state.value, "#3b82f6")
        self.sample_plot.setTitle(f"Sample pressure (ruby) — <span style='color:{color}'>{snap.state.value}</span>")

    def reset(self) -> None:
        self._t.clear()
        self._sample.clear()
        self._target.clear()
        self._membrane_set.clear()
        self._membrane_act.clear()
        self._t0 = None
        self.corr_trace.setData([], [])
        self.corr_current.setData([], [])


def _or_nan(value: float | None) -> float:
    return value if value is not None else float("nan")
