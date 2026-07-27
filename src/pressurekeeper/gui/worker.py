"""Background QThread that repeatedly calls controller.step() and hands each
resulting snapshot back to the GUI thread via a Qt signal.

Mirrors cli.py's `_ControllerRunner` (same crash-handling: on an unexpected
exception, abort the controller once, log/emit a final snapshot, then stop)
but Qt-native so cross-thread delivery to widgets is safe by construction.
`controller.step()` does blocking HTTP I/O (ruby + PACE5000 reads, up to
several seconds on timeout), so it must never run on the GUI thread.
"""
from __future__ import annotations

import threading
import time

from PyQt6.QtCore import QThread, pyqtSignal

from ..controller import OneSidedPressureController
from ..models import ControllerSnapshot


class ControllerWorker(QThread):
    snapshot_ready = pyqtSignal(object)  # ControllerSnapshot
    crashed = pyqtSignal(str)

    def __init__(self, controller: OneSidedPressureController, poll_interval_s: float, parent=None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._poll_interval_s = poll_interval_s
        self._stop_requested = threading.Event()

    def request_stop(self) -> None:
        self._stop_requested.set()

    def run(self) -> None:
        while not self._stop_requested.is_set():
            t0 = time.monotonic()
            try:
                snap: ControllerSnapshot | None = self._controller.step()
            except Exception as e:  # fail safe: latch abort, emit once more, stop the loop
                snap = None
                try:
                    self._controller.abort(f"controller loop crashed: {e!r}")
                    snap = self._controller.step()
                except Exception:
                    pass
                if snap is not None:
                    self.snapshot_ready.emit(snap)
                self.crashed.emit(repr(e))
                return
            self.snapshot_ready.emit(snap)
            elapsed = time.monotonic() - t0
            remaining_ms = int((self._poll_interval_s - elapsed) * 1000)
            if remaining_ms > 0:
                self._stop_requested.wait(remaining_ms / 1000.0)
