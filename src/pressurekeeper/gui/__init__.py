"""PyQt6 GUI for PressureKeeper.

Kept as an optional, self-contained subpackage (`pip install ".[gui]"`) so
the core control stack (config/controller/estimator/safety/gain/scheduler)
stays free of any GUI dependency and importable in headless/CI environments.

Setting PYQTGRAPH_QT_LIB here, before any submodule imports pyqtgraph,
pins pyqtgraph to the PyQt6 binding regardless of what else is on the
machine (PySide, PyQt5, ...).
"""
import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt6")
