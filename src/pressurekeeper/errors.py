"""Exception types raised by device I/O layers.

Kept separate from the underlying transport exceptions (requests.*, etc.) so
control/safety code depends only on this small vocabulary and never needs to
know whether a client talks HTTP, a serial port, or a simulator.
"""
from __future__ import annotations


class DeviceCommError(RuntimeError):
    """Base class for any I/O failure talking to an external controller/API."""


class RubyCommError(DeviceCommError):
    """Ruby fluorescence API unreachable, timed out, or returned malformed data."""


class MembraneCommError(DeviceCommError):
    """PACE5000 control app API unreachable, timed out, or rejected a command."""
