"""Configure API dialog: lets an operator edit the host/IP, port, and API
key for the two HTTP APIs (FluoraPressee ruby API, PACE5000 API) from the
GUI instead of hand-editing the config file.

This module only collects and validates form input (host/port/key for both
APIs, exposed as `ruby_url`/`ruby_api_key`/`pace_url`/`pace_api_key` once
accepted) -- it never mutates the live client objects or writes back to the
on-disk config file, so config/default.yaml's SITE-SPECIFIC comments and
values are untouched. All the policy lives in main_window.py's
`_on_configure_api()`: applying `RubyPressureClient.update_connection()` /
`Pace5000Client.update_connection()`, and -- for the PACE5000 endpoint
specifically -- re-acquiring the single-instance lock against the new
endpoint before switching (see instance_lock.py) so a changed connection is
still guarded against a second process. MainWindow only offers this dialog
while stopped (before the first "Start Control", or after "Stop Control") --
swapping endpoints mid-control would point the safety-checked loop at a
different physical device without warning.
"""
from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QMessageBox,
    QSpinBox,
    QVBoxLayout,
)

from ..clients import Pace5000Client, RubyPressureClient


def _split_host_port(base_url: str) -> tuple[str, int]:
    parts = urlsplit(base_url)
    host = parts.hostname or "127.0.0.1"
    port = parts.port or (443 if parts.scheme == "https" else 80)
    return host, port


def _join_host_port(base_url: str, host: str, port: int) -> str:
    parts = urlsplit(base_url)
    scheme = parts.scheme or "http"
    return urlunsplit((scheme, f"{host}:{port}", parts.path, parts.query, parts.fragment))


class _ApiFieldsGroup(QGroupBox):
    def __init__(self, title: str, base_url: str, api_key: str, parent=None) -> None:
        super().__init__(title, parent)
        host, port = _split_host_port(base_url)

        self.host_edit = QLineEdit(host)
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(port)
        self.key_edit = QLineEdit(api_key)
        self.key_edit.setEchoMode(QLineEdit.EchoMode.Password)

        form = QFormLayout(self)
        form.addRow("Host / IP:", self.host_edit)
        form.addRow("Port:", self.port_spin)
        form.addRow("API Key:", self.key_edit)

    def host(self) -> str:
        return self.host_edit.text().strip()

    def port(self) -> int:
        return self.port_spin.value()

    def api_key(self) -> str:
        return self.key_edit.text()


class ApiConfigDialog(QDialog):
    def __init__(self, ruby_client: RubyPressureClient, pace_client: Pace5000Client, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Configure API")
        self._ruby_client = ruby_client
        self._pace_client = pace_client

        self.ruby_group = _ApiFieldsGroup(
            "Ruby fluorescence API (FluoraPressee)",
            ruby_client.base_url, ruby_client.api_key,
        )
        self.pace_group = _ApiFieldsGroup(
            "PACE5000 API",
            pace_client.base_url, pace_client.api_key or "",
        )

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.ruby_group)
        layout.addWidget(self.pace_group)
        layout.addWidget(buttons)

    def _on_save(self) -> None:
        if not self.ruby_group.host():
            QMessageBox.warning(self, "Configure API", "Ruby API host must not be empty.")
            return
        if not self.pace_group.host():
            QMessageBox.warning(self, "Configure API", "PACE5000 API host must not be empty.")
            return
        if not self.ruby_group.api_key():
            QMessageBox.warning(self, "Configure API", "Ruby API key must not be empty.")
            return

        self.ruby_url = _join_host_port(self._ruby_client.base_url, self.ruby_group.host(), self.ruby_group.port())
        self.ruby_api_key = self.ruby_group.api_key()
        self.pace_url = _join_host_port(self._pace_client.base_url, self.pace_group.host(), self.pace_group.port())
        self.pace_api_key = self.pace_group.api_key() or None

        self.accept()
