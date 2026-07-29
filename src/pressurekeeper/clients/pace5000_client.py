"""HTTP client for the PACE5000 control app's API.

Talks to apps/PACE5000/pace5000_api.py in PF_BL18C_control — NOT to the
PACE5000 hardware directly. That app owns the serial link and all
device-safety logic (slew-rate verification, source-pressure checks); this
client is a thin, error-translating JSON/HTTP wrapper, exactly mirroring the
server's contract:

  GET  /api/v1/status         -> {"connected": bool, "pressure_mpa": ..., ...}
  POST /api/v1/pressure       <- {"pressure": float, "unit": "MPa",
                                   "rate": float, "rate_unit": "MPa/min"}
  POST /api/v1/control_mode   <- {"enabled": bool}

All pressures in this client's public interface are MPa, matching both the
remote API and this codebase's internal convention.
"""
from __future__ import annotations

import math
import time

import requests

from ..config import Pace5000ApiConfig
from ..errors import MembraneCommError
from ..models import MembraneStatus

_PREFIX = "/api/v1"


def _coerce_control_mode(value: object) -> bool | None:
    """Normalize whatever the wire sends for control_mode into bool|None.

    The PACE5000 control app is expected to send a real JSON bool, but its
    `get_output_state()` reads back the raw SCPI `:OUTP:STAT?` response,
    which is the string "0"/"1" -- `"0" is False` is never true in Python,
    so an un-coerced string silently reads as "control mode enabled"
    regardless of its actual value. Defend against that here regardless of
    whether the server-side type is ever fixed.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return None
        if float(value) == 1.0:
            return True
        if float(value) == 0.0:
            return False
        return None
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("1", "true", "on", "yes"):
            return True
        if v in ("0", "false", "off", "no"):
            return False
        return None
    return None


class Pace5000Client:
    def __init__(self, config: Pace5000ApiConfig, dry_run: bool = True) -> None:
        self._cfg = config
        self.dry_run = dry_run
        self._session = requests.Session()
        if config.api_key:
            self._session.headers["X-API-Key"] = config.api_key
        self.last_intended_setpoint_mpa: float | None = None

    def _url(self, path: str) -> str:
        return f"{self._cfg.base_url.rstrip('/')}{_PREFIX}{path}"

    def ensure_control_mode(self, enabled: bool = True) -> None:
        """Startup-only arm call, gated by ensure_control_mode_enabled."""
        if not self._cfg.ensure_control_mode_enabled:
            return
        self.set_control_mode(enabled)

    def set_control_mode(self, enabled: bool) -> None:
        if self.dry_run:
            return
        # pause()/abort() deliberately call this from an operator thread while
        # the polling thread may be blocked in read_status()/set_pressure().
        # requests.Session is not specified as thread-safe, so the emergency
        # STOP uses an independent one-shot request rather than sharing the
        # polling/command session.
        headers = {"X-API-Key": self._cfg.api_key} if self._cfg.api_key else None
        try:
            resp = requests.post(
                self._url("/control_mode"),
                json={"enabled": enabled},
                headers=headers,
                timeout=self._cfg.timeout_s,
            )
        except requests.RequestException as e:
            raise MembraneCommError(f"could not reach PACE5000 control app: {e}") from e
        self._parse_post_response(resp, "/control_mode")

    def read_status(self) -> MembraneStatus:
        # Also callable from stop_and_confirm() while the polling thread is
        # unwinding.  Use an independent request for the same thread-safety
        # reason as set_control_mode().
        headers = {"X-API-Key": self._cfg.api_key} if self._cfg.api_key else None
        try:
            resp = requests.get(
                self._url("/status"),
                headers=headers,
                timeout=self._cfg.timeout_s,
            )
        except requests.RequestException as e:
            raise MembraneCommError(f"could not reach PACE5000 control app: {e}") from e

        if resp.status_code != 200:
            raise MembraneCommError(f"PACE5000 API GET /status returned HTTP {resp.status_code}: {resp.text[:200]}")

        try:
            data = resp.json()
        except ValueError as e:
            raise MembraneCommError(f"PACE5000 API returned non-JSON response: {e}") from e
        if not isinstance(data, dict):
            raise MembraneCommError("PACE5000 API returned JSON that is not an object")

        connected = data.get("connected", False)
        if not isinstance(connected, bool):
            raise MembraneCommError("PACE5000 API response field 'connected' is not boolean")

        now = time.monotonic()
        if not connected:
            return MembraneStatus(t_mono=now, connected=False)

        try:
            return MembraneStatus(
                t_mono=now,
                connected=True,
                pressure_mpa=_optional_float(data.get("pressure_mpa")),
                target_pressure_mpa=_optional_float(data.get("target_pressure_mpa")),
                slew_rate_mpa_per_sec=_optional_float(data.get("slew_rate_mpa_per_sec")),
                control_mode=_coerce_control_mode(data.get("control_mode")),
                source_pressure_positive_mpa=_optional_float(data.get("source_pressure_positive_mpa")),
                effort_percent=_optional_float(data.get("effort_percent")),
            )
        except (TypeError, ValueError) as e:
            raise MembraneCommError(f"PACE5000 API status contains a non-numeric field: {e}") from e

    def set_pressure(self, pressure_mpa: float, rate_mpa_per_min: float) -> None:
        self.last_intended_setpoint_mpa = pressure_mpa
        if self.dry_run:
            return
        self._post("/pressure", {
            "pressure": pressure_mpa, "unit": "MPa",
            "rate": rate_mpa_per_min, "rate_unit": "MPa/min",
        })

    def _post(self, path: str, body: dict) -> dict:
        try:
            resp = self._session.post(self._url(path), json=body, timeout=self._cfg.timeout_s)
        except requests.RequestException as e:
            raise MembraneCommError(f"could not reach PACE5000 control app: {e}") from e

        return self._parse_post_response(resp, path)

    @staticmethod
    def _parse_post_response(resp: requests.Response, path: str) -> dict:
        if resp.status_code != 200:
            detail = resp.text[:300]
            try:
                detail = resp.json().get("error", detail)
            except ValueError:
                pass
            raise MembraneCommError(f"PACE5000 API POST {path} failed (HTTP {resp.status_code}): {detail}")

        try:
            data = resp.json()
        except ValueError as e:
            raise MembraneCommError(f"PACE5000 API returned non-JSON response: {e}") from e
        if not isinstance(data, dict):
            raise MembraneCommError("PACE5000 API returned JSON that is not an object")
        return data

    def close(self) -> None:
        self._session.close()

    @property
    def base_url(self) -> str:
        return self._cfg.base_url

    @property
    def api_key(self) -> str | None:
        return self._cfg.api_key

    def update_connection(self, *, base_url: str, api_key: str | None) -> None:
        """Repoint this client at a different host/key at runtime.

        Only for the GUI's "Configure API" dialog, and only meant to be
        called before the control loop has started -- swapping endpoints
        mid-control would point the safety-checked loop at a different
        physical PACE5000 without warning. `set_control_mode()`/
        `read_status()` build their headers fresh from `self._cfg` on every
        call, but `_post()` (used by `set_pressure()`) relies on the
        session's default headers, so those must be refreshed here too.
        """
        self._cfg.base_url = base_url
        self._cfg.api_key = api_key
        if api_key:
            self._session.headers["X-API-Key"] = api_key
        else:
            self._session.headers.pop("X-API-Key", None)

    def update_config(self, config: Pace5000ApiConfig) -> None:
        """Hot-swap timeout/status_poll_interval/default_rate/
        ensure_control_mode_enabled settings.

        Only for the GUI's "Configure Parameters" dialog, and only meant to
        be called before the control loop has started (same reasoning as
        `update_connection`). Callers must pass a config whose
        base_url/api_key already match this client's current live values
        (gui/parameters_config_dialog.py builds its merged config by starting
        from `controller.config`, which shares this same object graph, so
        they always do) -- otherwise this would silently revert a prior
        `update_connection()` call.
        """
        self._cfg = config
        if config.api_key:
            self._session.headers["X-API-Key"] = config.api_key
        else:
            self._session.headers.pop("X-API-Key", None)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("boolean is not a pressure/rate value")
    return float(value)
