"""HTTP client for the FluoraPressee ruby-fluorescence API.

Talks to lab_andor/FluoraPressee's FastAPI server (typically running on a
different PC than this control app). Each call to `read()` triggers one
fresh synchronous acquisition via `POST /acquire/pressure` — the remote API
has no "just give me the latest value" endpoint, so the achievable poll rate
is bounded by `exposure_time_s * accumulations` plus fit/network overhead
(the ~4 Hz ceiling assumed throughout this codebase).
"""
from __future__ import annotations

import time

import requests

from ..config import RubyApiConfig
from ..errors import RubyCommError
from ..models import RubyPressureSample


class RubyPressureClient:
    def __init__(self, config: RubyApiConfig) -> None:
        self._cfg = config
        self._session = requests.Session()
        self._session.headers["X-API-Key"] = config.api_key
        self._body = self._build_body(config)
        self._timeout_s = self._effective_timeout(config)

    @staticmethod
    def _build_body(config: RubyApiConfig) -> dict:
        acq = config.acquisition
        body: dict = {
            "configuration_id": acq.configuration_id,
            "axis_mode": acq.axis_mode,
            "exposure_time_s": acq.exposure_time_s,
            "accumulations": acq.accumulations,
            "dark": {"mode": acq.dark_mode, "data": None, "ignore_mismatch": False},
            "fit_function": acq.fit_function,
            "fit_peak_count": acq.fit_peak_count,
            "peak_sort_order": acq.peak_sort_order,
            "baseline_model": acq.baseline_model,
            "sensor": acq.sensor,
            "pressure_scale": acq.pressure_scale,
            "zero_pressure_peak": acq.zero_pressure_peak,
            "pressure_peak_index": acq.pressure_peak_index,
        }
        if acq.fit_range is not None:
            body["fit_range"] = {"start": acq.fit_range[0], "end": acq.fit_range[1]}
        return body

    @staticmethod
    def _effective_timeout(config: RubyApiConfig) -> float:
        """Read timeout for one /acquire/pressure round trip.

        `exposure_time_s * accumulations` is only the camera's dwell time --
        FluoraPressee's handler also pays for per-frame readout dead time, the
        peak fit itself, and however long its Qt event loop takes to service
        the acquisition on the GUI thread (see that repo's `gui_bridge.py`,
        which allows up to 60 s per acquisition as an outer safety valve
        against a stuck GUI thread, not a typical turnaround). A too-tight
        budget here makes a slow-but-working acquisition indistinguishable
        from a dead link, tripping `ruby_api_unreachable` even though
        FluoraPressee genuinely completed the measurement.

        Deliberately not stretched all the way to FluoraPressee's 60 s
        ceiling: a single stuck read would then block the control loop for a
        full minute, at odds with this app's own `max_stale_sample_s` /
        `ruby_error_pause_after_s` design intent (single-digit seconds). If a
        site's real readout/fit overhead exceeds this margin, raise
        `ruby_api.timeout_s` directly rather than stretching this further.
        """
        acq = config.acquisition
        accumulations = acq.accumulations or 1
        exposure_budget = (acq.exposure_time_s or 0.0) * accumulations
        per_frame_readout_margin = 0.5 * accumulations
        fit_and_scheduling_margin = 5.0
        return max(config.timeout_s, exposure_budget + per_frame_readout_margin + fit_and_scheduling_margin)

    def read(self) -> RubyPressureSample:
        try:
            resp = self._session.post(
                f"{self._cfg.base_url.rstrip('/')}/acquire/pressure",
                json=self._body,
                timeout=self._timeout_s,
            )
        except requests.RequestException as e:
            raise RubyCommError(f"could not reach ruby fluorescence API: {e}") from e

        # Receipt timestamps must describe the measurement we just received,
        # not when a potentially long exposure/request began.
        t_mono = time.monotonic()
        t_wall = time.time()

        if resp.status_code == 409:
            raise RubyCommError("ruby fluorescence API busy (concurrent acquisition in progress)")
        if resp.status_code != 200:
            detail = resp.text[:300]
            try:
                detail = resp.json().get("detail", detail)
            except ValueError:
                pass
            raise RubyCommError(f"ruby fluorescence API returned HTTP {resp.status_code}: {detail}")

        try:
            data = resp.json()
        except ValueError as e:
            raise RubyCommError(f"ruby fluorescence API returned non-JSON response: {e}") from e
        if not isinstance(data, dict):
            raise RubyCommError("ruby fluorescence API returned JSON that is not an object")

        fit = data.get("fit") or {}
        if not isinstance(fit, dict):
            raise RubyCommError("ruby fluorescence API response field 'fit' is not an object")
        fit_success_raw = fit.get("success", False)
        if not isinstance(fit_success_raw, bool):
            raise RubyCommError("ruby fluorescence API response field 'fit.success' is not boolean")
        fit_success = fit_success_raw
        fit_detail = fit.get("fit") or {}
        try:
            pressure_gpa = _optional_float(data.get("pressure_gpa"))
            pressure_err_gpa = _optional_float(data.get("pressure_err_gpa"))
            r2 = _optional_float(fit_detail.get("R2")) if isinstance(fit_detail, dict) else None
            intensity = self._pressure_peak_intensity(fit_detail)
        except (TypeError, ValueError) as e:
            raise RubyCommError(f"ruby fluorescence API response contains a non-numeric field: {e}") from e

        return RubyPressureSample(
            t_mono=t_mono,
            t_wall=t_wall,
            pressure_gpa=pressure_gpa,
            pressure_err_gpa=pressure_err_gpa,
            fit_success=fit_success,
            r2=r2,
            intensity=intensity,
            raw=data,
        )

    def _pressure_peak_intensity(self, fit_detail: object) -> float | None:
        """Intensity of the same peak the server used for pressure_gpa.

        Mirrors FluoraPressee's own `peaks[pressure_peak_index - 1]` lookup
        (see its `/acquire/pressure` handler) so `estimator.min_intensity`
        gates on the pressure-determining peak, not just any peak in the fit.
        """
        if not isinstance(fit_detail, dict):
            return None
        peaks = fit_detail.get("peaks")
        if not isinstance(peaks, list):
            return None
        idx = self._cfg.acquisition.pressure_peak_index - 1
        if idx < 0 or idx >= len(peaks):
            return None
        peak = peaks[idx]
        if not isinstance(peak, dict):
            return None
        intensity = peak.get("intensity")
        return float(intensity) if isinstance(intensity, (int, float)) else None

    def close(self) -> None:
        self._session.close()

    @property
    def base_url(self) -> str:
        return self._cfg.base_url

    @property
    def api_key(self) -> str:
        return self._cfg.api_key

    def update_connection(self, *, base_url: str, api_key: str) -> None:
        """Repoint this client at a different host/key at runtime.

        Only for the GUI's "Configure API" dialog, and only meant to be
        called before the control loop has started -- swapping endpoints
        mid-control would point the safety-checked loop at a different
        physical spectrometer PC without warning.
        """
        self._cfg.base_url = base_url
        self._cfg.api_key = api_key
        self._session.headers["X-API-Key"] = api_key

    def update_config(self, config: RubyApiConfig) -> None:
        """Hot-swap timeout/poll_interval/acquisition settings.

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
        self._session.headers["X-API-Key"] = config.api_key
        self._body = self._build_body(config)
        self._timeout_s = self._effective_timeout(config)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("boolean is not a numeric measurement")
    return float(value)
