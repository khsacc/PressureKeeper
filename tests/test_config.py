"""Config-load-time contract/validation tests.

Catches the kind of bug that otherwise only shows up as a 422 from the real
ruby API on the very first live acquisition, or as a config file that
silently accepts a NaN/Inf safety limit.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from pressurekeeper.clients.ruby_client import RubyPressureClient
from pressurekeeper.config import (
    ControlConfig,
    GainEstimationConfig,
    RubyAcquisitionConfig,
    RubyApiConfig,
    SafetyConfig,
    load_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"


def test_default_yaml_loads_cleanly():
    config = load_config(DEFAULT_CONFIG_PATH)
    assert config.ruby_api.acquisition.configuration_id is None
    assert config.ruby_api.acquisition.axis_mode is None
    assert config.approach.max_compression_rate_gpa_per_min == 0.5
    assert config.gain_estimation.step_sizing_mode == "adaptive_local"
    assert config.gain_estimation.interrupted_rate_learning_mode == "enforce"
    assert config.gain_estimation.interrupted_rate_safety_factor == 1.25
    assert config.gain_estimation.adaptive_probe_max_expected_gain == 1.0
    assert config.gain_estimation.adaptive_no_response_wait_s == 30.0
    assert config.safety.setpoint_mismatch_grace_s == 0.0
    # Never a real secret committed to git (see CLAUDE.md); --sim needs no
    # config at all, so this is a harmless literal placeholder rather than
    # a "${...}" that load_config() would insist on resolving.
    assert config.ruby_api.api_key == "REPLACE_ME_SITE_SPECIFIC_RUBY_API_KEY"
    assert config.pace5000_api.api_key is None


def test_interrupted_rate_learning_config_rejects_unsafe_or_unknown_values():
    GainEstimationConfig(interrupted_rate_learning_mode="enforce")
    with pytest.raises(ValidationError):
        GainEstimationConfig(interrupted_rate_learning_mode="automatic")
    with pytest.raises(ValidationError):
        GainEstimationConfig(interrupted_rate_safety_factor=0.99)


def test_adaptive_probe_configuration_is_self_consistent(tmp_path):
    from tests.helpers import make_config

    base = make_config(tmp_path)
    with pytest.raises(ValidationError, match="initial_probe_step_mpa"):
        type(base).model_validate({
            **base.model_dump(),
            "gain_estimation": {
                **base.gain_estimation.model_dump(),
                "step_sizing_mode": "adaptive_local",
                "initial_probe_step_mpa": 0.3,
                "max_probe_step_mpa": 0.2,
            },
        })


def test_env_var_placeholder_is_expanded(tmp_path, monkeypatch):
    from tests.helpers import make_config
    base = make_config(tmp_path)
    raw = base.model_dump()
    raw["ruby_api"]["api_key"] = "${TEST_PK_RUBY_KEY}"
    monkeypatch.setenv("TEST_PK_RUBY_KEY", "secret-value-from-env")
    config = type(base).model_validate(_expand_env_vars_for_test(raw))
    assert config.ruby_api.api_key == "secret-value-from-env"


def test_env_var_placeholder_raises_clearly_when_unset(tmp_path, monkeypatch):
    from tests.helpers import make_config
    base = make_config(tmp_path)
    raw = base.model_dump()
    raw["ruby_api"]["api_key"] = "${TEST_PK_UNSET_VAR}"
    monkeypatch.delenv("TEST_PK_UNSET_VAR", raising=False)
    with pytest.raises(ValueError, match="TEST_PK_UNSET_VAR"):
        _expand_env_vars_for_test(raw)


def test_partial_env_var_syntax_within_a_larger_string_is_left_alone(tmp_path):
    from tests.helpers import make_config
    base = make_config(tmp_path)
    raw = base.model_dump()
    raw["ruby_api"]["api_key"] = "prefix-${NOT_A_WHOLE_MATCH}-suffix"
    expanded = _expand_env_vars_for_test(raw)
    assert expanded["ruby_api"]["api_key"] == "prefix-${NOT_A_WHOLE_MATCH}-suffix"


def _expand_env_vars_for_test(raw: dict) -> dict:
    from pressurekeeper.config import _expand_env_vars
    return _expand_env_vars(raw)


def test_redact_api_keys_replaces_expanded_secrets_with_env_var_placeholders(tmp_path):
    from pressurekeeper.config import redact_api_keys
    from tests.helpers import make_config

    config = make_config(tmp_path)
    dumped = config.model_dump()
    assert dumped["ruby_api"]["api_key"] == "test-key"  # sanity: the literal secret is really there

    redacted = redact_api_keys(dumped)
    assert redacted["ruby_api"]["api_key"] == "${PRESSUREKEEPER_RUBY_API_KEY}"
    assert "test-key" not in str(redacted), "no literal secret may survive redaction"
    # Unmodified elsewhere, and the original dict passed in must not be mutated.
    assert redacted["safety"]["max_sample_pressure_gpa"] == dumped["safety"]["max_sample_pressure_gpa"]
    assert dumped["ruby_api"]["api_key"] == "test-key"


def test_redact_api_keys_leaves_a_null_pace5000_key_alone():
    from pressurekeeper.config import redact_api_keys

    dumped = {"ruby_api": {"api_key": "secret"}, "pace5000_api": {"api_key": None}}
    redacted = redact_api_keys(dumped)
    assert redacted["pace5000_api"]["api_key"] is None
    assert redacted["ruby_api"]["api_key"] == "${PRESSUREKEEPER_RUBY_API_KEY}"


def test_redact_api_keys_replaces_a_set_pace5000_key():
    from pressurekeeper.config import redact_api_keys

    dumped = {"ruby_api": {"api_key": "secret"}, "pace5000_api": {"api_key": "pace-secret"}}
    redacted = redact_api_keys(dumped)
    assert redacted["pace5000_api"]["api_key"] == "${PRESSUREKEEPER_PACE5000_API_KEY}"


def test_axis_mode_requires_explicit_configuration_id():
    RubyAcquisitionConfig(configuration_id="cfg-1", axis_mode="calibrated")  # ok
    with pytest.raises(ValidationError):
        RubyAcquisitionConfig(configuration_id=None, axis_mode="calibrated")


def test_ruby_client_body_never_violates_axis_mode_contract():
    cfg = RubyApiConfig(api_key="test-key")  # defaults: configuration_id=None, axis_mode=None
    body = RubyPressureClient._build_body(cfg)
    assert body["configuration_id"] is None
    assert body["axis_mode"] is None


def test_fluorapressee_schema_contract_if_sibling_repo_present():
    """Stronger drift protection: if lab_andor/FluoraPressee is checked out
    as a sibling repo (as it is in this lab's layout), validate our request
    body against its *real* pydantic schema instead of just our own mirror
    of the rule. Skips cleanly wherever that sibling repo isn't available
    (e.g. a bare CI checkout of just this repo)."""
    fluorapressee_root = REPO_ROOT.parent / "lab_andor" / "FluoraPressee"
    schemas_path = fluorapressee_root / "src" / "api" / "schemas.py"
    if not schemas_path.is_file():
        pytest.skip("lab_andor/FluoraPressee not present as a sibling checkout")

    import importlib.util
    import sys

    src_root = fluorapressee_root / "src"
    added = str(fluorapressee_root) not in sys.path
    if added:
        sys.path.insert(0, str(fluorapressee_root))
    try:
        spec = importlib.util.spec_from_file_location("_fluorapressee_schemas", schemas_path)
        if spec is None or spec.loader is None:
            pytest.skip("could not load FluoraPressee's schemas module")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            pytest.skip(f"FluoraPressee's schemas module could not be imported (missing deps?): {e}")
    finally:
        if added:
            sys.path.remove(str(fluorapressee_root))

    cfg = RubyApiConfig(api_key="test-key")
    body = RubyPressureClient._build_body(cfg)
    module.AcquireFitRequest.model_validate(body)  # must not raise


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_safety_config_rejects_non_finite_limits(bad):
    with pytest.raises(ValidationError):
        SafetyConfig(max_sample_pressure_gpa=bad)


def test_safety_config_rejects_non_finite_secondary_fields():
    with pytest.raises(ValidationError):
        SafetyConfig(max_sample_pressure_gpa=5.0, max_membrane_pressure_mpa=float("nan"))


def test_control_config_rejects_non_finite_default_target():
    with pytest.raises(ValidationError):
        ControlConfig(default_target_pressure_gpa=float("nan"))


def test_default_target_above_absolute_limit_rejected(tmp_path):
    from tests.helpers import make_config
    base = make_config(tmp_path, max_sample_pressure_gpa=1.0)
    # model_copy() doesn't re-run validators -- round-trip through
    # model_validate() to actually exercise Configuration's cross-field
    # check on a default target above the absolute sample-pressure limit.
    raw = base.model_dump()
    raw["control"]["default_target_pressure_gpa"] = 999.0
    with pytest.raises(ValidationError):
        type(base).model_validate(raw)


def test_active_setpoint_gap_limit_cannot_be_below_legal_step(tmp_path):
    from tests.helpers import make_config
    base = make_config(tmp_path)
    raw = base.model_dump()
    raw["safety"]["max_setpoint_actual_gap_mpa"] = (
        raw["safety"]["max_membrane_step_mpa_hard"] / 2
    )
    with pytest.raises(ValidationError):
        type(base).model_validate(raw)


def test_gain_region_rejects_non_finite_fields():
    from pressurekeeper.models import GainRegion
    with pytest.raises(ValueError):
        GainRegion(
            sample_pressure_min_gpa=0.0, sample_pressure_max_gpa=0.5,
            safe_gain=math.nan, max_sample_step_gpa=0.1,
            max_membrane_step=0.5, minimum_settle_time_s=5.0,
            settled_slope_threshold_gpa_s=0.01,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dark_mode": "provided"},
        {"fit_peak_count": 1, "pressure_peak_index": 2},
        {"fit_function": "Diamond Raman Edge", "fit_peak_count": 2},
        {"fit_range": (700.0, 690.0)},
        {"exposure_time_s": 0.0},
    ],
)
def test_ruby_acquisition_rejects_requests_the_server_cannot_accept(kwargs):
    with pytest.raises(ValidationError):
        RubyAcquisitionConfig(**kwargs)


@pytest.mark.parametrize(
    ("model", "kwargs"),
    [
        (RubyApiConfig, {"api_key": "key", "timeout_s": 0.0}),
        (ControlConfig, {"loop_min_interval_s": 0.0}),
        (SafetyConfig, {"max_sample_pressure_gpa": -1.0}),
        (SafetyConfig, {"max_sample_pressure_gpa": 5.0, "max_consecutive_invalid": 0}),
    ],
)
def test_runtime_interval_and_count_fields_reject_non_positive_values(model, kwargs):
    with pytest.raises(ValidationError):
        model(**kwargs)
