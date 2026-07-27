from pressurekeeper.clients.pace5000_client import _coerce_control_mode


def test_coerce_control_mode_bool_passthrough():
    assert _coerce_control_mode(True) is True
    assert _coerce_control_mode(False) is False


def test_coerce_control_mode_int():
    assert _coerce_control_mode(1) is True
    assert _coerce_control_mode(0) is False


def test_coerce_control_mode_scpi_string():
    # The real bug this guards against: PF_BL18C_control's get_output_state()
    # historically returned the raw SCPI ":OUTP:STAT?" response ("0"/"1")
    # instead of a JSON bool -- "0" is never `is False` in Python, so an
    # un-coerced string silently read as "control mode enabled".
    assert _coerce_control_mode("1") is True
    assert _coerce_control_mode("0") is False
    assert _coerce_control_mode("true") is True
    assert _coerce_control_mode("false") is False


def test_coerce_control_mode_unrecognized_or_missing():
    assert _coerce_control_mode(None) is None
    assert _coerce_control_mode("unknown") is None
    assert _coerce_control_mode([]) is None
    assert _coerce_control_mode(2) is None
    assert _coerce_control_mode(float("nan")) is None
