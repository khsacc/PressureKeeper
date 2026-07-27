from pressurekeeper.config import EstimatorConfig
from pressurekeeper.estimator import PressureEstimator
from pressurekeeper.models import RubyPressureSample


def _cfg(**overrides) -> EstimatorConfig:
    base = dict(
        outlier_median_window=5, smoothing_window_s=2.0, slope_window_s=4.0,
        min_points_for_valid=3, max_sample_age_s=1.0, max_jump_gpa=0.3,
        min_r2=None, min_intensity=None,
    )
    base.update(overrides)
    return EstimatorConfig(**base)


def sample(t, p, fit_success=True, r2=0.98):
    return RubyPressureSample(t_mono=t, t_wall=t, pressure_gpa=p, pressure_err_gpa=0.01,
                               fit_success=fit_success, r2=r2)


def test_outlier_suppressed_by_median_filter():
    est = PressureEstimator(_cfg())
    for i in range(6):
        est.update(sample(i * 0.25, 1.000))
    est.update(sample(6 * 0.25, 5.000))  # single wild outlier
    filtered = est.filtered_pressure()
    assert filtered is not None
    assert abs(filtered - 1.0) < 0.05, f"median filter should suppress the spike, got {filtered}"


def test_slope_is_positive_for_rising_pressure():
    est = PressureEstimator(_cfg())
    for i in range(20):
        est.update(sample(i * 0.25, 1.0 + 0.02 * i))
    slope = est.pressure_slope()
    assert slope is not None and slope > 0
    assert abs(slope - 0.08) < 0.02  # 0.02 GPa / 0.25 s = 0.08 GPa/s


def test_slope_zero_for_flat_pressure():
    est = PressureEstimator(_cfg())
    for i in range(20):
        est.update(sample(i * 0.25, 2.0))
    assert abs(est.pressure_slope()) < 1e-9


def test_invalid_sample_not_usable_counts_as_invalid_and_not_ingested():
    est = PressureEstimator(_cfg())
    est.update(sample(0.0, None, fit_success=False))
    assert est.consecutive_invalid == 1
    assert est.filtered_pressure() is None
    est.update(sample(0.25, 1.0))
    assert est.consecutive_invalid == 0


def test_is_valid_requires_min_points_and_freshness():
    est = PressureEstimator(_cfg(min_points_for_valid=3, max_sample_age_s=1.0))
    assert not est.is_valid(now=0.0)
    est.update(sample(0.0, 1.0))
    est.update(sample(0.25, 1.0))
    assert not est.is_valid(now=0.25)  # only 2 points
    est.update(sample(0.5, 1.0))
    assert est.is_valid(now=0.5)
    assert not est.is_valid(now=2.0)  # stale: > max_sample_age_s later


def test_min_r2_gate_rejects_low_quality_fit():
    est = PressureEstimator(_cfg(min_r2=0.95))
    est.update(sample(0.0, 1.0, r2=0.5))
    assert est.consecutive_invalid == 1
    est.update(sample(0.25, 1.0, r2=0.99))
    assert est.consecutive_invalid == 0


def test_min_r2_gate_fail_closed_on_missing_value():
    # Once a quality threshold is configured, a *missing* r2 must be treated
    # as failing it, not silently accepted -- fail-closed, not fail-open.
    est = PressureEstimator(_cfg(min_r2=0.95))
    est.update(sample(0.0, 1.0, r2=None))
    assert est.consecutive_invalid == 1


def test_min_intensity_gate_fail_closed_on_missing_value():
    est = PressureEstimator(_cfg(min_intensity=100.0))
    s = sample(0.0, 1.0)
    est.update(s)  # intensity defaults to None on the sample() helper
    assert est.consecutive_invalid == 1


def test_non_finite_pressure_rejected():
    est = PressureEstimator(_cfg())
    est.update(sample(0.0, float("nan")))
    assert est.consecutive_invalid == 1
    est.update(sample(0.25, float("inf")))
    assert est.consecutive_invalid == 2
    est.update(sample(0.5, 1.0))
    assert est.consecutive_invalid == 0


def test_predicted_pressure_ignores_negative_slope():
    est = PressureEstimator(_cfg())
    for i in range(20):
        est.update(sample(i * 0.25, 2.0 - 0.02 * i))  # falling
    predicted = est.predicted_pressure(horizon_s=5.0)
    filtered = est.filtered_pressure()
    assert predicted is not None and filtered is not None
    assert abs(predicted - filtered) < 1e-9, "negative slope must not extrapolate the prediction downward or upward"


def test_out_of_order_sample_is_dropped():
    est = PressureEstimator(_cfg())
    est.update(sample(1.0, 1.0))
    est.update(sample(0.5, 9.0))  # arrives "in the past" relative to prior sample
    assert est.consecutive_invalid == 1
    assert abs(est.filtered_pressure() - 1.0) < 1e-9


def test_jump_flag_set_on_large_single_step():
    est = PressureEstimator(_cfg(max_jump_gpa=0.05))
    for i in range(6):
        est.update(sample(i * 0.25, 1.0))
    assert not est.last_jump_flagged
    est.update(sample(6 * 0.25, 1.5))
    assert est.last_jump_flagged
