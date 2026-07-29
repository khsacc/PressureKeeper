from pressurekeeper.config import GainEstimationConfig
from pressurekeeper.gain import GainEstimator
from pressurekeeper.models import GainRegion, StepRecord


def make_region(safe_gain=0.2, rate_limit_gain=None):
    return GainRegion(0.0, 5.0, safe_gain=safe_gain, max_sample_step_gpa=0.1,
                       max_membrane_step=1.0, minimum_settle_time_s=1.0, settled_slope_threshold_gpa_s=0.01,
                       rate_limit_gain=rate_limit_gain)


def settled_step(step_id, before_mpa, after_mpa, before_gpa, after_gpa):
    step = StepRecord(step_id=step_id, t_command=0.0, membrane_pressure_before=before_mpa,
                       membrane_pressure_after=after_mpa, sample_pressure_before=before_gpa,
                       reason="test")
    step.settled = True
    step.sample_pressure_after = after_gpa
    step.t_settled = 10.0
    return step


def test_falls_back_to_prior_when_no_data():
    est = GainEstimator(GainEstimationConfig(bin_width_gpa=0.5, min_samples_for_estimate=3,
                                              safety_factor=1.0, upper_percentile=90.0, neighbor_bins=1))
    region = make_region(safe_gain=0.25)
    result = est.estimate(1.0, region)
    assert result.source == "prior"
    assert result.safe_gain == 0.25
    assert result.n_samples == 0


def test_observed_gain_used_once_enough_samples_and_is_conservative():
    est = GainEstimator(GainEstimationConfig(bin_width_gpa=1.0, min_samples_for_estimate=3,
                                              safety_factor=1.0, upper_percentile=90.0, neighbor_bins=0))
    region = make_region(safe_gain=0.10)
    # Observed gains all near 0.30 GPa/MPa, well above the conservative prior.
    for i, gain in enumerate([0.28, 0.30, 0.32]):
        step = settled_step(i, 1.0, 2.0, 0.5, 0.5 + gain)
        est.record_step(step)

    result = est.estimate(0.5, region)
    assert result.source == "observed"
    assert result.n_samples == 3
    # safe_gain must be a conservative (>=) estimate of the true sensitivity:
    # at least the observed median, and comfortably above the too-low prior.
    assert result.safe_gain >= result.estimated_gain
    assert abs(result.estimated_gain - 0.30) < 1e-9
    assert result.safe_gain > region.safe_gain


def test_bins_by_midpoint_pressure_prefer_nearby_data():
    est = GainEstimator(GainEstimationConfig(bin_width_gpa=0.5, min_samples_for_estimate=2,
                                              safety_factor=1.0, upper_percentile=90.0, neighbor_bins=0))
    region = make_region(safe_gain=0.10)
    # Low-pressure-band steps (small gain) and high-pressure-band steps (large gain).
    for i in range(3):
        est.record_step(settled_step(i, 0.0, 1.0, 0.0, 0.10))       # midpoint ~0.05 -> low gain 0.10
    for i in range(3, 6):
        est.record_step(settled_step(i, 5.0, 6.0, 2.0, 2.60))       # midpoint ~2.3 -> high gain 0.60

    low = est.estimate(0.1, region)
    high = est.estimate(2.3, region)
    assert low.source == "observed" and high.source == "observed"
    assert low.estimated_gain < high.estimated_gain


def test_zero_or_negative_delta_membrane_step_not_recorded():
    est = GainEstimator(GainEstimationConfig(bin_width_gpa=0.5, min_samples_for_estimate=1,
                                              safety_factor=1.0, upper_percentile=90.0, neighbor_bins=0))
    region = make_region()
    step = settled_step(0, 1.0, 1.0, 0.5, 0.6)  # no membrane change -> observed_gain is None
    est.record_step(step)
    result = est.estimate(0.5, region)
    assert result.source == "prior"


def test_observed_estimate_floored_at_prior_when_neighbor_bins_pull_in_lower_gain():
    # neighbor_bins=1 pulls data from the lower-pressure bin (already visited,
    # smaller true gain) into an as-yet-unvisited higher bin's estimate. In a
    # system where gain grows with pressure this must never push safe_gain
    # below the region's own conservative prior, since a too-low gain makes
    # the controller command an oversized membrane step for a given target
    # sample-pressure move (see gain.py's `membrane_step = step / safe_gain`).
    est = GainEstimator(GainEstimationConfig(bin_width_gpa=0.5, min_samples_for_estimate=3,
                                              safety_factor=1.0, upper_percentile=90.0, neighbor_bins=1))
    region = make_region(safe_gain=0.18)
    for i, gain in enumerate([0.04, 0.05, 0.06]):
        est.record_step(settled_step(i, 1.0, 2.0, 0.8, 0.8 + gain))  # midpoint ~0.8 -> low bin

    result = est.estimate(1.2, region)  # higher, still-empty bin; only neighbor data available
    assert result.source == "observed"
    assert result.n_samples == 3
    assert result.estimated_gain < region.safe_gain, "the raw observed signal really is lower than prior"
    assert result.safe_gain >= region.safe_gain, "but the safety floor must hold"


def test_rate_limit_gain_defaults_to_safe_gain_when_unset():
    est = GainEstimator(GainEstimationConfig(bin_width_gpa=0.5, min_samples_for_estimate=3,
                                              safety_factor=1.0, upper_percentile=90.0, neighbor_bins=1))
    region = make_region(safe_gain=0.28, rate_limit_gain=None)
    result = est.estimate(1.7, region)
    assert result.source == "prior"
    assert result.rate_limit_gain == result.safe_gain == 0.28


def test_rate_limit_gain_floors_the_prior_estimate_even_though_safe_gain_alone_would_be_lower():
    # Regression: a region's safe_gain prior can itself be optimistic
    # relative to real hardware (see config/default.yaml's gain_regions
    # note) -- rate_limit_gain must still floor the dynamic-rate-cap gain
    # even before any online observations exist to correct safe_gain itself.
    est = GainEstimator(GainEstimationConfig(bin_width_gpa=0.5, min_samples_for_estimate=3,
                                              safety_factor=1.0, upper_percentile=90.0, neighbor_bins=1))
    region = make_region(safe_gain=0.28, rate_limit_gain=0.75)
    result = est.estimate(1.7, region)
    assert result.source == "prior"
    assert result.safe_gain == 0.28, "step sizing still uses the plain (unfloored) prior"
    assert result.rate_limit_gain == 0.75, "the dynamic rate cap must use the separate, more conservative floor"


def test_rate_limit_gain_never_falls_below_the_online_safe_gain_estimate():
    # Even with a low configured rate_limit_gain, once real observations push
    # the online safe_gain estimate above it, the rate cap must track that
    # higher, more-informed value rather than staying pinned to the stale
    # configured floor.
    est = GainEstimator(GainEstimationConfig(bin_width_gpa=1.0, min_samples_for_estimate=3,
                                              safety_factor=1.0, upper_percentile=90.0, neighbor_bins=0))
    region = make_region(safe_gain=0.10, rate_limit_gain=0.15)
    for i, gain in enumerate([0.28, 0.30, 0.32]):
        step = settled_step(i, 1.0, 2.0, 0.5, 0.5 + gain)
        est.record_step(step)

    result = est.estimate(0.5, region)
    assert result.source == "observed"
    assert result.safe_gain > 0.15
    assert result.rate_limit_gain >= result.safe_gain
    assert result.rate_limit_gain == result.safe_gain, \
        "the online estimate already exceeds the configured floor, so it alone should govern"


def test_interrupted_rate_observation_is_visible_but_not_enforced_in_observe_mode():
    est = GainEstimator(GainEstimationConfig(
        bin_width_gpa=0.5,
        min_samples_for_estimate=3,
        safety_factor=1.0,
        upper_percentile=90.0,
        neighbor_bins=0,
        interrupted_rate_learning_mode="observe",
        interrupted_rate_safety_factor=1.25,
    ))
    region = make_region(safe_gain=0.20)
    est.record_interrupted_rate_observation(7, 0.3, 0.50)

    result = est.estimate(0.3, region)
    assert result.safe_gain == 0.20
    assert result.rate_limit_gain == 0.20
    assert result.learned_rate_floor == 0.625
    assert result.interrupted_rate_observation_count == 1
    assert result.rate_gain_source == "configured"


def test_interrupted_rate_observation_only_tightens_rate_limit_in_enforce_mode():
    est = GainEstimator(GainEstimationConfig(
        bin_width_gpa=0.5,
        min_samples_for_estimate=3,
        safety_factor=1.0,
        upper_percentile=90.0,
        neighbor_bins=0,
        interrupted_rate_learning_mode="enforce",
        interrupted_rate_safety_factor=1.25,
    ))
    region = make_region(safe_gain=0.20)
    est.record_interrupted_rate_observation(8, 0.3, 0.50)

    result = est.estimate(0.3, region)
    assert result.safe_gain == 0.20, "interrupted data must never resize static steps"
    assert result.rate_limit_gain == 0.625
    assert result.rate_gain_source == "interrupted"


def test_interrupted_rate_observation_updates_in_place_propagates_upward_and_can_be_discarded():
    est = GainEstimator(GainEstimationConfig(
        bin_width_gpa=0.5,
        min_samples_for_estimate=3,
        safety_factor=1.0,
        upper_percentile=90.0,
        neighbor_bins=0,
        interrupted_rate_learning_mode="enforce",
        interrupted_rate_safety_factor=1.0,
        interrupted_rate_propagate_upward=True,
    ))
    region = make_region(safe_gain=0.20)
    est.record_interrupted_rate_observation(9, 0.3, 0.40)
    est.record_interrupted_rate_observation(9, 0.3, 0.60)

    same_bin = est.estimate(0.3, region)
    higher_bin = est.estimate(1.3, region)
    assert same_bin.interrupted_rate_observation_count == 1
    assert same_bin.rate_limit_gain == 0.60
    assert higher_bin.rate_limit_gain == 0.60

    est.discard_interrupted_rate_observation(9)
    discarded = est.estimate(0.3, region)
    assert discarded.interrupted_rate_observation_count == 0
    assert discarded.rate_limit_gain == region.safe_gain


def test_adaptive_local_starts_in_probe_mode_then_uses_first_settled_response():
    est = GainEstimator(GainEstimationConfig(
        step_sizing_mode="adaptive_local",
        local_pressure_window_gpa=0.25,
        local_max_observations=5,
        local_gain_safety_factor=1.25,
    ))
    region = make_region(safe_gain=99.0)

    initial = est.estimate(0.2, region, forward_sample_step_gpa=0.03)
    assert initial.source == "probe"
    assert initial.safe_gain == 0.0

    step = settled_step(1, 1.0, 1.2, 0.20, 0.24)
    step.membrane_actual_before = 1.0
    step.membrane_actual_after = 1.2
    step.response_detected = True
    est.record_step(step)

    learned = est.estimate(0.24, region, forward_sample_step_gpa=0.03)
    assert learned.source == "observed"
    assert learned.n_samples == 1
    assert abs(learned.estimated_gain - 0.20) < 1e-9
    assert learned.safe_gain >= 0.25 - 1e-12
    assert learned.safe_gain < region.safe_gain, \
        "adaptive mode must not be floored by the legacy region calibration"


def test_adaptive_local_does_not_reuse_distant_low_pressure_gain():
    est = GainEstimator(GainEstimationConfig(
        step_sizing_mode="adaptive_local",
        local_pressure_window_gpa=0.1,
    ))
    region = make_region()
    step = settled_step(1, 1.0, 1.2, 0.10, 0.14)
    step.response_detected = True
    est.record_step(step)

    assert est.estimate(0.15, region).source == "observed"
    assert est.estimate(0.50, region).source == "probe"


def test_adaptive_local_adds_positive_curvature_allowance():
    cfg = GainEstimationConfig(
        step_sizing_mode="adaptive_local",
        local_pressure_window_gpa=1.0,
        local_gain_safety_factor=1.0,
        local_curvature_safety_factor=1.0,
    )
    est = GainEstimator(cfg)
    region = make_region()
    first = settled_step(1, 1.0, 1.2, 0.10, 0.14)  # gain 0.2
    first.response_detected = True
    second = settled_step(2, 1.2, 1.4, 0.14, 0.22)  # gain 0.4
    second.response_detected = True
    est.record_step(first)
    est.record_step(second)

    no_forward = est.estimate(0.22, region, forward_sample_step_gpa=0.0)
    forward = est.estimate(0.22, region, forward_sample_step_gpa=0.05)
    assert forward.local_gain_trend_per_gpa > 0
    assert forward.safe_gain > no_forward.safe_gain


def test_adaptive_local_trend_ignores_noise_from_closely_spaced_samples():
    # Regression for logs/run_20260729T164838_686358: two observations only
    # 0.005 GPa apart with a large gain jump between them is measurement
    # noise, not a real per-GPa curvature signal, and must not be read as one
    # -- on real hardware this pattern inflated safe_gain to ~4x the largest
    # gain ever actually observed in that run, collapsing the commanded
    # membrane step for the rest of the approach.
    cfg = GainEstimationConfig(
        step_sizing_mode="adaptive_local",
        local_pressure_window_gpa=1.0,
        local_gain_safety_factor=1.0,
        local_uncertainty_safety_factor=1.0,
        local_curvature_safety_factor=1.0,
        local_trend_min_span_gpa=0.05,
    )
    est = GainEstimator(cfg)
    region = make_region()
    first = settled_step(1, 1.0, 2.0, 0.950, 1.050)   # midpoint 1.000, gain 0.10
    first.response_detected = True
    second = settled_step(2, 1.0, 2.0, 0.855, 1.155)  # midpoint 1.005, gain 0.30
    second.response_detected = True
    est.record_step(first)
    est.record_step(second)

    result = est.estimate(1.005, region, forward_sample_step_gpa=0.03)
    assert result.local_observation_span_gpa < cfg.local_trend_min_span_gpa
    assert result.local_gain_trend_per_gpa == 0.0
    # Without the span gate this would be ~0.30 + 40 * 0.03 = 1.5.
    assert abs(result.safe_gain - 0.30) < 1e-9


def test_adaptive_local_trend_fires_once_span_is_wide_enough():
    cfg = GainEstimationConfig(
        step_sizing_mode="adaptive_local",
        local_pressure_window_gpa=1.0,
        local_gain_safety_factor=1.0,
        local_uncertainty_safety_factor=1.0,
        local_curvature_safety_factor=1.0,
        local_trend_min_span_gpa=0.05,
    )
    est = GainEstimator(cfg)
    region = make_region()
    first = settled_step(1, 1.0, 2.0, 0.10, 0.20)   # midpoint 0.15, gain 0.10
    first.response_detected = True
    second = settled_step(2, 1.0, 2.0, 0.30, 0.50)  # midpoint 0.40, gain 0.20
    second.response_detected = True
    est.record_step(first)
    est.record_step(second)

    result = est.estimate(0.40, region, forward_sample_step_gpa=0.03)
    assert result.local_observation_span_gpa >= cfg.local_trend_min_span_gpa
    assert result.local_gain_trend_per_gpa > 0.0


def test_step_observed_gain_prefers_actual_gas_delta_and_censors_no_response():
    step = settled_step(1, 1.0, 2.0, 0.1, 0.3)
    step.membrane_actual_before = 1.0
    step.membrane_actual_after = 1.5
    step.response_detected = True
    assert abs(step.observed_gain - 0.4) < 1e-9

    step.response_detected = False
    assert step.observed_gain is None
