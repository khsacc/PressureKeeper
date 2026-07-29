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
