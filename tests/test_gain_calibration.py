import numpy as np

from litjev.train.training import gain_calibration

# Five groups with known success rates, and the predicted gain set to each
# group's true log ratio. The rates are built exactly rather than sampled -- each
# block of ten questions carries the group's rate -- so what these tests measure
# is the statistic and not how a draw landed. A log ratio has no per-question
# value, so a sampled version of this would be testing the binning.
GROUPS = ((0.2, 0.8), (0.4, 0.8), (0.6, 0.9), (0.8, 0.9), (0.5, 0.5))
PER_GROUP = 1000


def exact_groups():
    """(true log ratio per question, fast correct, slow correct)."""
    truth, fast, slow = [], [], []
    for fast_rate, slow_rate in GROUPS:
        for index in range(PER_GROUP):
            truth.append(np.log(slow_rate) - np.log(fast_rate))
            fast.append(index % 10 < round(fast_rate * 10))
            slow.append(index % 10 < round(slow_rate * 10))
    return np.array(truth), np.array(fast, bool), np.array(slow, bool)


def test_perfectly_scaled_gain_has_no_error_and_unit_slope():
    truth, fast, slow = exact_groups()
    error, slope = gain_calibration(truth, fast, slow)
    assert error < 1e-9, error
    assert abs(slope - 1) < 1e-9, slope


def test_a_ranking_score_ranks_well_but_is_not_on_the_gain_scale():
    """Same ordering, five times the scale. AUROC cannot tell these apart and
    this must, because lambda is priced in the gain's own units."""
    truth, fast, slow = exact_groups()
    good_error, good_slope = gain_calibration(truth, fast, slow)
    bad_error, bad_slope = gain_calibration(truth * 5, fast, slow)

    assert bad_error > 1.0 > good_error
    assert abs(good_slope - 1) < 1e-9
    # Stretching the prediction fivefold flattens the regression to a fifth.
    assert abs(bad_slope - 0.2) < 1e-9, bad_slope


def test_the_realised_side_is_a_log_ratio_and_not_a_difference():
    """What this function compared against for one commit, after the gain became
    a log ratio. The two disagree by more than the statistic's own scale, so a
    value recorded on one does not read on the other."""
    truth, fast, slow = exact_groups()
    error, _ = gain_calibration(truth, fast, slow)

    # The old realised side, per bin: mean(slow) - mean(fast).
    order = np.argsort(truth, kind="stable")
    old = 0.0
    for chunk in np.array_split(order, 10):
        old += len(chunk) / len(truth) * abs(
            truth[chunk].mean() - (slow[chunk].mean() - fast[chunk].mean()))
    # On data whose predicted gain is exactly right, the log-ratio reading scores
    # zero error and the difference reading scores 0.241 -- a miscalibration of a
    # sixth of the gain range being reported for a perfectly calibrated head.
    assert error < 1e-9
    assert abs(old - 0.241) < 0.001, old


def test_empty_spread_gives_nan_slope_not_a_crash():
    error, slope = gain_calibration(np.zeros(10), np.zeros(10, bool), np.zeros(10, bool))
    assert error == 0.0 and np.isnan(slope)
