"""`tab:free`'s acc@ columns, and the tie that made one of its rows machine-dependent.

`log K` is the log of the option count: eight distinct values over the 346
held-out questions, 291 of them sharing the top one, and the 20% cut at rank 69
lands inside that block. Two machines printed two different rows from the same
file -- this one reproduced the paper's and exps's did not. `block_utils.auroc`
has used mid-ranks since it was written, which is why that row's AUROC agrees
across both and its acc@ columns do not.

So these pin two things: that the escalation set is now chosen deterministically,
and that when the cut cannot be resolved by the score the function says so with a
bracket rather than returning whichever answer the sort happened to give.
"""
import numpy as np

from conftest import load_script

free = load_script("free_baselines")


def test_a_cut_that_falls_between_distinct_scores_has_no_bracket():
    """The ordinary case: the four distribution statistics are like this.

    Four questions, scores all distinct, escalate the top two. Thinking rescues
    the top-scored one and ruins the second, so the accuracy is computable by
    hand: escalating gives slow on q0 (right) and q1 (wrong), fast on q2 (right)
    and q3 (wrong) -- two of four.
    """
    score = np.array([4.0, 3.0, 2.0, 1.0])
    fast = np.array([False, True, True, False])
    slow = np.array([True, False, True, False])

    accuracy, rescued, wasted, low, high, tied = free.escalated_accuracy(
        score, fast, slow, 0.5)
    assert accuracy == 0.5
    assert rescued == 1 and wasted == 0
    # Determined, so the bracket is a point and nothing is reported as tied.
    assert (low, high, tied) == (0.5, 0.5, 0)


def test_a_cut_inside_a_tie_reports_what_any_other_order_could_have_given():
    """The `log K` case, shrunk to where the arithmetic is checkable by hand.

    Six questions, all scoring the same, two of them escalate. One question is a
    rescue (fast wrong, slow right), one is a casualty (fast right, slow wrong),
    and four are unaffected. Baseline fast accuracy is 3 of 6.

    Escalating the rescue and one neutral gives 4 of 6; escalating the casualty
    and one neutral gives 2 of 6. So the bracket is 2/6 .. 4/6 and the printed
    number is one of three possibilities chosen by the sort.
    """
    score = np.ones(6)
    fast = np.array([False, True, True, True, False, False])
    slow = np.array([True, False, True, True, False, False])

    accuracy, _rescued, _wasted, low, high, tied = free.escalated_accuracy(
        score, fast, slow, 2 / 6)
    assert tied == 6, "every question is at the threshold"
    assert np.isclose(low, 2 / 6) and np.isclose(high, 4 / 6)
    # Whatever the sort chose has to lie inside the bracket, and the bracket has
    # to be wide enough to be worth printing -- the point of reporting it.
    assert low <= accuracy <= high
    assert high - low > 0.005
    # Never outside [0, 1]: the first version summed the tied questions twice and
    # returned 1.194.
    assert 0.0 <= low <= 1.0 and 0.0 <= high <= 1.0


def test_the_order_is_deterministic_rather_than_the_sort_implementation_s():
    """Same input, same answer, independent of how the tie is laid out in memory.

    A stable sort does not make a tied cut *meaningful* -- that is what the
    bracket is for -- but it does stop the same file giving two machines two
    numbers, which is the failure this row actually had.
    """
    rng = np.random.default_rng(0)
    fast = rng.random(200) < 0.5
    slow = rng.random(200) < 0.6
    # Three distinct values over 200 questions, so every cut lands in a tie.
    score = np.repeat([3.0, 2.0, 1.0], [80, 60, 60])

    first = free.escalated_accuracy(score, fast, slow, 0.2)
    for _ in range(5):
        assert free.escalated_accuracy(score, fast, slow, 0.2) == first

    # And a permutation of the tied block is still bounded by the same bracket,
    # which is the invariant that does not depend on the sort at all.
    order = rng.permutation(200)
    moved = free.escalated_accuracy(score[order], fast[order], slow[order], 0.2)
    assert np.isclose(moved[3], first[3]) and np.isclose(moved[4], first[4])
    assert first[3] <= moved[0] <= first[4]


def test_escalating_nothing_and_everything_are_both_well_defined():
    """The endpoints `rate=0` and `rate=1` take their own path and had no test."""
    score = np.array([3.0, 2.0, 1.0])
    fast = np.array([False, True, False])
    slow = np.array([True, True, False])

    none = free.escalated_accuracy(score, fast, slow, 0.0)
    assert none[0] == 1 / 3 and none[5] == 0
    every = free.escalated_accuracy(score, fast, slow, 1.0)
    assert every[0] == 2 / 3 and every[5] == 0
