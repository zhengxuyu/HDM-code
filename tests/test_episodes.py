import numpy as np
import pytest

from hde.episodes import (
    additive_score,
    allocate,
    episode_success,
    form_episodes,
    multiplicative_score,
    oracle_allocation,
)


def outcome(p_fast, p_slow):
    """A four-way outcome with the requested marginals, independence assumed."""
    return [p_fast * p_slow, p_fast * (1 - p_slow), (1 - p_fast) * p_slow, (1 - p_fast) * (1 - p_slow)]


def test_the_two_scores_disagree_on_which_step_to_escalate():
    # Raising 0.10 to 0.20 doubles the step's chance; raising 0.80 to 0.95 adds
    # more accuracy but multiplies by less.
    outcomes = np.array([outcome(0.10, 0.20), outcome(0.80, 0.95)])
    assert np.argmax(additive_score(outcomes)) == 1
    assert np.argmax(multiplicative_score(outcomes)) == 0


def test_episodes_stay_within_a_category_and_have_equal_length():
    categories = ["law"] * 7 + ["physics"] * 5
    episodes = form_episodes(categories, length=3, seed=0)
    assert [len(e) for e in episodes] == [3, 3, 3]          # 2 from law, 1 from physics
    for episode in episodes:
        assert len({categories[i] for i in episode}) == 1


def test_allocation_spends_the_whole_budget_on_the_top_scores():
    episodes = [np.array([0, 1, 2]), np.array([3, 4, 5])]
    score = np.array([0.1, 0.9, 0.5, 0.2, 0.3, 0.8])
    escalate = allocate(episodes, score, budget=2)
    assert list(escalate) == [False, True, True, False, True, True]
    assert not allocate(episodes, score, budget=0).any()


def test_success_counts_an_episode_only_when_every_step_is_right():
    episodes = [np.array([0, 1]), np.array([2, 3])]
    fast = np.array([True, False, True, True])
    slow = np.array([True, True, False, True])
    assert episode_success(episodes, np.zeros(4, bool), fast, slow) == 0.5
    assert episode_success(episodes, np.array([0, 1, 0, 0], bool), fast, slow) == 1.0


def test_the_oracle_only_spends_where_thinking_rescues_a_step():
    episodes = [np.array([0, 1, 2])]
    fast = np.array([False, False, True])
    slow = np.array([True, False, True])   # only step 0 is rescued
    escalate = oracle_allocation(episodes, fast, slow, budget=1)
    assert list(escalate) == [True, False, False]


def test_episode_length_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        form_episodes(["a"], length=0)


def test_allocation_is_sized_by_the_scores_not_by_the_episodes():
    # Question 6 is a leftover: it belongs to no episode, so it is never escalated.
    episodes = [np.array([0, 1, 2]), np.array([3, 4, 5])]
    score = np.arange(7, dtype=float)
    escalate = allocate(episodes, score, budget=1)
    assert len(escalate) == 7
    assert list(escalate) == [False, False, True, False, False, True, False]


def test_progress_is_a_sum_of_prefix_products():
    from hde.episodes import expected_progress
    assert expected_progress([1.0, 1.0, 1.0]) == pytest.approx(3.0)
    assert expected_progress([0.5, 1.0]) == pytest.approx(0.5 + 0.5)
    assert expected_progress([0.0, 1.0]) == pytest.approx(0.0)


def test_progress_allocation_prefers_the_earlier_step_at_equal_gain():
    from hde.episodes import greedy_progress_allocation
    # Both steps gain 0.3 from thinking, but fixing the first one also lets the
    # episode reach the second.
    episodes = [np.array([0, 1])]
    p_fast = np.array([0.4, 0.4])
    p_slow = np.array([0.7, 0.7])
    escalate = greedy_progress_allocation(episodes, p_fast, p_slow, budget=1)
    assert list(escalate) == [True, False]


def test_realised_progress_stops_at_the_first_wrong_step():
    from hde.episodes import realised_progress
    episodes = [np.array([0, 1, 2])]
    fast = np.array([True, False, True])
    slow = np.array([True, True, True])
    assert realised_progress(episodes, np.zeros(3, bool), fast, slow) == 1.0
    assert realised_progress(episodes, np.array([0, 1, 0], bool), fast, slow) == 3.0
