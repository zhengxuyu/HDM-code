import numpy as np

from litjev.online import online_thresholds, run_online, solve_values


def test_values_grow_with_steps_and_with_budget():
    values = solve_values([0.5] * 4, [0.9] * 4, steps=3, budget=2)
    assert values[3, 2] > values[2, 2] > values[1, 2]
    assert values[3, 2] > values[3, 1] > values[3, 0]


def test_no_budget_is_the_fast_mode_alone():
    values = solve_values([0.5] * 4, [0.9] * 4, steps=2, budget=1)
    # V(s, 0) = p + p^2 for a constant p, which is the fast mode walking unaided.
    assert values[2, 0] == np.float64(0.5 + 0.25)


def test_the_threshold_rises_when_budget_is_scarce_relative_to_steps():
    values = solve_values(np.linspace(0.2, 0.9, 30), np.linspace(0.4, 0.95, 30), steps=6, budget=3)
    thresholds = online_thresholds(values)
    # With one unit left, a long way to go makes that unit more precious than
    # it is with only one step remaining.
    assert thresholds[6, 1] > thresholds[1, 1]


def test_the_myopic_rule_spends_until_the_budget_is_gone():
    episodes = [np.array([0, 1, 2])]
    p_fast, p_slow = np.array([0.1, 0.1, 0.1]), np.array([0.9, 0.9, 0.9])
    fast = np.array([False, False, False])
    slow = np.array([True, True, True])
    assert run_online(episodes, p_fast, p_slow, fast, slow, budget=2) == 2.0
    assert run_online(episodes, p_fast, p_slow, fast, slow, budget=3) == 3.0


def test_an_episode_stops_at_its_first_wrong_step():
    episodes = [np.array([0, 1, 2])]
    p_fast, p_slow = np.ones(3) * 0.5, np.ones(3) * 0.5   # no reason to escalate
    fast = np.array([True, False, True])
    assert run_online(episodes, p_fast, p_slow, fast, fast, budget=3) == 1.0
