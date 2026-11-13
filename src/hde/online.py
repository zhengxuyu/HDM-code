"""Routing decided step by step, without seeing the rest of the episode.

An agent answers step t before step t+1 exists, so it cannot sort the episode
and spend its budget on the steps that deserve it most. It has to ration: spend
now, or keep the budget for a step it has not seen. The optimal online rule is a
dynamic program over the remaining steps and the remaining budget, and it is a
threshold on the log ratio whose level moves with the state.

This is the condition under which training for the decision can beat training
for the likelihood. With the whole episode in view the allocation is solved
exactly from the predicted probabilities, and nothing is left for a loss to add.
"""

import numpy as np

EPS = 1e-12


def solve_values(p_fast_sample, p_slow_sample, steps, budget):
    """Backward induction for V(s, b): expected steps completed from that state.

    The expectation is over the distribution of future steps, estimated by the
    sample of predicted probabilities passed in. That sample must come from the
    training split, since the rule is fixed before the episode is seen.
    """
    p_fast_sample = np.asarray(p_fast_sample, dtype=np.float64)
    p_slow_sample = np.asarray(p_slow_sample, dtype=np.float64)
    values = np.zeros((steps + 1, budget + 1))
    for s in range(1, steps + 1):
        for b in range(budget + 1):
            keep = p_fast_sample * (1.0 + values[s - 1, b])
            if b > 0:
                spend = p_slow_sample * (1.0 + values[s - 1, b - 1])
                values[s, b] = np.maximum(keep, spend).mean()
            else:
                values[s, b] = keep.mean()
    return values


def online_thresholds(values):
    """The log-ratio level above which escalating is worth it, per state.

    Escalating trades a step's survival probability up from p_fast to p_slow and
    spends one unit of budget, so it pays when
    p_slow (1 + V(s-1, b-1)) > p_fast (1 + V(s-1, b)).
    """
    steps, budget = values.shape[0] - 1, values.shape[1] - 1
    thresholds = np.full((steps + 1, budget + 1), np.inf)
    for s in range(1, steps + 1):
        for b in range(1, budget + 1):
            thresholds[s, b] = np.log(1.0 + values[s - 1, b]) - np.log(1.0 + values[s - 1, b - 1])
    return thresholds


def run_online(episodes, p_fast, p_slow, fast_correct, slow_correct, budget, thresholds=None):
    """Walk each episode, deciding one step at a time, and return mean progress.

    `thresholds` is the state-dependent rule from `online_thresholds`. Passing
    None gives the myopic rule: escalate whenever thinking is predicted to help
    and budget remains, which is what a per-question threshold can do.
    """
    p_fast = np.asarray(p_fast, dtype=np.float64)
    p_slow = np.asarray(p_slow, dtype=np.float64)
    fast_correct = np.asarray(fast_correct, dtype=bool)
    slow_correct = np.asarray(slow_correct, dtype=bool)
    total = 0.0
    for episode in episodes:
        remaining, done = budget, 0
        for position, index in enumerate(episode):
            steps_left = len(episode) - position
            ratio = np.log(max(p_slow[index], EPS)) - np.log(max(p_fast[index], EPS))
            if thresholds is None:
                level = 0.0
            else:
                level = thresholds[min(steps_left, thresholds.shape[0] - 1), remaining]
            escalate = remaining > 0 and ratio > level
            correct = slow_correct[index] if escalate else fast_correct[index]
            remaining -= int(escalate)
            if not correct:
                break
            done += 1
        total += done
    return total / len(episodes)
