"""Routing when several decisions share one episode and one budget.

A single question is a separable decision: escalating it adds its own gain to
the accuracy and its own tokens to the cost, so the optimal rule thresholds the
gain and nothing about one question changes what to do with another. An agentic
episode is not separable in the same way. When a task succeeds only if every
step succeeds, the reward is a product over steps, and the value of escalating
one step depends on how likely the others are to be right.

The two objectives rank escalations differently, which is the whole point:

  additive        maximise sum p_i      escalate by p_slow - p_fast
  multiplicative  maximise prod p_i     escalate by log(p_slow / p_fast)

Both rules are exactly optimal for their objective under a budget of m
escalations, because the base term is fixed and each escalation contributes its
own independent term. A step the fast mode answers with probability 0.10 rather
than 0.20 is worth more under the product than one it answers with 0.80 rather
than 0.95, and less under the sum.
"""

import numpy as np

EPS = 1e-12


def form_episodes(categories, length, seed=0):
    """Partition question indices into episodes of `length`, within a category.

    An agentic episode stays on one task, so steps are drawn from the same
    category. Questions left over after the last full episode of a category are
    dropped, so every episode has the same length and the budget means the same
    thing in each.
    """
    if length < 1:
        raise ValueError("episode length must be at least 1")
    rng = np.random.default_rng(seed)
    episodes = []
    for category in sorted(set(map(str, categories))):
        members = np.flatnonzero(np.asarray(categories, dtype=str) == category)
        rng.shuffle(members)
        usable = len(members) - len(members) % length
        episodes.extend(members[start : start + length] for start in range(0, usable, length))
    return [np.asarray(episode) for episode in episodes]


def mode_probabilities(outcomes):
    """P(fast correct) and P(slow correct) from the four-way joint outcome."""
    outcomes = np.asarray(outcomes, dtype=np.float64)
    return outcomes[:, 0] + outcomes[:, 1], outcomes[:, 0] + outcomes[:, 2]


def additive_score(outcomes):
    """The escalation gain: what the single-question rule ranks by."""
    fast, slow = mode_probabilities(outcomes)
    return slow - fast


def multiplicative_score(outcomes):
    """The log ratio: what an all-steps-correct reward ranks by."""
    fast, slow = mode_probabilities(outcomes)
    return np.log(np.maximum(slow, EPS)) - np.log(np.maximum(fast, EPS))


def allocate(episodes, score, budget):
    """Escalate the `budget` highest-scoring steps of each episode.

    Exactly optimal for both objectives above, each with its own score.
    """
    score = np.asarray(score)
    # Sized by the score, not by the episodes: questions left over after the last
    # full episode of a category are in the arrays but in no episode, and they are
    # never escalated and never scored.
    escalate = np.zeros(len(score), dtype=bool)
    if budget <= 0:
        return escalate
    for episode in episodes:
        take = min(budget, len(episode))
        chosen = episode[np.argsort(-score[episode], kind="stable")[:take]]
        escalate[chosen] = True
    return escalate


def episode_success(episodes, escalate, fast_correct, slow_correct):
    """Fraction of episodes in which every step is answered correctly."""
    answered = np.where(np.asarray(escalate), np.asarray(slow_correct), np.asarray(fast_correct))
    return float(np.mean([bool(answered[episode].all()) for episode in episodes]))


def step_accuracy(escalate, fast_correct, slow_correct):
    answered = np.where(np.asarray(escalate), np.asarray(slow_correct), np.asarray(fast_correct))
    return float(np.mean(answered))


def oracle_allocation(episodes, fast_correct, slow_correct, budget):
    """The best any allocation of `budget` escalations per episode can do.

    Escalating helps an episode only where the fast mode is wrong and the slow
    mode right, so the oracle spends the budget there and nowhere else.
    """
    fast_correct = np.asarray(fast_correct, dtype=bool)
    slow_correct = np.asarray(slow_correct, dtype=bool)
    return allocate(episodes, (~fast_correct & slow_correct).astype(float), budget)


def expected_progress(p_step):
    """Expected number of steps completed before the first failure.

    An agent that stops at its first wrong answer completes step t only if every
    step up to t is right, so the expectation is a sum of prefix products. This
    is what makes the steps depend on one another: the value of step t is
    multiplied by the chance of reaching it, and the value of every later step
    is multiplied by the chance of surviving step t.
    """
    p_step = np.asarray(p_step, dtype=np.float64)
    return float(np.cumprod(p_step).sum())


def greedy_progress_allocation(episodes, p_fast, p_slow, budget):
    """Escalate the `budget` steps of each episode that buy the most progress.

    Greedy on the exact marginal gain in expected progress. Unlike ranking by
    gain, this one is position-aware: the same gain is worth more early, and an
    unreliable early step suppresses the value of everything behind it.
    """
    p_fast = np.asarray(p_fast, dtype=np.float64)
    p_slow = np.asarray(p_slow, dtype=np.float64)
    escalate = np.zeros(len(p_fast), dtype=bool)
    for episode in episodes:
        chosen = np.zeros(len(episode), dtype=bool)
        for _ in range(min(budget, len(episode))):
            best, best_value = None, -np.inf
            for position in range(len(episode)):
                if chosen[position]:
                    continue
                trial = chosen.copy()
                trial[position] = True
                value = expected_progress(np.where(trial, p_slow[episode], p_fast[episode]))
                if value > best_value:
                    best, best_value = position, value
            if best is None:
                break
            chosen[best] = True
        escalate[episode[chosen]] = True
    return escalate


def realised_progress(episodes, escalate, fast_correct, slow_correct):
    """Mean steps completed before the first wrong answer."""
    answered = np.where(np.asarray(escalate), np.asarray(slow_correct), np.asarray(fast_correct))
    total = 0.0
    for episode in episodes:
        done = 0
        for index in episode:
            if not answered[index]:
                break
            done += 1
        total += done
    return total / len(episodes)
