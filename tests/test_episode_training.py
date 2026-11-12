import pytest
import torch

from litjev.train.episode_training import (
    EpisodeCDPOConfig,
    episode_loss,
    expected_progress_and_cost,
)


def test_progress_and_cost_match_the_closed_form_when_the_policy_is_certain():
    # Always escalate: every step is answered by the slow mode, and every step
    # the agent reaches is paid for.
    escalate = torch.ones(1, 3)
    fast = torch.zeros(1, 3)
    slow = torch.tensor([[1.0, 1.0, 0.0]])
    progress, cost = expected_progress_and_cost(escalate, fast, slow)
    assert progress.item() == 2.0          # two steps completed, then a failure
    assert cost.item() == 3.0              # reached and paid on all three


def test_an_escalation_behind_a_certain_failure_is_never_paid_for():
    escalate = torch.ones(1, 2)
    fast = torch.zeros(1, 2)
    slow = torch.tensor([[0.0, 1.0]])      # step 1 fails whatever we do
    progress, cost = expected_progress_and_cost(escalate, fast, slow)
    assert progress.item() == 0.0
    assert cost.item() == 1.0              # only the first step is ever reached


def test_never_escalating_reproduces_the_fast_mode_and_costs_nothing():
    escalate = torch.zeros(1, 3)
    fast = torch.tensor([[1.0, 1.0, 1.0]])
    slow = torch.zeros(1, 3)
    progress, cost = expected_progress_and_cost(escalate, fast, slow)
    assert progress.item() == 3.0
    assert cost.item() == 0.0


def test_the_loss_is_finite_and_differentiable():
    torch.manual_seed(0)
    logits = torch.randn(4, 3, 4, requires_grad=True)
    fast = torch.randint(0, 2, (4, 3)).float()
    slow = torch.randint(0, 2, (4, 3)).float()
    loss = episode_loss(logits, fast, slow, EpisodeCDPOConfig(), torch.Generator().manual_seed(0))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_the_anchor_labels_follow_the_four_way_convention():
    # 0 both right, 1 fast right slow wrong, 2 fast wrong slow right, 3 both wrong.
    fast = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    slow = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    labels = (2 * (1 - fast) + (1 - slow)).long().reshape(-1)
    assert list(labels.numpy()) == [0, 1, 2, 3]


def test_the_discounted_return_is_eq_return_and_not_discounted_progress():
    """`E[R exp(-lambda C)]`, checked against enumerating every trajectory.

    Two things made this worth enumerating rather than deriving. The natural
    cumprod form discounts each step by the spend that reached it, not by the
    episode's total, and on these episodes the two differ by up to 2.13 steps.
    And the first enumeration written here iterated the 2^n escalation vectors,
    which counts each failed prefix 2^(n-1-t) times -- it called the right formula
    wrong by 3.6 steps. Once an episode fails the later decisions are never made,
    so the enumeration has to stop with it.
    """
    import math

    import numpy as np
    import torch

    from litjev.train.episode_training import expected_progress_and_cost

    def exhaustive(p, fast, slow, lam):
        def walk(t, prob, reached, esc):
            if t == len(p):
                return prob * reached * math.exp(-lam * esc)
            total = 0.0
            for escalate in (0, 1):
                branch = prob * (p[t] if escalate else 1 - p[t])
                spent = esc + escalate
                if (slow[t] if escalate else fast[t]) >= 1:
                    total += walk(t + 1, branch, reached + 1, spent)
                else:
                    total += branch * reached * math.exp(-lam * spent)
            return total
        return walk(0, 1.0, 0, 0)

    def call(p, fast, slow, discount=None):
        row = (torch.tensor(p, dtype=torch.float64).unsqueeze(0),
               torch.tensor(fast, dtype=torch.float64).unsqueeze(0),
               torch.tensor(slow, dtype=torch.float64).unsqueeze(0))
        if discount is None:
            return expected_progress_and_cost(*row)
        return expected_progress_and_cost(
            *row, discount=torch.tensor([[discount]], dtype=torch.float64))

    rng = np.random.default_rng(0)
    for _ in range(60):
        steps = int(rng.integers(1, 6))
        p, lam = rng.random(steps), float(rng.uniform(-0.3, 0.6))
        fast = rng.integers(0, 2, steps).astype(float)
        slow = rng.integers(0, 2, steps).astype(float)

        closed, _ = call(p, fast, slow, math.exp(-lam))
        assert math.isclose(float(closed[0]), exhaustive(p, fast, slow, lam),
                            abs_tol=1e-9), (steps, lam, p, fast, slow)

        # A price of zero is no price: the discounted path must reproduce the
        # undiscounted one exactly, or the two branches have drifted.
        plain, plain_cost = call(p, fast, slow)
        at_zero, zero_cost = call(p, fast, slow, 1.0)
        assert math.isclose(float(plain[0]), float(at_zero[0]), abs_tol=1e-12)
        assert math.isclose(float(plain_cost[0]), float(zero_cost[0]), abs_tol=1e-12)


def test_a_price_cannot_make_progress_worth_less_than_none():
    """The subtraction `progress - price * cost` could; a discount cannot. An
    episode that completed nothing is worth zero at every price, so no price
    reorders a successful episode below a failed one."""
    import math

    import torch

    from litjev.train.episode_training import expected_progress_and_cost

    def value(fast, slow, lam):
        p = torch.full((1, len(fast)), 0.999, dtype=torch.float64)   # escalate
        progress, _ = expected_progress_and_cost(
            p, torch.tensor([fast], dtype=torch.float64),
            torch.tensor([slow], dtype=torch.float64),
            discount=torch.tensor([[math.exp(-lam)]], dtype=torch.float64))
        return float(progress[0])

    succeeded = [1.0] * 4
    failed_at_once = [0.0] * 4
    for lam in (0.0, 0.1, 0.5, 2.0, 10.0):
        assert value(succeeded, succeeded, lam) > value(failed_at_once, failed_at_once, lam)
    assert value(failed_at_once, failed_at_once, 0.0) == pytest.approx(0.0, abs=1e-6)
    # Monotone in the price, and still positive: thinking that worked is not
    # erased by what it cost.
    priced = [value(succeeded, succeeded, lam) for lam in (0.0, 0.25, 0.5, 1.0)]
    assert priced == sorted(priced, reverse=True) and priced[-1] > 0
