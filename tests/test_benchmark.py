import pathlib
import numpy as np

from hde._legacy.decision import RawFieldScores, SchemaDecisionEngine
from hde.bench.benchmark import evaluate
from hde.data.schema import Choice, DecisionSchema


def test_sequential_uses_ten_separate_single_question_calls():
    calls = []

    class Provider:
        def score(self, state, schema):
            calls.append(schema.names)
            return tuple(RawFieldScores(k, np.zeros(2), 10) for k in schema)

    engine = SchemaDecisionEngine(Provider(), readout="coded")
    schema = DecisionSchema(
        {
            f"q{i}": Choice(instructions=f"Question {i}", criteria={"A": None, "B": None})
            for i in range(10)
        }
    )
    evaluation, durations = evaluate(engine, schema, sequential=True)
    assert len(calls) == 10 and all(len(call) == 1 for call in calls)
    assert evaluation.diagnostics["forward_calls"] == 20
    assert len(evaluation.result.answers) == len(durations) == 10
    calls.clear()
    evaluation, durations = evaluate(engine, schema)
    assert len(calls) == 1 and len(calls[0]) == 10
    assert evaluation.diagnostics["forward_calls"] == 2
    assert durations == {}


def test_omitting_the_price_flag_asks_for_four_prices_not_none():
    """`--lambdas` defaults to a grid, which reads as "no prices" and is not.

    A run meant to measure the two endpoints with the readout paid measured six
    policies instead, because the flag was left off: `bench_policies` does build
    exactly `fast_only` and `slow_only` from an empty price list, and the list
    arrives non-empty unless the caller says `--lambdas ''`. The no-head path is
    safe -- `cli` empties the list and prints that it has -- so the trap only
    fires with a head attached, which is where a routed row is wanted least by a
    job asking for endpoints.

    This pins the default by parsing it rather than by reading the source, so a
    comment claiming otherwise fails here.
    """
    import re

    source = (pathlib.Path(__file__).resolve().parents[1] / "src" / "hde" / "cli.py").read_text()
    default = re.search(r'add_argument\("--lambdas", default="([^"]*)"\)', source)
    assert default, "the flag's default is no longer written where this test reads it"
    parsed = [float(x) for x in default.group(1).split(",") if x.strip()]
    assert len(parsed) == 4, (
        f"omitting --lambdas asks for {parsed}, so a job that wants the endpoints "
        f"alone has to pass --lambdas '' explicitly")
    assert [float(x) for x in "".split(",") if x.strip()] == [], (
        "and the empty string is a grid of no prices rather than a parse error")


def test_the_slow_only_sentinel_is_below_every_attainable_gain():
    """`slow_only` is a price, not a flag, so it is only slow-only while no gain reaches it.

    `ALL` was `-2.0`, justified in a comment as "below any gain in [-1, 1]". That
    was true of the old gain, the difference `P(slow right) - P(fast right)`, and
    false from the moment it became `log P(slow right) - log P(fast right)`: a log
    ratio is bounded only by the epsilon the logs are floored at, so it reaches
    `log(GAIN_EPS)`, about -13.82.

    Nothing measured was wrong, because the heads we have bottom out near -0.4. The
    failure it risked is the silent kind: a head confident that thinking would hurt
    a question would leave it unescalated in the row every bracket of
    Table~\\ref{tab:results} is measured against, and the caption's claim that a
    row escalating 100% is the slow-only policy would be false with nothing raised.

    So this asserts the bound against the gain function itself rather than against
    a number, and it fails if either the epsilon or the gain's definition moves
    again without the sentinel following.
    """
    import numpy as np

    from hde._legacy.routing import escalation_gain
    from hde.bench.routing_bench import ALL

    # The gain is minimised by putting all the mass on "fast right, slow wrong":
    # P(slow right) is then floored at the epsilon and P(fast right) is 1.
    worst = np.zeros(4)
    worst[1] = 1.0
    floor = escalation_gain(worst)
    assert ALL < floor, f"ALL={ALL} does not escalate a question whose gain is {floor}"

    # And no point of the simplex escapes it, which is the property the row needs.
    rng = np.random.default_rng(0)
    sample = rng.dirichlet(np.ones(4), size=20000)
    gains = np.array([escalation_gain(row) for row in sample])
    assert gains.min() > ALL
    # Not vacuous: the sample has to reach well below the old sentinel, or it would
    # pass against -2.0 as well and prove nothing.
    assert gains.min() < -2.0, "the sample never reaches the range this test is about"


def test_the_margin_cap_is_an_absolute_count_until_the_window_fills():
    """`cap_share * window` is a count, not a share of what has been seen.

    `MarginRouter.decide` tests `sum(recent) < cap_share * window`, so at the
    defaults it permits 200 escalations however few decisions have gone by. On a
    346-question benchmark the window never fills, and the effective ceiling is
    200/346 = 57.8% rather than the 20% the docstring used to claim.

    The existing test for the cap uses `window=10`, which fills after ten
    decisions, so it cannot tell the two readings apart. This one keeps the
    window longer than the run, which is the case every benchmark we have is in.

    The point is not the rule's behaviour, which is left alone -- "the last 1,000
    decisions" has no meaning at 346 -- but that a comparison against it must not
    claim operating points above the ceiling it is forbidden to reach.
    """
    from hde.core.block import MarginPolicy, MarginRouter

    # tau above any attainable margin, so every decision is "unsure" and only the
    # cap can stop an escalation.
    router = MarginRouter(MarginPolicy(tau=99.0, cap_share=0.2, window=1000, budget=256))
    decisions = [router.decide([0.5, 0.5]) for _ in range(346)]
    escalated = sum(decisions)

    assert escalated == 200, f"the cap let through {escalated}, not the absolute 200"
    assert escalated / 346 > 0.5, "a 20% share would have stopped at 69 of 346"
    # And the share the router reports is the share of the window it holds, which
    # is the other of the two notions living in this class.
    assert abs(router.escalated_share - 200 / 346) < 1e-9
