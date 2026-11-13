"""The walk's difficulty must be its depth and nothing else.

If the table leaks the answer, if chance moves with k, or if the gold answer is
wrong, then a fast-against-slow curve measured on it says nothing about serial
depth -- which is the only thing the task exists to vary.
"""

import pytest

from hde.data.depth_walk import (
    SYMBOLS,
    as_request,
    questions,
    state_text,
    sweep,
    table,
    walk,
)


def test_the_table_is_one_cycle_through_every_symbol():
    """Otherwise nominal depth is not real depth.

    A start on a cycle of length c is `k mod c` lookups from its answer. The first
    version drew cycles of 2, 4 and 10, so every start on the 2-cycle was one step
    away at every odd k, and the accuracy curve showed bumps at k = 5, 9 and 11
    that belonged to the table rather than to the model.
    """
    for seed in range(5):
        arrows = table(seed=seed)
        assert sorted(arrows) == list(range(SYMBOLS))
        assert sorted(arrows.values()) == list(range(SYMBOLS))
        assert all(source != target for source, target in arrows.items())

        here, length = 0, 0
        while True:
            here, length = arrows[here], length + 1
            if here == 0:
                break
        assert length == SYMBOLS, f"seed {seed} splits into cycles, shortest {length}"


def test_effective_depth_equals_k_for_every_depth_the_sweep_uses():
    arrows = table(seed=0)
    for start in sorted(arrows):
        seen = {start}
        here = start
        for steps in range(1, SYMBOLS):
            here = walk(arrows, start, steps)
            assert here not in seen, f"k={steps} from {start} revisits after {len(seen)}"
            seen.add(here)


def test_the_gold_answer_is_the_table_composed_k_times():
    arrows = table(seed=1)
    for start in sorted(arrows):
        here = start
        for steps in range(1, 9):
            here = arrows[here]
            assert walk(arrows, start, steps) == here


def test_every_arrow_is_on_the_page():
    """Nothing has to be known that is not written down; that is the premise."""
    arrows = table(seed=2)
    text = state_text(arrows)
    for source, target in arrows.items():
        assert f"{source} -> {target}" in text


def test_the_state_does_not_contain_the_answer_to_any_walk():
    """It gives single arrows. A k-step destination appearing beside its start
    would let the readout match a string instead of composing."""
    arrows = table(seed=4)
    state, schema, gold = questions(steps=5, instances=SYMBOLS, seed=4, arrows=arrows)
    # Whole lines, not substrings: "1 -> 15" occurs inside "11 -> 15", and the
    # first version of this test failed on exactly that rather than on the data.
    lines = set(state.splitlines())
    for name, answer in gold.items():
        start = int(schema[name].instructions.split("Start at ")[1].split(" ")[0])
        if f"{start} -> {answer}" in lines:
            assert arrows[start] == int(answer), "a five-step walk written as one arrow"


def test_chance_is_the_same_at_every_depth():
    """Every question offers all sixteen numbers, so accuracy is comparable across
    k. A candidate set that shrank with k would make the curve a fact about the
    options rather than about depth."""
    _, by_depth = sweep(range(1, 6), instances=4, seed=5)
    for _steps, (_state, schema, _gold) in by_depth.items():
        for question in schema.values():
            assert len(question.criteria) == SYMBOLS


def test_one_table_serves_the_whole_sweep():
    """k is the only thing that varies; a fresh table per depth would vary two."""
    arrows, by_depth = sweep([1, 4, 9], instances=3, seed=6)
    for _steps, (state, _schema, _gold) in by_depth.items():
        assert state == state_text(arrows)


def test_gold_spreads_over_the_symbols_rather_than_sitting_on_one():
    """A walk that always lands on the same number would be answerable by
    guessing that number, at any depth."""
    _, by_depth = sweep(range(1, 9), instances=SYMBOLS, seed=7)
    for steps, (_state, _schema, gold) in by_depth.items():
        assert len(set(gold.values())) == SYMBOLS, f"k={steps} collapsed"


def test_a_walk_of_zero_arrows_is_refused():
    with pytest.raises(ValueError, match="asks nothing"):
        questions(steps=0)


def test_the_request_body_is_what_the_endpoint_takes():
    state, schema, _gold = questions(steps=2, instances=2, seed=8)
    body = as_request(state, schema, lambda_=-1.0, budget=64)
    assert body["state"] == state
    assert body["routing"] == {"lambda": -1.0, "budget": 64}
    for question in body["questions"].values():
        assert question["type"] == "choice"
        assert len(question["criteria"]) == SYMBOLS
    assert "routing" not in as_request(state, schema)


def test_the_demo_is_one_file_with_the_trace_inside_it():
    """`build_compare` bakes a run into the page, the way `build_film` does, so the
    result opens with no server and no assets to fetch."""
    import json
    import re
    import tempfile
    from pathlib import Path

    from hde.games.film import build_compare

    trace = {
        "schema_version": "litjev.compare.v1",
        "chance": 1 / SYMBOLS,
        "arrows": {"0": 1, "1": 0},
        "settings": {"instances": 1, "budget": 64, "seed": 0,
                     "never_think": 1.0, "always_think": -1.0},
        "summary": [{"depth": 1, "n": 1, "fast_accuracy": 1.0, "slow_accuracy": 1.0,
                     "rescued": 0.0, "spoiled": 0.0, "slow_tokens": 4.0,
                     "mean_value_of_thinking": 0.01}],
        "rows": [{"depth": 1, "question": "k1_0", "gold": "1",
                  "instructions": "Start at 0 and follow 1 arrow. Which number do you end on?",
                  "fast": {"answer": "1", "correct": True, "tokens": 0, "mode": "one",
                           "probabilities": {"1": 1.0}, "reasoning": None},
                  "slow": {"answer": "1", "correct": True, "tokens": 4, "mode": "two",
                           "probabilities": {"1": 1.0},
                           "reasoning": "</script><b>not markup</b>"},
                  "value_of_thinking": 0.01, "confidence": 0.9}],
    }
    work = Path(tempfile.mkdtemp())
    source, page = work / "t.json", work / "c.html"
    source.write_text(json.dumps(trace))
    build_compare(source, page)
    html = page.read_text()

    inlined = re.search(r'<script id="trace-data"[^>]*>(.*?)</script>', html, re.S).group(1)
    assert json.loads(inlined)["rows"][0]["gold"] == "1", "the trace must be readable back"
    assert not re.search(r'<(link|script)[^>]*\b(src|href)=', html), "no asset may be fetched"

    # The slow mode's text is model output and goes straight into this page.
    assert "</script>" not in inlined
    assert "<b>not markup</b>" not in html


def test_a_trace_of_the_wrong_kind_is_refused():
    import json
    import tempfile
    from pathlib import Path

    import pytest as _pytest

    from hde.games.film import build_compare

    work = Path(tempfile.mkdtemp())
    source = work / "t.json"
    source.write_text(json.dumps({"schema_version": "litjev.trace.v1", "decisions": [1]}))
    with _pytest.raises(ValueError, match="litjev.compare.v1"):
        build_compare(source, work / "c.html")
