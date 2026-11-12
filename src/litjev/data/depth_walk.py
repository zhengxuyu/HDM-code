"""A task whose only difficulty is how many dependent lookups it needs.

A table of sixteen numbers, each pointing at another, is written into the state.
A question asks where a walk of `k` steps from a given start ends. Every arrow is
on the page, so nothing has to be known that is not there; what the question
demands is that the j-th lookup wait for the (j-1)-th, because which row to read
next is the answer to the row before.

That makes `k` a dial on serial depth with everything else held still. A single
forward pass has depth L whatever the question, while each generated token buys
another, so the fast-thinking mode should fail past some k and the slow-thinking
mode should keep going. The paper's appendix asserted exactly that on a 4B
backbone -- 43 per cent at k=1, chance at k=2, 99 per cent at k=14 with thinking
on -- and those figures were removed because nothing outside the paper recorded
them. This module is what a run that records them uses.

The gold answer is computed here, by composing the table k times, so the task can
be checked without a model.
"""

from __future__ import annotations

import json
from random import Random

from litjev.data.schema import Choice, DecisionSchema

SYMBOLS = 16


def table(symbols=SYMBOLS, seed=0):
    """A single cycle through all ``symbols``, as a mapping from each to its successor.

    One cycle, not just any permutation. A permutation splits into cycles, and a
    start on a cycle of length c is only ``k mod c`` lookups from its answer, so
    nominal depth stops being real depth: the first version of this drew cycles of
    2, 4 and 10, and every start on the 2-cycle was one step away at every odd k.
    Measured against effective depth the run was clean -- 0.720 at depth 1 and
    chance everywhere else -- but the k axis showed bumps at k = 5, 9 and 11 that
    were the short cycles and not the model.

    With one cycle of length `symbols`, effective depth equals k for every k below
    it, which is the only way the x axis means what the plot says it means.
    """
    rng = Random(seed)
    order = list(range(symbols))
    rng.shuffle(order)
    return {order[i]: order[(i + 1) % symbols] for i in range(symbols)}


def walk(arrows, start, steps):
    """Where `steps` arrows from `start` end. The gold answer, computed not asked."""
    here = start
    for _ in range(steps):
        here = arrows[here]
    return here


def state_text(arrows):
    """The table as it is written into the prompt: every arrow, in index order."""
    lines = [f"{source} -> {target}" for source, target in sorted(arrows.items())]
    return ("A table of arrows, each number pointing at one other number.\n"
            "Follow the arrows exactly as written; nothing outside this table is needed.\n\n"
            + "\n".join(lines))


def questions(steps, instances=8, symbols=SYMBOLS, seed=0, arrows=None):
    """`instances` walks of `steps` arrows each, with their gold answers.

    Returns `(state, schema, gold)`. Every question offers all sixteen numbers as
    candidates, so the readout scores the same set each time and chance is 1/16
    whatever `steps` is -- which is what makes accuracy comparable across k.
    """
    if steps < 1:
        raise ValueError("a walk of zero arrows asks nothing")
    arrows = table(symbols, seed) if arrows is None else arrows
    rng = Random((seed, steps).__hash__())
    starts = rng.sample(sorted(arrows), min(instances, len(arrows)))
    criteria = {str(symbol): str(symbol) for symbol in sorted(arrows)}
    schema, gold = {}, {}
    for index, start in enumerate(starts):
        name = f"k{steps}_{index}"
        schema[name] = Choice(
            instructions=(
                f"Start at {start} and follow {steps} "
                f"{'arrow' if steps == 1 else 'arrows'}. Which number do you end on?"
            ),
            criteria=dict(criteria),
        )
        gold[name] = str(walk(arrows, start, steps))
    return state_text(arrows), DecisionSchema(schema), gold


def sweep(depths, instances=8, symbols=SYMBOLS, seed=0):
    """One table, every depth in `depths`, so k is the only thing that varies."""
    arrows = table(symbols, seed)
    return arrows, {
        steps: questions(steps, instances, symbols, seed, arrows=arrows)
        for steps in depths
    }


def as_request(state, schema, model="litjev", lambda_=None, budget=512):
    """The body a `/v1/systemtwo/debug` call takes, so a caller writes no JSON."""
    body = {
        "model": model,
        "state": state,
        "questions": {
            name: json.loads(question.model_dump_json()) if hasattr(question, "model_dump_json")
            else {"type": question.type, "instructions": question.instructions,
                  "criteria": question.criteria}
            for name, question in schema.items()
        },
    }
    if lambda_ is not None:
        body["routing"] = {"lambda": lambda_, "budget": budget}
    return body
