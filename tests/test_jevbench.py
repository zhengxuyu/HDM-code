"""JevBench, the first typed benchmark here: 231 items, all three question types.

Two things make it worth its own loader rather than another multiple-choice entry.
Its items are typed as the schema types them -- 139 `choice`, 74 `noul`, 18 `score`
-- which is the limitation the paper states about its own evidence. And they carry
their own context, 35 of them as a JSON object, so the state is the benchmark's and
not a sentence this repository wrote.

The assertion that matters most here is the last one: the gold label has to be in
the form `routing_bench._answer_key` returns, which differs per type. If it is not,
nothing fails -- every answer simply scores wrong, and a benchmark reports a number
that is about the comparison and not about the model.
"""
import json

import pytest

from hde.data.datasets import load_jevbench
from hde.data.mmlu import own_state_requests, typed_question

MANIFEST = {
    "protocol": "jevbench::v1.2",
    "splits": [{"name": "original", "n": 2, "sha256": None},
               {"name": "easy", "n": 1, "sha256": None},
               {"name": "hard", "n": 1, "sha256": None}],
}

ITEMS = {
    "original": [
        {"id": "o-choice-0", "family": "intent", "state": "Where is my package?",
         "expected": "track", "labels": ["track", "cancel"],
         "question": {"type": "choice", "instructions": "Which intent?",
                      "criteria": {"track": "Wants to know where an order is",
                                   "cancel": "Wants to cancel"}}},
        {"id": "o-noul-0", "family": "policy", "state": "No receipt, bought 12 days ago.",
         "expected": "no", "labels": ["no", "yes"],
         "question": {"type": "noul", "instructions": "Is a refund permitted?",
                      "criteria": {"false": "A condition is missing",
                                   "true": "Every condition is established"}}},
    ],
    "easy": [
        {"id": "e-score-0", "family": "ordinal", "state": "The icon is misaligned.",
         "expected": 0, "labels": ["0", "1", "2"],
         "question": {"type": "score", "instructions": "Rate the impact.",
                      "criteria": ["Cosmetic only", "One user blocked", "Data loss"]}},
    ],
    "hard": [
        {"id": "h-choice-0", "family": "long_policy",
         # A context that arrived as an object, which is kept as one.
         "state": {"policy": ["Send email only to identified recipients."],
                   "conversation": [{"role": "user", "text": "Email Alex."}]},
         "expected": "ask", "labels": ["ask", "send"],
         "question": {"type": "choice", "instructions": "What should the agent do?",
                      "criteria": {"ask": "Ask which Alex", "send": "Send it"}}},
    ],
}


def fetcher(items=None, manifest=None):
    items = items or ITEMS

    def fetch(url):
        if url.endswith("manifest.json"):
            return json.dumps(manifest or MANIFEST).encode()
        tier = url.rsplit("/", 1)[-1].removesuffix(".jsonl")
        return "\n".join(json.dumps(row) for row in items.get(tier, [])).encode()

    return fetch


def test_all_three_question_types_load_with_their_own_states():
    loaded = load_jevbench(fetch=fetcher())
    assert loaded.provenance["types"] == {"choice": 2, "score": 1, "noul": 1}
    assert loaded.provenance["skipped"] == 0
    kinds = {q.question_id: q.question_type for q in loaded.questions}
    assert kinds == {"o-choice-0": "choice", "o-noul-0": "noul",
                     "e-score-0": "score", "h-choice-0": "choice"}
    structured = [q for q in loaded.questions if not isinstance(q.state, str)]
    assert [q.question_id for q in structured] == ["h-choice-0"], "the object state is kept"
    assert loaded.groups["o-choice-0"] == "intent"


def test_the_gold_label_is_in_the_form_the_benchmark_compares_against():
    """Per type, and this is the one that fails silently if it is wrong: the gold for
    `noul` is a boolean where the file writes yes/no, and for `score` an integer level
    where the schema's candidates are the strings "0".."n"."""
    from hde._legacy.decision import ChoiceAnswer, NoulAnswer, ScoreAnswer
    from hde.bench.routing_bench import _answer_key

    labels = load_jevbench(fetch=fetcher()).labels
    assert labels == {"o-choice-0": "track", "o-noul-0": False,
                      "e-score-0": 0, "h-choice-0": "ask"}

    right_choice = ChoiceAnswer(type="choice", choice="track", probabilities={}, confidence=0.0)
    assert _answer_key(right_choice) == labels["o-choice-0"]

    # p(true) below a half is the "no" this item expects.
    assert _answer_key(NoulAnswer(type="noul", noul=0.1)) == labels["o-noul-0"]
    assert _answer_key(NoulAnswer(type="noul", noul=0.9)) != labels["o-noul-0"]

    level_zero = ScoreAnswer(type="score", score=0.2, legend={}, confidence=0.0,
                             probabilities={"0": 0.8, "1": 0.1, "2": 0.1})
    assert _answer_key(level_zero) == labels["e-score-0"]


def test_each_question_becomes_the_schema_question_its_type_names():
    from hde.data.schema import Choice, Noul, Score

    by_id = {q.question_id: q for q in load_jevbench(fetch=fetcher()).questions}
    assert isinstance(typed_question(by_id["o-choice-0"]), Choice)
    assert isinstance(typed_question(by_id["o-noul-0"]), Noul)
    built = typed_question(by_id["e-score-0"])
    assert isinstance(built, Score)
    # Ordered low to high, from the list the benchmark gives, not from dict order.
    assert built.descriptions == ("Cosmetic only", "One user blocked", "Data loss")
    assert built.choices == ("0", "1", "2")
    assert typed_question(by_id["o-noul-0"]).choices == ("false", "true")


def test_one_request_per_item_with_the_item_s_own_state_and_no_padding():
    """`ten_question_batches` pairs ten questions with one synthetic state, which here
    would put unrelated documents in one prompt, and pads the last batch by repeating
    a question -- which would carry another item's context."""
    loaded = load_jevbench(fetch=fetcher())
    requests = list(own_state_requests(loaded.questions))
    assert len(requests) == len(loaded.questions)
    for request, scored in requests:
        assert len(request.questions) == len(scored) == 1
    states = {ids[0]: request.state for request, ids in requests}
    assert states["o-choice-0"] == "Where is my package?"
    assert isinstance(states["h-choice-0"], dict), "an object state reaches the request as one"


def test_a_file_that_does_not_match_the_manifest_is_refused():
    """The point of a pinned commit is that the content is fixed. A file that does not
    hash to what the manifest at that commit says is not that content."""
    manifest = {"protocol": "jevbench::v1.2",
                "splits": [{"name": "original", "n": 2, "sha256": "0" * 64}]}
    with pytest.raises(ValueError, match="manifest at this commit"):
        load_jevbench(split="original", fetch=fetcher(manifest=manifest))


def test_the_published_tiers_are_the_only_ones_and_the_pin_cannot_be_moved():
    with pytest.raises(ValueError, match="no published tier"):
        load_jevbench(split="router", fetch=fetcher())
    with pytest.raises(ValueError, match="pinned to commit"):
        load_jevbench(revision="deadbeef", fetch=fetcher())


def test_one_tier_loads_on_its_own():
    loaded = load_jevbench(split="easy", fetch=fetcher())
    assert [q.question_id for q in loaded.questions] == ["e-score-0"]
    assert loaded.provenance["tiers"] == ["easy"]


def test_the_digest_covers_the_state_and_survives_a_typed_gold():
    """A benchmark whose items differ only in their context would otherwise digest the
    same, and sorting rows whose gold mixes a string with an integer would raise."""
    loaded = load_jevbench(fetch=fetcher())
    first = loaded.digest()

    moved = dict(ITEMS)
    moved["original"] = [dict(ITEMS["original"][0], state="Different context."),
                         ITEMS["original"][1]]
    assert load_jevbench(fetch=fetcher(items=moved)).digest() != first
    assert load_jevbench(fetch=fetcher()).digest() == first


def test_without_a_head_the_bench_measures_the_endpoints_and_no_policy_between():
    """How a readout with no head trained for it yet still gets a fast-only column.
    `lambdas` is emptied rather than ignored, so no routed row can appear in a report
    for a policy that was never run."""
    import numpy as np

    from hde._legacy.decision import RawFieldScores, SchemaDecisionEngine
    from hde.bench.routing_bench import bench_policies
    from hde.data.schema import Choice, SystemOneRequest

    class Provider:
        def score(self, state, schema):
            return tuple(RawFieldScores(name, np.array([0.1, 0.9]), 7) for name in schema)

        def think(self, state, schema, names, budget):
            return tuple(RawFieldScores(name, np.array([0.9, 0.1]), 7, generated_tokens=5)
                         for name in names)

    engine = SchemaDecisionEngine(Provider(), readout="coded")
    request = SystemOneRequest(
        model="hde", state="s",
        questions={"q": Choice(instructions="Pick", criteria={"a": "1", "b": "2"})},
    )
    report = bench_policies(engine, [(request, ["q"])], {"q": "b"}, [], budget=8)
    assert set(report["policies"]) == {"fast_only", "slow_only"}
    assert report["policies"]["fast_only"]["accuracy"] == 1.0
    assert report["policies"]["slow_only"]["escalation_rate"] == 1.0


def test_a_row_says_which_primitive_each_question_was_asked_as():
    """So a per-type rate comes out of the rows, with no loader join.

    JevBench is the only venue with more than one type and the only one the paper
    makes a per-type claim about, and the rows carried `question_ids`,
    `escalated_by_question` and `tokens_by_question` but not the type. Recovering
    it meant re-running the selection through the loader to describe a run that
    had already happened, and the id prefix is no substitute: those are task
    names -- `sol`, `opus`, `intent` -- not primitives.

    The assertion is the whole point rather than the field's presence: a per-type
    escalation rate is computed here from one row and nothing else.
    """
    import numpy as np

    from hde._legacy.decision import RawFieldScores, SchemaDecisionEngine
    from hde.bench.routing_bench import bench_policies
    from hde.data.schema import Choice, Noul, Score, SystemOneRequest

    class Provider:
        def score(self, state, schema):
            return tuple(RawFieldScores(name, np.array([0.1, 0.9]), 7) for name in schema)

        def think(self, state, schema, names, budget):
            return tuple(RawFieldScores(name, np.array([0.9, 0.1]), 7, generated_tokens=5)
                         for name in names)

    engine = SchemaDecisionEngine(Provider(), readout="coded")
    request = SystemOneRequest(
        model="hde", state="s",
        questions={"c": Choice(instructions="Pick", criteria={"a": "1", "b": "2"}),
                   "n": Noul(instructions="Does it hold"),
                   "s": Score(instructions="How far", criteria=["low", "high"])},
    )
    report = bench_policies(engine, [(request, ["c", "n", "s"])], {}, [], budget=8)

    (fast, slow) = (row for row in report["batches"])
    assert fast["type_by_question"] == {"c": "choice", "n": "noul", "s": "score"}
    # On every policy's row, not only the first: a field written once per batch
    # and shared would pass the line above.
    assert slow["type_by_question"] == fast["type_by_question"]

    # The per-type escalation rate, from this row alone. The slow policy
    # escalates all three, so each type's rate is 1.0 -- what is under test is
    # that the grouping needs nothing the row does not carry.
    by_type = {}
    for key, kind in slow["type_by_question"].items():
        got, total = by_type.get(kind, (0, 0))
        by_type[kind] = (got + (key in slow["escalated_by_question"]), total + 1)
    assert {kind: got / total for kind, (got, total) in by_type.items()} == {
        "choice": 1.0, "noul": 1.0, "score": 1.0}


def test_the_benchmark_takes_the_split_names_jevbench_actually_has(monkeypatch):
    """`--split` was `choices=["validation", "test"]`, so `--split public` would have
    been refused at the command line for a dataset whose splits are named otherwise.
    It is per dataset now, and each loader refuses what it does not have."""
    import argparse

    from hde.cli import bench_header

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    assert parser.parse_args(["--split", "public"]).split == "public"

    # And the header still records it, so the rows file says which tier produced them.
    args = argparse.Namespace(
        model="m", backend="sglang", decision_head=None, dataset="jevbench",
        split="public", budget=1024, lambdas="0", batch_size=1, repeats=1,
        offset=0, stride=1, limit=231)
    assert bench_header(args, "coded", "sha", {}, [0.0])["split"] == "public"


def test_an_unknown_split_is_refused_by_the_loader_not_by_the_run():
    """Which is why the dataset is loaded before the model: the refusal costs seconds
    rather than a 27B load. 28170 paid the other order's bill on a different guard."""
    import inspect

    from hde.cli import bench_routing

    source = inspect.getsource(bench_routing)
    dataset_at = source.index("loaded = load(args.dataset")
    model_at = min(source.index("TransformersScorer.load("), source.index("SGLangScorer.load("))
    assert dataset_at < model_at, "the dataset has to be loaded before the weights"

    with pytest.raises(ValueError, match="no published tier"):
        load_jevbench(split="validation", fetch=fetcher())
