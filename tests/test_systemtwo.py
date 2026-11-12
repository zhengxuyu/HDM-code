import json
import math
import re

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from transformers import LlamaConfig, LlamaForCausalLM

from litjev._legacy.decision import HEAD_CONFIDENCE_METHOD, RawFieldScores, SchemaDecisionEngine
from litjev._legacy.heads import (
    STATS_DIM,
    DecisionHead,
    HeadMetadata,
    auroc,
    build_features,
    coverage_accuracy_curve,
    distribution_stats,
    outcome_labels,
    train_head,
)
from litjev._legacy.prompting import ANSWER_BOUNDARY, build_thinking_messages, question_body
from litjev._legacy.routing import RoutingPolicy, escalation_gain, fast_confidence, should_escalate
from litjev.api import create_app
from litjev.backends.backend import SuffixStop, TransformersScorer
from litjev.backends.slots import SLOT_FORMAT
from litjev.data.collect import MAX_OPTIONS, collect_batch, load_records, save_records
from litjev.data.schema import Choice, DecisionSchema, Noul, SystemOneRequest
from litjev.train.training import run_training


class CharTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, **kwargs):
        # Deterministic stand-in: content joined, with a visible marker for thinking mode.
        rendered = "|".join(m["content"] for m in messages)
        return rendered + ("<think>\n" if kwargs.get("enable_thinking") else "|")

    def encode(self, text, **kwargs):
        return [ord(char) + 2 for char in text.replace(": ", ":")]

    def decode(self, ids, **kwargs):
        return "".join(chr(i - 2) for i in ids)


def tiny_model(seed=4):
    torch.manual_seed(seed)
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=256,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
    ).eval()


def two_questions():
    return DecisionSchema(
        {
            "q1": Choice(instructions="Pick", criteria={"A": None, "B": None, "C": None}),
            "q2": Noul(),
        }
    )


# ---------------------------------------------------------------- head & routing


def test_stats_features_and_outcome_labels():
    stats = distribution_stats([0.7, 0.2, 0.1], "choice")
    assert stats.shape == (STATS_DIM,)
    assert stats[0] == pytest.approx(0.7)
    assert stats[2] == pytest.approx(0.5)
    assert list(stats[5:]) == [1.0, 0.0, 0.0]
    assert build_features(np.zeros((2, 4)), [0.5, 0.5], "noul").shape == (8 + STATS_DIM,)
    with pytest.raises(ValueError):
        distribution_stats([1.0], "essay")
    labels = outcome_labels([True, True, False, False], [True, False, True, False])
    assert labels.tolist() == [0, 1, 2, 3]


def test_routing_gain_and_confidence():
    outcome = np.array([0.5, 0.1, 0.3, 0.1])
    assert fast_confidence(outcome) == pytest.approx(0.6)
    # The gain is a log ratio, not a difference: P(slow ok) = 0.8 against
    # P(fast ok) = 0.6, so log(0.8/0.6), not 0.8 - 0.6.
    assert escalation_gain(outcome) == pytest.approx(math.log(0.8 / 0.6))
    assert escalation_gain(outcome) == pytest.approx(0.287682, abs=1e-6)
    assert should_escalate(outcome, RoutingPolicy(0.1, budget=16))
    assert not should_escalate(outcome, RoutingPolicy(0.3, budget=16))
    assert not should_escalate(outcome, RoutingPolicy(0.0, budget=0))
    with pytest.raises(ValueError):
        RoutingPolicy(float("nan"))


def test_head_roundtrip_and_serving_checks(tmp_path):
    meta = HeadMetadata("m", "r", hidden_size=4, feature_layers=(-1, 3), hidden_width=8)
    rng = np.random.default_rng(0)
    features = rng.normal(size=(64, meta.input_dim)).astype(np.float32)
    labels = rng.integers(0, 4, size=64)
    head, history = train_head(meta, features, labels, epochs=2, batch_size=16)
    assert len(history) == 2
    assert {"train_loss", "val_loss"} <= set(history[0])
    probs = head.predict(features[:3])
    assert probs.shape == (3, 4)
    np.testing.assert_allclose(probs.sum(1), 1.0, atol=1e-5)
    path = tmp_path / "head.safetensors"
    head.save(path)
    loaded = DecisionHead.load(path)
    assert loaded.metadata == meta
    np.testing.assert_allclose(loaded.predict(features[:3]), probs, atol=1e-6)
    loaded.metadata.check_serving("m", "r")
    with pytest.raises(ValueError):
        loaded.metadata.check_serving("other")
    # PCA bottleneck: whitened components replace the raw hidden block, stats pass through.
    pca_meta = HeadMetadata(
        "m", "r", hidden_size=4, feature_layers=(-1, 3), pca_dim=3, hidden_width=8
    )
    pca_head, _ = train_head(pca_meta, features, labels, epochs=2, batch_size=16)
    assert pca_head.projection.shape == (8, 3)
    assert pca_meta.model_dim == 3 + STATS_DIM
    pca_head.save(tmp_path / "pca.safetensors")
    reloaded = DecisionHead.load(tmp_path / "pca.safetensors")
    np.testing.assert_allclose(
        reloaded.predict(features[:2]), pca_head.predict(features[:2]), atol=1e-6
    )
    with pytest.raises(ValueError):
        DecisionHead(HeadMetadata("m", "r", hidden_size=4, feature_layers=(-1,), pca_dim=9))
    with pytest.raises(ValueError):
        loaded.metadata.check_serving("m", "different-revision")


def test_auroc_and_coverage_curve():
    assert auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert auroc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == 0.0
    assert auroc([0.5, 0.5, 0.5, 0.5], [True, False, True, False]) == 0.5
    assert np.isnan(auroc([0.1, 0.2], [True, True]))
    curve = coverage_accuracy_curve(
        gain=[0.4, -0.3, 0.1],
        fast_correct=[False, True, True],
        slow_correct=[True, False, True],
        lambdas=[0.0, 0.2, 1.0],
    )
    assert [row["escalation_rate"] for row in curve] == pytest.approx([2 / 3, 1 / 3, 0.0])
    assert [row["accuracy"] for row in curve] == pytest.approx([1.0, 1.0, 2 / 3])


# ---------------------------------------------------------------- backend


def test_fast_path_returns_readout_hidden_states():
    """The coded readout produces the state a decision head is fitted on.

    The content readout produces a different one, from a prompt that lists the
    options with no labels, which is why a head carries the readout it was trained
    under. `test_engine_construction` checks that contrast through the engine; this
    is where the guarantee for the real scorer lives, because a stub can only show
    that what it was given came back.
    """
    model = tiny_model()
    scorer = TransformersScorer(model, CharTokenizer(), feature_layers=(-1, 1))
    schema = two_questions()
    scores = scorer.score("state", schema)
    compiled = scorer._compile("state", schema)
    assert scorer.hidden_size == 32
    with torch.inference_mode():
        for i, row in enumerate(compiled.input_ids):
            assert scores[i].hidden.shape == (2, 32)
            full = model(torch.tensor([row]), use_cache=False, output_hidden_states=True)
            np.testing.assert_allclose(
                scores[i].hidden[0], full.hidden_states[-1][0, -1].numpy(), atol=1e-5
            )
            np.testing.assert_allclose(
                scores[i].hidden[1], full.hidden_states[1][0, -1].numpy(), atol=1e-5
            )
    assert scores[0].provenance["system"] == "one"
    assert scores[0].provenance["feature_layers"] == [-1, 1]


def test_slow_path_reads_same_boundary_after_thinking():
    model = tiny_model()
    tokenizer = CharTokenizer()
    scorer = TransformersScorer(model, tokenizer, feature_layers=(-1,))
    schema = two_questions()
    compiled = scorer._compile("state", schema)
    slow = scorer.think("state", schema, ("q2", "q1"), budget=5)
    assert tuple(row.name for row in slow) == ("q2", "q1")
    with torch.inference_mode():
        for row in slow:
            i = schema.names.index(row.name)
            assert row.provenance["system"] == "two"
            assert 0 <= row.generated_tokens <= 5
            thought = tokenizer.encode(row.provenance["thinking_text"])
            assert len(thought) == row.generated_tokens
            body = question_body(schema[row.name], compiled.candidate_codes[i])
            prompt = tokenizer.apply_chat_template(
                build_thinking_messages("state", body), enable_thinking=True
            )
            assert prompt.endswith("<think>\n") and "state\n\n" + body in prompt
            sequence = (
                tokenizer.encode(prompt)
                + thought
                + tokenizer.encode("\n</think>\n" + ANSWER_BOUNDARY)
            )
            assert row.input_tokens == len(sequence)
            full = model(torch.tensor([sequence]), use_cache=False, output_hidden_states=True)
            np.testing.assert_allclose(
                row.logits, full.logits[0, -1, compiled.candidates[i]].numpy(), atol=1e-5
            )
            np.testing.assert_allclose(
                row.hidden[0], full.hidden_states[-1][0, -1].numpy(), atol=1e-5
            )
    with pytest.raises(ValueError):
        scorer.think("state", schema, ("missing",), budget=5)
    with pytest.raises(ValueError):
        scorer.think("state", schema, ("q1",), budget=0)


def test_slow_path_strips_closing_delimiter_and_padding(monkeypatch):
    model = tiny_model()
    tokenizer = CharTokenizer()
    scorer = TransformersScorer(model, tokenizer)
    schema = two_questions()
    close = tokenizer.encode("</think>")
    body = tokenizer.encode("xy")

    def fake_generate(input_ids, attention_mask, **kwargs):
        width = input_ids.shape[1]
        tail = torch.tensor([body + close + [0, 0]] * input_ids.shape[0])
        return torch.cat([input_ids, tail], dim=1)[:, : width + len(body) + len(close) + 2]

    monkeypatch.setattr(scorer.model, "generate", fake_generate)
    (row,) = scorer.think("state", schema, ("q1",), budget=8)
    assert row.generated_tokens == 2
    assert row.provenance["thinking_text"] == "xy"


def test_suffix_stop_is_per_row():
    stop = SuffixStop([7, 8], prompt_width=2)
    ids = torch.tensor([[1, 1, 7, 8], [1, 1, 8, 7], [1, 1, 0, 0]])
    assert stop(ids, None).tolist() == [True, False, False]
    assert stop(torch.tensor([[1, 1, 7]]), None).tolist() == [False]


# ---------------------------------------------------------------- engine


class FakeProvider:
    def __init__(self):
        self.think_calls = []

    def score(self, state, schema):
        return (
            RawFieldScores("q1", np.array([2.0, 1.0, 0.0]), 20, {"system": "one"}, np.ones((1, 4))),
            RawFieldScores("q2", np.array([0.0, 1.0]), 20, {"system": "one"}, np.ones((1, 4))),
        )

    def think(self, state, schema, names, budget):
        self.think_calls.append((names, budget))
        return tuple(
            RawFieldScores(
                name,
                np.array([0.0, 0.0, 3.0]) if name == "q1" else np.array([3.0, 0.0]),
                40,
                {"system": "two", "thinking_text": "hmm"},
                np.ones((1, 4)),
                generated_tokens=7,
            )
            for name in names
        )


class FakeHead:
    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.metadata = HeadMetadata("m", "r", hidden_size=4, feature_layers=(-1,))

    def predict(self, features):
        return np.array([self.outcomes.pop(0)])


def test_engine_escalates_only_when_gain_exceeds_lambda():
    provider = FakeProvider()
    head = FakeHead([[0.2, 0.1, 0.6, 0.1], [0.8, 0.1, 0.05, 0.05]])
    engine = SchemaDecisionEngine(provider, head=head, routing=RoutingPolicy(0.2, budget=32), readout="coded")
    schema = two_questions()
    evaluation = engine.evaluate("state", schema)
    result = evaluation.result
    assert provider.think_calls == [(("q1",), 32)]
    assert result.answers["q1"].choice == "C"
    assert result.answers["q1"].confidence == pytest.approx(0.8)
    assert result.answers["q2"].noul == pytest.approx(1 / (1 + np.exp(-1)))
    assert result.usage.output_tokens == 7
    fields = evaluation.diagnostics["fields"]
    assert fields["q1"]["system"] == "two"
    assert fields["q1"]["fast_probabilities"]["A"] > 0.6
    # log P(slow ok) - log P(fast ok): the outcome is [0.2, 0.1, 0.6, 0.1], so
    # 0.8 against 0.3. Escalation is unchanged -- both readings clear lambda=0.2.
    assert fields["q1"]["escalation_gain"] == pytest.approx(math.log(0.8 / 0.3))
    assert fields["q2"]["system"] == "one"
    assert evaluation.diagnostics["routing"]["escalated"] == ["q1"]
    assert evaluation.diagnostics["forward_calls"] == 2 + 2 + 7
    assert evaluation.diagnostics["confidence_method"] == HEAD_CONFIDENCE_METHOD


def test_engine_without_routing_never_thinks_and_head_needs_hidden():
    provider = FakeProvider()
    head = FakeHead([[0.2, 0.1, 0.6, 0.1], [0.8, 0.1, 0.05, 0.05]])
    engine = SchemaDecisionEngine(provider, head=head, readout="coded")
    evaluation = engine.evaluate("state", two_questions())
    assert provider.think_calls == []
    assert evaluation.result.answers["q1"].confidence == pytest.approx(0.3)
    assert evaluation.result.usage.output_tokens == 0
    assert evaluation.diagnostics["forward_calls"] == 2
    with pytest.raises(ValueError):
        SchemaDecisionEngine(provider, routing=RoutingPolicy(0.0, 8), readout="coded").evaluate("s", two_questions())

    class NoHidden(FakeProvider):
        def score(self, state, schema):
            return tuple(
                RawFieldScores(row.name, row.logits, row.input_tokens)
                for row in super().score(state, schema)
            )

    with pytest.raises(RuntimeError):
        SchemaDecisionEngine(NoHidden(), head=FakeHead([[1, 0, 0, 0]]), readout="coded").evaluate(
            "s", two_questions()
        )

    class StatsOnlyHead(FakeHead):
        def __init__(self, outcomes):
            super().__init__(outcomes)
            self.metadata = HeadMetadata("m", "r", hidden_size=4, feature_layers=())

    # A stats-only head needs no hidden states from the scorer at all.
    stats_engine = SchemaDecisionEngine(
        NoHidden(), head=StatsOnlyHead([[0.7, 0.1, 0.1, 0.1], [0.2, 0.2, 0.5, 0.1]])
    , readout="coded")
    result = stats_engine.evaluate("s", two_questions()).result
    assert result.answers["q1"].confidence == pytest.approx(0.8)


# ---------------------------------------------------------------- collection & training


def test_a_record_scores_the_type_it_was_collected_under():
    """The 92 of JevBench's 231 items that could not be correct on any path.

    `question.choices` is strings for all three types -- an option key, a level
    index as a string, "false"/"true" -- while datasets.py normalises the gold
    label per type to a key, an int and a bool. So `fast_choice == label` was
    `'3' == 3` for every score item and `'true' == True` for every noul one,
    which are False whatever the model answered. 74 noul plus 18 score is exactly
    92, and every one of them read as a question both modes got wrong rather than
    as a comparison that could not succeed.

    `routing_bench._answer_key` has always converted correctly, which is why the
    bench path was right and only the collect path was wrong, and why the two
    disagreed by 60 accuracy points on the same items and model.
    """
    from litjev.data.collect import Record, choice_form, gold_form

    # A score question answers with a level; the gold is an int.
    assert gold_form("3", "score") == 3
    assert choice_form(3, "score") == "3"
    # A noul question answers true or false; the gold is a bool.
    assert gold_form("true", "noul") is True
    assert gold_form("false", "noul") is False
    assert choice_form(True, "noul") == "true"
    # A choice question is already in the gold's form.
    assert gold_form("B", "choice") == "B"
    assert choice_form("B", "choice") == "B"

    for question_type, chosen, gold in (("score", "3", 3), ("noul", "true", True),
                                        ("choice", "B", "B")):
        record = Record(
            question_id="q", category="c", question_type=question_type,
            hidden=np.zeros((1, 4), dtype=np.float32),
            stats=np.zeros(STATS_DIM, dtype=np.float32),
            fast_choice=gold_form(chosen, question_type),
            slow_choice=gold_form(chosen, question_type),
            label=gold, fast_probability=1.0, slow_probability=1.0, thinking_tokens=0,
        )
        assert record.fast_correct, f"{question_type}: a right answer scored wrong"
        assert record.slow_correct, f"{question_type}: a right answer scored wrong"

    # And the bug itself, pinned: the raw choice string against the normalised
    # gold is wrong for the two types that are not `choice`, whatever was answered.
    for question_type, chosen, gold in (("score", "3", 3), ("noul", "true", True)):
        uncorrected = Record(
            question_id="q", category="c", question_type=question_type,
            hidden=np.zeros((1, 4), dtype=np.float32),
            stats=np.zeros(STATS_DIM, dtype=np.float32),
            fast_choice=chosen, slow_choice=chosen, label=gold,
            fast_probability=1.0, slow_probability=1.0, thinking_tokens=0,
        )
        assert not uncorrected.fast_correct, (
            f"{question_type}: comparing the raw choice against a normalised gold "
            "now scores correct, so this test no longer pins the defect it was written for"
        )


def test_collect_save_load_and_train(tmp_path):
    model = tiny_model()
    scorer = TransformersScorer(model, CharTokenizer(), feature_layers=(-1, 1))
    request = SystemOneRequest(
        model="litjev",
        state="Answer.",
        questions={
            f"q{i}": Choice(instructions=f"Pick {i}", criteria={"A": "1", "B": "2"})
            for i in range(6)
        },
    )
    labels = {f"q{i}": "A" for i in range(6)}
    categories = {f"q{i}": "math" if i % 2 else "law" for i in range(6)}
    records = collect_batch(scorer, request, list(labels), labels, 3, categories)
    assert len(records) == 6
    assert records[0].hidden.shape == (2, 32)
    assert records[0].stats.shape == (STATS_DIM,)
    # Where the answer sat and how close the call was, so a question in the main
    # table can be audited without re-running the benchmark to find out.
    assert records[0].gold_index == 0, "gold is A, the first option"
    assert records[0].chosen_index in (0, 1)
    assert len(records[0].fast_probabilities) == 2
    assert abs(sum(records[0].fast_probabilities) - 1) < 1e-6
    metadata = {
        "model_id": "tiny",
        "revision": "r",
        "hidden_size": 32,
        "feature_layers": [-1, 1],
        "budget": 3,
    }
    path = tmp_path / "records.npz"
    save_records(path, records, metadata)
    loaded, meta = load_records([path, path])
    assert loaded["hidden"].shape == (12, 2, 32)
    assert loaded["chosen_index"].shape == (12,)
    assert loaded["fast_probabilities"].shape == (12, MAX_OPTIONS)
    assert loaded["option_count"].tolist() == [2] * 12
    assert not np.isnan(loaded["fast_probabilities"][:, :2]).any()
    assert np.isnan(loaded["fast_probabilities"][:, 2:]).all(), "padding is absence"
    assert meta["hidden_size"] == 32
    assert meta["count"] == 6
    with pytest.raises(ValueError):
        save_records(tmp_path / "empty.npz", [], metadata)

    rng = np.random.default_rng(0)
    loaded["fast_correct"] = rng.random(12) < 0.6
    loaded["slow_correct"] = rng.random(12) < 0.7
    head, report = run_training(
        loaded, meta, holdout_fraction=0.5, epochs=2, probe_epochs=1, selection_folds=2, pca_dim=2
    )
    assert report["split_level"] == "category"
    assert report["categories_seen"] == ["law", "math"]
    names = [c["name"] for c in report["candidates"]]
    # Per layer the sweep is a 2x2 in (projection, width), not a diagonal. It used
    # to be two cells, `layer_X` at width 256 with no projection and
    # `layer_X_pca{d}` at width 64 with one, so every PCA-versus-raw difference
    # was also a fourfold narrowing -- and the paper read one such difference as
    # PCA alone until exps caught it. The two extra cells break that tie.
    assert names == [
        "stats_only",
        "layer_-1", "layer_-1_pca2", "layer_-1_w64", "layer_-1_pca2_w256",
        "layer_1", "layer_1_pca2", "layer_1_w64", "layer_1_pca2_w256",
    ]
    assert report["chosen"] in names
    assert all({"validation", "test", "selection_score"} <= set(c) for c in report["candidates"])
    assert all(c["validation"]["folds"] == 2 for c in report["candidates"])
    # A single-category collection cannot hold out a category; fall back to examples.
    single = {**loaded, "category": np.array(["law"] * 12)}
    _, fallback = run_training(
        single, meta, holdout_fraction=0.5, epochs=1, probe_epochs=1, selection_folds=2, pca_dim=2
    )
    assert fallback["split_level"] == "example"
    assert 0 < fallback["test_count"] < 12
    assert set(report["by_objective"]) == {"supervised", "cdpo"}
    assert report["objective"] == "supervised"
    for result in report["by_objective"].values():
        assert 0 <= result["ece_fast_correct"] <= 1 and 0 <= result["brier_fast_correct"] <= 1
    assert 0 < report["train_count"] < 12
    assert set(report["head"]) >= {"auroc_fast_correct", "auroc_gain_vs_helps", "curve"}
    assert head.metadata.training["chosen_layers"] == report["chosen_layers"]
    out = tmp_path / "head.safetensors"
    head.save(out)
    assert DecisionHead.load(out).metadata.feature_layers == tuple(report["chosen_layers"])
    json.dumps(report)


def test_cdpo_objective_trains_and_prefers_escalating_helpful_questions():
    from litjev._legacy.heads import CDPOConfig, cdpo_loss

    rng = np.random.default_rng(1)
    meta = HeadMetadata("m", "r", hidden_size=4, feature_layers=(-1,), hidden_width=16)
    # Feature 0 marks questions where thinking helps (label 2); others are label 0.
    helps = rng.random(200) < 0.4
    features = rng.normal(size=(200, meta.input_dim)).astype(np.float32)
    features[:, 0] += 3 * helps
    labels = np.where(helps, 2, 0)
    head, history = train_head(
        meta, features, labels, epochs=60, batch_size=16, patience=20, objective="cdpo", seed=1
    )
    assert "val_loss" in history[0]
    outcome = head.predict(features)
    gain = outcome[:, 2] - outcome[:, 1]
    assert gain[helps].mean() > gain[~helps].mean() + 0.3
    assert auroc(gain, helps) > 0.9
    with pytest.raises(ValueError):
        train_head(meta, features, labels, epochs=1, objective="reinforce")
    # Loss is finite and the policy term rewards escalating exactly the helpful ones.
    logits = torch.zeros(4, 4)
    loss = cdpo_loss(
        logits,
        torch.tensor([0, 1, 2, 3]),
        torch.ones(4),
        CDPOConfig(lambda_range=(0.0, 0.0)),
        torch.Generator().manual_seed(0),
    )
    assert torch.isfinite(loss)


# ---------------------------------------------------------------- API


def test_systemtwo_endpoint_passes_routing_and_keeps_standard_shape():
    from litjev._legacy.decision import ChoiceAnswer, DecisionResponse, Evaluation, Usage

    class RecordingEngine:
        model_id = "litjev"

        def __init__(self):
            self.calls = []

        def evaluate(self, state, schema, routing=None):
            self.calls.append(routing)
            answer = ChoiceAnswer("choice", "A", {"A": 0.9, "B": 0.1}, 0.9)
            return Evaluation(
                DecisionResponse("tiny", {"q1": answer}, Usage(10, 5 if routing else 0)),
                {"fields": {}, "forward_calls": 2, "calibration_fitted": False},
            )

    engine = RecordingEngine()
    client = TestClient(create_app(lambda: engine))
    body = {
        "model": "litjev",
        "state": "s",
        "questions": {
            "q1": {"type": "choice", "instructions": "?", "criteria": {"A": "1", "B": "2"}}
        },
    }
    plain = client.post("/v1/systemtwo", json=body)
    assert plain.status_code == 200
    assert set(plain.json()) == {"model", "answers", "usage"}
    assert engine.calls[-1] is None
    routed = client.post("/v1/systemtwo", json={**body, "routing": {"lambda": 0.1, "budget": 64}})
    assert routed.status_code == 200
    assert routed.json()["usage"]["output_tokens"] == 5
    assert engine.calls[-1] == RoutingPolicy(0.1, 64)
    assert client.post("/v1/systemone", json={**body, "routing": {}}).status_code == 422
    assert client.post("/v1/systemtwo", json={**body, "routing": {"budget": 0}}).status_code == 422
    debug = client.post("/v1/systemtwo/debug", json={**body, "routing": {"lambda": 0.0}})
    assert debug.status_code == 200
    assert engine.calls[-1] == RoutingPolicy(0.0, 512)


# ---------------------------------------------------------------- routing benchmark


def test_bench_policies_measures_time_tokens_and_accuracy():
    from litjev.bench.routing_bench import bench_policies, format_table, mean_over_repeats

    class BenchHead(FakeHead):
        def predict(self, features):
            # q1: thinking helps (gain 0.5); q2: thinking hurts (gain -0.3).
            return (
                np.array([[0.2, 0.1, 0.6, 0.1]])
                if self.calls % 2 == 0
                else np.array([[0.6, 0.3, 0.0, 0.1]])
            )

        def __init__(self):
            super().__init__([])
            self.calls = -1

    head = BenchHead()
    original = head.predict

    def counting_predict(features):
        head.calls += 1
        return original(features)

    head.predict = counting_predict
    provider = FakeProvider()
    engine = SchemaDecisionEngine(provider, head=head, readout="coded")
    request = SystemOneRequest(
        model="litjev",
        state="s",
        questions={
            "q1": Choice(instructions="Pick", criteria={"A": "1", "B": "2", "C": "3"}),
            "q2": Choice(instructions="Pick", criteria={"A": "1", "B": "2"}),
        },
    )
    # Fast answers: q1 -> A (wrong), q2 -> B (right). Slow: q1 -> C (right), q2 -> A (wrong).
    labels = {"q1": "C", "q2": "B"}
    # 1.2 is above every gain the head returns here -- q1's is log(0.8/0.3) =
    # 0.981 -- so it escalates nothing. It was 0.6 when the gain was a difference.
    report = bench_policies(engine, [(request, ["q1", "q2"])], labels, [0.0, 1.2], budget=16)
    policies = report["policies"]
    assert set(policies) == {"fast_only", "slow_only", "routed_lambda_0", "routed_lambda_1.2"}
    assert policies["fast_only"]["accuracy"] == 0.5
    assert policies["fast_only"]["tokens_per_question"] == 0
    assert policies["slow_only"]["accuracy"] == 0.5
    assert policies["slow_only"]["escalation_rate"] == 1.0
    assert policies["slow_only"]["tokens_per_question"] == 7.0
    # lambda 0: only q1 escalates (gain 0.5 > 0 > -0.3): half the tokens, both right.
    routed = policies["routed_lambda_0"]
    assert routed["escalation_rate"] == 0.5
    assert routed["tokens_per_question"] == 3.5
    assert routed["accuracy"] == 1.0
    assert routed["gain_kept"] is None  # slow-only is no better than fast-only here
    assert routed["tokens_vs_slow"] == 0.5
    # lambda 1.2: nothing escalates; identical to fast-only.
    assert policies["routed_lambda_1.2"]["escalation_rate"] == 0.0
    assert policies["routed_lambda_1.2"]["accuracy"] == 0.5
    assert all(item["seconds_per_question"] >= 0 for item in policies.values())
    assert len(report["batches"]) == 4
    table = format_table(policies)
    assert "routed_lambda_0 " in table and "slow_only" in table
    merged = mean_over_repeats([report, report])
    assert merged["routed_lambda_0"]["tokens_per_question"] == 3.5
    assert merged["routed_lambda_0"]["seconds_std"] == 0.0
    # Where the answer sat, not only whether it was right. A readout with a
    # position prior and one reading the question give the same accuracy column.
    fast = next(row for row in report["batches"] if row["policy"] == "fast_only")
    positions = fast["position_by_question"]
    assert positions["q1"]["options"] == 3 and positions["q2"]["options"] == 2
    assert positions["q1"]["chosen_index"] == 0, "fast answers q1 with A"
    assert positions["q1"]["gold_index"] == 2, "gold for q1 is C"
    assert len(positions["q1"]["probabilities"]) == 3
    assert abs(sum(positions["q1"]["probabilities"]) - 1) < 1e-6
    slow = next(row for row in report["batches"] if row["policy"] == "slow_only")
    assert slow["position_by_question"]["q1"]["chosen_index"] == 2, "slow answers q1 with C"


class SteadyHead(FakeHead):
    """Predicts the same outcome every time, so the timing tests are about timing."""

    def predict(self, features):
        return np.array([[0.2, 0.1, 0.6, 0.1]])


def bench_request():
    """The two questions FakeProvider scores; a schema it does not is refused."""
    return SystemOneRequest(
        model="litjev", state="s",
        questions={"q1": Choice(instructions="Pick", criteria={"A": "1", "B": "2", "C": "3"}),
                   "q2": Choice(instructions="Pick", criteria={"A": "1", "B": "2"})})


def test_the_bench_says_where_it_is_on_every_batch(capsys):
    """It printed every tenth batch and nothing during warmup, so a 27B run with a
    budget of 8192 and seven policies to a batch could go ten hours without a line.
    A log that says nothing cannot distinguish a slow job from a hung one."""
    from litjev.bench.routing_bench import bench_policies

    provider = FakeProvider()
    engine = SchemaDecisionEngine(provider, head=SteadyHead([]), readout="coded")
    request = bench_request()
    batches = [(request, ["q1", "q2"])] * 3
    bench_policies(engine, batches, {"q1": "C", "q2": "B"}, [0.0], budget=8, label="warmup")
    lines = [line for line in capsys.readouterr().out.splitlines() if "warmup" in line]
    assert len(lines) == len(batches), lines
    assert "warmup 1/3" in lines[0] and "warmup 3/3" in lines[-1]
    # A wall clock, so a log read the next morning can be placed against when the
    # node was allocated; elapsed and remaining, so a slow run is legible as slow.
    assert re.match(r"^\d\d:\d\d:\d\d warmup 1/3", lines[0]), lines[0]
    assert "min in" in lines[0] and "min left" in lines[0] and "per batch" in lines[0]


def test_the_bench_can_still_be_asked_to_be_quiet(capsys):
    from litjev.bench.routing_bench import bench_policies

    engine = SchemaDecisionEngine(FakeProvider(), head=SteadyHead([]), readout="coded")
    bench_policies(engine, [(bench_request(), ["q1", "q2"])], {"q1": "C", "q2": "B"},
                   [0.0], budget=8, progress=None)
    assert "batch" not in capsys.readouterr().out


def test_records_from_before_the_position_columns_still_load(tmp_path):
    """The 1500-question MMLU-Pro set the paper's head is trained on is one of
    them, so a format bump that forced a re-collection would cost more than it
    bought. The columns come back as absent rather than as an error."""
    import json

    path = tmp_path / "v1.npz"
    count = 3
    np.savez_compressed(
        path,
        hidden=np.zeros((count, 2, 32), dtype=np.float16),
        stats=np.zeros((count, STATS_DIM), dtype=np.float32),
        fast_correct=np.ones(count, dtype=bool),
        slow_correct=np.ones(count, dtype=bool),
        fast_probability=np.full(count, 0.5),
        thinking_tokens=np.zeros(count, dtype=np.int64),
        question_id=np.array([f"q{i}" for i in range(count)]),
        category=np.array(["math"] * count),
        question_type=np.array(["choice"] * count),
        metadata=np.array(json.dumps({
            "model_id": "tiny", "revision": "r", "hidden_size": 32,
            "feature_layers": [-1, 1], "record_format": "head_records_v1",
            "slot_format": SLOT_FORMAT, "stats_dim": STATS_DIM, "count": count})))
    loaded, meta = load_records([path])
    assert meta["record_format"] == "head_records_v1"
    assert loaded["chosen_index"].tolist() == [-1] * count, "-1 says absent, not first"
    assert np.isnan(loaded["fast_probabilities"]).all()


def test_a_resumed_bench_produces_the_same_summary_as_an_uninterrupted_one(tmp_path):
    """The guarantee the streaming exists for. `bench_routing` wrote its JSON once at
    the end, which is how 27862 lost four hours and ten minutes of 27B with nothing
    on disk; the transfer reruns are longer than that.

    What is asserted is not that rows appear but that **resuming changes no number**.
    The summary is computed from the rows rather than accumulated as the run goes,
    so this holds by construction -- and this test is what says the construction is
    the one in the code.
    """
    import json

    from litjev.bench.routing_bench import bench_policies
    from litjev.run.streaming import Rows, completed

    def build():
        head = FakeHead([])
        head.predict = lambda features: np.array([[0.2, 0.1, 0.6, 0.1]])
        return SchemaDecisionEngine(FakeProvider(), head=head, readout="coded")

    request = SystemOneRequest(
        model="litjev", state="s",
        questions={"q1": Choice(instructions="Pick", criteria={"A": "1", "B": "2", "C": "3"}),
                   "q2": Choice(instructions="Pick", criteria={"A": "1", "B": "2"})},
    )
    batches = [(request, ["q1", "q2"])]
    labels = {"q1": "C", "q2": "B"}

    whole = bench_policies(build(), batches, labels, [0.0], budget=16)

    # A first pass that writes its rows, then a second that finds them all done.
    path = tmp_path / "rows.jsonl"
    rows_file = Rows(str(path), config={"run": "one"})
    first = bench_policies(build(), batches, labels, [0.0], budget=16,
                           rows_file=rows_file, done={})
    rows_file.close()
    done = completed(str(path))
    assert done, "the rows were written as the run went, not at the end"

    class Refuses:
        """Any evaluate here means a row that was already done was run again."""

        def evaluate(self, *arguments, **keywords):
            raise AssertionError("a completed row was re-run instead of being read back")

    resumed = bench_policies(Refuses(), batches, labels, [0.0], budget=16, done=done)

    for name in whole["policies"]:
        for figure in ("accuracy", "tokens_per_question", "escalation_rate", "questions"):
            assert resumed["policies"][name][figure] == whole["policies"][name][figure], (
                f"{name}.{figure} moved when the run was resumed")
            assert first["policies"][name][figure] == whole["policies"][name][figure]

    # Seconds are the one figure that cannot be reproduced, and the resumed run keeps
    # the measured ones rather than re-timing them in another process.
    assert resumed["policies"]["slow_only"]["seconds_per_question"] == \
        first["policies"]["slow_only"]["seconds_per_question"]

    # And the row carries what a killed run would otherwise lose.
    row = next(r for r in json.loads(json.dumps(list(done.values()))) if r["policy"] == "slow_only")
    assert row["question_ids"] == ["q1", "q2"]
    assert set(row["escalated_by_question"]) == {"q1", "q2"}
    assert set(row["tokens_by_question"]) == {"q1", "q2"}
    assert "reasoning_by_question" in row, "what the slow mode wrote, which 27866 lost"


def test_the_resume_key_is_the_questions_and_not_the_batch_number():
    """A run whose selection shifted would otherwise resume at the right count and the
    wrong questions, scoring one against another's gold answer -- the hazard
    `streaming.completed` was written for, worse here because a row covers a whole
    batch under one policy."""
    from litjev.bench.routing_bench import batch_key

    assert batch_key(["q1", "q2"]) != batch_key(["q2", "q1"]), "order is part of the work"
    assert batch_key(["q1", "q2"]) != batch_key(["q1", "q3"])
    assert batch_key(["q1", "q2"]) == batch_key(["q1", "q2"])


def test_a_resume_on_another_machine_says_so_rather_than_averaging_two_clocks():
    """code-reviewer caught the comment claiming a refusal the code did not perform:
    `describe_environment` records the interpreter and package versions, with no
    hostname, so a cross-node resume was never refused and the seconds column
    silently mixed two machines. Refusing would make resume useless on Slurm, where
    a resubmission rarely lands on the same node, so the row carries where it was
    measured and the summary says when they differ."""
    from litjev.bench.routing_bench import _summarise

    def row(policy, node, seconds):
        return {"policy": policy, "node": node, "seconds": seconds, "scored": 2,
                "in_batch": 2, "correct": 1, "tokens": 4, "escalated": 1}

    one = _summarise([row("fast_only", "507-13", 1.0), row("fast_only", "507-13", 1.2)],
                     ["fast_only"])
    assert one["fast_only"]["nodes"] == ["507-13"]
    assert one["fast_only"]["seconds_mixed_nodes"] is False

    two = _summarise([row("fast_only", "507-13", 1.0), row("fast_only", "507-14", 1.2)],
                     ["fast_only"])
    assert two["fast_only"]["nodes"] == ["507-13", "507-14"]
    assert two["fast_only"]["seconds_mixed_nodes"] is True
    # Accuracy and tokens do not depend on the machine, so they are not marked.
    assert two["fast_only"]["accuracy"] == one["fast_only"]["accuracy"]
    assert two["fast_only"]["tokens_per_question"] == one["fast_only"]["tokens_per_question"]


def test_every_bench_row_records_the_machine_it_was_measured_on():
    from litjev.bench.routing_bench import bench_policies

    head = FakeHead([])
    head.predict = lambda features: np.array([[0.2, 0.1, 0.6, 0.1]])
    engine = SchemaDecisionEngine(FakeProvider(), head=head, readout="coded")
    request = SystemOneRequest(
        model="litjev", state="s",
        questions={"q1": Choice(instructions="Pick", criteria={"A": "1", "B": "2", "C": "3"}),
                   "q2": Choice(instructions="Pick", criteria={"A": "1", "B": "2"})},
    )
    report = bench_policies(engine, [(request, ["q1", "q2"])], {"q1": "C", "q2": "B"},
                            [0.0], budget=16, node="507-13/H100")
    assert {row["node"] for row in report["batches"]} == {"507-13/H100"}
    assert report["policies"]["fast_only"]["nodes"] == ["507-13/H100"]


def test_the_rows_header_the_benchmark_writes_round_trips_and_refuses_another_run(tmp_path):
    """The resume test drives `bench_policies` directly, so the header the CLI writes
    was never exercised -- code-reviewer's note. This covers the part that can fail
    on the cluster and nowhere else: a header written by one process has to be
    accepted by the next, and refused when the run differs."""
    import argparse

    import pytest

    from litjev.cli import bench_header
    from litjev.run.streaming import Rows

    args = argparse.Namespace(
        model="qwen38-27b", backend="sglang", decision_head="head-27b-lastlayer.safetensors",
        dataset="gpqa-diamond", split="train", budget=1024, lambdas="0.1,0.05,0",
        batch_size=10, repeats=1, offset=0, stride=1, limit=198,
    )
    header = bench_header(args, "coded", "abc1234", {"python": "/x/bin/python"},
                          [0.1, 0.05, 0.0])
    path = tmp_path / "rows.jsonl"
    Rows(str(path), config=header).close()

    # The same run resumes into it.
    Rows(str(path), config=header, resume=True).close()

    # A different one does not, and the message names what differs.
    moved = dict(header, budget=8192)
    with pytest.raises(ValueError, match="budget"):
        Rows(str(path), config=moved, resume=True)

    # And the readout is part of that identity, since two readouts are two
    # measurements and a file holding both could not be read.
    with pytest.raises(ValueError, match="readout"):
        Rows(str(path), config=dict(header, readout="content"), resume=True)

    # The grid is stored parsed, so the same run typed with a space resumes into
    # itself. It used to be the raw string, which refused on whitespace.
    spaced = argparse.Namespace(**{**vars(args), "lambdas": "0.1, 0.05, 0"})
    Rows(str(path), config=bench_header(spaced, "coded", "abc1234",
                                        {"python": "/x/bin/python"}, [0.1, 0.05, 0.0]),
         resume=True).close()


def test_averaging_repeats_keeps_which_machines_the_seconds_came_from():
    """`mean_over_repeats` rebuilt the merged summary from a fixed key list, so with
    --repeats > 1 the thing tables are read from lost the node fields that had just
    been added one call down. code-reviewer's note; the transfer rerun uses
    --repeats 1 and so would not have shown it."""
    from litjev.bench.routing_bench import mean_over_repeats

    def report(nodes, mixed, seconds):
        return {"policies": {"fast_only": {
            "accuracy": 0.5, "tokens_per_question": 1.0, "seconds_per_question": seconds,
            "median_batch_seconds": seconds, "escalation_rate": 0.0, "questions": 2,
            "nodes": nodes, "seconds_mixed_nodes": mixed}}}

    same = mean_over_repeats([report(["507-13"], False, 1.0), report(["507-13"], False, 1.2)])
    assert same["fast_only"]["nodes"] == ["507-13"]
    assert same["fast_only"]["seconds_mixed_nodes"] is False

    # Two repeats, each clean on its own machine, are mixed once averaged.
    across = mean_over_repeats([report(["507-13"], False, 1.0), report(["507-14"], False, 1.2)])
    assert across["fast_only"]["nodes"] == ["507-13", "507-14"]
    assert across["fast_only"]["seconds_mixed_nodes"] is True

    # And a repeat that was itself mixed stays mixed.
    inherited = mean_over_repeats([report(["507-13", "507-14"], True, 1.0)])
    assert inherited["fast_only"]["seconds_mixed_nodes"] is True


def test_the_content_readout_returns_the_state_at_its_own_readout_position():
    """Where the guarantee lives for the real scorer, as the coded one has above.

    The position is the **last prefix token**, not the end of the row: a content row
    continues past it with the option's own tokens, so reading the row's end would
    give a state that depends on which option that row carried -- and a head fitted
    to that would be reading the options, not the question.

    Checked against the model's own `hidden_states`, and checked to be the same
    vector in every option's row, which is why one state for the question is the
    right shape to return.
    """
    from litjev._legacy.prompting import build_content_messages, unlabelled_suffix
    from litjev.data.schema import Choice

    model = tiny_model()
    tokenizer = CharTokenizer()
    scorer = TransformersScorer(model, tokenizer, feature_layers=(-1, 1))
    question = Choice(instructions="What is 2+2?", criteria={"A": "three", "B": "four"})

    scores, _, hidden = scorer.score_options("Answer.", question, content_free=False)
    assert hidden.shape == (2, 32), "one vector per layer, not one per option"

    prefix = tokenizer.apply_chat_template(
        build_content_messages("Answer."), tokenize=False,
        add_generation_prompt=True, enable_thinking=False) + unlabelled_suffix(question)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    position = len(prefix_ids) - 1

    with torch.inference_mode():
        for option in ("three", "four"):
            row = prefix_ids + tokenizer.encode(prefix + " " + option,
                                                add_special_tokens=False)[len(prefix_ids):]
            full = model(torch.tensor([row]), use_cache=False, output_hidden_states=True)
            # The same state in every row, because the prefix is identical and
            # attention is causal -- which is the argument for reading it from one.
            np.testing.assert_allclose(hidden[0], full.hidden_states[-1][0, position].numpy(),
                                       atol=1e-5)
            np.testing.assert_allclose(hidden[1], full.hidden_states[1][0, position].numpy(),
                                       atol=1e-5)
            assert not np.allclose(hidden[0], full.hidden_states[-1][0, -1].numpy()), (
                "the row's end is a different position and depends on the option")
    assert len(scores) == 2


def test_no_feature_layers_means_the_content_readout_returns_no_state():
    """And not a zero vector, which a head would read as a reading."""
    scorer = TransformersScorer(tiny_model(), CharTokenizer())
    from litjev.data.schema import Choice

    _, _, hidden = scorer.score_options(
        "Answer.", Choice(instructions="q", criteria={"A": "aa", "B": "bb"}),
        content_free=False)
    assert hidden is None


def test_the_margin_rule_is_anyjevs_published_one_in_both_its_halves():
    """AnyJev's adaptive mode, reimplemented so we can compare against it.

    Their README states it in two parts and both set the operating point:
    escalate when "the log-probability gap between the top two options is below
    `tau`" (their default 0.5), and "at most `max_cot_share` of the last
    `cot_window` decisions escalate" (their defaults 20% of 1000). A version with
    only the threshold would route a different share of a long run, and the
    comparison would then be against a rule they did not publish.
    """
    from litjev._legacy.routing import MarginPolicy, MarginRouter, top_two_margin

    # The margin is a log-probability gap taken from the normalised
    # distribution, so it means the same thing under either readout; the raw
    # logits are unnormalised and their differences would not be gaps.
    assert top_two_margin([0.9, 0.1]) == pytest.approx(np.log(9.0))
    assert top_two_margin([0.5, 0.5]) == pytest.approx(0.0)
    assert top_two_margin([0.4, 0.3, 0.3]) == pytest.approx(np.log(4 / 3))
    # One candidate is nothing to be unsure between, so it never escalates
    # rather than always. The other default would escalate every single-option
    # question a benchmark contains.
    assert top_two_margin([1.0]) == float("inf")
    assert top_two_margin([]) == float("inf")

    router = MarginRouter(MarginPolicy(tau=0.5, budget=1024))
    assert router.decide([0.5, 0.5]) is True, "no margin, so escalate"
    assert router.decide([0.95, 0.05]) is False, "log 19 is well above 0.5"
    assert MarginRouter(MarginPolicy(tau=0.0, budget=1024)).decide([0.5, 0.5]) is False, \
        "a tau of zero escalates nothing, since a gap is never below zero"

    capped = MarginRouter(MarginPolicy(tau=0.5, cap_share=0.2, window=10, budget=1024))
    taken = [capped.decide([0.5, 0.5]) for _ in range(10)]
    assert sum(taken) == 2, f"20% of a window of ten is two, got {taken}"
    assert taken[:2] == [True, True] and not any(taken[2:]), \
        "the cap binds once the window holds its share, and not retroactively"
    for _ in range(9):
        capped.decide([0.95, 0.05])
    assert capped.decide([0.5, 0.5]) is True, "the window rolled past them; there is room"

    off = MarginRouter(MarginPolicy(tau=0.5, budget=0))
    assert off.decide([0.5, 0.5]) is False
    assert off.escalated_share == 0.0, "a declined decision still enters the window"

    assert MarginPolicy(tau=0.5, budget=1024).enabled
    assert not MarginPolicy(tau=0.5, budget=0).enabled
    with pytest.raises(ValueError, match="tau must be finite"):
        MarginPolicy(tau=-1.0)
    with pytest.raises(ValueError, match="share of recent decisions"):
        MarginPolicy(cap_share=1.5)
    with pytest.raises(ValueError, match="at least one decision"):
        MarginPolicy(window=0)


def test_a_margin_router_and_a_priced_policy_cannot_both_decide_one_question():
    """Two rules on one question makes the escalation rate unattributable: the
    row would say neither what the head did nor what the threshold did."""
    from litjev._legacy.decision import RawFieldScores, SchemaDecisionEngine
    from litjev._legacy.routing import MarginPolicy, MarginRouter, RoutingPolicy
    from litjev.data.schema import Choice, DecisionSchema

    class TwoQuestions:
        def score(self, state, schema):
            return (RawFieldScores("q1", np.array([3.0, 1.0]), 12),
                    RawFieldScores("q2", np.array([0.0, 4.0]), 12))

    engine = SchemaDecisionEngine(TwoQuestions())
    schema = DecisionSchema({"q1": Choice(criteria={"A": None, "B": None}),
                             "q2": Choice(criteria={"A": None, "B": None})})
    router = MarginRouter(MarginPolicy(tau=0.5, budget=16))
    with pytest.raises(ValueError, match="one or the other"):
        engine.evaluate("state", schema, RoutingPolicy(0.1, budget=16), router=router)

    # Alone it is accepted, and routes without a head -- which is the point: it
    # fits nothing, so it is available wherever the fast mode is.
    assert engine.head is None
    evaluation = engine.evaluate("state", schema, router=router)
    assert evaluation is not None


def test_an_escalation_under_the_margin_rule_gets_a_budget_to_think_with():
    """code-reviewer's RED on 78b3c2c, and it would have decided an experiment.

    With a router, `evaluate` disables the priced policy -- and `policy.budget`
    is the same field that reaches `solve` and `think`. So every escalation the
    margin rule made would have thought with **zero tokens**, and the run would
    have compared a trained head against a threshold that cannot think. It would
    have read as the threshold failing.

    No test reached it before: replacing the router's `escalate.append` with
    `pass` left the suite green. This one records the budget the provider is
    handed, which is the quantity that was wrong.
    """
    from litjev._legacy.decision import RawFieldScores, SchemaDecisionEngine
    from litjev._legacy.routing import MarginPolicy, MarginRouter
    from litjev.data.schema import Choice, DecisionSchema

    spent_against = []

    class Unsure:
        """A tie on every question, so the margin rule escalates all of them."""

        def score(self, state, schema):
            return tuple(RawFieldScores(name, np.array([1.0, 1.0]), 12)
                         for name in schema.names)

        def think(self, state, schema, names, budget):
            spent_against.append(budget)
            return tuple(RawFieldScores(name, np.array([5.0, 0.0]), 12)
                         for name in names)

    engine = SchemaDecisionEngine(Unsure())
    schema = DecisionSchema({"q1": Choice(criteria={"A": None, "B": None})})
    router = MarginRouter(MarginPolicy(tau=0.5, budget=1024))

    engine.evaluate("state", schema, router=router)
    assert spent_against == [1024], (
        f"the slow mode was given {spent_against} tokens to think with; the "
        f"router's budget is 1024 and a zero here is the RED")

    # And the row says what was spent against, not the disabled policy's zero,
    # or the provenance would disagree with the run.
    evaluation = engine.evaluate("state", schema, router=MarginRouter(
        MarginPolicy(tau=0.5, budget=256)))
    routing = evaluation.diagnostics["routing"]
    assert routing["budget"] == 256, routing
