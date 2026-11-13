"""Whether a multiple-choice column is judgement or position, measured rather than argued.

The constructed candidate sets showed a readout taking the second option 99--100%
of the time. MMLU-Pro and AGIEval load their options in source order, so the same
prior there would put part of the fast column in the paper's main tables down to a
position. The accuracy bounds it loosely -- always picking one position scores how
often gold sits there, about a tenth on ten-way questions, against 0.538 and 0.665
-- but a few points is still worth knowing about, and the records carried no
position to check it with.

These run the script, on records built to hold a prior of a known size and on
records that hold none, because a diagnostic that cannot tell the two apart would
report the paper safe either way.
"""
import json
import subprocess
import sys

import numpy as np
import pytest
from conftest import repo_path  # noqa: E402
from test_systemtwo import CharTokenizer, tiny_model

from hde._legacy.heads import STATS_DIM
from hde.backends.backend import TransformersScorer
from hde.backends.slots import SLOT_FORMAT
from hde.data.collect import MAX_OPTIONS, RECORD_FORMAT
from hde.data.schema import Choice, SystemOneRequest

SCRIPT = str(repo_path("scripts", "position_bias.py"))
OPTIONS = 10


def records(path, chosen, gold, record_format=RECORD_FORMAT, positions=True, extra=None):
    count = len(gold)
    probabilities = np.full((count, MAX_OPTIONS), np.nan, dtype=np.float32)
    probabilities[:, :OPTIONS] = 1 / OPTIONS
    payload = {
        "hidden": np.zeros((count, 2, 8), dtype=np.float16),
        "stats": np.zeros((count, STATS_DIM), dtype=np.float32),
        "fast_correct": chosen == gold,
        "slow_correct": np.ones(count, dtype=bool),
        "fast_probability": np.full(count, 0.5),
        "thinking_tokens": np.zeros(count, dtype=np.int64),
        "question_id": np.array([f"q{index}" for index in range(count)]),
        "category": np.array(["math"] * count),
        "question_type": np.array(["choice"] * count),
        "metadata": np.array(json.dumps({
            "model_id": "tiny", "revision": "r", "hidden_size": 8,
            "feature_layers": [-1, 1], "record_format": record_format,
            "slot_format": SLOT_FORMAT, "stats_dim": STATS_DIM, "count": count,
            "dataset": "synthetic", **(extra or {})})),
    }
    if positions:
        payload.update(chosen_index=chosen, gold_index=gold,
                       option_count=np.full(count, OPTIONS),
                       fast_probabilities=probabilities)
    np.savez_compressed(path, **payload)
    return path


def run(tmp_path, path):
    out = tmp_path / "bias.json"
    result = subprocess.run([sys.executable, SCRIPT, str(path), "--output", str(out)],
                            capture_output=True, text=True)
    return result, (json.loads(out.read_text()) if out.exists() else None)


@pytest.fixture
def drawn():
    return np.random.default_rng(0)


def test_a_prior_on_one_position_is_visible(tmp_path, drawn):
    gold = drawn.integers(0, OPTIONS, 400)
    # Right when it is right, and otherwise always the second option.
    chosen = np.where(drawn.random(400) < 0.45, 1, gold)
    result, found = run(tmp_path, records(tmp_path / "biased.npz", chosen, gold))
    assert result.returncode == 0, result.stderr
    assert found["chosen_share"][1] > 0.35, found["chosen_share"]
    assert found["chosen_vs_gold_total_variation"] > 0.2
    # The shape a prior makes: one position much easier than the rest, because
    # every question whose answer sits there is answered by the prior alone.
    accuracy = [found["by_gold_position"][str(index)]["accuracy"] for index in range(OPTIONS)]
    assert accuracy[1] > max(accuracy[:1] + accuracy[2:]) + 0.2


def test_a_readout_with_no_prior_reports_none(tmp_path, drawn):
    """Otherwise the measurement says the paper is safe whatever is true."""
    gold = drawn.integers(0, OPTIONS, 400)
    wrong = (gold + drawn.integers(1, OPTIONS, 400)) % OPTIONS
    chosen = np.where(drawn.random(400) < 0.6, gold, wrong)
    _, found = run(tmp_path, records(tmp_path / "even.npz", chosen, gold))
    assert found["chosen_vs_gold_total_variation"] < 0.1, found["chosen_share"]
    assert max(found["chosen_share"]) < 0.2


def test_records_without_the_columns_say_so_rather_than_report_a_histogram(tmp_path, drawn):
    """Which option a readout chose is not recoverable from whether it was right."""
    gold = drawn.integers(0, OPTIONS, 20)
    path = records(tmp_path / "old.npz", gold.copy(), gold,
                   record_format="head_records_v1", positions=False)
    result, found = run(tmp_path, path)
    assert result.returncode != 0
    assert "re-collection" in result.stderr, result.stderr
    assert found is None, "it must not write a file it could not measure"


def test_a_fast_only_pass_records_the_positions_and_no_slow_answer(tmp_path):
    """The diagnostic wants the fast readout's positions on 1500 questions. Asking
    for the thinking too is most of the cost and none of the answer."""
    from hde.data.collect import ABSENT, collect_batch, load_records, save_records

    model = tiny_model()
    scorer = TransformersScorer(model, CharTokenizer(), feature_layers=(-1, 1))
    request = SystemOneRequest(
        model="hde", state="Answer.",
        questions={f"q{i}": Choice(instructions=f"Pick {i}", criteria={"A": "1", "B": "2"})
                   for i in range(4)})
    labels = {f"q{i}": "A" for i in range(4)}

    class NoThinking(TransformersScorer):
        def think(self, *arguments, **keywords):
            raise AssertionError("the slow path must not run")

    scorer.__class__ = NoThinking
    records = collect_batch(scorer, request, list(labels), labels, 3, fast_only=True)
    assert len(records) == 4
    assert all(record.chosen_index >= 0 for record in records)
    assert all(record.gold_index == 0 for record in records)
    assert all(record.slow_choice == ABSENT for record in records)
    # Unknown, not wrong. A head fitted to these would be fitted to a slow mode
    # that never got anything right.
    with pytest.raises(ValueError, match="unknown"):
        _ = records[0].slow_correct

    path = tmp_path / "fast-only.npz"
    save_records(path, records, {"model_id": "tiny", "revision": "r", "hidden_size": 32,
                                 "feature_layers": [-1, 1], "budget": 3})
    with pytest.raises(ValueError, match="fast-only"):
        load_records([path])
    merged, meta = load_records([path], allow_fast_only=True)
    assert meta["fast_only"] is True
    assert merged["has_slow"].tolist() == [False] * 4
    assert (merged["chosen_index"] >= 0).all()
    # Seven scripts np.load these files directly and read this column without
    # asking whether it means anything. A column of False would tell each of them
    # thinking was never right; its absence stops them at the line that wants it.
    assert "slow_correct" not in merged
    with np.load(path) as raw:
        assert "slow_correct" not in raw.files


def test_a_checkpoint_is_refused_the_way_a_fast_only_file_is(tmp_path, drawn):
    """`partial: True` was written every five batches and read by nothing.

    `hde-collect --checkpoint-every` defaults to 5 and rewrites the output as
    it goes, so a timeout keeps what was collected -- which is the right design
    and was never the gap. The gap is the read side: a collection killed at 900
    of 1500 left a file that loaded, trained and reported as though it held the
    whole selection, with no trace but a `count` nobody compares against the
    `limit` recorded beside it. The salvage is still reachable, deliberately.
    """
    from hde.data.collect import load_records

    gold = drawn.integers(0, OPTIONS, 900)
    path = records(tmp_path / "checkpoint.npz", gold.copy(), gold,
                   extra={"limit": 1500, "partial": True})
    with pytest.raises(ValueError, match="checkpoint"):
        load_records([path])
    # The message carries both numbers: "it is partial" does not tell a reader
    # whether they are missing four records or six hundred.
    with pytest.raises(ValueError, match="1500"):
        load_records([path])

    salvage, meta = load_records([path], allow_partial=True)
    assert meta["partial"] is True
    assert len(salvage["question_id"]) == 900

    # A finished file is unaffected, and the guard keys on the flag rather than
    # on `count` differing from `limit` -- a strided selection that runs out of
    # questions below its limit is the normal case, not a truncation.
    whole = records(tmp_path / "whole.npz", gold.copy(), gold, extra={"limit": 1500})
    loaded, _ = load_records([whole])
    assert len(loaded["question_id"]) == 900


def test_a_fast_only_file_is_not_concatenated_with_an_ordinary_one(tmp_path, drawn):
    """One slow column describing half the rows is worse than neither."""
    from hde.data.collect import load_records

    gold = drawn.integers(0, OPTIONS, 8)
    ordinary = records(tmp_path / "ordinary.npz", gold.copy(), gold)
    lone = records(tmp_path / "lone.npz", gold.copy(), gold)
    with np.load(lone) as raw:
        payload = {key: raw[key] for key in raw.files if key != "slow_correct"}
    meta = json.loads(str(payload.pop("metadata")))
    payload["has_slow"] = np.zeros(len(gold), dtype=bool)
    np.savez_compressed(lone, metadata=np.array(json.dumps({**meta, "fast_only": True})),
                        **payload)
    with pytest.raises(ValueError, match="disagree"):
        load_records([ordinary, lone], allow_fast_only=True)


def test_a_file_from_before_the_column_is_one_the_slow_path_ran_for(tmp_path, drawn):
    """-1 would be a truthy int standing in for a boolean nobody checks the dtype of."""
    from hde.data.collect import load_records

    gold = drawn.integers(0, OPTIONS, 6)
    old_file = records(tmp_path / "old.npz", gold.copy(), gold,
                       record_format="head_records_v1", positions=False)
    new_file = records(tmp_path / "new.npz", gold.copy(), gold)
    merged, _ = load_records([old_file, new_file])
    assert merged["has_slow"].dtype == bool
    assert merged["has_slow"].tolist() == [True] * 12


def test_records_with_and_without_a_slow_answer_cannot_be_saved_together(tmp_path):
    """Nothing downstream could tell which half a slow column described."""
    from hde.data.collect import ABSENT, Record, save_records

    def record(slow):
        return Record(question_id="q", category="m", question_type="choice",
                      hidden=np.zeros((2, 8), dtype=np.float16),
                      stats=np.zeros(STATS_DIM, dtype=np.float32),
                      fast_choice="A", slow_choice=slow, label="A",
                      fast_probability=0.5, slow_probability=0.5, thinking_tokens=1)

    with pytest.raises(ValueError, match="some of these records"):
        save_records(tmp_path / "mixed.npz", [record("A"), record(ABSENT)],
                     {"model_id": "t", "revision": "r", "hidden_size": 8,
                      "feature_layers": [-1]})
