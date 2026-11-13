import json
from collections import Counter

import pytest

from hde.data.datasets import (
    AGIEVAL_EN_MCQ,
    LoadedDataset,
    load,
    load_agieval,
    load_gpqa_diamond,
)
from hde.data.mmlu import McqQuestion


def fake_agieval(rows_by_task):
    def fetch(url):
        task = url.rsplit("/", 1)[-1].removesuffix(".jsonl")
        body = "\n".join(json.dumps(row) for row in rows_by_task[task])
        return body.encode("utf-8")

    return fetch


def test_agieval_strips_option_prefixes_and_keeps_the_label():
    rows = {
        "lsat-ar": [
            {
                "passage": "Eight students give reports.",
                "question": "Which schedule works?",
                "options": ["(A)Helen first", "(B)Robert first", "(C)Nina first"],
                "label": "C",
            }
        ]
    }
    loaded = load_agieval(tasks=("lsat-ar",), fetch=fake_agieval(rows))
    (question,) = loaded.questions
    assert question.options == {"A": "Helen first", "B": "Robert first", "C": "Nina first"}
    assert question.prompt == "Eight students give reports. Which schedule works?"
    assert loaded.labels[question.question_id] == "C"
    assert loaded.groups[question.question_id] == "lsat-ar"


def test_agieval_question_ids_follow_the_pinned_line_order():
    rows = {
        "sat-math": [
            {"question": f"Q{n}", "options": ["(A)1", "(B)2"], "label": "A"} for n in range(3)
        ]
    }
    loaded = load_agieval(tasks=("sat-math",), fetch=fake_agieval(rows))
    assert [q.question_id for q in loaded.questions] == [
        "sat-math-0000",
        "sat-math-0001",
        "sat-math-0002",
    ]


def test_agieval_skips_rows_that_are_not_usable_mcq():
    rows = {
        "sat-math": [
            {"question": "ok", "options": ["(A)1", "(B)2"], "label": "A"},
            {"question": "no label", "options": ["(A)1", "(B)2"], "label": None},
            {"question": "label out of range", "options": ["(A)1", "(B)2"], "label": "D"},
            {"question": "one option", "options": ["(A)1"], "label": "A"},
            {"question": "", "options": ["(A)1", "(B)2"], "label": "A"},
        ]
    }
    loaded = load_agieval(tasks=("sat-math",), fetch=fake_agieval(rows))
    assert len(loaded.questions) == 1
    assert loaded.provenance["skipped"] == 4


def test_agieval_pins_its_source_in_the_manifest():
    rows = {"sat-math": [{"question": "Q", "options": ["(A)1", "(B)2"], "label": "B"}]}
    manifest = load_agieval(tasks=("sat-math",), fetch=fake_agieval(rows)).manifest()
    assert manifest["revision"] == "84ab72d94318290aad2e4ec820d535a95a1f7552"
    assert manifest["source"] == "github.com/ruixiangcui/AGIEval"
    assert manifest["questions"] == 1
    assert manifest["groups"] == ["sat-math"]


def test_digest_is_stable_across_load_order_and_moves_with_content():
    def build(prompt):
        question = McqQuestion(question_id="q1", prompt=prompt, options={"A": "x", "B": "y"})
        return LoadedDataset("d", [question], {"q1": "A"}, {"q1": "g"})

    assert build("same").digest() == build("same").digest()
    assert build("same").digest() != build("different").digest()

    first = McqQuestion(question_id="a", prompt="p", options={"A": "x", "B": "y"})
    second = McqQuestion(question_id="b", prompt="p", options={"A": "x", "B": "y"})
    labels, groups = {"a": "A", "b": "B"}, {"a": "g", "b": "g"}
    forward = LoadedDataset("d", [first, second], labels, groups)
    reverse = LoadedDataset("d", [second, first], labels, groups)
    assert forward.digest() == reverse.digest()


def test_gpqa_shuffles_options_deterministically_and_tracks_the_answer():
    rows = [
        {
            "Question": "Which decays fastest?",
            "Correct Answer": "the muon",
            "Incorrect Answer 1": "the proton",
            "Incorrect Answer 2": "the electron",
            "Incorrect Answer 3": "the neutrino",
            "High-level domain": "Physics",
        }
    ]

    class FakeDatasets:
        @staticmethod
        def load_dataset(*_args, **_kwargs):
            return rows

    import sys

    sys.modules["datasets"] = FakeDatasets
    try:
        once = load_gpqa_diamond()
        twice = load_gpqa_diamond()
        moved = load_gpqa_diamond(seed=7)
    finally:
        del sys.modules["datasets"]

    (question,) = once.questions
    answer = once.labels[question.question_id]
    assert question.options[answer] == "the muon"
    assert sorted(question.options.values()) == sorted(
        ["the muon", "the proton", "the electron", "the neutrino"]
    )
    assert once.digest() == twice.digest()
    assert once.digest() != moved.digest()
    assert once.groups[question.question_id] == "Physics"


def test_load_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="Unknown dataset"):
        load("imagenet")


def test_english_mcq_task_list_excludes_cloze_and_chinese_tasks():
    assert "math" not in AGIEVAL_EN_MCQ
    assert "gaokao-mathcloze" not in AGIEVAL_EN_MCQ
    assert not [task for task in AGIEVAL_EN_MCQ if task.startswith(("gaokao", "jec-qa"))]


def test_gpqa_explains_that_the_source_is_gated():
    import sys

    class Gated:
        @staticmethod
        def load_dataset(*_args, **_kwargs):
            raise OSError("You are trying to access a gated dataset")

    sys.modules["datasets"] = Gated
    try:
        with pytest.raises(RuntimeError, match="gated"):
            load_gpqa_diamond()
    finally:
        del sys.modules["datasets"]


def test_agieval_refuses_a_revision_override():
    rows = {"sat-math": [{"question": "Q", "options": ["(A)1", "(B)2"], "label": "A"}]}
    fetch = fake_agieval(rows)
    with pytest.raises(ValueError, match="pinned to commit"):
        load_agieval(tasks=("sat-math",), fetch=fetch, revision="main")
    pinned = load_agieval(tasks=("sat-math",), fetch=fetch, revision=None)
    assert pinned.provenance["revision"] == "84ab72d94318290aad2e4ec820d535a95a1f7552"


def test_a_loader_with_one_split_refuses_another_rather_than_ignoring_it():
    """code-reviewer's finding, and the fifth of the same shape today: the help said
    every loader refuses a split it does not have, while these two swallowed it in
    `**_`. The data was right and the header was not, which is worse than an error --
    a run recorded `test` while reading `v1_1`, and nothing downstream could tell."""
    import pytest

    from hde.data.datasets import AGIEVAL_SPLIT, GPQA_SPLIT, load_agieval, load_gpqa_diamond

    with pytest.raises(ValueError, match="one split"):
        load_agieval(split="test", fetch=lambda url: b"")
    with pytest.raises(ValueError, match="one split"):
        load_gpqa_diamond(split="test")

    # Its own name, and None, are the two that pass. `fetch` returning nothing is
    # enough to show the split check runs before any data is read.
    loaded = load_agieval(split=AGIEVAL_SPLIT, tasks=(), fetch=lambda url: b"")
    assert loaded.provenance["split"] == AGIEVAL_SPLIT
    assert GPQA_SPLIT == "gpqa_diamond/train"


def test_the_bench_header_records_the_split_the_data_is():
    """Not the one asked for. `--split test` against AGIEval is accepted by argparse,
    refused by the loader now, and if it ever is not, the header has to say `v1_1`."""
    import argparse

    from hde.cli import bench_header

    args = argparse.Namespace(
        model="m", backend="transformers", decision_head=None, dataset="agieval",
        split="test", budget=1024, lambdas="0", batch_size=10, repeats=1,
        offset=0, stride=1, limit=100)
    assert bench_header(args, "coded", "sha", {}, [0.0], split="v1_1")["split"] == "v1_1"
    assert bench_header(args, "coded", "sha", {}, [0.0])["split"] == "test"


def test_an_unset_split_is_left_to_the_dataset_rather_than_forwarded():
    """`1b61be2` made two loaders refuse a split that is not their own, which was
    right, and broke every entry point, which was not: they all default `--split` to
    `test` and send it to whichever dataset was named. The Table 2 launcher failed at
    startup. `None` now means "whatever this one calls its own"."""
    import hde.data.datasets as datasets

    seen = {}
    original = dict(datasets.LOADERS)
    try:
        datasets.LOADERS["probe"] = lambda **kwargs: seen.update(kwargs) or "loaded"
        datasets.load("probe", split=None, revision="r")
        assert "split" not in seen, "an unset split is not the dataset's problem"
        seen.clear()
        datasets.load("probe", split="test")
        assert seen["split"] == "test", "a split that was asked for still arrives"
    finally:
        datasets.LOADERS.clear()
        datasets.LOADERS.update(original)


def test_the_pinned_loaders_accept_an_unset_split():
    from hde.data.datasets import AGIEVAL_SPLIT, load_agieval

    loaded = load_agieval(split=None, tasks=(), fetch=lambda url: b"")
    assert loaded.provenance["split"] == AGIEVAL_SPLIT


def test_no_entry_point_sends_a_split_the_dataset_may_not_have():
    """Every `--split` that reaches `load()` defaults to None. The one exception is
    `hde-mmlu`, which reads MMLU-Pro straight from `load_dataset`, where None means
    every split rather than one -- so it keeps its own default and its choices.

    A source test because the parsers are built inside the functions that run them,
    and because this is a default rather than a behaviour: the failure it prevents is
    a flag nobody passed."""
    import pathlib
    import re

    from conftest import repo_path

    for path in (repo_path("src/hde/cli.py"),
                 repo_path("scripts/content_vs_coded.py"),
                 repo_path("scripts/propose_gate.py"),
                 repo_path("scripts/open_three_policies.py"),
                 repo_path("scripts/settling_layer.py")):
        source = pathlib.Path(path).read_text()
        for match in re.finditer(r'add_argument\(\s*\n?\s*"--split",([^)]*)\)', source):
            body = match.group(1)
            if "choices=" in body:
                assert "hde-mmlu" in source or "load_dataset(" in source, path
                continue
            assert "default=None" in body, f"{path} sends a split the dataset may not have"


def test_banking77_is_twenty_way_and_says_what_it_could_not_match():
    """AnyJev's banking20, as far as their published parameters go.

    Their results file gives `task: banking20`, `n: 300`, `seed: 0`, `k: 20`. It
    does **not** give which 20 of the 77 intents, which 300 items, or the
    ordering the seed draws over, and that is not in their README or `bench/`
    either. So the draw here is ours and the provenance has to say which half is
    which, or a reader will take our number as item-matched to their 0.803.
    """
    pytest.importorskip("datasets")

    from hde.data.datasets import load

    found = load("banking77")
    assert len(found.questions) == 300, "their n"
    assert len({group for group in found.groups.values()}) == 20, "their 20 classes"

    # **Twenty options, not ten.** The first draft zipped the intents against
    # `LABELS`, MMLU-Pro's ten letters, and silently produced a ten-way task
    # that looked right everywhere except the option count.
    for question in found.questions:
        assert len(question.options) == 20, f"{question.question_id} is not 20-way"
        assert set(question.options) == set("ABCDEFGHIJKLMNOPQRST")
    assert all(found.labels[q.question_id] in q.options for q in found.questions)

    # Stratified, so a 20-way accuracy is not dominated by one intent.
    counts = Counter(found.groups.values())
    assert max(counts.values()) - min(counts.values()) <= 1, dict(counts)

    # The provenance separates what matches their setting from what cannot.
    where = found.provenance
    assert where["classes"] == 20 and where["items"] == 300
    assert where["selection_seed"] == 0
    assert set(where["matches_anyjev"]) == {"classes", "items", "seed_value"}
    assert set(where["not_matched"]) == {"which intents", "which items",
                                         "their draw order"}
    assert len(where["intents"]) == 20
    # Theirs says `calib: 200` and we do not know what it calibrates; ours is
    # 300 test items with nothing held out, because nothing here is calibrated.
    assert where["calibration_items"] == 0

    # **The draw must not be the first twenty label ids.** It was, because every
    # intent has exactly 40 test items so "most frequent" selected nothing and
    # the tie-break was label order -- and those twenty contain **all ten**
    # `card_*` intents in the dataset, where a random twenty expects 2.6 and 2000
    # random draws never exceeded 7 of any one prefix. That block is the most
    # confusable twenty available and would have depressed accuracy for a reason
    # unrelated to the method. code-reviewer caught it.
    clusters = Counter(intent.split("_")[0] for intent in where["intents"])
    assert max(clusters.values()) <= 7, (
        f"the draw clusters near-duplicate intents: {clusters.most_common(3)}; a "
        f"random twenty reached 7 at most over 2000 draws")
    assert sum(1 for i in where["intents"] if i.startswith("card")) < 10, \
        "all ten card_* intents means this is the alphabetical block again"
    # And the sampling error that bounds any comparison at this n, recorded with
    # the data rather than left to the write-up: sqrt(.8*.2/300) = 0.023, so two
    # draws of this construction differ by ~0.05 at two standard errors, which is
    # the size of the improvement they report.
    assert where["standard_error_at_0.8"] == pytest.approx(0.023, abs=5e-4)
    # **The comparison needs the SE of a difference of two independent draws**,
    # which is sqrt(2) times one draw's, so two of those is 0.065. My first
    # version used 0.023 and drew bands too narrow at both ends; code-reviewer
    # caught it, and the conclusion it supports is stronger, not weaker: **no
    # outcome of this run licenses ranking our number against theirs.**
    assert where["item_sampling_difference_se"] == pytest.approx(0.033, abs=5e-4)
    assert where["item_sampling_difference_se"] > where["standard_error_at_0.8"]
    assert "intents" in where["uncertainty_not_included"]


def test_the_head_can_read_a_twenty_way_question_though_collection_cannot():
    """`collect.MAX_OPTIONS` is 16 and banking20 has 20, which matters for
    training records and not for evaluation.

    `distribution_stats` is scale-free in the option count by construction -- it
    reduces any K to five scalars and a type one-hot -- so an existing head
    reads a 20-way question with nothing truncated. The stored probability row
    in `collect` is fixed width and would truncate, which is why this dataset is
    evaluation-only until that constant moves. Both halves are asserted so the
    distinction cannot rot into "20 options do not work".
    """
    import numpy as np

    from hde._legacy.heads import distribution_stats
    from hde.data.collect import MAX_OPTIONS

    assert MAX_OPTIONS == 16 < 20, "the constraint this test is about"

    wide = distribution_stats(np.full(20, 1 / 20), "choice")
    narrow = distribution_stats(np.full(10, 1 / 10), "choice")

    # **The width is what is K-independent, and that is what matters**: the head
    # is a fixed-shape MLP, so a 20-way question has to produce the same number
    # of features as a 10-way one, which it does.
    assert wide.shape == narrow.shape == (8,)
    assert np.isfinite(wide).all()

    # My first version of this test asserted that *only* log K differs between
    # the two. That is false and the test caught it: index 0 is the top
    # probability, which for a uniform distribution is 1/20 against 1/10. The
    # statistics are scale-free in the sense of being comparable across K, not
    # invariant to it -- several of them depend on K by definition, and a test
    # claiming otherwise was asserting a property I had not checked.
    assert wide[0] == pytest.approx(0.05) and narrow[0] == pytest.approx(0.10)
    assert wide[4] == pytest.approx(np.log(20)), "log K"

    # Nothing is dropped on the way in: 20 probabilities all reach the summary,
    # which is the claim that makes a 20-way evaluation sound. A truncating
    # version would give the same answer for these two inputs.
    peaked = np.concatenate([[0.9], np.full(19, 0.1 / 19)])
    tail_only = np.concatenate([np.full(19, 0.1 / 19), [0.9]])
    assert distribution_stats(peaked, "choice")[0] == pytest.approx(0.9)
    assert distribution_stats(tail_only, "choice")[0] == pytest.approx(0.9), \
        "the top probability is found wherever it sits, including past index 16"


def test_any_prefix_of_banking77_is_a_sample_and_not_a_subset_of_intents():
    """The defect that invalidated every intermediate read of 29122.

    Each intent occupies a contiguous block of indices in BANKING77, and the
    loader sorted the draw by index, so the twenty intents came out in twenty
    consecutive runs. The first 160 items covered 11 intents, the last 140
    covered 10, and they overlapped by **one**: a prefix was a near-disjoint
    *different task*, not a sample of the same one. Across prefixes of 29122
    `helps` read 0, then 1, then 10 and the accuracy gap read 0.080, 0.056,
    0.044, 0.007 -- four populations, not a convergence.

    It is also a defect and not only a reading hazard: `--limit` below the full
    count silently selected a biased intent subset.
    """
    pytest.importorskip("datasets")

    from hde.data.datasets import load

    found = load("banking77")
    ids = [question.question_id for question in found.questions]
    assert len(ids) == 300

    # Every prefix a caller might take must see the whole label set.
    for size in (100, 160, 200):
        seen = {found.groups[name] for name in ids[:size]}
        assert len(seen) == 20, (
            f"the first {size} items cover {len(seen)} of 20 intents; sorting the "
            f"draw by dataset index is what made a prefix a different task")

    # And the halves must not be near-disjoint, which is the shape that caused it.
    first = {found.groups[name] for name in ids[:160]}
    last = {found.groups[name] for name in ids[160:]}
    assert len(first & last) == 20, f"halves share {len(first & last)} of 20 intents"

    # The ordering is not the dataset's, which is the thing being asserted.
    indices = [int(name.split("-")[1]) for name in ids]
    assert indices != sorted(indices), "sorted by index again"


def test_collect_and_bench_select_the_same_questions_for_one_category_filter():
    """`hde-collect --categories` has to agree with the bench's identical flag.

    The two commands build a run's question set independently, and a record file
    whose questions are not the ones the bench will serve is a head fitted on a
    different population than it is evaluated on. The bench applies the filter
    *after* `--stride`/`--limit`; this pins that collect does the same, because
    filtering first would change which questions the stride lands on.

    The flag is also a cost flag: collecting extra held-out questions in three
    categories costs about a quarter as much with it, since the slow pass runs on
    everything collected.
    """
    from hde.data.datasets import select_questions

    class Q:
        def __init__(self, qid):
            self.question_id = qid

    questions = [Q(f"q{i}") for i in range(40)]
    groups = {f"q{i}": ("keep" if i % 4 == 0 else "drop") for i in range(40)}
    keep = {"keep"}

    def after_stride(offset, stride, limit):
        chosen = select_questions(questions, offset, stride, limit)
        return [q.question_id for q in chosen if groups.get(q.question_id) in keep]

    def before_stride(offset, stride, limit):
        filtered = [q for q in questions if groups.get(q.question_id) in keep]
        return [q.question_id for q in select_questions(filtered, offset, stride, limit)]

    # The order matters, which is why both commands have to choose the same one.
    assert after_stride(0, 8, 40) == ["q0", "q8", "q16", "q24", "q32"]
    assert before_stride(0, 8, 40) == ["q0", "q32"], "filtering first changes the selection"
    assert after_stride(0, 8, 40) != before_stride(0, 8, 40)

    # **And `--offset` does not give a disjoint sample**, which is the trap this
    # test exists to record. `select_questions` strides first and then offsets
    # *inside* the strided sample, deliberately, so that shards tile one sample --
    # see its docstring. So offsets at one stride overlap almost completely, and a
    # second collection at `--offset 1 --stride 8` would re-measure the questions
    # the first one already has rather than adding new ones.
    base = after_stride(0, 8, 40)
    shifted = after_stride(1, 8, 40)
    assert shifted == base[1:], "offset indexes within the strided sample"
    assert len(set(base) & set(shifted)) == len(base) - 1

    # `phase` is what asks for a disjoint sample, and the phases partition the
    # split, so extra held-out questions come from a phase and never an offset.
    def by_phase(phase):
        chosen = select_questions(questions, 0, 8, 40, phase=phase)
        return [q.question_id for q in chosen]

    assert by_phase(0) == base
    seen = set()
    for phase in range(8):
        got = set(by_phase(phase))
        assert seen.isdisjoint(got), f"phase {phase} overlaps an earlier one"
        seen |= got
    assert len(seen) == len(questions), "the phases do not partition the split"
