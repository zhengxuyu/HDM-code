"""Scoring an option by its own text, and the three ways of normalising it.

The coded readout scores single-token letters, which is why it has a preference
over positions at all: a code is a token, a token carries a prior, and the prior
lands on whichever option sits in that slot. Scoring content removes the label
rather than correcting for it.

The three normalisations are kept apart because they fail differently, and these
plant each failure rather than asserting that they differ: sum prefers short
options, mean prefers long ones, and PMI is the only one that is not a length
artefact. A comparison that reported three accuracies and not their
disagreements could not tell which of those was doing the work.
"""
import json
import math

import pytest

from litjev.readout.content_readout import (
    OptionScore,
    as_record,
    chosen_index,
    option_token_ids,
    rank,
    summed_logprob,
)


def score(index, total, tokens, prior=None):
    return OptionScore(index=index, text=f"o{index}", tokens=tokens, total=total, prior=prior)


class Tokenizer:
    """Characters as tokens, so a boundary merge can be arranged deliberately."""

    def __init__(self, merges=()):
        self.merges = set(merges)

    def encode(self, text, add_special_tokens=False):
        for merge in self.merges:
            if merge in text:
                # One token stands for the merged pair, so the prefix's own ids
                # stop being a prefix of the joint ids -- which is the fault.
                text = text.replace(merge, "\x00")
        return [ord(character) for character in text]


# ---------------------------------------------------------------- normalisation


def test_sum_prefers_the_shorter_option():
    """Every extra token subtracts, so a long right answer loses to a short wrong one."""
    short, long = score(0, -6.0, 2), score(1, -9.0, 6)
    assert chosen_index([short, long], "sum") == 0
    assert long.mean > short.mean, "which is exactly what mean is for"


def test_mean_prefers_the_longer_option():
    """A long option's improbable first token is averaged away by its tail."""
    short, long = score(0, -4.0, 1), score(1, -9.0, 6)
    assert chosen_index([short, long], "sum") == 0
    assert chosen_index([short, long], "mean") == 1


def test_pmi_is_the_one_that_is_not_about_length():
    """Two options of very different lengths whose scores are entirely their
    priors: PMI calls them level, and sum and mean do not."""
    scores = [score(0, -4.0, 1, prior=-4.0), score(1, -40.0, 10, prior=-40.0)]
    assert scores[0].pmi == scores[1].pmi == 0.0
    assert chosen_index(scores, "sum") == 0
    assert chosen_index(scores, "mean") == 0, "mean is not length-free either"


def test_pmi_without_the_content_free_pass_is_refused():
    """`None` rather than zero, because zero is a score and absence is not."""
    scores = [score(0, -4.0, 1), score(1, -9.0, 6)]
    assert scores[0].pmi is None
    with pytest.raises(ValueError, match="content-free"):
        rank(scores, "pmi")


def test_an_unknown_normalisation_is_named():
    with pytest.raises(ValueError, match="sum, mean or pmi"):
        rank([score(0, -1.0, 1)], "median")


def test_the_record_keeps_all_three_and_not_the_winner():
    """Which normalisation wins over a benchmark is a weaker fact than where they
    disagree, and the disagreement is not recoverable from a file that kept the
    choice alone."""
    scores = [score(1, -9.0, 6, prior=-8.0), score(0, -4.0, 1, prior=-3.0)]
    rows = as_record(scores)
    assert [row["index"] for row in rows] == [0, 1], "in option order, not score order"
    for row in rows:
        assert set(row) == {"index", "tokens", "sum", "mean", "prior", "pmi"}


# ---------------------------------------------------------------- the boundary


def test_an_option_is_tokenised_after_the_prompt_and_not_alone():
    tokenizer = Tokenizer()
    ids = option_token_ids(tokenizer, "Answer: ", "42")
    assert ids == [ord("4"), ord("2")]


def test_a_merge_across_the_boundary_is_refused():
    """A tokenizer that joins the prompt's last character to the option's first
    makes the option's token count unknowable, so every normalisation would
    divide by the wrong number and the first token's probability would be folded
    into a token that is partly prompt."""
    tokenizer = Tokenizer(merges=[" 4"])
    with pytest.raises(ValueError, match="merges across the boundary"):
        option_token_ids(tokenizer, "Answer: ", "42")


def test_an_option_that_adds_nothing_is_refused():
    with pytest.raises(ValueError, match="nothing to score"):
        option_token_ids(Tokenizer(), "Answer: ", "")


# ---------------------------------------------------------------- the sum itself


def test_only_the_options_own_tokens_are_counted():
    """Counting the prompt's would score the question, identically for every
    option, which hides the error in a constant that cancels in the ranking and
    not in the numbers."""
    vocabulary = 4
    # Position i predicts token i+1, so a prefix of length 3 has its first
    # option token predicted from index 2.
    logprobs = [[math.log(0.1)] * vocabulary for _ in range(6)]
    logprobs[2][1] = math.log(0.5)
    logprobs[3][2] = math.log(0.25)
    total = summed_logprob(logprobs, prefix_length=3, option_ids=[1, 2])
    assert total == pytest.approx(math.log(0.5) + math.log(0.25))


# ---------------------------------------------------------------- end to end


def test_the_scorer_returns_one_score_per_option(tmp_path):
    from test_systemtwo import CharTokenizer, tiny_model

    from litjev.backends.backend import TransformersScorer
    from litjev.data.schema import Choice

    scorer = TransformersScorer(tiny_model(), CharTokenizer())
    question = Choice(instructions="What is 2+2?",
                      criteria={"A": "three", "B": "four", "C": "five point one"})
    scores, rewritten, _ = scorer.score_options("Answer.", question)
    assert rewritten == 0
    assert [s.index for s in scores] == [0, 1, 2]
    assert [s.text for s in scores] == ["three", "four", "five point one"]
    assert [s.tokens for s in scores] == [5, 4, 14], "its own characters, none of the prompt"
    assert all(s.total < 0 for s in scores)
    assert all(s.prior is not None for s in scores), "the content-free pass ran"
    # Nothing was generated: the scores come from a forward pass over given text.
    assert all(isinstance(s.total, float) for s in scores)


def test_the_content_free_pass_can_be_turned_off():
    from test_systemtwo import CharTokenizer, tiny_model

    from litjev.backends.backend import TransformersScorer
    from litjev.data.schema import Choice

    scorer = TransformersScorer(tiny_model(), CharTokenizer())
    question = Choice(instructions="q", criteria={"A": "aa", "B": "bb"})
    scores, _, _ = scorer.score_options("Answer.", question, content_free=False)
    assert all(s.prior is None for s in scores)


def test_the_prompt_carries_no_label_for_an_option():
    """The whole point: there is no code for a prior to attach to."""
    from litjev._legacy.prompting import unlabelled_suffix
    from litjev.data.schema import Choice

    question = Choice(instructions="What is 2+2?", criteria={"A": "3", "B": "4"})
    rendered = unlabelled_suffix(question)
    assert rendered == "Question: What is 2+2?\nOptions:\n3\n4\nAnswer:"
    assert '"code"' not in rendered and '"option"' not in rendered
    # The criteria keys are A and B and neither reaches the prompt.
    assert "A" not in rendered.replace("Answer:", "") and "B" not in rendered


def test_the_comparison_script_runs_end_to_end(tmp_path, monkeypatch, pytestconfig):
    """The script, not its parts. Two of the three faults on this path were a
    thing that worked and a caller that could not reach it."""
    import subprocess
    import sys

    # The tiny model is imported inside the stub, in the subprocess, since that
    # is the process the script runs in.
    stub = tmp_path / "stub.py"
    stub.write_text('''
import json, sys, types
sys.path.insert(0, "tests")
from test_systemtwo import tiny_model, CharTokenizer
from litjev.backends.backend import TransformersScorer, ModelSettings
from litjev.data.datasets import LoadedDataset
import litjev.data.datasets as datasets
from litjev.data.mmlu import McqQuestion

TransformersScorer.load = classmethod(
    lambda cls, settings: cls(tiny_model(), CharTokenizer()))
questions = [McqQuestion(question_id=f"q{i}", prompt=f"what is {i}?",
                         options={"A": "aa", "B": "bbb", "C": "c"}) for i in range(4)]
datasets.LOADERS["mmlu-pro"] = lambda **kw: LoadedDataset(
    name="mmlu-pro", questions=questions,
    labels={q.question_id: "B" for q in questions},
    groups={q.question_id: "math" for q in questions}, provenance={})
import runpy
sys.argv = ["content_vs_coded.py", "--limit", "4", "--stride", "1",
            "--output", sys.argv[1]]
runpy.run_path("scripts/content_vs_coded.py", run_name="__main__")
''')
    out = tmp_path / "found.json"
    result = subprocess.run([sys.executable, str(stub), str(out)],
                            cwd=pytestconfig.rootpath, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    found = json.loads(out.read_text())
    assert found["questions"] == 4
    assert set(found["summary"]) == {"coded", "sum", "mean", "pmi"}
    for way, figures in found["summary"].items():
        assert 0.0 <= figures["accuracy"] <= 1.0, way
        assert 0.0 <= figures["chosen_vs_gold_total_variation"] <= 1.0, way
    # Where they disagree, which is the part an accuracy table cannot show.
    assert "coded|sum" in found["agreement"]
    assert found["seconds_per_question"]["content"] > 0
    # Every option's three scores, in option order, not just the winner.
    assert len(found["rows"][0]["scores"]) == 3
    assert [s["index"] for s in found["rows"][0]["scores"]] == [0, 1, 2]


# ---------------------------------------------------------------- a real tokenizer

REAL = ["Oxygen", "Carbon dioxide", "42", r"\frac{1}{2}", r"(3, \pi/2)", "x = 2", "-5",
        "The answer is B"]


def bpe():
    """A real subword tokenizer, since the stub above is character-level and a
    character-level tokenizer cannot have the fault this guards against."""
    from transformers import AutoTokenizer

    for name in ("Qwen/Qwen2.5-0.5B", "gpt2", "Qwen/Qwen3.5-4B"):
        try:
            return AutoTokenizer.from_pretrained(name, local_files_only=True)
        except Exception:  # noqa: BLE001  any failure here means "not cached"
            continue
    return None


@pytest.mark.parametrize("option", REAL)
def test_a_real_tokenizer_separates_the_option_from_the_prompt(option):
    """The fault a character-level stub cannot show. BPE binds a separating space
    to the word after it, so a prompt ending in one stops being a prefix of the
    joint encoding: measured on gpt2, none of these options was separable with the
    trailing space and all of them are without it."""
    tokenizer = bpe()
    if tokenizer is None:
        pytest.skip("no tokenizer cached locally")
    ids = option_token_ids(tokenizer, "Question: q\nOptions:\na\nb\nAnswer:", " " + option)
    assert ids, option
    assert tokenizer.decode(ids).strip() == option.strip()


def test_a_prompt_ending_in_a_space_is_what_would_have_broken_it():
    """Asserted so the fix cannot be quietly undone by re-adding the space."""
    tokenizer = bpe()
    if tokenizer is None:
        pytest.skip("no tokenizer cached locally")
    broken = sum(
        tokenizer.encode("Answer: " + option, add_special_tokens=False)[
            : len(tokenizer.encode("Answer: ", add_special_tokens=False))]
        != tokenizer.encode("Answer: ", add_special_tokens=False)
        for option in REAL)
    assert broken, "this tokenizer would not have shown the fault; the test guards nothing"

    from litjev._legacy.prompting import unlabelled_suffix
    from litjev.data.schema import Choice

    rendered = unlabelled_suffix(Choice(instructions="q", criteria={"A": "a", "B": "b"}))
    assert not rendered.endswith(" "), "the space belongs to the option"


def test_the_prompt_shows_the_text_that_is_scored():
    r"""Through json.dumps the model reads "\\frac{1}{2}", escaped and quoted,
    while the score is for \frac{1}{2}: the system prompt asks for an option
    exactly as written and the thing scored is not what was written."""
    from litjev._legacy.prompting import unlabelled_suffix
    from litjev.data.schema import Choice

    option = r"\frac{1}{2}"
    rendered = unlabelled_suffix(Choice(instructions="q", criteria={"A": option, "B": "3"}))
    assert f"\n{option}\n" in rendered, rendered
    assert '\\\\' not in rendered, "no JSON escaping between the prompt and the score"
    assert '"' not in rendered


def test_the_listing_is_one_line_per_option_however_the_option_arrived():
    """The count the prompt shows and the count being scored have to agree. A
    carriage return does it as surely as a newline."""
    from litjev._legacy.prompting import listable_options, unlabelled_body
    from litjev.data.schema import Choice

    fine = Choice(instructions="q", criteria={"A": "3", "B": "4"})
    assert unlabelled_body(fine).count("\n") == 3, "instructions, Options:, two options"

    for bad in ("two\nlines", "carriage\rreturn", "many\n\nbreaks"):
        question = Choice(instructions="q", criteria={"A": bad, "B": "4"})
        texts, rewritten = listable_options(question)
        assert rewritten == 1 and "\n" not in texts[0] and "\r" not in texts[0], bad
        assert unlabelled_body(question).count("\n") == 3, bad


def test_the_comparison_reports_a_paired_test_not_two_accuracies():
    """Two accuracies over the same questions invite a difference of 35 in 1500 to
    be read as 2.3 points, when it rests on 313 disagreements and does not clear
    significance. The questions both readouts answer alike carry no information
    about which is better."""
    import math

    # The 4B result, as a worked case: mean beat coded by 35 questions and the
    # paired test puts that at p = 0.055.
    wins, losses = 174, 139
    z = (abs(wins - losses) - 1) / math.sqrt(wins + losses)
    assert math.erfc(z / math.sqrt(2)) == pytest.approx(0.055, abs=0.002)

    from conftest import script_source
    source = script_source("content_vs_coded")
    assert "paired_against" in source, "the result must carry it, not just print it"
    assert "McNemar" in source


def test_the_comparison_runs_a_line_break_under_the_pipelines_own_policy(tmp_path,
                                                                        pytestconfig):
    """It used to refuse, which was right while this compared two readouts on a
    subset. It now picks the default normalisation for the unified path, so it has
    to run under that path's policy -- rewrite to a space in listing and scoring
    alike -- or it would choose a default by measuring something else. Twelve of
    MMLU-Pro's 12032 test questions carry one and none are in the stride-8 1500."""
    import subprocess
    import sys

    stub = tmp_path / "stub.py"
    stub.write_text('''
import sys
sys.path.insert(0, "tests")
from test_systemtwo import tiny_model, CharTokenizer
from litjev.backends.backend import TransformersScorer
from litjev.data.datasets import LoadedDataset
import litjev.data.datasets as datasets
from litjev.data.mmlu import McqQuestion

TransformersScorer.load = classmethod(
    lambda cls, settings: cls(tiny_model(), CharTokenizer()))
questions = [McqQuestion(question_id="q0", prompt="p",
                         options={"A": "two" + chr(10) + "lines", "B": "fine"})]
datasets.LOADERS["mmlu-pro"] = lambda **kw: LoadedDataset(
    name="mmlu-pro", questions=questions, labels={"q0": "B"},
    groups={"q0": "math"}, provenance={})
import runpy
sys.argv = ["content_vs_coded.py", "--limit", "1", "--stride", "1",
            "--no-coded", "--no-content-free", "--output", "x.json"]
runpy.run_path("scripts/content_vs_coded.py", run_name="__main__")
''')
    result = subprocess.run([sys.executable, str(stub)], cwd=pytestconfig.rootpath,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "line break" in result.stdout, result.stdout
    found = json.loads((pytestconfig.rootpath / "x.json").read_text())
    assert found["questions_with_a_line_break"] == ["q0"]
    assert found["rewritten_options"] == 1, "counted, not silent"
    (pytestconfig.rootpath / "x.json").unlink()


def test_the_count_of_rewritten_options_is_returned_not_left_on_the_scorer():
    """As instance state it was overwritten by every later call, including by the
    content-free pass two lines down, so what a caller read afterwards was
    whatever the last question happened to need."""
    from test_systemtwo import CharTokenizer, tiny_model

    from litjev.backends.backend import TransformersScorer
    from litjev.data.schema import Choice

    scorer = TransformersScorer(tiny_model(), CharTokenizer())
    assert not hasattr(scorer, "rewritten_options"), "it must not be instance state"

    broken = Choice(instructions="q",
                    criteria={"A": "two" + chr(10) + "lines", "B": "fine"})
    scores, rewritten, _ = scorer.score_options("Answer.", broken, content_free=False)
    assert rewritten == 1
    assert [s.text for s in scores] == ["two lines", "fine"], "the listed form is scored"
    # The count is the rewritten option's. This stub eats the space at the
    # boundary the way BPE merges there, so it is nine and not ten -- what
    # matters is that no newline was ever tokenised.
    assert scores[0].tokens == len("two lines")

    # A later question does not inherit the earlier one's count.
    _, after, _ = scorer.score_options(
        "Answer.", Choice(instructions="q", criteria={"A": "a", "B": "b"}),
        content_free=False)
    assert after == 0


def test_the_content_free_pass_does_not_report_its_own_rewriting():
    """It runs on a copy of the question, so its count is about the same options
    -- but the number returned must be the real question's, not the last pass's."""
    from test_systemtwo import CharTokenizer, tiny_model

    from litjev.backends.backend import TransformersScorer
    from litjev.data.schema import Choice

    scorer = TransformersScorer(tiny_model(), CharTokenizer())
    question = Choice(instructions="q",
                      criteria={"A": "two" + chr(10) + "lines", "B": "fine"})
    scores, rewritten, _ = scorer.score_options("Answer.", question, content_free=True)
    assert rewritten == 1
    assert all(s.prior is not None for s in scores)


def test_the_coded_column_can_be_skipped(tmp_path, pytestconfig):
    """Roughly half the work, and no information at all about which content
    normalisation is better -- sum and mean come from the same pass."""
    import subprocess
    import sys

    stub = tmp_path / "stub.py"
    stub.write_text('''
import sys
sys.path.insert(0, "tests")
from test_systemtwo import tiny_model, CharTokenizer
from litjev.backends.backend import TransformersScorer
from litjev.data.datasets import LoadedDataset
import litjev.data.datasets as datasets
from litjev.data.mmlu import McqQuestion

class NoCoded(TransformersScorer):
    def score(self, state, schema):
        raise AssertionError("the coded readout must not run")

TransformersScorer.load = classmethod(
    lambda cls, settings: NoCoded(tiny_model(), CharTokenizer()))
questions = [McqQuestion(question_id=f"q{i}", prompt=f"what is {i}?",
                         options={"A": "aa", "B": "bbb", "C": "c"}) for i in range(4)]
datasets.LOADERS["mmlu-pro"] = lambda **kw: LoadedDataset(
    name="mmlu-pro", questions=questions,
    labels={q.question_id: "B" for q in questions},
    groups={q.question_id: "math" for q in questions}, provenance={})
import runpy
sys.argv = ["content_vs_coded.py", "--limit", "4", "--stride", "1",
            "--no-coded", "--no-content-free", "--output", sys.argv[1]]
runpy.run_path("scripts/content_vs_coded.py", run_name="__main__")
''')
    out = tmp_path / "found.json"
    result = subprocess.run([sys.executable, str(stub), str(out)],
                            cwd=pytestconfig.rootpath, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    found = json.loads(out.read_text())
    assert found["coded"] is False
    assert set(found["summary"]) == {"sum", "mean"}, "PMI off too, and no coded"
    # The paired test falls back to sum as the baseline, so the run still says
    # how sure the sum-against-mean difference is.
    assert found["paired_against"]["baseline"] == "sum"
    assert "mean" in found["paired_against"]


def test_the_comparison_takes_a_backend_and_records_which_one_ran(tmp_path, pytestconfig):
    """A readout implemented on both backends is not the same as one reachable on
    both from the script that measures it. This script built `TransformersScorer`
    itself, so the SGLang content readout existed and no benchmark could use it.

    The recorded backend is half the point: the two implementations' scores differ
    by a few percent, so a row that does not say which produced it cannot be put
    beside 28031 or 28037.
    """
    import subprocess
    import sys

    stub = tmp_path / "stub.py"
    stub.write_text('''
import sys
sys.path.insert(0, "tests")
from test_systemtwo import tiny_model, CharTokenizer
from litjev.backends.backend import TransformersScorer
from litjev.data.datasets import LoadedDataset
import litjev.data.datasets as datasets
from litjev.data.mmlu import McqQuestion

TransformersScorer.load = classmethod(
    lambda cls, settings: cls(tiny_model(), CharTokenizer()))
questions = [McqQuestion(question_id=f"q{i}", prompt=f"what is {i}?",
                         options={"A": "aa", "B": "bbb", "C": "c"}) for i in range(2)]
datasets.LOADERS["mmlu-pro"] = lambda **kw: LoadedDataset(
    name="mmlu-pro", questions=questions,
    labels={q.question_id: "B" for q in questions},
    groups={q.question_id: "math" for q in questions}, provenance={})
import runpy
sys.argv = ["content_vs_coded.py", "--limit", "2", "--stride", "1",
            "--backend", "transformers", "--output", sys.argv[1]]
runpy.run_path("scripts/content_vs_coded.py", run_name="__main__")
''')
    out = tmp_path / "found.json"
    result = subprocess.run([sys.executable, str(stub), str(out)],
                            cwd=pytestconfig.rootpath, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(out.read_text())["backend"] == "transformers"

    from conftest import script_source
    source = script_source("content_vs_coded")
    assert "load_proposer(" in source, (
        "through the shared loader, which is what gives it both backends")
    assert "TransformersScorer" not in source, (
        "constructing one backend directly is how the script lost the other")
    assert '"--backend"' in source
