"""What the distribution over codes does and does not establish.

It does not separate a letter preference from a slot one, and this file said it
did. `candidate_codes` hands out `A, B, C…` in slot order, so a distribution over
letters is a distribution over slots; "slot 6 with probability two thirds, else
slot 1" gives the same marginal as a preference for `G` with `B` behind it.

What it does establish is that the preference is over a few indices rather than
one, which rules out "always the k-th" and "always the last". That is what these
check: one planted preference concentrated on a single index, and one spread over
several, since a reading that comes out the same either way is not a reading.
"""
import csv
import json
import string
import subprocess
import sys

from conftest import repo_path  # noqa: E402

SCRIPT = str(repo_path("scripts", "readout_codes.py"))
LETTERS = string.ascii_uppercase


def write(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "question_id", "temperature", "rank", "degenerate", "option_count",
            "readout_index", "readout_code"])
        writer.writeheader()
        for index, (width, chosen) in enumerate(rows):
            writer.writerow({
                "question_id": f"q{index}", "temperature": "1.0", "rank": "likelihood",
                "degenerate": 0, "option_count": width, "readout_index": chosen,
                "readout_code": LETTERS[chosen]})
    return path


def run(tmp_path, path, *extra):
    out = tmp_path / "codes.json"
    result = subprocess.run([sys.executable, SCRIPT, str(path), "--output", str(out), *extra],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return json.loads(out.read_text()), result.stdout


def test_one_index_at_each_width_leaves_nothing_behind_the_mode(tmp_path):
    """Always the last candidate: every question at a width agrees, so nothing is
    held at the index the mode used to be. This is the shape the data does not have."""
    rows = [(width, width - 1) for width in range(2, 11) for _ in range(20)]
    found, printed = run(tmp_path, write(tmp_path / "slot.csv", rows))
    assert found["held_behind_the_mode"] == {}
    assert "the preference is on one index at each width" in printed


def test_a_preference_over_several_indices_leaves_mass_behind_the_mode(tmp_path):
    """Index 1 where there is no index 6, index 6 once it exists, and index 1 still
    holding a quarter. Consistent with a letter preference and with a slot mixture
    alike -- which is the point, and why only P3 settles it."""
    rows = []
    for width in range(2, 11):
        for index in range(20):
            if width <= 6:
                rows.append((width, 1))
            else:
                rows.append((width, 6 if index >= 5 else 1))
    found, printed = run(tmp_path, write(tmp_path / "code.csv", rows))
    held = found["held_behind_the_mode"]
    assert set(held) == {"7", "8", "9", "10"}, held
    assert held["9"]["mode"] == "G"
    assert held["9"]["still_held"]["B"] == 0.25
    assert "B 0.25 keeps mass behind it" in printed


def test_the_first_listed_candidate_is_counted(tmp_path):
    """The one number that is a fact about the readout whatever explains it."""
    rows = [(5, 0)] * 3 + [(5, 1)] * 97
    found, printed = run(tmp_path, write(tmp_path / "some.csv", rows))
    assert found["first_listed_taken"] == 3
    assert "3 times (3.0%)" in printed


def test_degenerate_questions_have_no_code_to_contribute(tmp_path):
    path = tmp_path / "mixed.csv"
    write(path, [(5, 1)] * 10)
    with open(path, "a", newline="") as handle:
        handle.write("d,1.0,likelihood,1,1,,\n")
    found, _ = run(tmp_path, path)
    assert found["questions"] == 10


def test_a_rank_that_is_not_there_is_said_rather_than_summarised(tmp_path):
    path = write(tmp_path / "one.csv", [(5, 1)] * 5)
    result = subprocess.run([sys.executable, SCRIPT, str(path), "--rank", "frequency"],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "no non-degenerate rows" in result.stderr


# --------------------------------------------- the two backends read the same ids
#
# code-reviewer found the coded readout agreeing on only 78.1% of MMLU-Pro's
# stride-8 1500 between 28037 (transformers) and 28284 (SGLang), with systematic
# shifts rather than near-ties. This is the cause: the two paths computed the
# candidate ids differently, and both "worked".


class SpacedCodeTokenizer:
    """`" A"` and `"A"` are different entries, as they are on a real BPE tokenizer.

    The whole defect lives in that difference, so a tokenizer that collapses it --
    characters as tokens, say -- cannot show it.
    """

    def encode(self, text, **kwargs):
        head, marker, tail = text.rpartition("Answer:")
        prefix = [ord(c) for c in head + marker]
        if tail:
            if tail.startswith(" ") and tail[1:].isalpha():
                # The space-bound code: one token, from a region of the vocabulary
                # the bare letters do not share.
                return prefix + [10000 + sum(ord(c) * 26**i for i, c in enumerate(tail[1:]))]
            return prefix + [ord(c) for c in tail]
        return prefix


def test_both_backends_read_the_same_candidate_token_ids():
    """The one assertion that would have caught it. Each path was self-consistent;
    what nothing checked was that they agree with each other."""
    from inference.sglang_backend import SGLangScorer, SGLangSettings
    from litjev.backends.slots import compile_prefix_slots
    from litjev.data.schema import Choice, DecisionSchema

    tokenizer = SpacedCodeTokenizer()
    schema = DecisionSchema({"q": Choice(instructions="Pick",
                                         criteria={"a": "one", "b": "two", "c": "three"})})
    compiled = compile_prefix_slots(tokenizer, schema, [1])
    transformers_ids = list(compiled.candidates[0])

    scorer = SGLangScorer(engine=None, tokenizer=tokenizer,
                          settings=SGLangSettings(model_id="m", feature_layers=()),
                          hidden_size=8)
    sglang_ids = scorer._candidate_ids(compiled.candidate_codes[0])

    assert sglang_ids == transformers_ids, (
        "the backends read different rows of the same logit vector")
    # And specifically the space-bound token, which is what the model would emit
    # after a colon and what `candidate_codes` chose the code set against.
    assert all(token >= 10000 for token in sglang_ids)
    assert all(token != ord(code) for token, code in zip(sglang_ids, "ABC"))
