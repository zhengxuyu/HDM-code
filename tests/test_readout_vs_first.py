"""Telling three readings of one inversion apart, on records built to hold each.

The readout sits below candidate one at high temperatures and above it at low
ones. Three things produce that and the gate's own two numbers cannot separate
them: the construction (at a low temperature the candidates are near-duplicates,
so taking the second costs nothing), the denominator (58% of questions are
degenerate at 0.1 and excluded, so the temperatures are scored on different
question sets), and the readout itself.

Each is planted here in turn, because a diagnostic that reports the same thing
whichever is true would settle the question by assertion.
"""
import json
import subprocess
import sys

import pytest
from conftest import repo_path  # noqa: E402

SCRIPT = str(repo_path("scripts", "readout_vs_first.py"))


def gate(path, temperature, entries, rank="likelihood"):
    """A `propose_gate --output` file holding the given per-question entries."""
    rows = [{"id": key, "gold": "7", "by_rank": {rank: entry}}
            for key, entry in entries.items()]
    path.write_text(json.dumps({"temperature": temperature, "rows": rows}))
    return path


def entry(reason=None, candidates=3, first_right=False, readout_right=False, index=1):
    if reason is not None:
        return {"reason": reason, "candidates": candidates, "first": None,
                "gold_rank": None, "gold_in_proposals": False}
    return {"reason": None, "candidates": candidates, "first": "7", "gold_rank": 0,
            "gold_in_proposals": True, "readout": "7", "readout_index": index,
            "first_right": first_right, "readout_right": readout_right}


def run(tmp_path, *paths):
    out = tmp_path / "cut.json"
    result = subprocess.run(
        [sys.executable, SCRIPT, *[str(p) for p in paths], "--output", str(out)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return json.loads(out.read_text()), result.stdout


def test_a_gap_that_is_the_denominator_disappears_on_a_shared_question_set(tmp_path):
    """The questions excluded at 0.1 are the ones the readout is bad at, so the
    two temperatures were never scored on the same thing."""
    hard = {f"h{i}": entry(first_right=True, readout_right=False) for i in range(20)}
    easy = {f"e{i}": entry(first_right=True, readout_right=True) for i in range(20)}
    # At 0.1 the hard ones are degenerate and drop out; at 1.0 everything counts.
    low = {**{key: entry(reason="degenerate") for key in hard}, **easy}
    high = {**hard, **easy}
    found, _ = run(tmp_path,
                   gate(tmp_path / "low.json", 0.1, low),
                   gate(tmp_path / "high.json", 1.0, high))
    assert found["shared_questions"] == 20
    low_found, high_found = found["by_temperature"]["t=0.1"], found["by_temperature"]["t=1.0"]
    # As reported: the readout looks better at 0.1 than at 1.0.
    assert low_found["reported"]["readout"] > high_found["reported"]["readout"]
    # On the same questions: identical, so it was the denominator.
    assert low_found["shared"]["readout"] == pytest.approx(high_found["shared"]["readout"])


def test_a_gap_that_survives_the_shared_set_and_moves_with_the_candidate_count(tmp_path):
    """The construction reading: two near-duplicate candidates cost nothing to
    confuse, three real ones do."""
    pairs = {f"p{i}": entry(candidates=2, first_right=True, readout_right=True)
             for i in range(20)}
    triples = {f"t{i}": entry(candidates=3, first_right=True, readout_right=False)
               for i in range(20)}
    both = {**pairs, **triples}
    found, printed = run(tmp_path,
                         gate(tmp_path / "low.json", 0.1, both),
                         gate(tmp_path / "high.json", 1.0, both))
    assert found["shared_questions"] == 40
    counted = found["by_temperature"]["t=1.0"]["by_candidate_count"]
    assert counted["2"]["readout"] == 1.0 and counted["3"]["readout"] == 0.0
    assert "shared, 2 candidates" in printed and "shared, 3 candidates" in printed


def test_a_readout_that_is_simply_worse_survives_both_cuts(tmp_path):
    """Then it is neither the questions nor the candidates, and the second forward
    pass does not earn its place."""
    ids = [f"q{i}" for i in range(20)]
    low = {key: entry(first_right=True, readout_right=True) for key in ids}
    high = {key: entry(first_right=True, readout_right=False) for key in ids}
    found, _ = run(tmp_path,
                   gate(tmp_path / "low.json", 0.1, low),
                   gate(tmp_path / "high.json", 1.0, high))
    assert found["shared_questions"] == 20
    high_found = found["by_temperature"]["t=1.0"]
    assert high_found["shared"]["readout"] == 0.0 and high_found["shared"]["first"] == 1.0
    assert high_found["by_candidate_count"]["3"]["readout"] == 0.0


def test_the_readout_index_share_is_over_questions_that_had_a_choice(tmp_path):
    """A question excluded for having no choice has no index to contribute."""
    entries = {**{f"q{i}": entry(index=1) for i in range(9)},
               "d": entry(reason="degenerate")}
    found, _ = run(tmp_path, gate(tmp_path / "one.json", 0.6, entries))
    assert found["by_temperature"]["t=0.6"]["readout_index_share"]["1"] == pytest.approx(1.0)
    assert found["by_temperature"]["t=0.6"]["reported"]["n"] == 9


def test_a_rank_the_file_does_not_hold_is_named_rather_than_averaged_over_nothing(tmp_path):
    path = gate(tmp_path / "one.json", 0.6, {"q": entry()}, rank="likelihood")
    result = subprocess.run([sys.executable, SCRIPT, str(path), "--rank", "frequency"],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "likelihood" in result.stderr, result.stderr
