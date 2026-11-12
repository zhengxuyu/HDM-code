"""Constructing a candidate set, and the two ways it can fail to be one.

The construction is only useful if what comes out is a choice: several distinct
answers, ordered as the model ranked them. The tests that matter are the ones
where it is not — beams that differ by a full stop, beams that collapse to one
answer — because those are what a real beam returns and what would otherwise
reach the head as a question with options that are the same option.
"""
import pytest

from litjev.candidates.construct import (
    DEGENERATE,
    EMPTY,
    candidates_from_beams,
    normalise,
    schema_from_beams,
)


def test_beams_that_differ_only_in_writing_are_one_answer():
    beams = [("42", -0.1), (" 42 ", -0.3), ("42.", -0.5), ("42.0", -0.9)]
    assert [text for text, _ in candidates_from_beams(beams)] == ["42"]


def test_a_single_surviving_answer_is_not_a_choice():
    question, texts, reason = schema_from_beams(
        [("42", -0.1), ("42.", -0.2)], instructions="what is it")
    assert question is None and reason == DEGENERATE and texts == ["42"]


def test_no_beams_at_all_is_reported_rather_than_crashing():
    question, texts, reason = schema_from_beams([("   ", -0.1)], instructions="x")
    assert question is None and reason == EMPTY and texts == []


def test_the_beam_order_survives_into_the_options():
    """Beam one has to stay first: whether the readout departs from it is the question."""
    question, texts, reason = schema_from_beams(
        [("seven", -2.0), ("six", -0.5), ("five", -1.0)], instructions="how many")
    assert reason is None
    assert texts == ["six", "five", "seven"]
    assert question.descriptions == ("six", "five", "seven")


def test_more_beams_than_codes_are_cut_at_the_limit_not_silently_kept():
    beams = [(f"answer {i}", -float(i)) for i in range(80)]
    kept = candidates_from_beams(beams, limit=52)
    assert len(kept) == 52
    assert kept[0][0] == "answer 0"


def test_unicode_minus_and_hyphen_are_the_same_number():
    assert normalise("−5") == normalise("-5")


def test_the_constructed_question_is_an_ordinary_choice():
    """Nothing downstream may need to know where the candidates came from."""
    question, _, _ = schema_from_beams(
        [("a", -0.1), ("b", -0.2)], instructions="pick")
    assert question.type == "choice"
    # Keyed by the answer, not by its position. The key is what comes back as the
    # answer, so an index key would make the system's output "1" where the answer
    # is "b" -- an index standing for an answer, which is what the no-thinking
    # output must never be.
    assert question.choices == ("a", "b")
    assert question.descriptions == ("a", "b")


def test_a_constructed_question_carries_no_trace_for_the_slow_mode_to_read():
    """The slow mode renders its own prompt; the beams must live only in the choice.

    If the construction ever wrote the candidates into the instructions, the slow
    mode would inherit them through the question body and stop being the policy
    the always-thinking baseline measures.
    """
    question, texts, _ = schema_from_beams(
        [("alpha", -0.1), ("beta", -0.2)], instructions="what is it")
    assert question.instructions == "what is it"
    for text in texts:
        assert text not in str(question.instructions)


def test_scores_order_rather_than_the_order_they_arrive_in():
    kept = candidates_from_beams([("low", -5.0), ("high", -0.1), ("mid", -1.0)])
    assert [text for text, _ in kept] == ["high", "mid", "low"]


def test_an_empty_limit_is_refused():
    with pytest.raises(ValueError):
        from litjev.data.schema import Choice

        Choice(instructions="x", criteria={})


def test_frequency_ranks_agreement_above_likelihood():
    """A sampler's repeats are a signal a beam cannot produce; the rank must use them."""
    from litjev.candidates.construct import FREQUENCY, LIKELIHOOD

    proposals = [("7", -0.1), ("4", -2.0), ("4", -2.1), ("4", -2.3), ("9", -3.0)]
    assert [t for t, _ in candidates_from_beams(proposals, rank=LIKELIHOOD)][0] == "7"
    assert [t for t, _ in candidates_from_beams(proposals, rank=FREQUENCY)][0] == "4"


def test_frequency_breaks_ties_on_likelihood():
    from litjev.candidates.construct import FREQUENCY

    proposals = [("a", -1.0), ("a", -1.0), ("b", -0.2), ("b", -0.3)]
    assert [t for t, _ in candidates_from_beams(proposals, rank=FREQUENCY)][0] == "b"


def test_duplicates_collapse_into_one_candidate_under_either_rank():
    from litjev.candidates.construct import FREQUENCY, LIKELIHOOD

    proposals = [("42", -0.5)] * 9 + [("41", -0.9)]
    for rank in (LIKELIHOOD, FREQUENCY):
        kept = candidates_from_beams(proposals, rank=rank)
        assert [text for text, _ in kept] == ["42", "41"]


def test_an_unknown_rank_is_refused():
    with pytest.raises(ValueError, match="unknown rank"):
        candidates_from_beams([("a", 0.0), ("b", -1.0)], rank="vibes")


def test_a_sampler_that_collapsed_is_reported_as_degenerate():
    """Temperature zero returns one answer width times; that is not a choice."""
    question, texts, reason = schema_from_beams([("42", -0.1)] * 10, instructions="x")
    assert question is None and reason == DEGENERATE


def test_integers_need_no_symbolic_comparison():
    from litjev.candidates.construct import equivalent

    assert equivalent("042", "42", symbolic=False)
    assert equivalent("42.0", "42", symbolic=False)
    assert not equivalent("41", "42", symbolic=False)


def test_expressions_written_differently_are_one_answer():
    pytest.importorskip("math_verify", reason="expression answers need the maths extra")
    from litjev.candidates.construct import equivalent

    assert equivalent("3/4", "\\frac{3}{4}")
    assert equivalent("0.75", "\\frac{3}{4}")
    assert equivalent("\\frac34", "\\frac{3}{4}")
    assert not equivalent("\\frac{1}{4}", "\\frac{3}{4}")


def test_an_unparseable_answer_is_not_a_match_rather_than_an_error():
    pytest.importorskip("math_verify", reason="expression answers need the maths extra")
    from litjev.candidates.construct import equivalent

    assert not equivalent("\\frac{", "\\frac{3}{4}")


def test_a_missing_comparator_says_how_to_install_it_rather_than_guessing(monkeypatch):
    """Comparing expressions as strings would undercount coverage and fail the gate."""
    from litjev.candidates.construct import equivalent

    monkeypatch.setitem(__import__("sys").modules, "math_verify", None)
    with pytest.raises(RuntimeError, match="math_verify"):
        equivalent("3/4", "\\frac{3}{4}")


def test_strict_and_lenient_comparison_are_not_the_same_test():
    """One decides a ceiling and the other an accuracy column; they cannot share a lean."""
    pytest.importorskip("math_verify", reason="expression answers need the maths extra")
    from litjev.candidates.construct import correct, equivalent

    # 2*pi against 2\pi parses only outside maths delimiters, so the lenient test
    # that tries both accepts it and the strict one does not.
    assert equivalent("2*pi", "2\\pi")
    assert not correct("2*pi", "2\\pi")
    # Anything the strict test accepts, the lenient one accepts too.
    for candidate, gold in (("3/4", "\\frac{3}{4}"), ("42", "42")):
        assert not correct(candidate, gold) or equivalent(candidate, gold)


def test_an_answer_is_taken_from_the_marker_the_model_was_asked_for():
    from litjev.candidates.construct import extract_answer

    assert extract_answer("work \\boxed{7} more work")[0] == "7"
    assert extract_answer("thinking\nAnswer: 3/4")[0] == "3/4"
    assert extract_answer("no marker at all\n12")[0] == "12"
    assert extract_answer("")[0] == ""


def test_a_boxed_answer_wins_over_a_sentence_about_the_answer():
    from litjev.candidates.construct import extract_answer

    assert extract_answer("the answer is probably wrong\n\\boxed{5}")[0] == "5"


def test_an_answer_abandoned_inside_the_thinking_is_not_the_answer():
    """A trace says "maybe 5, no wait"; the answer is what it settled on afterwards."""
    from litjev.candidates.construct import extract_answer

    answer, closed = extract_answer("<think>maybe \\boxed{5}\nno wait</think>\nAnswer: 6")
    assert (answer, closed) == ("6", True)


def test_a_boxed_answer_several_levels_deep_is_matched():
    from litjev.candidates.construct import extract_answer

    answer, _ = extract_answer("work </think> \\boxed{\\frac{\\sqrt{3}}{2}} done")
    assert answer == "\\frac{\\sqrt{3}}{2}"


def test_answered_is_not_the_word_answer():
    from litjev.candidates.construct import extract_answer

    answer, _ = extract_answer("</think>\nI answered it: 12")
    assert answer == "I answered it: 12"


def test_a_response_that_never_stopped_thinking_says_so():
    """Scoring the last line of a truncated thought as the answer credits a trace."""
    from litjev.candidates.construct import extract_answer

    _, closed = extract_answer("<think>still working on it")
    assert closed is False


def test_the_two_ranks_order_the_same_set():
    """Coverage cannot move with the rank: reordering does not change membership."""
    from litjev.candidates.construct import FREQUENCY, LIKELIHOOD

    proposals = [("7", -0.1), ("4", -2.0), ("4", -2.1), ("4", -2.3), ("9", -3.0)]
    sets = {rank: {text for text, _ in candidates_from_beams(proposals, rank=rank)}
            for rank in (LIKELIHOOD, FREQUENCY)}
    assert sets[LIKELIHOOD] == sets[FREQUENCY] == {"7", "4", "9"}
    assert [t for t, _ in candidates_from_beams(proposals, rank=LIKELIHOOD)] != \
           [t for t, _ in candidates_from_beams(proposals, rank=FREQUENCY)]


def test_a_sentence_after_the_answer_is_not_the_answer():
    """The forced continuation appends the marker and lets the model keep writing."""
    from litjev.candidates.construct import extract_answer

    answer, closed = extract_answer(
        "</think>\nAnswer: 12\n\nThis answer uses the formula.")
    assert (answer, closed) == ("12", True)


def test_delimiters_around_an_answer_are_stripped_from_both_ends():
    """A leading dollar survived rstrip and was then wrapped again by the comparison."""
    from litjev.candidates.construct import extract_answer

    assert extract_answer("</think>The answer is $\\frac{1}{2}$.")[0] == "\\frac{1}{2}"


def test_a_sentence_continuing_on_the_same_line_is_cut():
    """The newline in the earlier test was doing the work; a model needs no newline."""
    from litjev.candidates.construct import extract_answer

    assert extract_answer("</think>Answer: 12. This answer uses the formula")[0] == "12"
    assert extract_answer("</think>The answer is 12 because x=3.")[0] == "12"


def test_the_cut_leaves_answers_that_contain_punctuation_alone():
    """A decimal has a full stop and a coordinate pair has a comma."""
    from litjev.candidates.construct import extract_answer

    assert extract_answer("</think>Answer: 3.14")[0] == "3.14"
    assert extract_answer("</think>Answer: (1, 2)")[0] == "(1, 2)"
    assert extract_answer("</think>Answer: 1, 2, 3")[0] == "1, 2, 3"


def test_the_module_defines_each_name_once():
    """Twice now a redefinition has silently won over the definition above it.

    Both times the tests passed, because the two definitions agreed at the
    moment they were written and diverged afterwards. This fails on the
    duplication rather than on the divergence.
    """
    import ast
    import collections
    import pathlib

    import litjev.candidates.construct as module

    tree = ast.parse(pathlib.Path(module.__file__).read_text())
    names = [node.name for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    names += [target.id for node in tree.body if isinstance(node, ast.Assign)
              for target in node.targets if isinstance(target, ast.Name)]
    repeated = [name for name, count in collections.Counter(names).items() if count > 1]
    assert not repeated, f"defined more than once: {repeated}"


def test_a_cut_is_refused_inside_braces():
    """MATH-500 answers are often \\text{...} and the words inside are prose."""
    from litjev.candidates.construct import balanced, extract_answer

    answer, _ = extract_answer("</think>Answer: \\text{Sunday as well}")
    assert answer == "\\text{Sunday as well}"
    assert balanced(answer)


def test_a_cut_after_a_closing_brace_still_fires():
    from litjev.candidates.construct import extract_answer

    assert extract_answer(
        "</think>Answer: \\text{no solution}. This follows.")[0] == "\\text{no solution}"


def test_a_decimal_followed_by_a_lowercase_word_is_not_a_sentence_break():
    """IGNORECASE made "before a capital" mean "before any letter"."""
    from litjev.candidates.construct import extract_answer

    assert extract_answer("</think>Answer: 3. x is irrelevant")[0] == "3. x is irrelevant"


def test_unbalanced_braces_are_detectable():
    from litjev.candidates.construct import balanced

    assert balanced("\\frac{1}{2}")
    assert not balanced("\\text{Sunday")
    assert not balanced("}{")
