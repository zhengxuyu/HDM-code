"""Reading a schema-constrained proposal without JSON eating the answer.

Every backslash sequence JSON spells is one LaTeX spells too, and `json.loads`
is right about all of them and wrong for this. `{"answer": "\\frac{1}{2}"}` is
valid JSON whose value is formfeed followed by `rac{1}{2}`; `\\boxed`, `\\theta`,
`\\neq` and `\\right` go the same way, and none of them errors. On MATH-500,
where the gold answers are LaTeX, every one of those would have been quietly
wrong in the coverage and accuracy columns.

So the field is found and unescaped for the two sequences JSON needs and LaTeX
does not use alone, rather than parsed.
"""
import json

import pytest

from inference.sglang_backend import PROPOSAL_SCHEMA, answer_of

# The five code-reviewer found, plus the shapes around them.
LATEX = [r"\frac{1}{2}", r"\boxed{5}", r"\theta", r"\neq", r"\right)",
         r"\begin{pmatrix}1\\2\end{pmatrix}", r"2\sqrt{3}"]


def wire(answer):
    """What the engine emits: the answer between quotes, backslashes as written."""
    return '{"answer": "' + answer + '"}'


@pytest.mark.parametrize("answer", LATEX)
def test_latex_survives_the_read(answer):
    assert answer_of(wire(answer)) == answer


@pytest.mark.parametrize("answer", LATEX)
def test_json_loads_would_have_corrupted_it(answer):
    """The reason this file exists, asserted so it cannot be quietly undone."""
    try:
        through_json = json.loads(wire(answer))["answer"]
    except json.JSONDecodeError:
        return  # rejected outright is survivable; silently changed is not
    assert through_json != answer, (
        f"{answer!r} is unharmed by json.loads, so this case no longer guards anything")


@pytest.mark.parametrize("answer", ["42", "x + 1", "-3/4", "0.75", "B", "2"])
def test_plain_answers_come_back_unchanged(answer):
    assert answer_of(wire(answer)) == answer


def test_a_backslash_is_taken_as_the_model_wrote_it():
    r"""No unescaping, because `\\` is an escaped backslash to JSON and a line break
    to LaTeX -- the same two characters -- so no reader is right about both. The
    constraint is a regex with no escaping layer under it, and the prompt tells the
    model to write the answer literally."""
    assert answer_of(wire(r"\frac{1}{2}")) == r"\frac{1}{2}"
    assert answer_of(wire(r"\begin{pmatrix}1\\2\end{pmatrix}")) == \
        r"\begin{pmatrix}1\\2\end{pmatrix}"


def test_an_answer_cannot_contain_a_quote():
    """Which is the price of reading raw, and no MATH-500 answer pays it."""
    from inference.sglang_backend import PROPOSAL_REGEX

    assert '[^"' in PROPOSAL_REGEX, "the constraint must exclude the delimiter"


def test_a_response_that_is_not_the_object_asked_for_is_refused():
    """It means the constraint did not apply, and taking the text anyway would put
    an unconstrained run inside a record that says it was constrained."""
    for text in ("42", "", '{"result": "42"}', "Answer: 42"):
        with pytest.raises(ValueError, match="schema was not enforced"):
            answer_of(text)


def test_the_schema_fixes_the_shape_and_not_the_content():
    """The comment that said otherwise was wrong: `{"answer": "B"}` satisfies it,
    and no pattern could rule that out, since 2, x and C are real MATH answers.
    What keeps proposals index-free is the option-free prompt and looks_like_codes."""
    schema = json.loads(PROPOSAL_SCHEMA)
    assert schema["properties"]["answer"] == {"type": "string"}
    assert "pattern" not in schema["properties"]["answer"]
    assert answer_of(wire("B")) == "B", "the schema does not exclude a letter"

    from hde.candidates.proposing import looks_like_codes

    assert looks_like_codes([("B", -0.1), ("C", -0.2), ("A", -0.3)]), (
        "the guard that does exclude them must still be the one doing it")


def test_an_answer_is_compared_as_content_and_never_as_an_index():
    """A proposal is matched against the gold answer by value. Nothing maps it
    back to a position, which is what would make the letter meaningful again."""
    from hde.candidates.construct import correct

    # symbolic=False is a string comparison, which is the point: nothing here
    # turns "B" into a position and compares positions.
    assert correct("B", "B", symbolic=False)
    assert not correct("B", "1", symbolic=False), "a letter is not an index into anything"
    assert not correct("2", "C", symbolic=False), "nor is a digit"


# ---------------------------------------------------------------- the code rotation


def test_rotating_the_codes_moves_the_letters_and_not_the_options():
    """Which is the whole of it: the options keep their order, the readout keeps
    reading logits in slot order, and only the letter beside each option moves.
    So a preference that follows the letter moves with it and one that follows the
    slot stays put -- the two readings every run so far has confounded."""
    from hde.backends.slots import rotated

    codes = list("ABCDEFGHIJ")
    assert rotated(codes, 0) == codes, "zero is every run to date"
    assert rotated(codes, 1)[0] == "B" and rotated(codes, 1)[1] == "C"
    assert rotated(codes, 3)[0] == "D"
    assert sorted(rotated(codes, 4)) == codes, "a rotation, so no code is lost or repeated"
    assert rotated(list("ABC"), 1) == ["B", "C", "A"], "it wraps at the width"
    assert rotated(codes, 10) == codes and rotated(codes, 12) == rotated(codes, 2)


def test_the_offset_reaches_the_slots_from_the_settings():
    """Between the flag and the prompt there are four hops, and a run that sets the
    offset and gets slot order back would look exactly like 'position, not letter'."""
    import inspect

    from hde.backends.backend import ModelSettings, TransformersScorer
    from hde.backends.slots import compile_prefix_slots, compile_slots

    assert "code_offset" in inspect.signature(ModelSettings).parameters
    assert "code_offset" in inspect.signature(TransformersScorer.__init__).parameters
    for carrier in (compile_slots, compile_prefix_slots):
        assert "code_offset" in inspect.signature(carrier).parameters

    scorer = TransformersScorer.__new__(TransformersScorer)
    scorer.code_offset = 2
    assert scorer.code_offset == 2
