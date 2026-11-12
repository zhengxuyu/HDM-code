"""A question that arrives without its answers, expressed in the schema.

Which path a question takes -- read out the options it came with, or build some
first -- used to be decided by which script was run. That is a fact about the
operator rather than about the question, and it meant the two paths could not be
reached through one entry point. `criteria=None` puts it in the request.
"""
import pytest
from pydantic import ValidationError

from litjev.data.schema import Choice, DecisionSchema, SystemOneRequest


def test_a_question_may_arrive_without_its_answers():
    question = Choice(instructions="What is 2+2?", criteria=None)
    assert question.criteria is None
    assert not question.enumerated


def test_omitting_the_field_is_not_the_same_as_asking_for_construction():
    """A forgotten field would otherwise mean "generate candidates", which is the
    costly path: one readout reads logits the pass already produced, the other
    runs a proposal pass first. A caller asking for it writes null and means it."""
    with pytest.raises(ValidationError):
        Choice.model_validate({"type": "choice", "instructions": "x"})
    assert Choice.model_validate(
        {"type": "choice", "instructions": "x", "criteria": None}).criteria is None


def test_a_question_that_arrived_with_them_says_so():
    question = Choice(instructions="Pick", criteria={"A": "1", "B": "2"})
    assert question.enumerated
    assert question.choices == ("A", "B")
    assert question.descriptions == ("1", "2")


def test_asking_for_choices_that_are_not_there_says_what_to_do():
    """Rather than returning an empty tuple, which reads as a question with no
    answers instead of one whose answers have not been built."""
    question = Choice(instructions="What is 2+2?", criteria=None)
    with pytest.raises(ValueError, match="no candidate set"):
        question.choices
    with pytest.raises(ValueError, match="no candidate set"):
        question.descriptions


def test_an_empty_mapping_is_still_refused():
    """`None` means "none were offered"; `{}` would mean "a set with nothing in
    it", which is not a question anyone can answer."""
    with pytest.raises(ValidationError):
        Choice(instructions="Pick", criteria={})


def test_it_round_trips_through_a_request():
    request = SystemOneRequest(
        model="litjev", state="s",
        questions={"open": Choice(instructions="What is 2+2?", criteria=None),
                   "closed": Choice(instructions="Pick", criteria={"A": "1", "B": "2"})})
    again = SystemOneRequest.model_validate_json(request.model_dump_json())
    assert again.questions["open"].criteria is None
    assert again.questions["closed"].criteria == {"A": "1", "B": "2"}
    assert [q.enumerated for q in again.questions.values()] == [False, True]


def test_a_schema_can_hold_both_kinds():
    schema = DecisionSchema({"open": Choice(instructions="q", criteria=None),
                             "closed": Choice(instructions="q", criteria={"A": "1", "B": "2"})})
    assert [schema[name].enumerated for name in schema.names] == [False, True]
