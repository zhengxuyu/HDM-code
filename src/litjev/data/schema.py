"""Canonical Jev questions, shared by HTTP, Python and prompt compilation."""

from collections.abc import Iterator, Mapping
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

MAX_CHOICES = 255
Content = str | dict[str, JsonValue] | list[JsonValue] | None
State = str | dict[str, JsonValue] | list[JsonValue]


class QuestionBase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)
    instructions: Content = None

    @property
    def enumerated(self):
        """Whether this question arrived with its answers.

        True for every type but `choice`: a `noul` is answered true or false and a
        `score` by one of its levels, and both sets are the type rather than the
        caller's to supply. Only a `choice` can arrive without one.
        """
        return True


class Choice(QuestionBase):
    """A question answered by picking one of a set, which the caller may not have.

    `criteria` is `None` when the caller has no candidates to offer. That is the
    request type this system is for as much as the enumerated one -- a maths
    answer is not drawn from a set anybody can write down -- and the fast mode
    reaches such a question by proposing candidates for itself.

    `None` rather than an empty mapping, and rather than a second question type,
    because it is the one thing that decides which path a question takes and it
    should be readable off the request. It used to be decided by which script was
    run, which is a fact about the operator and not about the question.

    Required, and nullable: a caller asking for construction writes
    `"criteria": null` and means it. With a default of `None` a forgotten field
    would have meant the same thing silently, and the two paths are not
    comparable in cost -- one reads logits the pass already produced, the other
    generates candidates first.
    """

    type: Literal["choice"] = "choice"
    criteria: dict[str, Content] | None = Field(min_length=1, max_length=MAX_CHOICES)

    @property
    def enumerated(self):
        """Whether this question arrived with its answers, or needs them built."""
        return self.criteria is not None

    @property
    def choices(self):
        if self.criteria is None:
            raise ValueError(
                "this question has no candidate set, so it has no choices to read out "
                "yet. Construct them first -- the engine does it on the way past -- or "
                "ask `enumerated` before assuming there are any")
        return tuple(self.criteria)

    @property
    def descriptions(self):
        if self.criteria is None:
            raise ValueError(
                "this question has no candidate set, so it has no option text yet")
        return tuple(self.criteria.values())


class Score(QuestionBase):
    type: Literal["score"] = "score"
    criteria: list[Content] = Field(min_length=2, max_length=10)

    @property
    def choices(self):
        return tuple(str(i) for i in range(len(self.criteria)))

    @property
    def descriptions(self):
        return tuple(self.criteria)


class Noul(QuestionBase):
    type: Literal["noul"] = "noul"
    criteria: dict[Literal["true", "false"], Content] | None = None

    @property
    def choices(self):
        return ("false", "true")

    @property
    def descriptions(self):
        return tuple((self.criteria or {}).get(key) for key in self.choices)


Question = Annotated[Choice | Score | Noul, Field(discriminator="type")]
QUESTION_ADAPTER = TypeAdapter(Question)


class DecisionSchema(Mapping[str, Question]):
    """Ids are bookkeeping only and must never enter model prompts."""

    def __init__(self, questions: Mapping[str, Question]):
        if not questions:
            raise ValueError("Questions must contain at least one question")
        self._questions = {
            key: QUESTION_ADAPTER.validate_python(value).model_copy(deep=True)
            for key, value in questions.items()
        }
        if any(not isinstance(key, str) for key in self._questions):
            raise ValueError("Question IDs must be strings")

    @classmethod
    def from_mapping(cls, questions):
        return cls(questions)

    def to_mapping(self):
        return {key: value.model_dump() for key, value in self.items()}

    @property
    def names(self):
        return tuple(self._questions)

    def __getitem__(self, key):
        return self._questions[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._questions)

    def __len__(self):
        return len(self._questions)


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    model: str = Field(min_length=1)
    state: State
    questions: dict[str, Question] = Field(min_length=1)

    def to_schema(self):
        return DecisionSchema(self.questions)
