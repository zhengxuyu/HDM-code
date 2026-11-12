"""Gold labels stay outside model requests."""

from typing import Literal

from pydantic import BaseModel

from litjev.data.schema import Choice, Noul, Score, State, SystemOneRequest

DATASET_ID = "TIGER-Lab/MMLU-Pro"
LABELS = "ABCDEFGHIJ"


class McqQuestion(BaseModel):
    """Dataset preparation record, not an HTTP request schema.

    `state` and `question_type` are for benchmarks whose items are typed and carry
    their own context, JevBench being the first. They default to the shape every
    earlier loader produces -- no state of its own, and a `choice` -- so one record
    type covers both rather than a second one being added beside it.

    A `score` question's levels are kept here as `{"0": ..., "1": ...}` and a
    `noul`'s as `{"false": ..., "true": ...}`, which is the order their `choices`
    property returns, so `options` stays one shape and the digest stays one rule.
    """

    question_id: str
    prompt: str
    options: dict[str, str]
    # The same type the request schema takes, because 35 of JevBench's 231 items
    # carry a JSON object as their context rather than a string, and flattening those
    # to text would be this repository editing the benchmark's prompts.
    state: State | None = None
    question_type: Literal["choice", "score", "noul"] = "choice"


def convert_rows(rows):
    questions, labels, skipped = [], {}, []
    for row in rows:
        question_id = str(row.get("question_id", ""))
        options = row.get("options")
        answer = row.get("answer_index")
        valid = (
            isinstance(options, list)
            and 2 <= len(options) <= 10
            and all(isinstance(text, str) and text.strip() for text in options)
            and type(answer) is int
            and 0 <= answer < len(options)
            and isinstance(row.get("question"), str)
            and row["question"].strip()
        )
        if not valid:
            skipped.append({"id": question_id, "reason": "invalid MCQ or missing label"})
            continue
        if question_id in labels:
            raise ValueError(f"Duplicate question ID: {question_id}")
        questions.append(
            McqQuestion(
                question_id=question_id, prompt=row["question"], options=dict(zip(LABELS, options))
            )
        )
        labels[question_id] = LABELS[answer]
    return questions, labels, skipped


def typed_question(record):
    """The request-schema question a dataset record describes.

    One place, because the mapping from a record to a `Choice`, `Score` or `Noul`
    is the only thing that decides what the readout reads, and two copies of it
    would be two benchmarks measuring different things under one name.
    """
    if record.question_type == "choice":
        return Choice(instructions=record.prompt, criteria=dict(record.options))
    if record.question_type == "score":
        levels = [record.options[key] for key in sorted(record.options, key=int)]
        return Score(instructions=record.prompt, criteria=levels)
    if record.question_type == "noul":
        return Noul(instructions=record.prompt,
                    criteria={key: record.options[key] for key in ("false", "true")
                              if key in record.options} or None)
    raise ValueError(f"unknown question type {record.question_type!r}")


def own_state_requests(questions):
    """One request per question, each with the state the question came with.

    `ten_question_batches` pairs ten questions with one synthetic state, which is
    what MMLU-Pro and the other multiple-choice sets need: they have no context of
    their own, so the state is a sentence this repository wrote. A JevBench item is
    a context and a question about it, so its state cannot be shared and batching
    ten of them together would be putting ten unrelated documents in one prompt.

    The cost of that is real and is not hidden: the shared-prefix saving the fast
    mode is built around does not apply here, and a `seconds` column from this
    dataset is a batch-of-one column. Padding is absent for the same reason -- a
    padded question would carry another item's context.
    """
    for record in questions:
        yield (
            SystemOneRequest(
                model="litjev",
                state=record.state if record.state is not None else "",
                questions={record.question_id: typed_question(record)},
            ),
            [record.question_id],
        )


def ten_question_batches(questions, size=10):
    """Pad the final batch by repeating real questions with distinct request IDs."""
    if size <= 0:
        raise ValueError("Batch size must be positive")
    for start in range(0, len(questions), size):
        group = questions[start : start + size]
        valid_ids = [q.question_id for q in group]
        while len(group) < size:
            padding_id = f"__padding_{start}_{len(group)}"
            if any(q.question_id == padding_id for q in questions):
                raise ValueError("Question ID collides with padding namespace")
            group.append(group[0].model_copy(update={"question_id": padding_id}))
        yield (
            SystemOneRequest(
                model="litjev",
                state="Answer each question using its listed options.",
                questions={
                    q.question_id: Choice(instructions=q.prompt, criteria=q.options) for q in group
                },
            ),
            valid_ids,
        )
